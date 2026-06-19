from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import secrets
import string
import time
import wave
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .api_generator import (
    GenerationInput as ApiGenerationInput,
    KieAIConfigError,
    KieAIResponseError,
    generate_song as generate_api_song,
    generate_vocal_song,
    get_callback_path,
    get_default_trim_duration_seconds,
    trim_audio,
    validate_kieai_config,
)
from .music_generator import (
    TARGET_SECONDS,
    build_prompt as build_music_prompt,
    generate_music,
    load_model,
)
from .auth import AuthError, login as auth_login, signup as auth_signup, user_email_for_token, user_name_for_email, user_name_for_token
from .composer import compose_song
from .transcriber import TranscriptionError, transcribe_audio
from .agents.lyrics_agent import generate_lyrics as generate_agent_lyrics
from .agents.copyright_agent.main import check_copyright
from .agents.copyright_agent.models.request_model import CopyrightCheckRequest
from .ace_step_generator import generate_with_ace_step
from .sheet_generator.music_sheet import _choose_chords, _choose_key, generate_music_sheet_pdf


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
GENERATED_DIR = BASE_DIR / "generated"
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = PROJECT_DIR / "data"
USER_DATA_DIR = DATA_DIR / "users"
USER_STATE_FILE = DATA_DIR / "library_state.json"
DEFAULT_HISTORY_FILE = BASE_DIR / "song_history.json"
HISTORY_FILE = DEFAULT_HISTORY_FILE
CALLBACK_FILE = BASE_DIR / "kiai_callbacks.json"
LEGACY_CALLBACK_FILE = BASE_DIR / "kieai_callbacks.json"
DEFAULT_USER_EMAIL = "test@echo.com"
DEFAULT_USER_ID = hashlib.sha256(DEFAULT_USER_EMAIL.encode("utf-8")).hexdigest()[:12]

load_dotenv(PROJECT_DIR / ".env")
load_dotenv(BASE_DIR / ".env")

DEFAULT_GENERATION_ESTIMATE_SECONDS = 120
ROTATING_STATUS_MESSAGES = [
    "Analyzing mood",
    "Composing melody",
    "Creating atmosphere",
    "Writing musical ideas",
    "Adding emotional texture",
    "Finalizing audio",
]
LYRICS_FALLBACK = "Lyrics generation is unavailable for this song, but the music was generated successfully."
FIREBASE_ENV_MAP = {
    "apiKey": "FIREBASE_API_KEY",
    "authDomain": "FIREBASE_AUTH_DOMAIN",
    "projectId": "FIREBASE_PROJECT_ID",
    "storageBucket": "FIREBASE_STORAGE_BUCKET",
    "messagingSenderId": "FIREBASE_MESSAGING_SENDER_ID",
    "appId": "FIREBASE_APP_ID",
    "measurementId": "FIREBASE_MEASUREMENT_ID",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [Echo Echo] %(message)s",
)
logger = logging.getLogger(__name__)

model = None
processor = None
model_status = "loading"
model_error = ""

model_lock = Lock()
generation_lock = Lock()
generation_status: dict[str, Any] = {
    "stage": "Idle",
    "progress": 0,
    "started_at": None,
    "elapsed_seconds": 0,
    "estimated_remaining_seconds": 0,
    "active": False,
    "estimated_total_seconds": DEFAULT_GENERATION_ESTIMATE_SECONDS,
}


class GenerateRequest(BaseModel):
    prompt: str = ""
    mode: str | None = None
    duration: int = Field(default=TARGET_SECONDS, ge=1, le=600)
    fast: bool = False

    mood: str | None = None
    theme: str | None = None
    style: str | None = None
    moods: list[str] = Field(default_factory=list)
    genres: list[str] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)
    instruments: list[str] = Field(default_factory=list)
    tempo: int = Field(default=90, ge=20, le=300)
    complexity: Literal["Simple", "Moderate", "Rich"] = "Moderate"
    energy: int = Field(default=4, ge=1, le=10)
    custom_prompt: str = ""
    lyrics: str = ""


class LoginRequest(BaseModel):
    email: str
    password: str


class SignupRequest(BaseModel):
    name: str = ""
    email: str
    password: str


class ComposeRequest(BaseModel):
    mood: str = ""
    genre: str = ""
    theme: str = ""
    instrument: str = ""
    style: str = ""
    tempoFeel: str = ""
    bpm: int = Field(default=90, ge=40, le=220)
    prompt: str = ""


class KieVocalRequest(BaseModel):
    mood: str = "Energetic"
    genre: str = "Pop"
    theme: str = ""
    instrument: str = ""
    style: str = ""
    tempoFeel: str = ""
    bpm: int = Field(default=90, ge=40, le=220)
    promptText: str = ""


class TrimRequest(BaseModel):
    duration_seconds: int | None = Field(default=None, ge=1, le=600)
    force: bool = False


class TrackUpdateRequest(BaseModel):
    favorite: bool | None = None


class CopyrightCheckApiRequest(BaseModel):
    song_id: str | None = None
    lyrics: str = ""
    title: str = ""
    prompt: str = ""
    description: str = ""


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    ensure_files()
    try:
        validate_kieai_config()
    except Exception as exc:
        logger.warning("KieAI config invalid — API generation disabled: %s", exc)
    Thread(target=warm_model, daemon=True).start()
    yield


app = FastAPI(title="Echo Echo", version="2.0.0", lifespan=lifespan, docs_url="/api/docs", redoc_url="/api/redoc")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def ensure_files() -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not USER_STATE_FILE.exists():
        USER_STATE_FILE.write_text(json.dumps({"users": {}, "songs": {}}, indent=2), encoding="utf-8")
    default_user_file = user_songs_path(DEFAULT_USER_ID)
    if not default_user_file.exists():
        legacy_history = []
        if HISTORY_FILE == DEFAULT_HISTORY_FILE and HISTORY_FILE.exists():
            legacy_history = read_json_list_no_ensure(HISTORY_FILE)
        default_user_file.write_text(json.dumps(legacy_history, indent=2), encoding="utf-8")
    if HISTORY_FILE != DEFAULT_HISTORY_FILE and not HISTORY_FILE.exists():
        HISTORY_FILE.write_text("[]", encoding="utf-8")
    if LEGACY_CALLBACK_FILE.exists() and not CALLBACK_FILE.exists():
        LEGACY_CALLBACK_FILE.replace(CALLBACK_FILE)
    if not CALLBACK_FILE.exists():
        CALLBACK_FILE.write_text("[]", encoding="utf-8")

def read_json_list_no_ensure(path: Path) -> list[dict[str, Any]]:
    try:
        raw = path.read_text(encoding="utf-8").strip()
        data = json.loads(raw or "[]")
        return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        logger.exception("%s is not valid JSON", path.name)
        return []


def read_json_list(path: Path) -> list[dict[str, Any]]:
    ensure_files()
    return read_json_list_no_ensure(path)


def write_json_list(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_user_state() -> dict[str, Any]:
    ensure_files()
    try:
        data = json.loads(USER_STATE_FILE.read_text(encoding="utf-8") or "{}")
    except json.JSONDecodeError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    users = data.get("users") if isinstance(data.get("users"), dict) else {}
    songs = data.get("songs") if isinstance(data.get("songs"), dict) else {}
    return {"users": users, "songs": songs}


def write_user_state(state: dict[str, Any]) -> None:
    USER_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    USER_STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def sanitize_user_id(user_id: str) -> str:
    cleaned = "".join(ch for ch in user_id.strip().lower() if ch.isalnum() or ch in {"_", "-"})
    return cleaned or DEFAULT_USER_ID


def user_id_from_email(email: str) -> str:
    normalized = email.strip().lower() or DEFAULT_USER_EMAIL
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def user_songs_path(user_id: str) -> Path:
    if HISTORY_FILE != DEFAULT_HISTORY_FILE and sanitize_user_id(user_id) == DEFAULT_USER_ID:
        return HISTORY_FILE
    return USER_DATA_DIR / f"user_{sanitize_user_id(user_id)}.json"


def load_user_songs(user_id: str) -> list[dict[str, Any]]:
    path = user_songs_path(user_id)
    if not path.exists():
        write_json_list(path, [])
    try:
        raw = json.loads(path.read_text(encoding="utf-8") or "[]")
    except (FileNotFoundError, json.JSONDecodeError):
        raw = []
    if isinstance(raw, dict):
        tracks = raw.get("tracks")
        return tracks if isinstance(tracks, list) else []
    return raw if isinstance(raw, list) else []


def write_user_songs(user_id: str, songs: list[dict[str, Any]]) -> None:
    write_json_list(user_songs_path(user_id), songs)


def generate_song_id(existing_ids: set[str] | None = None) -> str:
    existing = existing_ids or set()
    alphabet = string.ascii_uppercase
    for _ in range(1000):
        song_id = "".join(secrets.choice(alphabet) for _ in range(4))
        if song_id not in existing:
            return song_id
    raise RuntimeError("Could not create a unique song ID.")


def library_song_summary(record: dict[str, Any], owner_email: str = "") -> dict[str, Any]:
    song = song_detail_payload(record, owner_email=owner_email)
    return {
        "songId": song.get("songId"),
        "song_id": song.get("song_id"),
        "id": song.get("id"),
        "ownerEmail": owner_email,
        "title": song.get("title"),
        "displayTitle": song.get("displayTitle"),
        "provider": song.get("provider"),
        "mode": song.get("mode"),
        "audioUrl": song.get("audioUrl"),
        "audio_url": song.get("audio_url"),
        "downloadAudioUrl": song.get("downloadAudioUrl"),
        "duration": song.get("duration"),
        "durationSeconds": song.get("durationSeconds") or song.get("duration"),
        "duration_seconds": song.get("duration_seconds") or song.get("duration"),
        "mood": song.get("mood"),
        "genre": song.get("genre"),
        "bpm": song.get("bpm"),
        "key": song.get("key"),
        "chords": song.get("chords") or [],
        "musicSheet": song.get("musicSheet") or [],
        "lyrics": song.get("lyrics"),
        "lyricsAvailable": song.get("lyricsAvailable"),
        "sheetAvailable": song.get("sheetAvailable"),
        "lyricsSheetDownloadUrl": song.get("lyricsSheetDownloadUrl"),
        "createdAt": song.get("created_at"),
        "created_at": song.get("created_at"),
    }


def remember_user_song(user_email: str, user_name: str, record: dict[str, Any]) -> None:
    email = user_email.strip().lower()
    if not email:
        return
    song_id = record_code(record)
    if not song_id:
        return
    state = read_user_state()
    users = state["users"]
    songs = state["songs"]
    user_entry = users.setdefault(email, {"name": user_name or email.split("@")[0], "email": email, "songIds": []})
    user_entry["name"] = user_name or user_entry.get("name") or email.split("@")[0]
    user_entry["email"] = email
    song_ids = user_entry.setdefault("songIds", [])
    if song_id not in song_ids:
        song_ids.append(song_id)
    songs[song_id] = song_detail_payload(record, owner_email=email)
    write_user_state(state)


def remember_user_profile(user_email: str, user_name: str, song_ids: list[str] | None = None) -> None:
    email = user_email.strip().lower()
    if not email:
        return
    state = read_user_state()
    user_entry = state["users"].setdefault(email, {"name": user_name or email.split("@")[0], "email": email, "songIds": []})
    user_entry["name"] = user_name or user_entry.get("name") or email.split("@")[0]
    user_entry["email"] = email
    if song_ids is not None:
        existing = list(user_entry.get("songIds") or [])
        for song_id in song_ids:
            if song_id and song_id not in existing:
                existing.append(song_id)
        user_entry["songIds"] = existing
    write_user_state(state)


def save_user_song(
    user_id: str,
    song_data: dict[str, Any],
    *,
    user_email: str = "",
    user_name: str = "",
) -> dict[str, Any]:
    attach_sheet(song_data)
    songs = load_user_songs(user_id)
    existing_ids = {record_code(song) for song in songs}
    song_id = record_code(song_data)
    if not song_id or song_id in existing_ids:
        song_id = generate_song_id(existing_ids)
        song_data["song_id"] = song_id
        song_data["code"] = song_id
        song_data["id"] = song_id

    song_data.setdefault("song_id", song_id)
    song_data.setdefault("code", song_id)
    song_data.setdefault("id", song_id)
    song_data.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    song_data.setdefault("title", record_title(song_data))
    song_data.setdefault("genre", song_data.get("style") or "")
    song_data.setdefault("bpm", song_data.get("tempo") or 90)
    song_data.setdefault("duration", 0)
    song_data.setdefault("durationSeconds", song_data.get("duration"))
    song_data.setdefault("duration_seconds", song_data.get("duration"))
    song_data.setdefault("audio_url", "")
    song_data.setdefault("lyrics", "")
    song_data.setdefault("favorite", bool(song_data.get("favorite") or song_data.get("fav") or False))
    song_data["fav"] = bool(song_data.get("favorite"))
    if user_email:
        song_data["ownerEmail"] = user_email.strip().lower()
    songs.append(song_data)
    write_user_songs(user_id, songs)
    remember_user_song(user_email, user_name, song_data)
    return song_data


def auth_token_from_request(request: Request | None) -> str:
    if request is None:
        return ""
    value = request.headers.get("authorization", "")
    if value.lower().startswith("bearer "):
        return value.split(" ", 1)[1].strip()
    return ""


def current_user_id(request: Request | None) -> str:
    email = user_email_for_token(auth_token_from_request(request))
    if email:
        return user_id_from_email(email)
    return DEFAULT_USER_ID


def current_user_email(request: Request | None) -> str:
    return user_email_for_token(auth_token_from_request(request)) or DEFAULT_USER_EMAIL


def current_user_name(request: Request | None) -> str:
    token_name = user_name_for_token(auth_token_from_request(request))
    return token_name or user_name_for_email(current_user_email(request))


def iter_user_song_files() -> list[Path]:
    ensure_files()
    files = list(USER_DATA_DIR.glob("user_*.json"))
    if HISTORY_FILE != DEFAULT_HISTORY_FILE and HISTORY_FILE.exists():
        files.append(HISTORY_FILE)
    return files


def read_history(user_id: str = DEFAULT_USER_ID) -> list[dict[str, Any]]:
    return load_user_songs(user_id)


def write_history(history: list[dict[str, Any]], user_id: str = DEFAULT_USER_ID) -> None:
    write_user_songs(user_id, history)


def append_history(record: dict[str, Any], user_id: str = DEFAULT_USER_ID) -> None:
    save_user_song(user_id, record)


ensure_files()


def read_callbacks() -> list[dict[str, Any]]:
    return read_json_list(CALLBACK_FILE)


def write_callbacks(callbacks: list[dict[str, Any]]) -> None:
    write_json_list(CALLBACK_FILE, callbacks)


def first_or_join(values: list[str], fallback: str) -> str:
    cleaned = [value.strip() for value in values if value and value.strip()]
    return " + ".join(cleaned) if cleaned else fallback


def model_to_dict(model_instance: Any) -> dict[str, Any]:
    if hasattr(model_instance, "model_dump"):
        return model_instance.model_dump()
    if hasattr(model_instance, "dict"):
        return model_instance.dict()
    return dict(model_instance)


def lyrics_context(request: GenerateRequest, prompt: str) -> dict[str, Any]:
    duration = int(request.duration or TARGET_SECONDS)
    max_lines = 8 if duration <= 30 else 16 if duration <= 60 else 24 if duration <= 90 else 48
    return {
        "title": request.title or "Generated Track",
        "prompt": prompt,
        "mood": first_or_join(request.moods or ([request.mood] if request.mood else []), ""),
        "theme": first_or_join(request.themes or ([request.theme] if request.theme else []), ""),
        "style": first_or_join(request.genres or ([request.style] if request.style else []), ""),
        "genre": first_or_join(request.genres or ([request.style] if request.style else []), ""),
        "tempo": request.tempo,
        "bpm": request.tempo,
        "durationSeconds": duration,
        "maxLines": max_lines,
        "complexity": request.complexity,
        "energy": request.energy,
        "instruments": request.instruments,
    }


def generate_lyrics_section(request: GenerateRequest, prompt: str) -> tuple[dict[str, str], bool]:
    try:
        lyrics = generate_agent_lyrics(lyrics_context(request, prompt))
        return {
            "text": lyrics.get("text", "").strip() or LYRICS_FALLBACK,
            "structure": lyrics.get("structure", "verse/chorus"),
        }, True
    except Exception as exc:
        logger.warning("Lyrics generation unavailable: %s", exc)
        return {
            "text": f"{LYRICS_FALLBACK} Reason: {exc}",
            "structure": "unavailable",
        }, False


def copyright_unavailable(message: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "similarity_score": None,
        "notes": message,
        "matches": [],
    }


def generate_copyright_section(lyrics_text: str, lyrics_available: bool) -> dict[str, Any]:
    if not lyrics_available:
        return copyright_unavailable("Lyrics generation failed; copyright check was skipped.")
    try:
        response = check_copyright(CopyrightCheckRequest(lyrics=lyrics_text))
        result = model_to_dict(response)
        return normalize_copyright_response(result)
    except Exception as exc:
        logger.warning("Copyright check unavailable: %s", exc)
        return copyright_unavailable(str(exc))


def normalize_copyright_response(result: dict[str, Any]) -> dict[str, Any]:
    status = "safe" if result.get("safe") else "risky"
    notes = result.get("recommendation") or result.get("details") or ""
    matches = []
    if result.get("song_title") or result.get("artist"):
        matches.append(
            {
                "title": result.get("song_title", ""),
                "artist": result.get("artist", ""),
                "risk": result.get("risk", ""),
                "copyright_status": result.get("copyright_status", ""),
            }
        )
    return {
        "status": status,
        "similarity_score": result.get("confidence"),
        "notes": notes,
        "matches": matches,
        "risk": result.get("risk"),
        "copyright_status": result.get("copyright_status"),
        "song_title": result.get("song_title", ""),
        "artist": result.get("artist", ""),
        "details": result.get("details", ""),
    }


def copyright_check_text(payload: CopyrightCheckApiRequest, record: dict[str, Any] | None = None) -> str:
    lyrics_text = payload.lyrics.strip()
    if lyrics_text:
        return lyrics_text

    record = record or {}
    if isinstance(record.get("lyrics"), str):
        saved_lyrics_text = str(record.get("lyrics") or "").strip()
        if saved_lyrics_text:
            return saved_lyrics_text
    record_lyrics = record.get("lyrics") if isinstance(record.get("lyrics"), dict) else {}
    saved_lyrics = str(record_lyrics.get("text") or "").strip()
    if saved_lyrics and record_lyrics.get("structure") != "unavailable":
        return saved_lyrics

    parts = [
        payload.title,
        payload.prompt,
        payload.description,
        str(record.get("prompt") or ""),
        str(record.get("mood") or ""),
        str(record.get("theme") or ""),
        str(record.get("style") or ""),
    ]
    return " ".join(part.strip() for part in parts if part and part.strip())


def add_agent_sections(record: dict[str, Any], request: GenerateRequest, prompt: str) -> None:
    update_generation_status("Writing lyrics", 93)
    lyrics, lyrics_available = generate_lyrics_section(request, prompt)
    update_generation_status("Checking copyright", 96)
    record["lyrics"] = lyrics
    record["copyright"] = generate_copyright_section(lyrics["text"], lyrics_available)


def provider_label(mode: str | None) -> str:
    labels = {
        "api": "KIE.AI",
        "kie-vocal": "KIE.AI",
        "ace-step": "Ace AI",
        "musicgen": "MusicGen",
    }
    return labels.get((mode or "musicgen").strip().lower(), mode or "MusicGen")


def has_real_lyrics(record: dict[str, Any]) -> bool:
    lyrics = record.get("lyrics")
    if isinstance(lyrics, dict):
        text = str(lyrics.get("text") or "").strip()
        structure = str(lyrics.get("structure") or "").strip().lower()
    else:
        text = str(lyrics or "").strip()
        structure = ""
    if not text:
        return False
    if structure == "unavailable":
        return False
    if text.startswith(LYRICS_FALLBACK):
        return False
    return True


def one_verse_lyrics(record: dict[str, Any]) -> dict[str, str] | None:
    if not has_real_lyrics(record):
        return None
    lyrics = record.get("lyrics")
    text = str(lyrics.get("text") if isinstance(lyrics, dict) else lyrics or "").strip()
    lines: list[str] = []
    collecting = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if collecting and lines:
                break
            continue
        if line.startswith("[") and line.endswith("]"):
            if collecting and lines:
                break
            collecting = "verse" in line.lower() or not lines
            continue
        if collecting or not lines:
            lines.append(line)
        if len(lines) >= 6:
            break
    verse = "\n".join(lines).strip() or text
    return {"label": "Verse", "text": verse}


def clean_short_verse(text: str) -> dict[str, str] | None:
    record = {"lyrics": {"text": text, "structure": "real"}}
    return one_verse_lyrics(record)


def maybe_transcribe_lyrics(audio_path: Path, provider: str, vocals_enabled: bool) -> dict[str, str] | None:
    if not vocals_enabled or not audio_path.exists():
        return None
    try:
        content_type = "audio/mpeg" if audio_path.suffix.lower() == ".mp3" else "audio/wav"
        text = transcribe_audio(audio_path.read_bytes(), audio_path.name, content_type)
        lyrics = clean_short_verse(text)
        if lyrics:
            logger.info("Transcribed lyrics for %s using %s", audio_path.name, provider)
        return lyrics
    except Exception as exc:
        logger.warning("Lyric transcription skipped for %s: %s", audio_path.name, exc)
        return None


def build_music_sheet_lines(record: dict[str, Any], chords: list[str]) -> list[str]:
    instruments = record.get("instruments") or []
    if isinstance(instruments, list):
        instrument_text = ", ".join(str(item) for item in instruments if str(item).strip())
    else:
        instrument_text = str(instruments or "").strip()
    instrument_text = instrument_text or "lead melody, chords, bass, and light rhythm"
    chord_text = " - ".join(chords)
    return [
        "0:00-0:10 soft intro that states the mood clearly",
        "0:10-0:35 main melodic idea over the core progression",
        "0:35-0:55 fuller arrangement with " + instrument_text,
        "Main chord movement: " + chord_text,
        "Keep the progression simple and loop-friendly for a 45-60 second inspiration clip",
    ]


def sheet_context(record: dict[str, Any]) -> dict[str, Any]:
    sheet = record.get("sheet") if isinstance(record.get("sheet"), dict) else {}
    lyrics = sheet.get("lyrics") if isinstance(sheet.get("lyrics"), dict) else None
    lyrics_text = str((lyrics or {}).get("text") or "")
    raw_lyrics = record.get("lyrics")
    if not lyrics_text and isinstance(raw_lyrics, dict):
        lyrics_text = str(raw_lyrics.get("text") or "")
    elif not lyrics_text:
        lyrics_text = str(raw_lyrics or "")
    mood = str(record.get("mood") or record.get("mood_tag") or sheet.get("mood") or "")
    genre = str(record.get("genre") or record.get("style") or sheet.get("genre") or "")
    bpm = record.get("bpm") or record.get("tempo") or sheet.get("bpm") or 90
    key = str(record.get("key") or sheet.get("key") or _choose_key(mood, genre))
    chords = record.get("chords") or sheet.get("chords") or _choose_chords(mood)
    if not isinstance(chords, list):
        chords = _choose_chords(mood)
    chords = [str(chord).strip() for chord in chords if str(chord).strip()] or _choose_chords(mood)
    instruments = record.get("instruments") or []
    if isinstance(instruments, list):
        instrument_text = ", ".join(str(item).strip() for item in instruments if str(item).strip())
    else:
        instrument_text = str(instruments or "").strip()
    return {
        "title": record_title(record),
        "mood": mood or "Original",
        "genre": genre or "AI-generated",
        "bpm": bpm,
        "key": key,
        "timeSignature": str(record.get("timeSignature") or sheet.get("timeSignature") or "4/4"),
        "duration": int(float(record.get("duration") or record.get("duration_seconds") or 60)),
        "instruments": instrument_text or "piano/synth, bass, drums, and lead texture",
        "lyrics": lyrics_text.strip(),
        "chords": chords,
    }


def clean_sheet_text(value: str) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("PROVIDER IK", "").replace("MAIN CH", "Main chord movement")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def section_bars(chords: list[str], offset: int = 0, rows: int = 1) -> list[str]:
    if not chords:
        chords = ["C", "Am", "F", "G"]
    lines = []
    for row in range(rows):
        rotated = chords[(offset + row) % len(chords):] + chords[: (offset + row) % len(chords)]
        while len(rotated) < 4:
            rotated += chords
        lines.append("| " + " | ".join(rotated[:4]) + " |")
    return lines


def build_chord_sheet(record: dict[str, Any]) -> str:
    ctx = sheet_context(record)
    chords = ctx["chords"]
    lines = [
        "CHORD SHEET",
        f"Key: {ctx['key']} · BPM: {ctx['bpm']} · Time Signature: {ctx['timeSignature']}",
        "",
        "[Intro]",
        *section_bars(chords, 0, 1),
        "",
        "[Verse]",
        *section_bars(chords, 0, 2),
        "",
        "[Hook]",
        *section_bars(chords, 2, 2),
        "",
        "Performance Notes:",
        f"Keep the feel {str(ctx['mood']).lower()} and let the {ctx['genre']} groove breathe.",
    ]
    return clean_sheet_text("\n".join(lines))


def build_music_sheet(record: dict[str, Any]) -> dict[str, Any]:
    ctx = sheet_context(record)
    key_root = re.sub(r"m$|maj.*$|min.*$", "", str(ctx["key"])) or "C"
    notes = {
        "C": ["C", "D", "E", "G", "A"],
        "D": ["D", "E", "F#", "A", "B"],
        "E": ["E", "F#", "G", "B", "C"],
        "F": ["F", "G", "A", "C", "D"],
        "G": ["G", "A", "B", "D", "E"],
        "A": ["A", "B", "C", "E", "G"],
        "B": ["B", "C#", "D#", "F#", "G#"],
    }.get(key_root[:1].upper(), ["C", "D", "E", "G", "A"])
    music_sheet = {
        "title": ctx["title"],
        "key": ctx["key"],
        "bpm": ctx["bpm"],
        "timeSignature": ctx["timeSignature"],
        "scale": f"{ctx['key']} major/minor modal color",
        "mood": ctx["mood"],
        "genre": ctx["genre"],
        "melodySketch": [
            f"Bar 1: {notes[0]} - {notes[2]} - {notes[4]} - {notes[2]}",
            f"Bar 2: {notes[1]} - {notes[2]} - {notes[3]}",
            f"Bar 3: {notes[4]} - {notes[3]} - {notes[2]} - {notes[1]}",
        ],
        "arrangement": [
            f"Piano/Synth: Main harmonic bed using {', '.join(ctx['chords'][:4])}",
            "Bass: Root-note movement that follows each bar",
            "Drums: Light rhythmic support with space for the vocal or lead",
            f"Lead Texture: Reinforce the {str(ctx['mood']).lower()} mood with {ctx['instruments']}",
        ],
        "productionNotes": [
            "Keep transitions clean and loop-friendly.",
            "Use the hook section as the loudest point of the preview.",
            "Avoid overfilling the midrange so the main melody stays readable.",
        ],
    }
    return music_sheet


def attach_sheet(record: dict[str, Any]) -> None:
    try:
        mood = str(record.get("mood") or record.get("mood_tag") or "Selected mood")
        genre = str(record.get("genre") or record.get("style") or "Selected style")
        key = str(record.get("key") or _choose_key(mood, genre))
        chords = record.get("chords")
        if not isinstance(chords, list) or not chords:
            chords = _choose_chords(mood)
        chords = [str(chord) for chord in chords if str(chord).strip()]
        sheet = {
            "title": record_title(record),
            "provider": provider_label(str(record.get("mode") or "")),
            "mood": mood,
            "genre": genre,
            "bpm": record.get("bpm") or record.get("tempo") or 90,
            "key": key,
            "chords": chords,
            "musicSheet": build_music_sheet_lines(record, chords),
        }
        lyrics = one_verse_lyrics(record)
        if lyrics:
            sheet["lyrics"] = lyrics
        record["sheet"] = sheet
        record["sheetAvailable"] = True
        record["lyricsAvailable"] = bool(lyrics)
        record["chords"] = chords
        record["key"] = key
        record.setdefault("timeSignature", "4/4")
        record.setdefault("chordSheet", build_chord_sheet(record))
        record.setdefault("musicSheet", build_music_sheet(record))
    except Exception as exc:
        logger.warning("Sheet generation failed for %s: %s", record_code(record), exc)
        record["sheet"] = None
        record["sheetAvailable"] = False
        record["lyricsAvailable"] = False
        record["sheet_error"] = "Sheet could not be generated for this inspiration."


def normalize_mode(value: str | None) -> str:
    mode = (value or "musicgen").strip().lower()
    if mode not in {"api", "musicgen", "ace-step"}:
        raise HTTPException(status_code=400, detail="Unknown generation mode. Use 'api', 'musicgen', or 'ace-step'.")
    return mode


def record_code(record: dict[str, Any]) -> str:
    return str(record.get("code") or record.get("song_id") or "").upper()


def original_filename(record: dict[str, Any]) -> str | None:
    for key in ("original_audio_filename", "audio_filename", "filename"):
        filename = record.get(key)
        if filename:
            return Path(str(filename)).name
    output_file = record.get("output_file")
    if output_file:
        return Path(str(output_file)).name
    code = record_code(record)
    if code and record.get("mode") == "api":
        return f"ECHO_{code}_original.mp3"
    if code:
        return f"{code}.wav"
    return None


def trimmed_filename(record: dict[str, Any]) -> str | None:
    filename = record.get("trimmed_audio_filename")
    return Path(str(filename)).name if filename else None


def media_type_for(filename: str) -> str:
    return "audio/mpeg" if filename.lower().endswith(".mp3") else "audio/wav"


def generated_url(filename: str) -> str:
    return f"/generated/{Path(filename).name}"


def generated_static_url(path: str | Path) -> str:
    raw_path = Path(path)
    try:
        relative = raw_path.resolve().relative_to(GENERATED_DIR.resolve())
    except ValueError:
        relative = Path(raw_path.name)
    return "/generated/" + relative.as_posix()


def generated_audio_path(filename: str) -> Path:
    return GENERATED_DIR / Path(filename).name


def audio_duration_seconds(path: Path, fallback_seconds: int | float | None = None) -> int:
    try:
        if path.suffix.lower() == ".wav":
            with wave.open(str(path), "rb") as wav_file:
                frame_rate = wav_file.getframerate()
                if frame_rate > 0:
                    return max(1, int(round(wav_file.getnframes() / frame_rate)))

        from pydub import AudioSegment

        audio = AudioSegment.from_file(path)
        return max(1, int(round(len(audio) / 1000)))
    except Exception as exc:
        logger.warning("Could not read audio duration for %s: %s", path, exc)
        if fallback_seconds:
            return max(1, int(round(float(fallback_seconds))))
        return 0


def record_title(record: dict[str, Any]) -> str:
    title = str(record.get("title") or "").strip()
    code = record_code(record)
    mood = str(record.get("mood") or record.get("mood_tag") or "").strip()
    theme = str(record.get("theme") or "").strip()
    style = str(record.get("style") or record.get("genre") or "").strip()
    parts = [part for part in (mood, theme or style) if part]
    generic_title = " Â· ".join(parts)
    too_generic = title.lower() in {"dreamy Â· inspiration", "dreamy · inspiration", "generated track"}
    if title and title != generic_title and not too_generic:
        return title
    if code:
        prefix = "Inspiration" if record.get("mode") == "api" else "Song"
        return f"{prefix} {code}"
    return " · ".join(parts) or "Generated Track"


def finalize_generated_record(
    record: dict[str, Any],
    *,
    fallback_duration: int | float | None = None,
    title: str | None = None,
    genre: str | None = None,
    bpm: int | None = None,
) -> dict[str, Any]:
    filename = original_filename(record)
    if not filename:
        raise RuntimeError("Generated audio metadata did not include a filename.")

    output_path = generated_audio_path(filename)
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Generated audio file was not found on disk: {output_path}")

    duration = audio_duration_seconds(output_path, fallback_duration)
    audio_url = generated_url(filename)
    relative_path = f"generated/{filename}"
    song_id = record_code(record)

    record.update(
        {
            "id": song_id or record.get("id"),
            "title": title or record_title(record),
            "genre": genre or record.get("genre") or record.get("style") or "",
            "bpm": bpm or record.get("bpm") or record.get("tempo"),
            "duration": duration,
            "durationSeconds": duration,
            "duration_seconds": duration,
            "audio_path": relative_path,
            "audio_url": audio_url,
            "audioUrl": audio_url,
            "filename": filename,
            "audio_filename": filename,
            "original_audio_filename": record.get("original_audio_filename") or filename,
            "output_file": relative_path,
        }
    )

    logger.info("Generated file: %s", output_path)
    logger.info("Audio URL: %s", audio_url)
    logger.info("Duration: %s", duration)
    return record


def find_history_record(code: str, user_id: str = DEFAULT_USER_ID) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    normalized = code.upper()
    history = read_history(user_id)
    for record in history:
        if record_code(record) == normalized:
            return history, record
    return history, None


def state_user_entry(request: Request) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    email = current_user_email(request).strip().lower()
    name = current_user_name(request)
    state = read_user_state()
    user = state["users"].setdefault(email, {"name": name or email.split("@")[0], "email": email, "songIds": []})
    user["name"] = name or user.get("name") or email.split("@")[0]
    user["email"] = email
    user.setdefault("songIds", [])
    return email, name, state, user


def migrate_history_to_state(request: Request) -> dict[str, Any]:
    email, name, state, user = state_user_entry(request)
    changed = False
    for record in read_history(current_user_id(request)):
        song_id = record_code(record)
        if not song_id:
            continue
        if song_id not in user["songIds"]:
            user["songIds"].append(song_id)
            changed = True
        existing_owner = str((state["songs"].get(song_id) or {}).get("ownerEmail") or "").strip().lower()
        if song_id not in state["songs"] or existing_owner in {"", email}:
            state["songs"][song_id] = song_detail_payload(record, owner_email=email)
            changed = True
    if changed:
        write_user_state(state)
    return state


def state_song_for_user(song_id: str, request: Request) -> dict[str, Any] | None:
    normalized = song_id.strip().upper()
    email = current_user_email(request).strip().lower()
    _, history_record = find_history_record(normalized, current_user_id(request))
    if history_record:
        return song_detail_payload(history_record, owner_email=email)

    email, _, state, user = state_user_entry(request)
    if normalized not in set(str(item).upper() for item in user.get("songIds") or []):
        return None
    song = state["songs"].get(normalized)
    if not isinstance(song, dict):
        return None
    owner = str(song.get("ownerEmail") or email).strip().lower()
    if owner and owner != email:
        return None
    return song_detail_payload(song, owner_email=email)


def update_state_song_for_user(song_id: str, request: Request, updates: dict[str, Any]) -> dict[str, Any] | None:
    normalized = song_id.strip().upper()
    email = current_user_email(request).strip().lower()
    user_id = current_user_id(request)
    history = read_history(user_id)
    changed = False
    updated_record: dict[str, Any] | None = None
    for record in history:
        if record_code(record) == normalized:
            record.update(updates)
            changed = True
            updated_record = record
    if changed:
        write_user_songs(user_id, history)
        remember_user_song(email, current_user_name(request), updated_record or {})
        return song_detail_payload(updated_record or {}, owner_email=email)

    email, _, state, user = state_user_entry(request)
    if normalized not in set(str(item).upper() for item in user.get("songIds") or []):
        return None
    song = state["songs"].get(normalized)
    if not isinstance(song, dict):
        return None
    owner = str(song.get("ownerEmail") or email).strip().lower()
    if owner and owner != email:
        return None
    song.update(updates)
    state["songs"][normalized] = song
    write_user_state(state)
    return song_detail_payload(song, owner_email=email)


def with_urls(record: dict[str, Any]) -> dict[str, Any]:
    attach_sheet(record)
    code = record_code(record)
    original = original_filename(record)
    trimmed = trimmed_filename(record)
    source_url = record.get("audio_source_url") or None
    saved_audio_url = (
        record.get("audio_url")
        or record.get("audioUrl")
        or record.get("fileUrl")
        or record.get("file_url")
        or record.get("outputUrl")
        or record.get("output_url")
        or record.get("generatedAudioUrl")
        or record.get("generated_audio_url")
        or record.get("musicgenUrl")
        or record.get("musicgen_url")
        or record.get("aceAudioUrl")
        or record.get("ace_audio_url")
        or record.get("mp3Url")
        or record.get("mp3_url")
        or record.get("url")
        or record.get("path")
        or None
    )

    local_original_exists = bool(original and generated_audio_path(original).exists())
    original_url = generated_url(original) if original else saved_audio_url or source_url

    local_trimmed_exists = bool(trimmed and generated_audio_path(trimmed).exists())
    trimmed_url = generated_url(trimmed) if trimmed else None
    playable_url = trimmed_url or original_url

    return {
        **record,
        "id": record.get("id") or code,
        "title": record_title(record),
        "displayTitle": record_title(record),
        "genre": record.get("genre") or record.get("style") or "",
        "bpm": record.get("bpm") or record.get("tempo"),
        "code": code,
        "song_id": code,
        "songId": code,
        "filename": original,
        "audio_filename": original,
        "audio_path": f"generated/{original}" if original else record.get("audio_path"),
        "audio_url": playable_url,
        "audioUrl": playable_url,
        "fileUrl": playable_url,
        "generatedAudioUrl": playable_url,
        "musicgenUrl": playable_url if str(record.get("mode") or "").lower() == "musicgen" else record.get("musicgenUrl"),
        "aceAudioUrl": playable_url if str(record.get("mode") or "").lower() in {"ace-step", "ace_step", "ace"} else record.get("aceAudioUrl"),
        "mp3Url": playable_url if playable_url and str(playable_url).lower().endswith(".mp3") else record.get("mp3Url"),
        "original_audio_url": original_url,
        "original_download_url": f"/download/{code}/original" if code and original else None,
        "download_url": f"/download/{code}/original" if code and original else None,
        "downloadAudioUrl": f"/download/{code}/original" if code and original else None,
        "trimmed_audio_url": trimmed_url,
        "trimmed_download_url": f"/download/{code}/trimmed" if trimmed else None,
        "sheetAvailable": bool(record.get("sheetAvailable")),
        "lyricsAvailable": bool(record.get("lyricsAvailable")),
        "sheet": record.get("sheet"),
    }


def song_detail_payload(record: dict[str, Any], owner_email: str = "") -> dict[str, Any]:
    persisted_chord_sheet = record.get("chordSheet") if isinstance(record.get("chordSheet"), str) else ""
    persisted_music_sheet = record.get("musicSheet") if isinstance(record.get("musicSheet"), dict) else None
    song = with_urls(record)
    sheet = song.get("sheet") if isinstance(song.get("sheet"), dict) else {}
    lyrics_sheet = sheet.get("lyrics") if isinstance(sheet.get("lyrics"), dict) else None
    lyrics_text = str((lyrics_sheet or {}).get("text") or "").strip()
    has_lyrics = bool(song.get("lyricsAvailable") and lyrics_text)
    code = record_code(song)
    inline_music_sheet = persisted_music_sheet
    return {
        **song,
        "songId": code,
        "song_id": code,
        "id": song.get("id") or code,
        "ownerEmail": owner_email or song.get("ownerEmail") or "",
        "displayTitle": song.get("displayTitle") or song.get("title") or (f"Song {code}" if code else "Generated Track"),
        "provider": provider_label(str(song.get("mode") or "")),
        "audioUrl": song.get("audioUrl") or song.get("audio_url") or "",
        "audio_url": song.get("audio_url") or song.get("audioUrl") or "",
        "downloadAudioUrl": song.get("downloadAudioUrl") or song.get("download_url"),
        "duration": song.get("duration") or song.get("durationSeconds") or song.get("duration_seconds") or song.get("lengthSeconds") or 0,
        "durationSeconds": song.get("duration") or song.get("durationSeconds") or song.get("duration_seconds") or song.get("lengthSeconds") or 0,
        "duration_seconds": song.get("duration") or song.get("durationSeconds") or song.get("duration_seconds") or song.get("lengthSeconds") or 0,
        "lengthSeconds": song.get("lengthSeconds") or song.get("durationSeconds") or song.get("duration_seconds") or song.get("duration") or 0,
        "mood": song.get("mood") or song.get("vibe") or song.get("emotion") or song.get("mood_tag") or "",
        "genre": song.get("genre") or song.get("style") or "",
        "bpm": song.get("bpm") or song.get("BPM") or song.get("beatsPerMinute") or song.get("tempo") or 90,
        "BPM": song.get("BPM") or song.get("bpm") or song.get("tempo") or 90,
        "beatsPerMinute": song.get("beatsPerMinute") or song.get("bpm") or song.get("tempo") or 90,
        "instruments": song.get("instruments") or song.get("instrumentTags") or song.get("instrumentation") or song.get("instrument") or "",
        "isInstrumental": song.get("isInstrumental") if song.get("isInstrumental") is not None else song.get("instrumental"),
        "instrumental": song.get("instrumental") if song.get("instrumental") is not None else song.get("isInstrumental"),
        "vocalsEnabled": song.get("vocalsEnabled"),
        "vocalMode": song.get("vocalMode") or ("instrumental" if song.get("vocalsEnabled") is False else ""),
        "key": song.get("key") or sheet.get("key") or "",
        "chords": sheet.get("chords") or song.get("chords") or [],
        "chordSheet": persisted_chord_sheet,
        "chord_sheet": persisted_chord_sheet,
        "musicSheet": inline_music_sheet,
        "music_sheet": inline_music_sheet,
        "sheetMusic": inline_music_sheet,
        "sheetMusicPdfUrl": song.get("sheetMusicPdfUrl") or song.get("musicSheetPdfUrl") or "",
        "musicSheetPdfUrl": song.get("musicSheetPdfUrl") or song.get("sheetMusicPdfUrl") or "",
        "sheet_music_pdf_url": song.get("sheetMusicPdfUrl") or song.get("musicSheetPdfUrl") or "",
        "timeSignature": song.get("timeSignature") or sheet.get("timeSignature") or "4/4",
        "musicSheetLines": sheet.get("musicSheet") if isinstance(sheet.get("musicSheet"), list) else [],
        "lyrics": lyrics_sheet if has_lyrics else None,
        "lyricsAvailable": has_lyrics,
        "sheetAvailable": bool(song.get("sheetAvailable") and (sheet.get("musicSheet") or sheet.get("chords"))),
        "lyricsSheetDownloadUrl": f"/api/song/{code}/lyrics-sheet" if code and has_lyrics else None,
        "createdAt": song.get("created_at") or song.get("createdAt"),
    }


def unified_response(record: dict[str, Any]) -> dict[str, Any]:
    song = song_detail_payload(record)
    return {
        "success": True,
        "ok": True,
        "mode": song.get("mode") or "musicgen",
        "songId": song.get("songId"),
        "displayTitle": song.get("displayTitle"),
        "provider": song.get("provider"),
        "audioUrl": song.get("audioUrl"),
        "audio_url": song.get("audio_url"),
        "filename": song.get("filename"),
        "song_id": song.get("song_id"),
        "duration": song.get("duration"),
        "mood": song.get("mood"),
        "genre": song.get("genre"),
        "bpm": song.get("bpm"),
        "key": song.get("key"),
        "chords": song.get("chords"),
        "musicSheet": song.get("musicSheet"),
        "prompt_used": song.get("prompt"),
        "generation_time_seconds": song.get("generation_time_seconds"),
        "generation_time": song.get("generation_time"),
        "output_file": song.get("output_file") or (f"generated/{song['filename']}" if song.get("filename") else None),
        "download_filename": song.get("filename"),
        "downloadAudioUrl": song.get("downloadAudioUrl"),
        "lyrics": song.get("lyrics"),
        "lyricsAvailable": bool(song.get("lyricsAvailable")),
        "sheetAvailable": bool(song.get("sheetAvailable")),
        "lyricsSheetDownloadUrl": song.get("lyricsSheetDownloadUrl"),
        "sheet": song.get("sheet"),
        "copyright": song.get("copyright"),
        "song": song,
        "record": song,
    }


def get_generation_estimate_seconds() -> int:
    recent_times = [
        int(record["generation_time_seconds"])
        for record in read_history()[-5:]
        if isinstance(record.get("generation_time_seconds"), (int, float))
        and record.get("generation_time_seconds", 0) > 0
    ]
    if recent_times:
        return max(60, int(sum(recent_times) / len(recent_times)))
    return DEFAULT_GENERATION_ESTIMATE_SECONDS


def update_generation_status(stage: str, progress: int) -> None:
    progress = max(0, min(progress, 100))
    with generation_lock:
        started_at = generation_status.get("started_at")
        if started_at is None:
            started_at = time.perf_counter()
            generation_status["started_at"] = started_at
        elapsed = int(time.perf_counter() - started_at) if started_at else 0
        estimated_total = generation_status.get("estimated_total_seconds") or DEFAULT_GENERATION_ESTIMATE_SECONDS
        generation_status.update(
            {
                "stage": stage,
                "progress": progress,
                "elapsed_seconds": elapsed,
                "estimated_remaining_seconds": 0 if progress >= 100 else max(1, int(estimated_total - elapsed)),
                "active": progress < 100 and stage != "Failed",
            }
        )


def reset_generation_status(total_seconds: int | None = None) -> None:
    estimated_total = total_seconds or get_generation_estimate_seconds()
    with generation_lock:
        generation_status.update(
            {
                "stage": "Idle",
                "progress": 0,
                "started_at": time.perf_counter(),
                "elapsed_seconds": 0,
                "estimated_remaining_seconds": estimated_total,
                "active": True,
                "estimated_total_seconds": estimated_total,
            }
        )


def warm_model() -> None:
    global model, processor, model_status, model_error
    logger.info("Loading MusicGen model...")
    try:
        loaded_model, loaded_processor = load_model()
        with model_lock:
            model = loaded_model
            processor = loaded_processor
            model_status = "ready"
            model_error = ""
        logger.info("MusicGen loaded and cached.")
    except Exception as error:
        logger.exception("MusicGen warmup failed")
        with model_lock:
            model = None
            processor = None
            model_status = "error"
            model_error = str(error)


def create_song_id(extension: str, user_id: str = DEFAULT_USER_ID) -> str:
    existing_ids = {record_code(record) for record in read_history(user_id)}
    for _ in range(1000):
        song_id = generate_song_id(existing_ids)
        if not (GENERATED_DIR / f"{song_id}.{extension}").exists():
            return song_id
    raise RuntimeError("Could not create a unique song ID.")


def format_seconds(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    minutes = total_seconds // 60
    remainder = total_seconds % 60
    return f"{minutes}m {remainder:02d}s" if minutes else f"{remainder}s"


def api_input_from_request(request: GenerateRequest) -> ApiGenerationInput:
    mood = request.mood or first_or_join(request.moods, "Dreamy")
    theme = request.theme or first_or_join(request.themes, request.prompt or "Inspiration")
    style = request.style or first_or_join(request.genres, "Lo-fi")
    return ApiGenerationInput(
        mood=mood,
        theme=theme,
        style=style,
        instruments=request.instruments,
        tempo=request.tempo,
        energy=request.energy,
    )


async def generate_with_api(
    request: GenerateRequest,
    user_id: str = DEFAULT_USER_ID,
    *,
    user_email: str = "",
    user_name: str = "",
) -> dict[str, Any]:
    reset_generation_status(60)
    update_generation_status("Submitting API generation", 10)
    try:
        history = read_history(user_id)
        existing_ids = {record_code(item) for item in history}
        generated = await generate_api_song(
            api_input_from_request(request),
            generated_dir=GENERATED_DIR,
            existing_ids=existing_ids,
        )
        generated["mode"] = "api"
        generated["fast"] = request.fast
        filename = original_filename(generated)
        generated["filename"] = filename
        generated["audio_filename"] = filename
        finalize_generated_record(
            generated,
            fallback_duration=request.duration,
            title=record_title(generated),
            genre=generated.get("style") or request.style or first_or_join(request.genres, ""),
            bpm=generated.get("tempo") or request.tempo,
        )
        generated["lyrics"] = None
        generated["copyright"] = copyright_unavailable("No real lyrics were returned for this KIE.AI inspiration.")
        attach_sheet(generated)
        save_user_song(user_id, generated, user_email=user_email, user_name=user_name)
        update_generation_status("Completed", 100)
        return unified_response(generated)
    except KieAIConfigError as exc:
        update_generation_status("Failed", 100)
        logger.error("KieAI configuration error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except KieAIResponseError as exc:
        update_generation_status("Failed", 100)
        logger.error("KieAI request error: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        update_generation_status("Failed", 100)
        logger.exception("API generation failed")
        raise HTTPException(status_code=500, detail="API song generation failed. Please try again.") from exc


def generate_with_musicgen(
    request: GenerateRequest,
    user_id: str = DEFAULT_USER_ID,
    *,
    user_email: str = "",
    user_name: str = "",
) -> dict[str, Any]:
    total_start = time.perf_counter()
    reset_generation_status()
    update_generation_status("Preparing prompt", 0)
    try:
        song_id = create_song_id("wav", user_id)
        filename = f"{song_id}.wav"
        output_path = GENERATED_DIR / filename
    except Exception as error:
        update_generation_status("Failed", 100)
        raise HTTPException(status_code=500, detail=f"Song ID assignment failed: {error}") from error

    with model_lock:
        ready_model = model
        ready_processor = processor
        current_status = model_status

    if current_status != "ready" or ready_model is None or ready_processor is None:
        update_generation_status("Failed", 100)
        detail = (
            "MusicGen is unavailable on this deployment — it requires a GPU and is not installed."
            if current_status == "error"
            else "MusicGen is still loading, please try again in a moment."
        )
        raise HTTPException(status_code=503, detail=detail)

    try:
        custom_prompt = request.custom_prompt.strip() or request.prompt.strip()
        prompt = build_music_prompt(
            moods=request.moods or ([request.mood] if request.mood else []),
            genres=request.genres or ([request.style] if request.style else []),
            themes=request.themes or ([request.theme] if request.theme else []),
            instruments=request.instruments,
            tempo=request.tempo,
            complexity=request.complexity,
            energy=request.energy,
            custom_prompt=custom_prompt,
            duration_seconds=request.duration,
        )
        generation_result = generate_music(
            ready_model,
            ready_processor,
            prompt,
            output_path,
            progress_callback=update_generation_status,
            duration_seconds=request.duration,
        )
    except RuntimeError as error:
        update_generation_status("Failed", 100)
        if "out of memory" in str(error).lower():
            raise HTTPException(status_code=500, detail="MusicGen ran out of memory.") from error
        raise HTTPException(status_code=500, detail=f"Music generation failed: {error}") from error
    except Exception as error:
        update_generation_status("Failed", 100)
        logger.exception("MusicGen generation failed")
        raise HTTPException(status_code=500, detail=f"Unexpected generation error: {error}") from error

    total_seconds = time.perf_counter() - total_start
    record = {
        "mode": "musicgen",
        "code": song_id,
        "song_id": song_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt,
        "mood": first_or_join(request.moods or ([request.mood] if request.mood else []), ""),
        "theme": first_or_join(request.themes or ([request.theme] if request.theme else []), ""),
        "style": first_or_join(request.genres or ([request.style] if request.style else []), ""),
        "instruments": request.instruments,
        "tempo": request.tempo,
        "complexity": request.complexity,
        "duration": request.duration,
        "durationSeconds": request.duration,
        "duration_seconds": request.duration,
        "energy": request.energy,
        "isInstrumental": True,
        "vocalsEnabled": False,
        "aiVocals": False,
        "vocalMode": "instrumental",
        "generation_time_seconds": round(total_seconds),
        "generation_time": format_seconds(total_seconds),
        "filename": filename,
        "audio_filename": filename,
        "output_file": f"generated/{filename}",
        "audio_url": f"/generated/{filename}",
        "inference_seconds": round(float(generation_result.get("inference_seconds", 0)), 2),
    }
    finalize_generated_record(
        record,
        fallback_duration=request.duration,
        title=record_title(record),
        genre=record.get("style") or first_or_join(request.genres, ""),
        bpm=request.tempo,
    )
    record["lyrics"] = None
    record["copyright"] = copyright_unavailable("MusicGen does not produce lyrics for this instrumental inspiration.")
    attach_sheet(record)
    save_user_song(user_id, record, user_email=user_email, user_name=user_name)
    update_generation_status("Completed", 100)
    return unified_response(record)


async def generate_with_ace_step_mode(
    request: GenerateRequest,
    user_id: str = DEFAULT_USER_ID,
    *,
    user_email: str = "",
    user_name: str = "",
) -> dict[str, Any]:
    reset_generation_status(600)
    update_generation_status("Generating lyrics for ACE-Step", 10)
    try:
        song_id = create_song_id("wav", user_id)
        filename = f"ECHO_{song_id}_ace.wav"
        output_path = GENERATED_DIR / filename

        mood = first_or_join(request.moods or ([request.mood] if request.mood else []), "dreamy")
        theme = first_or_join(request.themes or ([request.theme] if request.theme else []), "inspiration")
        style = first_or_join(request.genres or ([request.style] if request.style else []), "pop")
        prompt_str = f"{mood} {style}, {request.tempo}bpm, {theme}"

        # Use user-provided lyrics if given; only auto-generate if empty
        if request.lyrics and request.lyrics.strip():
            update_generation_status("Using provided lyrics", 20)
            lyrics_text = request.lyrics.strip()
            lyrics_result = clean_short_verse(lyrics_text) or {"text": lyrics_text, "structure": "user-provided"}
            lyrics_ok = True
        else:
            update_generation_status("Writing lyrics with AI", 20)
            lyrics_result, lyrics_ok = generate_lyrics_section(request, prompt_str)
            lyrics_text = lyrics_result.get("text", "")
            lyrics_result = clean_short_verse(lyrics_text) or lyrics_result

        update_generation_status("Sending to ACE-Step (this may take a few minutes)", 35)
        ace_result = await asyncio.to_thread(generate_with_ace_step,
            prompt=prompt_str,
            lyrics=lyrics_text,
            duration=request.duration or 30,
            mood=mood,
            style=style,
            instruments=request.instruments,
            tempo=request.tempo,
            energy=request.energy,
            output_path=output_path,
        )
        generated_path = Path(str(ace_result.get("audio_path") or output_path))
        if generated_path.parent == GENERATED_DIR:
            filename = generated_path.name
        else:
            filename = Path(filename).name

        update_generation_status("Checking copyright", 90)
        copyright_result = generate_copyright_section(lyrics_text, lyrics_ok)

        record: dict[str, Any] = {
            "mode": "ace-step",
            "code": song_id,
            "song_id": song_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "prompt": prompt_str,
            "mood": mood,
            "theme": theme,
            "style": style,
            "instruments": request.instruments,
            "tempo": request.tempo,
            "complexity": request.complexity,
            "duration": request.duration,
            "energy": request.energy,
            "filename": filename,
            "audio_filename": filename,
            "original_audio_filename": filename,
            "output_file": f"generated/{filename}",
            "audio_url": f"/generated/{filename}",
            "lyrics": lyrics_result,
            "copyright": copyright_result,
        }
        finalize_generated_record(
            record,
            fallback_duration=request.duration or 30,
            title=record_title(record),
            genre=style,
            bpm=request.tempo,
        )
        attach_sheet(record)
        save_user_song(user_id, record, user_email=user_email, user_name=user_name)
        update_generation_status("Completed", 100)
        return unified_response(record)
    except Exception as exc:
        update_generation_status("Failed", 100)
        logger.exception("ACE-Step generation failed")
        if "gradio_client" in str(exc):
            raise HTTPException(
                status_code=503,
                detail="ACE-Step dependency gradio_client is missing. Install requirements.txt and restart the server.",
            ) from exc
        detail = (
            f"ACE-Step HuggingFace Space is unavailable — it may be sleeping. "
            f"Try again in a few minutes. ({exc})"
        )
        raise HTTPException(status_code=500, detail=detail) from exc


@app.post("/api/auth/login")
def api_auth_login(request: LoginRequest) -> dict[str, Any]:
    try:
        return auth_login(request.email, request.password)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@app.post("/api/auth/signup")
def api_auth_signup(request: SignupRequest) -> dict[str, Any]:
    if os.getenv("ECHO_SIGNUPS_OPEN", "").strip() != "1":
        raise HTTPException(
            status_code=403,
            detail="Sign-ups are temporarily closed — Echo Echo is in invite-only beta.",
        )
    try:
        return auth_signup(request.name, request.email, request.password)
    except AuthError as exc:
        status_code = 409 if "already exists" in str(exc) else 400
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc


@app.get("/api/firebase-config")
def api_firebase_config() -> dict[str, str]:
    config = {
        firebase_key: os.getenv(env_key, "").strip()
        for firebase_key, env_key in FIREBASE_ENV_MAP.items()
    }
    missing = [
        env_key
        for firebase_key, env_key in FIREBASE_ENV_MAP.items()
        if firebase_key != "measurementId" and not config[firebase_key]
    ]
    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"Firebase configuration is missing: {', '.join(missing)}",
        )
    return {key: value for key, value in config.items() if value}


class AceStepLyricsRequest(BaseModel):
    mood: str = ""
    genre: str = ""
    theme: str = ""
    instruments: str = ""
    vocal: str = ""
    tempo: int = 90
    energy: int = 5
    duration: int = 60
    prompt: str = ""
    promptText: str = ""


@app.post("/api/ace-step/lyrics")
def api_ace_step_lyrics(request: AceStepLyricsRequest) -> dict[str, Any]:
    """Generate ACE-Step formatted lyrics via Groq based on user selections."""
    import os
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY not configured")

    try:
        from groq import Groq
        client = Groq(api_key=groq_key)
    except ImportError:
        raise HTTPException(status_code=503, detail="groq package not installed")

    energy_words = {1:"very calm",2:"calm",3:"mellow",4:"relaxed",5:"moderate",
                    6:"upbeat",7:"energetic",8:"intense",9:"very intense",10:"extreme"}
    energy_desc = energy_words.get(request.energy, "moderate")
    duration = int(request.duration or 60)
    max_lines = 8 if duration <= 30 else 16 if duration <= 60 else 24 if duration <= 90 else 48

    system = (
        "You are a professional songwriter. Write original song lyrics for an AI-generated song preview. "
        "Use ONLY these section tags exactly as written: [verse], [pre-chorus], [chorus], [bridge]. "
        "Each section tag must be on its own line. Write short, singable lines. "
        "Do not write a full song for a short preview. Do not output one paragraph. "
        "Never use [Verse 1] or numbered variants — only [verse] and [chorus]. "
        "Output ONLY the lyrics, no explanations."
    )
    user_prompt = (
        f"Write lyrics that fit inside {duration} seconds.\n"
        f"Maximum lyric lines: {max_lines}.\n"
        "Use section labels and 2-4 short lines per section.\n"
        f"Mood: {request.mood or 'emotional'}\n"
        f"Genre: {request.genre or 'pop'}\n"
        f"Instruments: {request.instruments or 'piano, guitar'}\n"
        f"Vocal style: {request.vocal or 'smooth'}\n"
        f"Tempo: {request.tempo} BPM ({energy_desc} energy)\n"
        f"Theme/prompt: {request.prompt or request.promptText or request.theme or 'life and journey'}\n\n"
        "Make the lyrics singable, emotional, and original."
    )

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
        max_tokens=600,
        temperature=0.85,
    )
    lyrics = completion.choices[0].message.content.strip()
    return {"ok": True, "lyrics": lyrics}


@app.post("/api/kie-vocal")
async def kie_vocal_generate(request: KieVocalRequest, http_request: Request) -> dict[str, Any]:
    """Generate a full vocal singing track via Kie.ai → Suno (instrumental=False)."""
    reset_generation_status(180)
    update_generation_status("Submitting vocal request to Kie.ai", 8)
    try:
        user_id = current_user_id(http_request)
        user_email = current_user_email(http_request)
        user_name = current_user_name(http_request)
        history = read_history(user_id)
        existing_ids = {record_code(item) for item in history}

        style = request.style or request.genre or "Pop"
        instruments = [request.instrument] if request.instrument else []
        theme = request.theme or request.promptText or "life"

        data = ApiGenerationInput(
            mood=request.mood or "Energetic",
            theme=theme,
            style=style,
            instruments=instruments,
            tempo=request.bpm,
            energy=7,
        )

        update_generation_status("Generating vocal song — this takes 1–3 minutes", 15)
        generated = await generate_vocal_song(
            data,
            generated_dir=GENERATED_DIR,
            existing_ids=existing_ids,
            extra_prompt=request.promptText,
        )

        trimmed = generated.get("trimmed_audio_filename")
        original = generated.get("original_audio_filename")
        serve_file = trimmed or original
        generated["mode"] = "kie-vocal"
        generated["filename"] = serve_file
        generated["audio_filename"] = serve_file
        generated["genre"] = request.genre
        generated["mood_tag"] = request.mood
        generated["bpm"] = request.bpm
        finalize_generated_record(
            generated,
            fallback_duration=60,
            title=record_title(generated),
            genre=request.genre or generated.get("style") or "",
            bpm=request.bpm,
        )
        real_lyrics = one_verse_lyrics(generated)
        if not real_lyrics and serve_file:
            real_lyrics = maybe_transcribe_lyrics(GENERATED_DIR / serve_file, "kie-vocal", vocals_enabled=True)
        generated["lyrics"] = real_lyrics
        generated["lyricsAvailable"] = bool(real_lyrics)
        attach_sheet(generated)

        save_user_song(user_id, generated, user_email=user_email, user_name=user_name)
        update_generation_status("Done", 100)
        return {"ok": True, "track": song_detail_payload(generated, owner_email=user_email)}

    except KieAIConfigError as exc:
        update_generation_status("Failed", 100)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except KieAIResponseError as exc:
        update_generation_status("Failed", 100)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        update_generation_status("Failed", 100)
        logger.exception("Vocal generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/compose")
def api_compose(request: ComposeRequest) -> dict[str, Any]:
    song = compose_song(request.model_dump())
    return {"ok": True, "song": song}


@app.post("/api/transcribe")
async def api_transcribe(file: UploadFile = File(...)) -> dict[str, Any]:
    audio = await file.read()
    try:
        text = transcribe_audio(audio, file.filename or "", file.content_type or "")
    except TranscriptionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "text": text}


@app.post("/generate")
async def generate(request: GenerateRequest, http_request: Request) -> dict[str, Any]:
    user_id = current_user_id(http_request)
    user_email = current_user_email(http_request)
    user_name = current_user_name(http_request)
    mode = normalize_mode(request.mode)
    if mode == "api":
        return await generate_with_api(request, user_id, user_email=user_email, user_name=user_name)
    if mode == "ace-step":
        return await generate_with_ace_step_mode(request, user_id, user_email=user_email, user_name=user_name)
    return generate_with_musicgen(request, user_id, user_email=user_email, user_name=user_name)


@app.post("/api/generate")
async def api_generate(request: GenerateRequest, http_request: Request) -> dict[str, Any]:
    request.mode = "api"
    request.fast = True
    return await generate_with_api(
        request,
        current_user_id(http_request),
        user_email=current_user_email(http_request),
        user_name=current_user_name(http_request),
    )


@app.post("/generate-inspiration")
async def generate_inspiration(request: GenerateRequest, http_request: Request) -> dict[str, Any]:
    return await api_generate(request, http_request)


@app.get("/health")
def health() -> dict[str, Any]:
    with model_lock:
        return {"status": model_status}


@app.get("/status")
def status(request: Request) -> dict[str, Any]:
    user_id = current_user_id(request)
    with model_lock:
        return {
            "ok": True,
            "service": "Echo Echo",
            "history_count": len(read_history(user_id)),
            "status": model_status,
            "model_loaded": model is not None,
            "processor_loaded": processor is not None,
            "error": model_error,
        }


@app.get("/generation-status")
def get_generation_status() -> dict[str, Any]:
    with generation_lock:
        status_snapshot = dict(generation_status)
        started_at = status_snapshot.get("started_at")

    if status_snapshot.get("active") and started_at:
        elapsed = int(time.perf_counter() - started_at)
        progress = int(status_snapshot.get("progress", 0))
        estimated_total = int(status_snapshot.get("estimated_total_seconds") or DEFAULT_GENERATION_ESTIMATE_SECONDS)
        if 20 <= progress < 85:
            generation_window = max(1, estimated_total - 20)
            generated_progress = 20 + int(min(65, (elapsed / generation_window) * 65))
            progress = max(progress, min(generated_progress, 84))
            status_snapshot["stage"] = ROTATING_STATUS_MESSAGES[elapsed % len(ROTATING_STATUS_MESSAGES)]
        status_snapshot["progress"] = progress
        status_snapshot["elapsed_seconds"] = elapsed
        status_snapshot["estimated_remaining_seconds"] = max(1, estimated_total - elapsed) if progress < 100 else 0

    status_snapshot.pop("started_at", None)
    return status_snapshot


@app.get("/history")
def history(request: Request) -> dict[str, Any]:
    email = current_user_email(request).strip().lower()
    records = [with_urls({**item, "ownerEmail": email}) for item in read_history(current_user_id(request))]
    return {"songs": list(reversed(records))}


@app.get("/api/tracks")
def api_tracks(request: Request) -> list[dict[str, Any]]:
    email = current_user_email(request).strip().lower()
    name = current_user_name(request)
    records = read_history(current_user_id(request))
    remember_user_profile(email, name, [record_code(record) for record in records if record_code(record)])
    return [song_detail_payload(with_urls({**record, "ownerEmail": email}), owner_email=email) for record in reversed(records)]


@app.get("/api/library")
def api_library(request: Request) -> dict[str, Any]:
    songs = api_tracks(request)
    email, _, _, user = state_user_entry(request)
    song_ids = [record_code(song) for song in songs if record_code(song)]
    return {
        "ok": True,
        "user": {
            "name": current_user_name(request),
            "email": email,
            "songIds": song_ids,
        },
        "songs": songs,
    }


@app.get("/songs")
def songs(request: Request) -> dict[str, Any]:
    return history(request)


@app.get("/inspirations")
def inspirations(request: Request) -> dict[str, Any]:
    return history(request)


@app.get("/library/refresh")
def refresh_library(request: Request) -> dict[str, Any]:
    return history(request)


@app.post("/api/copyright/check")
def api_copyright_check(request: CopyrightCheckApiRequest, http_request: Request) -> dict[str, Any]:
    user_id = current_user_id(http_request)
    history_records: list[dict[str, Any]] = []
    record: dict[str, Any] | None = None
    if request.song_id:
        history_records, record = find_history_record(request.song_id, user_id)

    text = copyright_check_text(request, record)
    if not text:
        result = copyright_unavailable("No lyrics, prompt, title, or metadata were provided for copyright checking.")
    else:
        try:
            response = check_copyright(CopyrightCheckRequest(lyrics=text))
            result = normalize_copyright_response(model_to_dict(response))
        except Exception as exc:
            logger.warning("Copyright API check unavailable: %s", exc)
            result = copyright_unavailable(str(exc))

    if record is not None:
        record["copyright"] = result
        write_history(history_records, user_id)

    return result


@app.get("/song/{song_id}")
def song(song_id: str, request: Request) -> dict[str, Any]:
    record = state_song_for_user(song_id, request)
    if not record:
        migrate_history_to_state(request)
        record = state_song_for_user(song_id, request)
    if record:
        return {"song": record}
    raise HTTPException(status_code=404, detail="Song not found")


@app.get("/api/song/{song_id}")
def api_song(song_id: str, request: Request) -> dict[str, Any]:
    record = state_song_for_user(song_id, request)
    if not record:
        migrate_history_to_state(request)
        record = state_song_for_user(song_id, request)
    if record:
        return {"ok": True, "song": record}
    raise HTTPException(status_code=404, detail="Song not found")


@app.get("/tracks/{song_id}")
def api_track(song_id: str, request: Request) -> dict[str, Any]:
    record = state_song_for_user(song_id, request)
    if not record:
        migrate_history_to_state(request)
        record = state_song_for_user(song_id, request)
    if record:
        return {"ok": True, "track": record, "song": record}
    raise HTTPException(status_code=404, detail="Track not found")


@app.patch("/tracks/{song_id}")
@app.patch("/api/tracks/{song_id}")
def api_update_track(song_id: str, payload: TrackUpdateRequest, request: Request) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    if payload.favorite is not None:
        updates["favorite"] = payload.favorite
        updates["fav"] = payload.favorite
    if not updates:
        record = state_song_for_user(song_id, request)
        if record:
            return {"ok": True, "track": record, "song": record}
        raise HTTPException(status_code=404, detail="Track not found")

    updated = update_state_song_for_user(song_id, request, updates)
    if not updated:
        raise HTTPException(status_code=404, detail="Track not found")
    return {"ok": True, "track": updated, "song": updated}


@app.delete("/tracks/{song_id}")
@app.delete("/api/tracks/{song_id}")
def api_delete_track(song_id: str, request: Request) -> dict[str, Any]:
    normalized = song_id.strip().upper()
    user_id = current_user_id(request)
    history = read_history(user_id)
    kept = [record for record in history if record_code(record) != normalized]
    if len(kept) == len(history):
        raise HTTPException(status_code=404, detail="Track not found")
    write_user_songs(user_id, kept)

    email, _, state, user = state_user_entry(request)
    user["songIds"] = [item for item in user.get("songIds") or [] if str(item).upper() != normalized]
    song = state["songs"].get(normalized)
    if isinstance(song, dict) and str(song.get("ownerEmail") or email).strip().lower() == email:
        state["songs"].pop(normalized, None)
    write_user_state(state)
    return {"ok": True}


@app.post("/api/song/{song_id}/generate-chords")
@app.post("/tracks/{song_id}/generate-chords")
def api_song_generate_chords(song_id: str, request: Request) -> dict[str, Any]:
    record = state_song_for_user(song_id, request)
    if not record:
        migrate_history_to_state(request)
        record = state_song_for_user(song_id, request)
    if not record:
        raise HTTPException(status_code=404, detail="Song not found")
    chord_sheet = build_chord_sheet(record)
    updated = update_state_song_for_user(song_id, request, {"chordSheet": chord_sheet})
    if not updated:
        raise HTTPException(status_code=404, detail="Song not found")
    return {"ok": True, "chordSheet": chord_sheet, "song": updated}


@app.post("/api/song/{song_id}/generate-music-sheet")
@app.post("/tracks/{song_id}/generate-music-sheet")
def api_song_generate_music_sheet(song_id: str, request: Request) -> dict[str, Any]:
    record = state_song_for_user(song_id, request)
    if not record:
        migrate_history_to_state(request)
        record = state_song_for_user(song_id, request)
    if not record:
        raise HTTPException(status_code=404, detail="Song not found")
    music_sheet = build_music_sheet(record)
    updates = {
        "musicSheet": music_sheet,
        "timeSignature": music_sheet.get("timeSignature") or "4/4",
    }
    updated = update_state_song_for_user(song_id, request, updates)
    if not updated:
        raise HTTPException(status_code=404, detail="Song not found")
    return {"ok": True, "musicSheet": music_sheet, "song": updated}


@app.post("/api/song/{song_id}/generate-sheet-music-pdf")
@app.post("/tracks/{song_id}/generate-sheet-music-pdf")
def api_song_generate_sheet_music_pdf(song_id: str, request: Request) -> dict[str, Any]:
    record = state_song_for_user(song_id, request)
    if not record:
        migrate_history_to_state(request)
        record = state_song_for_user(song_id, request)
    if not record:
        raise HTTPException(status_code=404, detail="Song not found")

    existing_url = record.get("sheetMusicPdfUrl") or record.get("musicSheetPdfUrl")
    if existing_url:
        return {
            "ok": True,
            "pdfUrl": existing_url,
            "sheetMusicPdfUrl": existing_url,
            "filename": Path(str(existing_url)).name or f"{song_id.lower()}-sheet-music.pdf",
            "song": record,
        }

    ctx = sheet_context(record)
    ctx.update({
        "song_title": ctx["title"],
        "title": ctx["title"],
        "song_id": record_code(record) or song_id,
        "tempo": ctx["bpm"],
        "bpm": ctx["bpm"],
        "timeSignature": ctx["timeSignature"],
        "chords": ctx["chords"],
    })
    try:
        result = generate_music_sheet_pdf(ctx)
    except Exception as exc:
        logger.exception("Sheet music PDF generation failed for %s", song_id)
        raise HTTPException(status_code=500, detail="Sheet music PDF could not be generated.") from exc

    pdf_path = Path(result.get("music_sheet_pdf") or "")
    if not pdf_path.exists():
        raise HTTPException(status_code=500, detail="Sheet music PDF could not be generated.")
    pdf_url = generated_static_url(pdf_path)
    filename = pdf_path.name
    updates = {
        "sheetMusicPdfUrl": pdf_url,
        "musicSheetPdfUrl": pdf_url,
        "sheetMusicPdfPath": str(pdf_path),
    }
    updated = update_state_song_for_user(song_id, request, updates)
    if not updated:
        raise HTTPException(status_code=404, detail="Song not found")
    return {"ok": True, "pdfUrl": pdf_url, "sheetMusicPdfUrl": pdf_url, "filename": filename, "song": updated}


@app.get("/api/song/{song_id}/lyrics-sheet")
def api_song_lyrics_sheet(song_id: str, request: Request) -> Response:
    record = state_song_for_user(song_id, request)
    if not record:
        migrate_history_to_state(request)
        record = state_song_for_user(song_id, request)
    if not record:
        raise HTTPException(status_code=404, detail="Song not found")
    song_record = song_detail_payload(record, owner_email=current_user_email(request))
    sheet = song_record.get("sheet") or {}
    lyrics = sheet.get("lyrics") if isinstance(sheet.get("lyrics"), dict) else None
    if not lyrics or not str(lyrics.get("text") or "").strip():
        raise HTTPException(status_code=404, detail="Lyrics sheet is not available for this song.")
    title = record_title(song_record)
    lines = [
        "EchoEcho Lyrics Sheet",
        "",
        f"Title: {title}",
        f"Provider: {sheet.get('provider') or provider_label(str(song_record.get('mode') or ''))}",
        f"Mood: {sheet.get('mood') or song_record.get('mood') or ''}",
        f"Genre: {sheet.get('genre') or song_record.get('genre') or ''}",
        f"BPM: {sheet.get('bpm') or song_record.get('bpm') or ''}",
        "",
        f"[{lyrics.get('label') or 'Verse'}]",
        str(lyrics.get("text") or "").strip(),
        "",
        "Chords: " + " - ".join(sheet.get("chords") or song_record.get("chords") or []),
    ]
    safe_title = "".join(ch.lower() if ch.isalnum() else "_" for ch in title).strip("_") or song_id
    return Response(
        "\n".join(lines),
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="echoecho_{safe_title}_lyrics_sheet.txt"'},
    )


@app.get("/api/song/{song_id}/audio")
def api_song_audio(song_id: str, request: Request) -> FileResponse:
    record = state_song_for_user(song_id, request)
    if not record:
        migrate_history_to_state(request)
        record = state_song_for_user(song_id, request)
    if not record:
        raise HTTPException(status_code=404, detail="Song not found")
    filename = original_filename(record)
    if not filename:
        raise HTTPException(status_code=404, detail="Audio file not found")
    path = GENERATED_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path, media_type=media_type_for(filename), filename=filename)


@app.post("/api/library/{code}/trim")
def trim_library_item(code: str, http_request: Request, request: TrimRequest | None = None) -> dict[str, Any]:
    user_id = current_user_id(http_request)
    request = request or TrimRequest()
    history, record = find_history_record(code, user_id)
    if not record:
        raise HTTPException(status_code=404, detail="Library item not found.")

    normalized = record_code(record)
    source_name = original_filename(record)
    if not source_name:
        raise HTTPException(status_code=404, detail="Original audio file is missing from metadata.")
    if not source_name.lower().endswith(".mp3"):
        raise HTTPException(status_code=400, detail="Only API-generated MP3 files can be trimmed.")

    source = GENERATED_DIR / source_name
    if not source.exists():
        raise HTTPException(status_code=404, detail="Original audio file was not found on disk.")

    destination_name = f"ECHO_{normalized}_trimmed.mp3"
    destination = GENERATED_DIR / destination_name
    if destination.exists() and not request.force:
        record["trimmed_audio_filename"] = destination_name
        write_history(history, user_id)
        return {"ok": True, "song": with_urls(record)}

    duration = request.duration_seconds or get_default_trim_duration_seconds()
    try:
        trim_audio(source, destination, duration_seconds=duration)
    except RuntimeError as exc:
        if "FFmpeg is required for trimming" in str(exc):
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        raise

    record["trimmed_audio_filename"] = destination_name
    record["trimmed_duration_seconds"] = duration
    record["trimmed_created_at"] = datetime.now(timezone.utc).isoformat()
    write_history(history, user_id)
    return {"ok": True, "song": with_urls(record)}


def callback_task_id(payload: dict[str, Any]) -> str | None:
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    task_id = data.get("taskId") or data.get("task_id")
    return str(task_id) if task_id else None


def callback_type(payload: dict[str, Any]) -> str | None:
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    value = data.get("callbackType") or data.get("status")
    return str(value) if value else None


def store_callback(payload: dict[str, Any]) -> dict[str, Any]:
    received_at = datetime.now(timezone.utc).isoformat()
    task_id = callback_task_id(payload)
    event = {
        "received_at": received_at,
        "task_id": task_id,
        "status": callback_type(payload),
        "payload": payload,
    }
    callbacks = read_callbacks()
    callbacks.append(event)
    write_callbacks(callbacks)

    if task_id:
        for song_file in iter_user_song_files():
            history_records = read_json_list(song_file)
            updated = False
            for record in history_records:
                if record.get("task_id") == task_id:
                    record["callback_status"] = event["status"]
                    record["callback_received_at"] = received_at
                    record["callback_payload"] = payload
                    updated = True
            if updated:
                write_json_list(song_file, history_records)
    return event


def kieai_callback(payload: dict[str, Any]) -> dict[str, Any]:
    event = store_callback(payload)
    logger.info("KieAI callback received task_id=%s status=%s", event.get("task_id"), event.get("status"))
    return {"ok": True, "status": "received"}


for callback_route in dict.fromkeys((get_callback_path(), "/api/kieai/callback", "/kieai/callback")):
    app.add_api_route(callback_route, kieai_callback, methods=["POST"])


@app.get("/api/kieai/callbacks")
def kieai_callbacks() -> dict[str, Any]:
    return {"callbacks": read_callbacks()}


@app.get("/download/{code}")
def download(code: str, request: Request) -> FileResponse:
    return download_original(code, request)


@app.get("/download/{code}/original")
def download_original(code: str, request: Request) -> FileResponse:
    _, record = find_history_record(code, current_user_id(request))
    if not record:
        raise HTTPException(status_code=404, detail="Library item not found.")
    filename = original_filename(record)
    if not filename:
        raise HTTPException(status_code=404, detail="Original audio file not found")
    path = GENERATED_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Original audio file not found")
    return FileResponse(path, media_type=media_type_for(filename), filename=filename)


@app.get("/download/{code}/trimmed")
def download_trimmed(code: str, request: Request) -> FileResponse:
    _, record = find_history_record(code, current_user_id(request))
    if not record:
        raise HTTPException(status_code=404, detail="Library item not found.")
    filename = trimmed_filename(record)
    if not filename:
        raise HTTPException(status_code=404, detail="Trimmed audio file not found")
    path = GENERATED_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Trimmed audio file not found")
    return FileResponse(path, media_type="audio/mpeg", filename=filename)


@app.get("/audio/{song_id}.wav")
def get_audio(song_id: str) -> FileResponse:
    cleaned_song_id = song_id.strip().upper()
    if len(cleaned_song_id) != 4 or not cleaned_song_id.isalpha():
        raise HTTPException(status_code=400, detail="Invalid song ID.")
    path = GENERATED_DIR / f"{cleaned_song_id}.wav"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found.")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


@app.get("/docs", include_in_schema=False)
def docs_page() -> FileResponse:
    path = FRONTEND_DIR / "docs.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Docs page not found.")
    return FileResponse(path, media_type="text/html")


@app.get("/docs/{path:path}", include_in_schema=False)
def docs_subpage(path: str) -> FileResponse:
    doc = FRONTEND_DIR / "docs.html"
    if not doc.exists():
        raise HTTPException(status_code=404, detail="Docs page not found.")
    return FileResponse(doc, media_type="text/html")


@app.get("/")
def serve_frontend() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(index_path)


@app.get("/login", include_in_schema=False)
def serve_login_page() -> FileResponse:
    path = FRONTEND_DIR / "login.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Login page not found.")
    return FileResponse(path, media_type="text/html")


@app.get("/signup", include_in_schema=False)
def serve_signup_page() -> FileResponse:
    path = FRONTEND_DIR / "signup.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Signup page not found.")
    return FileResponse(path, media_type="text/html")


@app.get("/{page}.html", include_in_schema=False)
def serve_page(page: str) -> FileResponse:
    path = FRONTEND_DIR / f"{Path(page).name}.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Page not found.")
    return FileResponse(path, media_type="text/html")


@app.get("/wave-bg.js", include_in_schema=False)
def wave_bg() -> FileResponse:
    path = FRONTEND_DIR / "wave-bg.js"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Script not found")
    return FileResponse(path, media_type="text/javascript")


@app.get("/three.module.min.js", include_in_schema=False)
def three_js() -> FileResponse:
    path = STATIC_DIR / "three.module.min.js"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Three.js not found")
    return FileResponse(path, media_type="text/javascript")


@app.get("/logo.svg", include_in_schema=False)
def logo() -> FileResponse:
    path = FRONTEND_DIR / "logo.svg"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Logo not found")
    return FileResponse(path, media_type="image/svg+xml")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    path = STATIC_DIR / "favicon.ico"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Favicon not found")
    return FileResponse(path)


app.mount("/generated", StaticFiles(directory=GENERATED_DIR), name="generated")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
