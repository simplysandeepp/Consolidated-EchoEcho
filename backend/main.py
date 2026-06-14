from __future__ import annotations

import json
import logging
import os
import secrets
import string
import time
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
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
from .auth import AuthError, login as auth_login, signup as auth_signup
from .composer import compose_song
from .transcriber import TranscriptionError, transcribe_audio
from .agents.lyrics_agent import generate_lyrics as generate_agent_lyrics
from .agents.copyright_agent.main import check_copyright
from .agents.copyright_agent.models.request_model import CopyrightCheckRequest
from .ace_step_generator import generate_with_ace_step


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
GENERATED_DIR = BASE_DIR / "generated"
STATIC_DIR = BASE_DIR / "static"
HISTORY_FILE = BASE_DIR / "song_history.json"
CALLBACK_FILE = BASE_DIR / "kiai_callbacks.json"
LEGACY_CALLBACK_FILE = BASE_DIR / "kieai_callbacks.json"

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
    if not HISTORY_FILE.exists():
        HISTORY_FILE.write_text("[]", encoding="utf-8")
    if LEGACY_CALLBACK_FILE.exists() and not CALLBACK_FILE.exists():
        LEGACY_CALLBACK_FILE.replace(CALLBACK_FILE)
    if not CALLBACK_FILE.exists():
        CALLBACK_FILE.write_text("[]", encoding="utf-8")


ensure_files()


def read_json_list(path: Path) -> list[dict[str, Any]]:
    ensure_files()
    try:
        raw = path.read_text(encoding="utf-8").strip()
        data = json.loads(raw or "[]")
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        logger.exception("%s is not valid JSON", path.name)
        return []


def write_json_list(path: Path, data: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_history() -> list[dict[str, Any]]:
    return read_json_list(HISTORY_FILE)


def write_history(history: list[dict[str, Any]]) -> None:
    write_json_list(HISTORY_FILE, history)


def append_history(record: dict[str, Any]) -> None:
    history = read_history()
    history.append(record)
    write_history(history)


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
    return {
        "prompt": prompt,
        "mood": first_or_join(request.moods or ([request.mood] if request.mood else []), ""),
        "theme": first_or_join(request.themes or ([request.theme] if request.theme else []), ""),
        "style": first_or_join(request.genres or ([request.style] if request.style else []), ""),
        "genre": first_or_join(request.genres or ([request.style] if request.style else []), ""),
        "tempo": request.tempo,
        "bpm": request.tempo,
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


def find_history_record(code: str) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    normalized = code.upper()
    history = read_history()
    for record in history:
        if record_code(record) == normalized:
            return history, record
    return history, None


def with_urls(record: dict[str, Any]) -> dict[str, Any]:
    code = record_code(record)
    original = original_filename(record)
    trimmed = trimmed_filename(record)
    original_url = f"/generated/{original}" if original else None
    return {
        **record,
        "code": code,
        "song_id": code,
        "filename": original,
        "audio_filename": original,
        "audio_url": original_url,
        "original_audio_url": original_url,
        "original_download_url": f"/download/{code}/original" if code else None,
        "download_url": f"/download/{code}/original" if code else None,
        "trimmed_audio_url": f"/generated/{trimmed}" if trimmed else None,
        "trimmed_download_url": f"/download/{code}/trimmed" if trimmed else None,
    }


def unified_response(record: dict[str, Any]) -> dict[str, Any]:
    song = with_urls(record)
    return {
        "success": True,
        "ok": True,
        "mode": song.get("mode") or "musicgen",
        "audio_url": song.get("audio_url"),
        "filename": song.get("filename"),
        "song_id": song.get("song_id"),
        "prompt_used": song.get("prompt"),
        "generation_time_seconds": song.get("generation_time_seconds"),
        "generation_time": song.get("generation_time"),
        "output_file": song.get("output_file") or (f"generated/{song['filename']}" if song.get("filename") else None),
        "download_filename": song.get("filename"),
        "lyrics": song.get("lyrics"),
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


def create_song_id(extension: str) -> str:
    existing_ids = {record_code(record) for record in read_history()}
    alphabet = string.ascii_uppercase
    for _ in range(1000):
        song_id = "".join(secrets.choice(alphabet) for _ in range(4))
        if song_id not in existing_ids and not (GENERATED_DIR / f"{song_id}.{extension}").exists():
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


async def generate_with_api(request: GenerateRequest) -> dict[str, Any]:
    reset_generation_status(60)
    update_generation_status("Submitting API generation", 10)
    try:
        history = read_history()
        existing_ids = {record_code(item) for item in history}
        generated = await generate_api_song(
            api_input_from_request(request),
            generated_dir=GENERATED_DIR,
            existing_ids=existing_ids,
        )
        generated["mode"] = "api"
        generated["fast"] = request.fast
        generated["duration"] = request.duration
        filename = original_filename(generated)
        generated["filename"] = filename
        generated["audio_filename"] = filename
        generated["audio_url"] = f"/generated/{filename}" if filename else None
        add_agent_sections(generated, request, str(generated.get("prompt") or ""))
        append_history(generated)
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


def generate_with_musicgen(request: GenerateRequest) -> dict[str, Any]:
    total_start = time.perf_counter()
    reset_generation_status()
    update_generation_status("Preparing prompt", 0)
    try:
        song_id = create_song_id("wav")
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
        raise HTTPException(status_code=503, detail="MusicGen is still loading.")

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
        "energy": request.energy,
        "generation_time_seconds": round(total_seconds),
        "generation_time": format_seconds(total_seconds),
        "filename": filename,
        "audio_filename": filename,
        "output_file": f"generated/{filename}",
        "audio_url": f"/generated/{filename}",
        "inference_seconds": round(float(generation_result.get("inference_seconds", 0)), 2),
    }
    add_agent_sections(record, request, prompt)
    append_history(record)
    update_generation_status("Completed", 100)
    return unified_response(record)


async def generate_with_ace_step_mode(request: GenerateRequest) -> dict[str, Any]:
    reset_generation_status(600)
    update_generation_status("Generating lyrics for ACE-Step", 10)
    try:
        song_id = create_song_id("mp3")
        filename = f"ECHO_{song_id}_ace.mp3"
        output_path = GENERATED_DIR / filename

        mood = first_or_join(request.moods or ([request.mood] if request.mood else []), "dreamy")
        theme = first_or_join(request.themes or ([request.theme] if request.theme else []), "inspiration")
        style = first_or_join(request.genres or ([request.style] if request.style else []), "pop")
        prompt_str = f"{mood} {style}, {request.tempo}bpm, {theme}"

        # Use user-provided lyrics if given; only auto-generate if empty
        if request.lyrics and request.lyrics.strip():
            update_generation_status("Using provided lyrics", 20)
            lyrics_text = request.lyrics.strip()
            lyrics_result = {"text": lyrics_text, "structure": "user-provided"}
            lyrics_ok = True
        else:
            update_generation_status("Writing lyrics with AI", 20)
            lyrics_result, lyrics_ok = generate_lyrics_section(request, prompt_str)
            lyrics_text = lyrics_result.get("text", "")

        update_generation_status("Sending to ACE-Step (this may take a few minutes)", 35)
        ace_result = generate_with_ace_step(
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
        append_history(record)
        update_generation_status("Completed", 100)
        return unified_response(record)
    except Exception as exc:
        update_generation_status("Failed", 100)
        logger.exception("ACE-Step generation failed")
        raise HTTPException(status_code=500, detail=f"ACE-Step generation failed: {exc}") from exc


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


class AceStepLyricsRequest(BaseModel):
    mood: str = ""
    genre: str = ""
    instruments: str = ""
    vocal: str = ""
    tempo: int = 90
    energy: int = 5
    prompt: str = ""


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

    system = (
        "You are a professional songwriter. Write original song lyrics in ACE-Step format. "
        "Use ONLY these section tags exactly as written: [verse], [pre-chorus], [chorus], [bridge]. "
        "Each section tag must be on its own line. Write 3-4 lines per section. "
        "Never use [Verse 1] or numbered variants — only [verse] and [chorus]. "
        "Output ONLY the lyrics, no explanations."
    )
    user_prompt = (
        f"Write a full song with [verse], [chorus], [verse], [chorus], [bridge], [chorus] structure.\n"
        f"Mood: {request.mood or 'emotional'}\n"
        f"Genre: {request.genre or 'pop'}\n"
        f"Instruments: {request.instruments or 'piano, guitar'}\n"
        f"Vocal style: {request.vocal or 'smooth'}\n"
        f"Tempo: {request.tempo} BPM ({energy_desc} energy)\n"
        f"Theme/prompt: {request.prompt or 'life and journey'}\n\n"
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
async def kie_vocal_generate(request: KieVocalRequest) -> dict[str, Any]:
    """Generate a full vocal singing track via Kie.ai → Suno (instrumental=False)."""
    reset_generation_status(180)
    update_generation_status("Submitting vocal request to Kie.ai", 8)
    try:
        history = read_history()
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
        generated["audio_url"] = f"/generated/{serve_file}" if serve_file else None
        generated["genre"] = request.genre
        generated["mood_tag"] = request.mood
        generated["bpm"] = request.bpm

        append_history(generated)
        update_generation_status("Done", 100)
        return {"ok": True, "track": generated}

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
async def generate(request: GenerateRequest) -> dict[str, Any]:
    mode = normalize_mode(request.mode)
    if mode == "api":
        return await generate_with_api(request)
    if mode == "ace-step":
        return await generate_with_ace_step_mode(request)
    return generate_with_musicgen(request)


@app.post("/api/generate")
async def api_generate(request: GenerateRequest) -> dict[str, Any]:
    request.mode = "api"
    request.fast = True
    return await generate_with_api(request)


@app.post("/generate-inspiration")
async def generate_inspiration(request: GenerateRequest) -> dict[str, Any]:
    return await api_generate(request)


@app.get("/health")
def health() -> dict[str, Any]:
    with model_lock:
        return {"status": model_status}


@app.get("/status")
def status() -> dict[str, Any]:
    with model_lock:
        return {
            "ok": True,
            "service": "Echo Echo",
            "history_count": len(read_history()),
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
def history() -> dict[str, Any]:
    records = [with_urls(item) for item in read_history()]
    return {"songs": list(reversed(records))}


@app.get("/songs")
def songs() -> dict[str, Any]:
    return history()


@app.get("/inspirations")
def inspirations() -> dict[str, Any]:
    return history()


@app.get("/library/refresh")
def refresh_library() -> dict[str, Any]:
    return history()


@app.post("/api/copyright/check")
def api_copyright_check(request: CopyrightCheckApiRequest) -> dict[str, Any]:
    history_records: list[dict[str, Any]] = []
    record: dict[str, Any] | None = None
    if request.song_id:
        history_records, record = find_history_record(request.song_id)

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
        write_history(history_records)

    return result


@app.get("/song/{song_id}")
def song(song_id: str) -> dict[str, Any]:
    _, record = find_history_record(song_id)
    if record:
        return {"song": with_urls(record)}
    raise HTTPException(status_code=404, detail="Song not found")


@app.post("/api/library/{code}/trim")
def trim_library_item(code: str, request: TrimRequest | None = None) -> dict[str, Any]:
    request = request or TrimRequest()
    history, record = find_history_record(code)
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
        write_history(history)
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
    write_history(history)
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
        history_records = read_history()
        updated = False
        for record in history_records:
            if record.get("task_id") == task_id:
                record["callback_status"] = event["status"]
                record["callback_received_at"] = received_at
                record["callback_payload"] = payload
                updated = True
        if updated:
            write_history(history_records)
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
def download(code: str) -> FileResponse:
    return download_original(code)


@app.get("/download/{code}/original")
def download_original(code: str) -> FileResponse:
    _, record = find_history_record(code)
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
def download_trimmed(code: str) -> FileResponse:
    _, record = find_history_record(code)
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
