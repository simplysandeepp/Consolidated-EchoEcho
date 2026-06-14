from __future__ import annotations

import asyncio
import logging
import os
import random
import shutil
import string
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_DIR / ".env")

logger = logging.getLogger(__name__)

KIE_GENERATE_URL = "https://api.kie.ai/api/v1/generate"
KIE_RECORD_INFO_URL = "https://api.kie.ai/api/v1/generate/record-info"
POLL_INTERVAL_SECONDS = 5
POLL_TIMEOUT_SECONDS = 600
TARGET_DURATION_MS = 30_000
DEFAULT_TRIM_DURATION_SECONDS = 30
FAILED_STATUSES = {
    "CREATE_TASK_FAILED",
    "GENERATE_AUDIO_FAILED",
    "CALLBACK_EXCEPTION",
    "SENSITIVE_WORD_ERROR",
}
DEFAULT_CALLBACK_PATH = "/api/kieai/callback"


class KieAIConfigError(RuntimeError):
    pass


class KieAIResponseError(RuntimeError):
    pass


@dataclass
class GenerationInput:
    mood: str
    theme: str
    style: str
    instruments: list[str]
    tempo: int
    energy: int


def build_prompt(data: GenerationInput) -> str:
    instrument_text = ", ".join(data.instruments) if data.instruments else "soft layered textures"
    return (
        f"Create a 30 second instrumental songwriting inspiration sketch. "
        f"Mood: {data.mood}. Theme: {data.theme}. Style: {data.style}. "
        f"Instruments: {instrument_text}. Tempo: {data.tempo} BPM. "
        f"Energy: {data.energy}/10. Keep it memorable, compact, melodic, and emotionally clear."
    )


def build_style(data: GenerationInput) -> str:
    instrument_text = ", ".join(data.instruments) if data.instruments else "minimal instrumentation"
    return f"{data.style}, {data.mood}, {instrument_text}, {data.tempo} BPM, energy {data.energy}/10"


def is_kieai_generation_enabled() -> bool:
    return bool(os.getenv("KIEAI_API_KEY", "").strip())


def get_callback_path() -> str:
    path = os.getenv("KIEAI_CALLBACK_PATH", DEFAULT_CALLBACK_PATH).strip() or DEFAULT_CALLBACK_PATH
    return path if path.startswith("/") else f"/{path}"


def build_callback_url() -> str:
    public_base_url = os.getenv("PUBLIC_BASE_URL", "").strip()
    if not public_base_url:
        raise KieAIConfigError(
            "PUBLIC_BASE_URL is required for KieAI generation because KieAI must POST to a public callback URL. "
            "Localhost is not reachable from KieAI; use ngrok or cloudflared for local development."
        )
    if "localhost" in public_base_url.lower() or "127.0.0.1" in public_base_url:
        raise KieAIConfigError(
            "PUBLIC_BASE_URL must be publicly reachable for KieAI callbacks. "
            "Localhost/127.0.0.1 will not work unless it is exposed through a tunnel such as ngrok or cloudflared."
        )
    return f"{public_base_url.rstrip('/')}{get_callback_path()}"


def validate_kieai_config() -> None:
    if is_kieai_generation_enabled():
        build_callback_url()


def generate_song_id(existing_ids: set[str]) -> str:
    for _ in range(10_000):
        song_id = "".join(random.choices(string.ascii_uppercase, k=4))
        if song_id not in existing_ids:
            return song_id
    raise RuntimeError("Unable to generate a unique song ID")


def extract_audio_url(record: dict[str, Any]) -> str | None:
    response = record.get("response") or {}
    suno_data = response.get("sunoData") or []
    if not suno_data:
        return None
    first_track = suno_data[0] or {}
    return first_track.get("audioUrl") or first_track.get("streamAudioUrl")


def find_audio_converter() -> str | None:
    return shutil.which("ffmpeg") or shutil.which("avconv")


def require_audio_converter() -> str:
    converter = find_audio_converter()
    if converter:
        return converter
    raise RuntimeError(
        "FFmpeg is required for trimming. Original audio is still available."
    )


def get_default_trim_duration_seconds() -> int:
    raw = os.getenv("TRIM_DURATION_SECONDS", str(DEFAULT_TRIM_DURATION_SECONDS)).strip()
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid TRIM_DURATION_SECONDS=%r; using %s", raw, DEFAULT_TRIM_DURATION_SECONDS)
        return DEFAULT_TRIM_DURATION_SECONDS
    return max(1, value)


def sanitized_response_body(response: httpx.Response) -> dict[str, Any] | str:
    try:
        return response.json()
    except ValueError:
        return response.text[:1000]


async def submit_to_suno(client: httpx.AsyncClient, api_key: str, data: GenerationInput, prompt: str) -> str:
    callback_url = build_callback_url()
    payload = {
        "prompt": prompt,
        "customMode": True,
        "instrumental": True,
        "model": "V4",
        "style": build_style(data),
        "title": f"EchoEcho {data.mood} {data.theme}",
        "negativeTags": "harsh noise, distorted vocals, abrupt ending",
        "callBackUrl": callback_url,
    }
    started = time.perf_counter()
    response = await client.post(
        KIE_GENERATE_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    elapsed = time.perf_counter() - started
    logger.info("KieAI submit request time %.2fs", elapsed)
    body = sanitized_response_body(response)
    if not response.is_success:
        logger.error("KieAI submit failed status=%s response=%r", response.status_code, body)
        raise KieAIResponseError(
            f"KieAI generation request failed with HTTP {response.status_code}. "
            "Check your KieAI configuration and callback URL."
        )
    if not isinstance(body, dict):
        logger.error("KieAI submit returned non-JSON response=%r", body)
        raise KieAIResponseError("KieAI generation request returned an unexpected response.")
    if body.get("code") not in (None, 200):
        logger.error("KieAI submit returned API error response=%r", body)
        raise KieAIResponseError(f"KieAI rejected the generation request: {body.get('msg') or 'Unknown error'}")
    task_id = (body.get("data") or {}).get("taskId")
    if not task_id:
        logger.error("KieAI submit response missing taskId response=%r", body)
        raise KieAIResponseError("KieAI accepted the request but did not return a taskId.")
    logger.info("KieAI generation task submitted task_id=%s callback_url=%s", task_id, callback_url)
    return task_id


async def poll_suno(client: httpx.AsyncClient, api_key: str, task_id: str) -> dict[str, Any]:
    started = time.perf_counter()
    deadline = started + POLL_TIMEOUT_SECONDS
    while time.perf_counter() < deadline:
        response = await client.get(
            KIE_RECORD_INFO_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            params={"taskId": task_id},
            timeout=60,
        )
        response.raise_for_status()
        body = response.json()
        data = body.get("data") or {}
        status = data.get("status")
        if status == "SUCCESS":
            logger.info("Song generation time %.2fs", time.perf_counter() - started)
            return data
        if status in FAILED_STATUSES:
            raise RuntimeError(data.get("errorMessage") or f"KieAI generation failed with status {status}")
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError("Timed out waiting for Suno generation to complete")


async def download_audio(client: httpx.AsyncClient, audio_url: str, destination: Path) -> None:
    started = time.perf_counter()
    response = await client.get(audio_url, timeout=120)
    response.raise_for_status()
    destination.write_bytes(response.content)
    logger.info("Download time %.2fs", time.perf_counter() - started)


def trim_audio(source: Path, destination: Path, duration_seconds: int | None = None) -> None:
    converter = require_audio_converter()
    from pydub import AudioSegment

    AudioSegment.converter = converter
    duration_ms = (duration_seconds or get_default_trim_duration_seconds()) * 1000
    started = time.perf_counter()
    audio = AudioSegment.from_file(source)
    if len(audio) > duration_ms:
        audio = audio[:duration_ms]
    audio.export(destination, format="mp3")
    logger.info("Trimming time %.2fs", time.perf_counter() - started)


async def generate_song(
    data: GenerationInput,
    generated_dir: Path,
    existing_ids: set[str],
) -> dict[str, Any]:
    api_key = os.getenv("KIEAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("KIEAI_API_KEY is missing. Add it to .env before generating songs.")

    generated_dir.mkdir(parents=True, exist_ok=True)
    code = generate_song_id(existing_ids)
    prompt = build_prompt(data)
    original_audio_filename = f"ECHO_{code}_original.mp3"
    original_file = generated_dir / original_audio_filename
    total_started = time.perf_counter()

    async with httpx.AsyncClient(follow_redirects=True) as client:
        task_id = await submit_to_suno(client, api_key, data, prompt)
        record = await poll_suno(client, api_key, task_id)
        audio_url = extract_audio_url(record)
        if not audio_url:
            raise RuntimeError("KieAI completed the task but did not provide an audio URL")
        await download_audio(client, audio_url, original_file)

    total_elapsed = time.perf_counter() - total_started

    result = {
        "code": code,
        "song_id": code,
        "task_id": task_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **asdict(data),
        "prompt": prompt,
        "original_audio_filename": original_audio_filename,
        "trimmed_audio_filename": None,
    }
    logger.info(
        "Completed code=%s task_id=%s prompt=%r file=%s total=%.2fs",
        code,
        task_id,
        prompt,
        original_file,
        total_elapsed,
    )
    return result


async def submit_vocal_to_suno(
    client: httpx.AsyncClient,
    api_key: str,
    data: GenerationInput,
    extra_prompt: str = "",
) -> str:
    """Submit a vocal (non-instrumental) generation to Kie.ai/Suno."""
    instrument_text = ", ".join(data.instruments) if data.instruments else ""
    style_parts = [data.style, data.mood]
    if instrument_text:
        style_parts.append(instrument_text)
    style = ", ".join(p for p in style_parts if p)

    description = (
        f"A {data.mood} {data.style} song"
        f"{' about ' + data.theme if data.theme else ''}"
        f"{'. ' + extra_prompt if extra_prompt else ''}."
        f" {data.tempo} BPM, energy {data.energy}/10."
        " Catchy melody, memorable chorus, clear singing vocals."
    )

    payload: dict[str, Any] = {
        "prompt": description,
        "customMode": False,
        "instrumental": False,
        "model": "V4",
        "style": style or "pop, melodic, vocals",
        "title": f"EchoEcho {data.mood} {data.theme or 'Song'}",
        "negativeTags": "harsh noise, abrupt ending, spoken word",
    }

    # Attach callback URL only when PUBLIC_BASE_URL is configured
    try:
        payload["callBackUrl"] = build_callback_url()
    except KieAIConfigError:
        logger.info("No PUBLIC_BASE_URL set — submitting without callback URL")

    response = await client.post(
        KIE_GENERATE_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    body = sanitized_response_body(response)
    if not response.is_success:
        raise KieAIResponseError(
            f"KieAI vocal generation failed with HTTP {response.status_code}."
        )
    if not isinstance(body, dict):
        raise KieAIResponseError("KieAI returned an unexpected non-JSON response.")
    if body.get("code") not in (None, 200):
        raise KieAIResponseError(f"KieAI rejected the request: {body.get('msg') or 'Unknown error'}")
    task_id = (body.get("data") or {}).get("taskId")
    if not task_id:
        raise KieAIResponseError("KieAI accepted the request but did not return a taskId.")
    logger.info("KieAI vocal task submitted task_id=%s", task_id)
    return task_id


async def generate_vocal_song(
    data: GenerationInput,
    generated_dir: Path,
    existing_ids: set[str],
    extra_prompt: str = "",
) -> dict[str, Any]:
    """Generate a full vocal singing track via Kie.ai → Suno."""
    api_key = os.getenv("KIEAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("KIEAI_API_KEY is missing. Add it to .env before generating songs.")

    generated_dir.mkdir(parents=True, exist_ok=True)
    code = generate_song_id(existing_ids)
    original_audio_filename = f"ECHO_{code}_vocal.mp3"
    original_file = generated_dir / original_audio_filename
    total_started = time.perf_counter()

    async with httpx.AsyncClient(follow_redirects=True) as client:
        task_id = await submit_vocal_to_suno(client, api_key, data, extra_prompt)
        record = await poll_suno(client, api_key, task_id)
        audio_url = extract_audio_url(record)
        if not audio_url:
            raise RuntimeError("KieAI completed but did not provide an audio URL.")
        await download_audio(client, audio_url, original_file)

    # Trim to 60 seconds
    trimmed_audio_filename = f"ECHO_{code}_vocal_60s.mp3"
    trimmed_file = generated_dir / trimmed_audio_filename
    try:
        trim_audio(original_file, trimmed_file, duration_seconds=60)
        logger.info("Trimmed vocal to 60s: %s", trimmed_file)
    except Exception as exc:
        logger.warning("Trim failed, using original: %s", exc)
        trimmed_audio_filename = None
        trimmed_file = None

    logger.info(
        "Vocal generation done code=%s task_id=%s file=%s total=%.2fs",
        code, task_id, original_file, time.perf_counter() - total_started,
    )
    return {
        "code": code,
        "song_id": code,
        "task_id": task_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **asdict(data),
        "original_audio_filename": original_audio_filename,
        "trimmed_audio_filename": trimmed_audio_filename,
    }
