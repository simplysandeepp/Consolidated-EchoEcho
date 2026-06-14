from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")
ACE_SPACE = "ACE-Step/ACE-Step"


def _build_prompt(mood: str, style: str, instruments: list[str], tempo: int, energy: int) -> str:
    """Build comma-separated ACE-Step tag prompt from Echo Echo inputs."""
    parts = [p.strip() for p in [mood, style] if p.strip()]
    if instruments:
        parts.extend(i.strip() for i in instruments if i.strip())
    if tempo:
        parts.append(f"{tempo} BPM")
    energy_words = {1: "very calm", 2: "calm", 3: "mellow", 4: "relaxed",
                    5: "moderate", 6: "upbeat", 7: "energetic", 8: "intense",
                    9: "very intense", 10: "extreme"}
    if energy in energy_words:
        parts.append(energy_words[energy])
    return ", ".join(parts)


def _format_lyrics(raw: str) -> str:
    """Normalize any lyric format to ACE-Step [tag] format."""
    import re
    if not raw or not raw.strip():
        return "[verse]\nInstrumental\n[chorus]\nInstrumental"

    text = raw.strip()

    # Normalize bracketed section headers (e.g. [Verse 1], [CHORUS], [Pre-Chorus 2])
    text = re.sub(r'\[verse\s*\d*\]', '[verse]', text, flags=re.IGNORECASE)
    text = re.sub(r'\[pre[\s\-]chorus\s*\d*\]', '[pre-chorus]', text, flags=re.IGNORECASE)
    text = re.sub(r'\[chorus\s*\d*\]', '[chorus]', text, flags=re.IGNORECASE)
    text = re.sub(r'\[bridge\s*\d*\]', '[bridge]', text, flags=re.IGNORECASE)
    text = re.sub(r'\[outro\s*\d*\]', '[outro]', text, flags=re.IGNORECASE)
    text = re.sub(r'\[intro\s*\d*\]', '[intro]', text, flags=re.IGNORECASE)
    text = re.sub(r'\[hook\s*\d*\]', '[chorus]', text, flags=re.IGNORECASE)

    # If already in ACE-Step format after normalization, return as-is
    if any(t in text for t in ["[verse]", "[chorus]", "[bridge]"]):
        return text

    # Fall back: line-by-line conversion for plain-text section headers
    lines = text.split("\n")
    out: list[str] = []
    for line in lines:
        s = line.strip()
        lo = s.lower()
        if not s:
            out.append("")
            continue
        if any(lo.startswith(k) for k in ("verse", "v1", "v2", "v 1", "v 2")):
            out.append("[verse]")
        elif any(lo.startswith(k) for k in ("pre-chorus", "pre chorus")):
            out.append("[pre-chorus]")
        elif any(lo.startswith(k) for k in ("chorus", "hook", "refrain")):
            out.append("[chorus]")
        elif any(lo.startswith(k) for k in ("bridge", "outro", "intro")):
            out.append("[bridge]")
        else:
            out.append(s)
    return "\n".join(out)


def generate_with_ace_step(
    *,
    prompt: str,
    lyrics: str,
    duration: int = 30,
    mood: str = "",
    style: str = "",
    instruments: list[str] | None = None,
    tempo: int = 90,
    energy: int = 5,
    output_path: Path | None = None,
) -> dict:
    """Call the ACE-Step HuggingFace Space and return the generated audio path."""
    try:
        from gradio_client import Client
    except ImportError as exc:
        raise RuntimeError("gradio_client is not installed.") from exc

    logger.info("ACE-Step: connecting to %s", ACE_SPACE)
    client = Client(
        ACE_SPACE,
        headers={"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {},
        httpx_kwargs={"timeout": 300.0},
    )

    ace_prompt = _build_prompt(mood or prompt, style, instruments or [], tempo, energy)
    formatted_lyrics = _format_lyrics(lyrics)

    logger.info("ACE-Step: prompt=%r  duration=%ss", ace_prompt, duration)
    logger.info("ACE-Step: lyrics=\n%s", formatted_lyrics)

    result = client.predict(
        audio_duration=float(min(max(duration, 10), 240)),
        prompt=ace_prompt,
        lyrics=formatted_lyrics,
        infer_step=60,
        guidance_scale=15.0,
        scheduler_type="euler",
        cfg_type="apg",
        omega_scale=10.0,
        manual_seeds=None,
        guidance_interval=0.5,
        guidance_interval_decay=0.0,
        min_guidance_scale=3.0,
        use_erg_tag=True,
        use_erg_lyric=False,
        use_erg_diffusion=True,
        oss_steps=None,
        guidance_scale_text=0.0,
        guidance_scale_lyric=0.0,
        audio2audio_enable=False,
        ref_audio_strength=0.5,
        ref_audio_input=None,
        lora_name_or_path="none",
        api_name="/__call__",
    )

    # Returns (audio_filepath, parameters_json)
    audio_src = result[0] if isinstance(result, (list, tuple)) else result
    if not audio_src:
        raise RuntimeError("ACE-Step returned no audio")

    logger.info("ACE-Step: audio ready at %s", audio_src)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(audio_src), str(output_path))
        logger.info("ACE-Step: saved to %s", output_path)
        return {"audio_path": str(output_path)}

    return {"audio_path": str(audio_src)}
