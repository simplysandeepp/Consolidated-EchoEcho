from __future__ import annotations

import os
from typing import Any

try:
    from crewai import Agent, Crew, LLM, Task
except ImportError:  # pragma: no cover - exercised only when optional dependency is missing.
    Agent = Crew = LLM = Task = None  # type: ignore[assignment]


class LyricsGenerationError(RuntimeError):
    pass


def create_lyrics_agent(llm: LLM) -> Agent:
    if Agent is None:
        raise LyricsGenerationError("CrewAI is not installed.")

    return Agent(
        role="Lyricist",
        goal=(
            "Write lyrics that complement the mood, melody, and chord progression. "
            "Include a verse and a chorus. Lyrics should match the rhythmic feel of the melody."
        ),
        backstory=(
            "You are an award-winning lyricist who has written songs across pop, indie, "
            "R&B, and folk. You craft words that feel natural to sing and deepen the "
            "emotional impact of the music."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


def _default_model() -> str | None:
    if os.getenv("GROQ_API_KEY"):
        return "groq/llama-3.1-8b-instant"
    return None


def _create_llm() -> LLM:
    if LLM is None:
        raise LyricsGenerationError("CrewAI is not installed.")

    model = _default_model()
    if not model:
        raise LyricsGenerationError("No lyrics LLM API key is configured.")
    return LLM(model=model)


def generate_lyrics(context: dict[str, Any]) -> dict[str, str]:
    if Crew is None or Task is None:
        raise LyricsGenerationError("CrewAI is not installed.")

    mood = context.get("mood") or "the selected mood"
    genre = context.get("genre") or context.get("style") or "the selected style"
    theme = context.get("theme") or context.get("prompt") or "the song idea"
    tempo = context.get("tempo") or context.get("bpm") or "the chosen"
    duration = int(context.get("durationSeconds") or context.get("duration") or 60)
    max_lines = int(context.get("maxLines") or (8 if duration <= 30 else 16 if duration <= 60 else 24 if duration <= 90 else 48))
    title = context.get("title") or "Generated Track"
    instruments = context.get("instruments") or []
    prompt = context.get("prompt") or ""

    llm = _create_llm()
    agent = create_lyrics_agent(llm)
    task = Task(
        description=(
            "You are a professional songwriter.\n\n"
            "Create lyrics for an AI-generated song preview.\n\n"
            "Track metadata:\n"
            f"Title: {title}\n"
            f"Mood: {mood}\n"
            f"Genre: {genre}\n"
            f"BPM: {tempo}\n"
            f"Duration: {duration} seconds\n"
            f"Theme: {theme}\n"
            f"Instruments: {instruments}\n"
            f"Music prompt: {prompt}\n\n"
            "Strict rules:\n"
            f"- Lyrics must fit inside {duration} seconds.\n"
            f"- Maximum lines: {max_lines}\n"
            "- Use section labels.\n"
            "- Do not write one paragraph.\n"
            "- Do not write a full song for a short preview.\n"
            "- Keep every line short and singable.\n"
            "- Return only formatted lyrics.\n\n"
            "Required format:\n\n"
            "[Verse 1]\n"
            "Line one\n"
            "Line two\n"
            "Line three\n"
            "Line four\n\n"
            "[Chorus]\n"
            "Line one\n"
            "Line two\n"
            "Line three\n"
            "Line four\n\n"
            "No explanation. Avoid quoting or imitating existing songs."
        ),
        expected_output=(
            f"Original formatted lyrics with section labels and no more than {max_lines} lines."
        ),
        agent=agent,
    )
    result = Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
    text = str(result).strip()
    if not text:
        raise LyricsGenerationError("Lyrics agent returned an empty response.")
    return {"text": text, "structure": "verse/chorus"}
