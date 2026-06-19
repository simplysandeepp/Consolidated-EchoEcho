from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
MUSIC_SHEET_DIR = BASE_DIR / "generated" / "music_sheets"

MAJOR_KEYS = ["C", "G", "D", "A", "F", "Bb", "Eb"]
MINOR_KEYS = ["Am", "Em", "Dm", "Bm", "Fm", "Cm"]

PROGRESSIONS = {
    "happy": ["C", "G", "Am", "F"],
    "hopeful": ["C", "Em", "F", "G"],
    "dreamy": ["Am", "Fmaj7", "C", "G"],
    "melancholic": ["Am", "Em", "F", "G"],
    "sad": ["Dm", "Bb", "F", "C"],
    "dark": ["Em", "C", "G", "D"],
    "romantic": ["Cmaj7", "Am7", "Dm7", "G7"],
    "epic": ["Dm", "Bb", "F", "C"],
}


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_")
    return cleaned[:60] or "music_sheet"


def _pdf_safe_text(value: str) -> str:
    return value.encode("latin-1", "replace").decode("latin-1")


def _choose_key(mood: str, genre: str) -> str:
    text = f"{mood} {genre}".lower()
    if any(word in text for word in ("sad", "dark", "melancholic", "heartbreak")):
        return MINOR_KEYS[abs(hash(text)) % len(MINOR_KEYS)]
    return MAJOR_KEYS[abs(hash(text)) % len(MAJOR_KEYS)]


def _choose_chords(mood: str) -> list[str]:
    text = mood.lower()
    for keyword, chords in PROGRESSIONS.items():
        if keyword in text:
            return chords
    return ["C", "Am", "F", "G"]


def _sheet_summary(context: dict[str, Any]) -> dict[str, str]:
    title = str(context.get("song_title") or context.get("title") or context.get("song_id") or "EchoEcho Music Sheet")
    mood = str(context.get("mood") or "Selected mood")
    genre = str(context.get("genre") or context.get("style") or "Selected style")
    theme = str(context.get("theme") or "Song idea")
    tempo = str(context.get("tempo") or context.get("bpm") or "90")
    instruments = context.get("instruments") or []
    instrument_text = ", ".join(instruments) if isinstance(instruments, list) else str(instruments)
    lyrics = str(context.get("lyrics") or "")
    key = _choose_key(mood, genre)
    chords = _choose_chords(mood)
    return {
        "title": title,
        "mood": mood,
        "genre": genre,
        "theme": theme,
        "tempo": tempo,
        "instruments": instrument_text or "Lead melody and chords",
        "key": key,
        "time_signature": "4/4",
        "chords": " | ".join(chords),
        "lyrics": lyrics,
    }


def music_sheet_preview(context: dict[str, Any]) -> str:
    summary = _sheet_summary(context)
    return (
        f"Title: {summary['title']}\n"
        f"Key: {summary['key']}    Time: {summary['time_signature']}    Tempo: {summary['tempo']} BPM\n"
        f"Style: {summary['genre']}    Mood: {summary['mood']}\n"
        f"Instruments: {summary['instruments']}\n"
        f"Chord sketch: {summary['chords']}\n"
        "Notation: lead melody staff with chord symbols for musician reference."
    )


def _lyric_words(lyrics: str) -> list[str]:
    cleaned = re.sub(r"\[[^\]]+\]", " ", lyrics)
    words = re.findall(r"[A-Za-z']+", cleaned)
    return words[:48] or ["melody", "moves", "with", "the", "song"]


def _draw_staff(pdf: Any, x: float, y: float, width: float, chords: list[str], words: list[str], row: int) -> None:
    staff_gap = 3.2
    clef_width = 18
    measure_width = (width - clef_width) / 4

    pdf.set_font("Helvetica", "", 24)
    pdf.text(x + 1, y + 13, "G")
    pdf.set_font("Helvetica", "B", 8)
    pdf.text(x + 11, y + 5, "4")
    pdf.text(x + 11, y + 12, "4")

    for line_index in range(5):
        yy = y + line_index * staff_gap
        pdf.line(x + clef_width, yy, x + width, yy)

    for bar_index in range(5):
        xx = x + clef_width + bar_index * measure_width
        pdf.line(xx, y, xx, y + staff_gap * 4)

    pdf.set_font("Helvetica", "", 9)
    note_offsets = [9, 6, 3, 7, 2, 10, 5, 1]
    for measure in range(4):
        chord = chords[(measure + row) % len(chords)]
        measure_x = x + clef_width + measure * measure_width
        pdf.text(measure_x + 4, y - 3, _pdf_safe_text(chord))
        for note_index in range(2):
            note_x = measure_x + 12 + note_index * 16
            note_y = y + note_offsets[(measure * 2 + note_index + row) % len(note_offsets)]
            pdf.ellipse(note_x, note_y, 4.2, 3.0, style="F")
            pdf.line(note_x + 4.2, note_y + 1.5, note_x + 4.2, note_y - 11)
            word = words[(row * 8 + measure * 2 + note_index) % len(words)]
            pdf.set_font("Helvetica", "", 7)
            pdf.text(note_x - 2, y + 22, _pdf_safe_text(word[:10]))
            pdf.set_font("Helvetica", "", 9)


def generate_music_sheet_pdf(context: dict[str, Any]) -> dict[str, str]:
    from fpdf import FPDF

    summary = _sheet_summary(context)
    title = summary["title"].strip() or "EchoEcho Music Sheet"
    MUSIC_SHEET_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{_safe_filename(title)}_{uuid4().hex[:8]}.pdf"
    output_path = MUSIC_SHEET_DIR / filename

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()
    pdf.set_draw_color(20, 20, 20)
    pdf.set_fill_color(20, 20, 20)

    pdf.set_font("Helvetica", "B", 19)
    pdf.cell(0, 12, _pdf_safe_text(title), align="C", ln=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(
        0,
        7,
        _pdf_safe_text(
            f"Music Sheet  |  Key {summary['key']}  |  {summary['time_signature']}  |  {summary['tempo']} BPM"
        ),
        ln=1,
    )
    pdf.cell(0, 7, _pdf_safe_text(f"Style: {summary['genre']}  |  Mood: {summary['mood']}"), ln=1)
    pdf.ln(10)

    chords = summary["chords"].split(" | ")
    words = _lyric_words(summary["lyrics"])
    y = 58
    for row in range(5):
        pdf.set_font("Helvetica", "", 7)
        pdf.text(9, y + 5, str(row * 2 + 1))
        _draw_staff(pdf, 18, y, 174, chords, words, row)
        y += 38

    pdf.ln(4)
    pdf.set_y(248)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "Chord Sketch", ln=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 7, _pdf_safe_text(summary["chords"]))

    pdf.output(str(output_path))
    return {
        "summary": music_sheet_preview(context),
        "music_sheet_pdf": str(output_path),
    }
