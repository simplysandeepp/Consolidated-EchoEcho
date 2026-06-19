from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4
from typing import Any

from .music_sheet import _choose_chords, _pdf_safe_text


BASE_DIR = Path(__file__).resolve().parents[1]
CHORD_SHEET_DIR = BASE_DIR / "generated" / "chord_sheets"

CHORD_SHAPES = {
    "A": [0, 2, 2, 2, 0, -1],
    "Am": [0, 1, 2, 2, 0, -1],
    "A7": [0, 2, 0, 2, 0, -1],
    "Bb": [1, 3, 3, 3, 1, -1],
    "B7": [2, 0, 2, 1, 2, -1],
    "Bm": [2, 3, 4, 4, 2, -1],
    "C": [0, 1, 0, 2, 3, -1],
    "Cmaj7": [0, 0, 0, 2, 3, -1],
    "D": [2, 3, 2, 0, -1, -1],
    "D7": [2, 1, 2, 0, -1, -1],
    "Dm": [1, 3, 2, 0, -1, -1],
    "E": [0, 0, 1, 2, 2, 0],
    "Em": [0, 0, 0, 2, 2, 0],
    "Em7": [0, 3, 0, 0, 2, 0],
    "F": [1, 1, 2, 3, 3, 1],
    "Fmaj7": [0, 1, 2, 3, -1, -1],
    "G": [3, 0, 0, 0, 2, 3],
    "G7": [1, 0, 0, 0, 2, 3],
}


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_")
    return cleaned[:60] or "chord_sheet"


def _context_summary(context: dict[str, Any]) -> dict[str, str]:
    title = str(context.get("song_title") or context.get("title") or context.get("song_id") or "EchoEcho Chord Sheet")
    mood = str(context.get("mood") or "Selected mood")
    genre = str(context.get("genre") or context.get("style") or "Selected style")
    tempo = str(context.get("tempo") or context.get("bpm") or "90")
    chords = _choose_chords(mood)
    return {
        "title": title,
        "mood": mood,
        "genre": genre,
        "tempo": tempo,
        "chords": " | ".join(chords),
    }


def chord_sheet_preview(context: dict[str, Any]) -> str:
    summary = _context_summary(context)
    return (
        f"Title: {summary['title']}\n"
        f"Tempo: {summary['tempo']} BPM    Style: {summary['genre']}    Mood: {summary['mood']}\n"
        f"Chord progression: {summary['chords']}\n"
        "Format: beginner guitar chord boxes with suggested progression."
    )


def _draw_chord_box(pdf: Any, x: float, y: float, chord: str) -> None:
    shape = CHORD_SHAPES.get(chord, CHORD_SHAPES.get(chord.replace("maj7", ""), [0, 2, 2, 2, 0, -1]))
    width = 22
    height = 28
    string_gap = width / 5
    fret_gap = height / 5

    pdf.set_font("Helvetica", "B", 12)
    pdf.text(x + 7, y - 5, _pdf_safe_text(chord))
    pdf.set_line_width(0.45)
    for string_index in range(6):
        xx = x + string_index * string_gap
        pdf.line(xx, y, xx, y + height)
    for fret_index in range(6):
        yy = y + fret_index * fret_gap
        pdf.line(x, yy, x + width, yy)

    pdf.set_fill_color(15, 15, 15)
    pdf.set_font("Helvetica", "", 6)
    for string_index, fret in enumerate(shape):
        xx = x + string_index * string_gap
        if fret == -1:
            pdf.text(xx - 1, y - 1, "x")
        elif fret == 0:
            pdf.text(xx - 1, y - 1, "o")
        else:
            dot_y = y + (fret - 0.5) * fret_gap
            pdf.ellipse(xx - 1.7, dot_y - 1.7, 3.4, 3.4, style="F")


def generate_chord_sheet_pdf(context: dict[str, Any]) -> dict[str, str]:
    from fpdf import FPDF

    summary = _context_summary(context)
    title = summary["title"].strip() or "EchoEcho Chord Sheet"
    chords = summary["chords"].split(" | ")
    unique_chords = list(dict.fromkeys(chords))
    CHORD_SHEET_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{_safe_filename(title)}_{uuid4().hex[:8]}.pdf"
    output_path = CHORD_SHEET_DIR / filename

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.set_draw_color(20, 20, 20)

    pdf.set_font("Helvetica", "B", 22)
    pdf.cell(0, 14, _pdf_safe_text(f"{title} Chord Sheet"), align="C", ln=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7, _pdf_safe_text(f"{summary['genre']}  |  {summary['mood']}  |  {summary['tempo']} BPM"), align="C", ln=1)
    pdf.ln(10)

    x_positions = [25, 68, 111, 154]
    y = 48
    for index, chord in enumerate(unique_chords):
        x = x_positions[index % len(x_positions)]
        if index and index % len(x_positions) == 0:
            y += 50
        _draw_chord_box(pdf, x, y, chord)

    pdf.set_y(208)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Suggested Progression", ln=1)
    pdf.set_font("Helvetica", "", 12)
    for row in range(4):
        rotated = chords[row % len(chords):] + chords[: row % len(chords)]
        pdf.cell(0, 8, _pdf_safe_text("  |  ".join(rotated)), ln=1)

    pdf.output(str(output_path))
    return {
        "summary": chord_sheet_preview(context),
        "chord_sheet_pdf": str(output_path),
    }
