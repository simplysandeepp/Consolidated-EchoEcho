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
    time_signature = str(context.get("timeSignature") or context.get("time_signature") or "4/4")
    instruments = context.get("instruments") or []
    instrument_text = ", ".join(instruments) if isinstance(instruments, list) else str(instruments)
    lyrics = str(context.get("lyrics") or "")
    key = str(context.get("key") or _choose_key(mood, genre))
    chords_value = context.get("chords") or _choose_chords(mood)
    if isinstance(chords_value, list):
        chords = [str(chord).strip() for chord in chords_value if str(chord).strip()]
    else:
        chords = [part.strip() for part in str(chords_value).replace("-", "|").split("|") if part.strip()]
    chords = chords or _choose_chords(mood)
    return {
        "title": title,
        "mood": mood,
        "genre": genre,
        "theme": theme,
        "tempo": tempo,
        "instruments": instrument_text or "Lead melody and chords",
        "key": key,
        "time_signature": time_signature,
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


def _pdf_escape(value: str) -> str:
    return _pdf_safe_text(value).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_text(x: float, y: float, size: int, text: str, *, bold: bool = False) -> str:
    font = "F2" if bold else "F1"
    return f"BT /{font} {size} Tf {x:.2f} {y:.2f} Td ({_pdf_escape(text)}) Tj ET\n"


def _pdf_line(x1: float, y1: float, x2: float, y2: float) -> str:
    return f"{x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S\n"


def _write_pdf(path: Path, commands: str) -> None:
    stream = commands.encode("latin-1", "replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref_at = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode("ascii")
    )
    path.write_bytes(output)


def _generate_simple_music_sheet_pdf(summary: dict[str, str], output_path: Path) -> None:
    chords = summary["chords"].split(" | ")
    commands = "0 0 0 RG 0 0 0 rg 1.1 w\n"
    title = summary["title"][:52]
    commands += _pdf_text(250 - len(title) * 2.4, 800, 20, title, bold=True)
    commands += _pdf_text(170, 779, 10, f"Key: {summary['key']}    Tempo: {summary['tempo']} BPM    Time Signature: {summary['time_signature']}")
    commands += _pdf_text(238, 762, 9, "Generated by Echo Echo")

    staff_left = 58
    staff_right = 535
    clef_width = 35
    measure_width = (staff_right - staff_left - clef_width) / 4
    note_steps = [0, 2, 4, 1, 3, 5, 2, 4]
    system_tops = [700, 585, 470, 355, 240]
    for system_index, top in enumerate(system_tops):
        gap = 8
        for line_index in range(5):
            y = top - line_index * gap
            commands += _pdf_line(staff_left + clef_width, y, staff_right, y)
        commands += _pdf_text(staff_left + 5, top - 28, 34, "G", bold=True)
        commands += _pdf_text(staff_left + 24, top - 11, 9, "4", bold=True)
        commands += _pdf_text(staff_left + 24, top - 25, 9, "4", bold=True)

        for bar_index in range(5):
            x = staff_left + clef_width + bar_index * measure_width
            commands += _pdf_line(x, top, x, top - gap * 4)

        for measure in range(4):
            chord = chords[(measure + system_index) % len(chords)]
            mx = staff_left + clef_width + measure * measure_width
            commands += _pdf_text(mx + 8, top + 18, 10, chord, bold=True)
            for note_index in range(2):
                nx = mx + 28 + note_index * 33
                step = note_steps[(system_index * 2 + measure + note_index) % len(note_steps)]
                ny = top - 8 - step * 4
                commands += f"{nx:.2f} {ny:.2f} 7.50 5.00 re f\n"
                commands += _pdf_line(nx + 7.5, ny + 2, nx + 7.5, ny + 34)
            bass_y = top - 58
            if measure == 0:
                for line_index in range(5):
                    y = bass_y - line_index * gap
                    commands += _pdf_line(staff_left + clef_width, y, staff_right, y)
                commands += _pdf_text(staff_left + 8, bass_y - 26, 24, "F", bold=True)
            bx = mx + 25
            commands += f"{bx:.2f} {bass_y - 15:.2f} 8.50 5.50 re f\n"
            commands += _pdf_line(bx + 8.5, bass_y - 13, bx + 8.5, bass_y + 18)
    _write_pdf(output_path, commands)


def generate_music_sheet_pdf(context: dict[str, Any]) -> dict[str, str]:
    summary = _sheet_summary(context)
    title = summary["title"].strip() or "EchoEcho Music Sheet"
    MUSIC_SHEET_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{_safe_filename(title)}_{uuid4().hex[:8]}.pdf"
    output_path = MUSIC_SHEET_DIR / filename

    try:
        from fpdf import FPDF
    except ModuleNotFoundError:
        _generate_simple_music_sheet_pdf(summary, output_path)
        return {
            "summary": music_sheet_preview(context),
            "music_sheet_pdf": str(output_path),
        }

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
