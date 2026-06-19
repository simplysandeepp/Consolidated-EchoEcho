from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4


BASE_DIR = Path(__file__).resolve().parents[1]
LYRICS_DIR = BASE_DIR / "generated" / "lyrics"


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_")
    return cleaned[:60] or "lyrics_sheet"


def _is_section_header(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("[") and stripped.endswith("]") and len(stripped) > 2


def _pdf_safe_text(value: str) -> str:
    return value.encode("latin-1", "replace").decode("latin-1")


def _pdf_escape(value: str) -> str:
    return _pdf_safe_text(value).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _write_basic_pdf(output_path: Path, title: str, lyrics_text: str) -> None:
    lines = [title, "Lyrics Sheet", ""] + lyrics_text.splitlines()
    y = 790
    content = ["BT", "/F1 18 Tf", "50 790 Td", f"({_pdf_escape(title)}) Tj"]
    content.extend(["/F1 14 Tf", "0 -28 Td", "(Lyrics Sheet) Tj", "/F1 11 Tf"])
    y -= 56

    for line in lines[3:]:
        if y < 60:
            break
        safe_line = _pdf_escape(line.strip())
        if not safe_line:
            content.append("0 -14 Td")
            y -= 14
            continue
        content.append(f"0 -16 Td ({safe_line}) Tj")
        y -= 16

    content.append("ET")
    stream = "\n".join(content).encode("latin-1", "replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    output_path.write_bytes(pdf)


def _write_fpdf_pdf(output_path: Path, title: str, lyrics_text: str) -> None:
    from fpdf import FPDF

    title = _pdf_safe_text(title)
    lyrics_text = _pdf_safe_text(lyrics_text)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.multi_cell(0, 10, title)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Lyrics Sheet", ln=1)
    pdf.ln(4)

    for line in lyrics_text.splitlines():
        if _is_section_header(line):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(0, 8, line.strip())
            pdf.set_font("Helvetica", "", 12)
        elif line.strip():
            pdf.set_font("Helvetica", "", 12)
            pdf.multi_cell(0, 8, line.strip())
        else:
            pdf.ln(4)

    pdf.output(str(output_path))


def generate_lyrics_pdf(song_title: str, lyrics: str) -> str:
    title = song_title.strip() or "Untitled Song"
    lyrics_text = lyrics.strip() or "No lyrics provided."
    LYRICS_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"{_safe_filename(title)}_{uuid4().hex[:8]}.pdf"
    output_path = LYRICS_DIR / filename
    try:
        _write_fpdf_pdf(output_path, title, lyrics_text)
    except ImportError:
        _write_basic_pdf(output_path, title, lyrics_text)
    return str(output_path)
