"""ingest.py - PDF reading, title detection, section detection and chunking.

Steps (guide, Project 2):
  1. extract text page by page, clean line breaks and hyphenation
  2. detect the paper title (largest text on page 1, with fallbacks)
  3. label sections (Abstract, Introduction, Method, Results, Conclusion ...)
  4. chunk (about 800 characters, 150 overlap), ignore tiny chunks, skip references
"""
import os
import re
from pypdf import PdfReader

CHUNK_SIZE = 600
OVERLAP = 100
MIN_CHUNK = 50

# heading text (lower case) -> label stored in the chunk metadata
HEADINGS = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "related work": "Related Work",
    "background": "Background",
    "preliminaries": "Background",
    "method": "Method", "methods": "Method", "methodology": "Method",
    "approach": "Method", "proposed method": "Method", "proposed approach": "Method",
    "experiments": "Experiments", "experimental setup": "Experiments",
    "results": "Results", "experimental results": "Results",
    "experiments and results": "Results", "results and discussion": "Results",
    "evaluation": "Results",
    "discussion": "Discussion",
    "conclusion": "Conclusion", "conclusions": "Conclusion",
    "conclusion and future work": "Conclusion", "conclusions and future work": "Conclusion",
    "references": "References", "bibliography": "References",
}

_NUM_PREFIX = re.compile(r"^(?:(?:\d+|[IVX]+)(?:\.\d+)*\.?\s+)?")
_ABSTRACT = re.compile(r"^abstract(?:\s*[:.\u2014\u2013\-]\s*|\s*$)(.*)$", re.I)


def detect_heading(line):
    """Return (section_label, rest_of_line). label is None if the line is normal text."""
    s = line.strip()
    m = _ABSTRACT.match(s)
    if m:
        return "Abstract", m.group(1).strip()
    if not s or len(s) > 60 or s.endswith("."):
        return None, s
    core = _NUM_PREFIX.sub("", s).strip().rstrip(":").lower()
    core = re.sub(r"\s+", " ", core)
    label = HEADINGS.get(core)
    if label:
        return label, ""
    return None, s


def clean_text(text):
    """Join words that were split with a hyphen at the end of a line."""
    text = text.replace("\r", "")
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def split_segments(raw, section):
    """Split page text into (section, text) segments.
    Returns (segments, current_section, reached_references)."""
    segments, buf = [], []

    def flush():
        text = re.sub(r"\s+", " ", " ".join(buf)).strip()
        if text:
            segments.append((section, text))
        buf.clear()

    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        label, rest = detect_heading(line)
        if label:
            flush()
            if label == "References":
                return segments, section, True      # stop: skip the reference list
            section = label
            if rest:
                buf.append(rest)
            continue
        buf.append(line)
    flush()
    return segments, section, False


def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    """Overlapping chunks that do not cut words; chunks shorter than MIN_CHUNK are ignored."""
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            space = text.rfind(" ", start, end)
            if space > start + size // 2:
                end = space
        piece = text[start:end].strip()
        if len(piece) >= MIN_CHUNK:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


# ---------------------------- title detection ----------------------------
def _title_from_fonts(page):
    """The title is normally the text with the biggest font on page 1."""
    parts = []

    def visitor(text, cm, tm, font_dict, font_size):
        t = (text or "").strip()
        if len(t) < 2:
            return
        try:
            size = abs(float(font_size) * float(tm[3]) * float(cm[3]))
        except Exception:
            return
        if size > 0:
            parts.append((t, size))

    try:
        page.extract_text(visitor_text=visitor)
    except Exception:
        return ""
    if not parts:
        return ""
    biggest = max(s for _, s in parts)
    big = [t for t, s in parts if s >= biggest * 0.92]
    return re.sub(r"\s+", " ", " ".join(big)).strip()


def _title_from_lines(page):
    """Fallback: first lines of page 1 until authors / abstract start."""
    try:
        lines = [l.strip() for l in (page.extract_text() or "").split("\n") if l.strip()]
    except Exception:
        return ""
    title = []
    for line in lines[:10]:
        low = line.lower()
        if re.match(r"^(arxiv|https?:|page \d|\d+$)", low):
            continue
        if low.startswith("abstract") or "@" in line or re.search(
                r"university|institute|department|school of|laborator", low):
            break
        title.append(line)
        if len(title) == 3:
            break
    return re.sub(r"\s+", " ", " ".join(title)).strip()


def detect_title(reader, fallback):
    first = reader.pages[0]
    candidates = []
    try:
        candidates.append(_title_from_fonts(first))
    except Exception:
        pass
    try:
        candidates.append(_title_from_lines(first))
    except Exception:
        pass
    if reader.metadata:
        try:
            meta_title = (getattr(reader.metadata, "title", "") or "").strip()
            if meta_title:
                candidates.append(meta_title)
        except Exception:
            pass
    for candidate in candidates:
        if candidate and 8 <= len(candidate) <= 250 and candidate.lower() != "abstract":
            return candidate
    return os.path.splitext(fallback)[0]


# ------------------------------- main entry -------------------------------
def ingest_pdf(path):
    """Read one paper. Returns {"paper_id","title","pages","chunks"}.
    Raises ValueError if the PDF has no extractable text (scanned file)."""
    file_name = os.path.basename(path)
    reader = PdfReader(path)
    if len(reader.pages) == 0:
        raise ValueError("The PDF has no pages.")
    title = detect_title(reader, file_name)

    chunks, section, done = [], "Front matter", False
    for page_no, page in enumerate(reader.pages, start=1):
        if done:
            break
        raw = clean_text(page.extract_text() or "")
        segments, section, done = split_segments(raw, section)
        for sec, text in segments:
            for piece in chunk_text(text):
                chunks.append({"paper_id": file_name, "paper": title, "page": page_no,
                               "section": sec, "text": piece})
    if not chunks:
        raise ValueError("No text found. This looks like a scanned PDF (images only).")
    return {"paper_id": file_name, "title": title, "pages": len(reader.pages), "chunks": chunks}
