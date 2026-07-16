#!/usr/bin/env python3
"""
pdf_to_md.py — Convert compliance PDFs to structured Markdown.

Preserves: chapter/section headings (by font size), numbered clauses,
bullet lists, tables (as proper Markdown tables, never flattened),
and reading order. Built for dense legal/regulatory PDFs, not generic docs.

Usage:
    python pdf_to_md.py input.pdf output.md --framework "DPDP Act 2023"

Dependencies:
    pip install pdfplumber --break-system-packages
"""

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — tune per document family if needed
# ─────────────────────────────────────────────────────────────────────────────

# Font-size thresholds relative to the document's most common (body) size.
# A line whose font size is this many points larger than body text is a heading.
H1_DELTA = 6.0   # chapter / part level → "#"
H2_DELTA = 3.0   # section level        → "##"
H3_DELTA = 1.2   # sub-section level    → "###"

BULLET_CHARS = ("•", "●", "○", "▪", "‣", "·", "-", "*")

# Ordered by nesting depth — matched top-down, first match wins.
# (a)/(b)        → depth 1
# (i)/(ii)       → depth 2 (roman, lowercase, in parens)
# (A)/(B)        → depth 2 (single capital letter, in parens) — checked after roman
# 1./2.          → depth 0 (top-level numbered)
# 1.1/1.1.1      → depth = number of dot-segments - 1
NUMBERED_PATTERNS = [
    (re.compile(r"^\s*\((?:[ivxlcdm]{1,6})\)\s+", re.IGNORECASE), 2),   # (i) (ii) (iii)
    (re.compile(r"^\s*\([A-Z]\)\s+"), 2),                               # (A) (B)
    (re.compile(r"^\s*\([a-z]\)\s+"), 1),                               # (a) (b)
    (re.compile(r"^\s*\(\d{1,3}\)\s+"), 1),                             # (1) (2)
    (re.compile(r"^\s*\d{1,2}(?:\.\d{1,2}){2,}\s+"), 2),                # 1.1.1
    (re.compile(r"^\s*\d{1,2}\.\d{1,2}\s+"), 1),                        # 1.1
    (re.compile(r"^\s*\d{1,3}[.)]\s+"), 0),                             # 1. or 1)
]
SECTION_NUM_RE = re.compile(
    r"^\s*((?:Section|Clause|Article|Annex(?:ure)?|Chapter|Part)\s+[\w.\-]+|"
    r"\d{1,2}(?:\.\d{1,2}){0,3})\b",
    re.IGNORECASE,
)

# Lines matching these are structural noise (page numbers), not content.
NOISE_PAGE_NUM_RE = re.compile(r"^\s*(page\s+)?\d{1,4}(\s*/\s*\d{1,4})?\s*$", re.IGNORECASE)

# Indian Gazette masthead/letterhead noise — legacy non-Unicode Hindi font glyphs
# (render as Latin-lookalike garbage) plus repeating English masthead lines.
# These carry zero legal content and must never appear in a knowledge-base doc.
GAZETTE_NOISE_RE = re.compile(
    r"(vlk/kkj|Hkkx|izkf/kdkj|PUBLISHED\s+BY\s+AUTHORITY|EXTRAORDINARY|"
    r"PART\s+II\s*[—-]?\s*Section|REGISTERED\s+NO|jftLVª|xxxGID|CG-DL-E|"
    r"lañ\s*\d|bl\s+Hkkx|MINISTRY\s+OF\s+LAW\s+AND\s+JUSTICE|"
    r"\(Legislative\s+Department\)|THE\s+GAZETTE\s+OF\s+INDIA)",
    re.IGNORECASE,
)

# A line repeating on >= this fraction of pages at the same vertical band
# (top 8% or bottom 8% of page height) is treated as a running header/footer.
HEADER_FOOTER_REPEAT_THRESHOLD = 0.6
HEADER_FOOTER_BAND_FRACTION = 0.08

# Multi-column detection: if text x-positions cluster into 2+ distinct groups
# separated by a wide gap, treat the page as multi-column and read column-by-column.
COLUMN_GAP_MIN_PT = 24.0


# ─────────────────────────────────────────────────────────────────────────────
# DATA MODEL
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Line:
    text: str
    size: float
    top: float
    page: int
    bold: bool = False
    x0: float = 0.0
    x1: float = 0.0
    plain_text: str = ""   # text with inline ** markers stripped — used for classification


@dataclass
class Block:
    kind: str            # "heading1" | "heading2" | "heading3" | "para" | "bullet" | "numbered" | "table"
    text: str = ""
    rows: list = field(default_factory=list)   # for tables
    page: int = 0
    list_depth: int = 0
    list_marker: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — extract lines with font metadata, per page
# ─────────────────────────────────────────────────────────────────────────────

def extract_lines(page, page_no: int) -> list[Line]:
    """
    Group pdfplumber chars into lines, keeping the dominant font size,
    bold flag, and horizontal span for each line. The x-span is what
    lets us later detect and correctly order multi-column layouts.
    """
    chars = page.chars
    if not chars:
        return []

    lines_map: dict[float, list] = {}
    for ch in chars:
        key = round(ch["top"], 1)
        lines_map.setdefault(key, []).append(ch)

    lines: list[Line] = []
    for top in sorted(lines_map.keys()):
        row_chars = sorted(lines_map[top], key=lambda c: c["x0"])
        plain_text = "".join(c["text"] for c in row_chars).strip()
        if not plain_text:
            continue
        text = line_with_inline_bold(top, row_chars)
        sizes = [c["size"] for c in row_chars]
        avg_size = sum(sizes) / len(sizes)
        bold = any("Bold" in (c.get("fontname") or "") for c in row_chars)
        x0 = min(c["x0"] for c in row_chars)
        x1 = max(c["x1"] for c in row_chars)
        lines.append(Line(text=text, plain_text=plain_text, size=avg_size, top=top,
                          page=page_no, bold=bold, x0=x0, x1=x1))

    return lines


def detect_columns(lines: list[Line], page_width: float) -> list[tuple[float, float]]:
    """
    Detect column bands on a page by clustering line x0 start positions.
    Returns a list of (band_start, band_end) tuples, left to right.
    A single-column page returns one band spanning the full width.
    """
    if not lines:
        return [(0.0, page_width)]

    starts = sorted(ln.x0 for ln in lines)
    # cluster starts into groups separated by a gap >= COLUMN_GAP_MIN_PT
    clusters: list[list[float]] = [[starts[0]]]
    for x in starts[1:]:
        if x - clusters[-1][-1] > COLUMN_GAP_MIN_PT:
            clusters.append([x])
        else:
            clusters[-1].append(x)

    # Require each cluster to have a meaningful number of lines to count as
    # a real column (avoids false positives from the occasional indented quote).
    significant = [c for c in clusters if len(c) >= max(3, len(lines) * 0.08)]
    if len(significant) < 2:
        return [(0.0, page_width)]

    bands = []
    for i, c in enumerate(significant):
        band_start = min(c) - 2
        band_end = (min(significant[i + 1]) - 2) if i + 1 < len(significant) else page_width
        bands.append((band_start, band_end))
    return bands


def order_lines_by_columns(lines: list[Line], page_width: float) -> list[Line]:
    """
    Multi-column-aware reading order: assign each line to a column band,
    then emit column 1 (top→bottom) fully, then column 2, etc.
    Falls back to plain top-to-bottom order for single-column pages.
    """
    bands = detect_columns(lines, page_width)
    if len(bands) <= 1:
        return sorted(lines, key=lambda ln: ln.top)

    ordered: list[Line] = []
    for (start, end) in bands:
        col_lines = [ln for ln in lines if start <= ln.x0 < end]
        col_lines.sort(key=lambda ln: ln.top)
        ordered.extend(col_lines)
    return ordered


def merge_wrapped_lines(lines: list[Line]) -> list[Line]:
    """
    pdfplumber sometimes splits a single visual line into near-identical
    'top' clusters. Merge lines that are extremely close in vertical
    position, share the same size, and are horizontally adjacent
    (handles font kerning artifacts without merging across columns).
    """
    if not lines:
        return []
    merged = [lines[0]]
    for ln in lines[1:]:
        prev = merged[-1]
        same_band = abs(ln.top - prev.top) < 1.0 and abs(ln.size - prev.size) < 0.5
        horizontally_adjacent = ln.x0 >= prev.x0   # same reading line, not a column jump
        if same_band and horizontally_adjacent and abs(ln.x0 - prev.x1) < 40:
            prev.text = (prev.text + " " + ln.text).strip()
            prev.x1 = ln.x1
        else:
            merged.append(ln)
    return merged


def detect_running_headers_footers(
    pages_lines: list[list[Line]], page_heights: list[float]
) -> set[str]:
    """
    Find lines that repeat near-identically across most pages, anchored in
    the top or bottom band of the page (running headers/footers, doc title
    restated on every page, "Confidential" watermarks, etc). These are
    structural noise for a knowledge-base document and must be stripped
    before they pollute the chunk text.

    Returns a set of normalized line texts to exclude.
    """
    total_pages = len(pages_lines)
    if total_pages < 3:
        return set()  # not enough pages to detect a repeating pattern reliably

    def normalize(text: str) -> str:
        # collapse page numbers / dates so "Page 3 of 40" and "Page 4 of 40"
        # are recognized as the same repeating template
        t = re.sub(r"\d+", "#", text.strip().lower())
        return t

    counts: Counter = Counter()
    for lines, height in zip(pages_lines, page_heights):
        band = height * HEADER_FOOTER_BAND_FRACTION
        seen_this_page: set[str] = set()
        for ln in lines:
            in_top_band = ln.top <= band
            in_bottom_band = ln.top >= (height - band)
            if not (in_top_band or in_bottom_band):
                continue
            norm = normalize(ln.plain_text or ln.text)
            if norm and norm not in seen_this_page:
                counts[norm] += 1
                seen_this_page.add(norm)

    threshold = max(3, int(total_pages * HEADER_FOOTER_REPEAT_THRESHOLD))
    noise = {norm for norm, c in counts.items() if c >= threshold}
    return noise


def is_noise_line(ln: Line, page_height: float, noise_norms: set[str]) -> bool:
    plain = ln.plain_text or ln.text
    if GAZETTE_NOISE_RE.search(plain):
        return True
    if NOISE_PAGE_NUM_RE.match(plain):
        band = page_height * HEADER_FOOTER_BAND_FRACTION
        if ln.top <= band or ln.top >= (page_height - band):
            return True
    norm = re.sub(r"\d+", "#", plain.strip().lower())
    return norm in noise_norms


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — determine body font size (the mode), then classify each line
# ─────────────────────────────────────────────────────────────────────────────

def body_font_size(all_lines: list[Line]) -> float:
    sizes = [round(ln.size, 1) for ln in all_lines if len(ln.text) > 25]
    if not sizes:
        sizes = [round(ln.size, 1) for ln in all_lines]
    if not sizes:
        return 10.0
    return Counter(sizes).most_common(1)[0][0]


def numbered_depth(text: str) -> int | None:
    """Returns the nesting depth (0=top, 1, 2...) if the line opens a
    numbered/lettered clause, else None."""
    for pattern, depth in NUMBERED_PATTERNS:
        if pattern.match(text):
            return depth
    return None


def classify_line(ln: Line, body_size: float) -> tuple[str, int]:
    """Returns (kind, list_depth). list_depth is only meaningful for
    'bullet' and 'numbered' kinds. Classification always runs against
    plain_text (no ** markers) so inline bold-wrapping never corrupts
    heading/bullet/numbered pattern matching."""
    delta = ln.size - body_size
    plain = ln.plain_text or ln.text
    looks_like_section_heading = bool(SECTION_NUM_RE.match(plain)) and len(plain) < 120

    if delta >= H1_DELTA:
        return "heading1", 0
    if delta >= H2_DELTA or (ln.bold and looks_like_section_heading and delta >= 0.5):
        return "heading2", 0
    if delta >= H3_DELTA or (ln.bold and looks_like_section_heading):
        return "heading3", 0

    depth = numbered_depth(plain)
    if depth is not None:
        return "numbered", depth

    if plain.lstrip()[:1] in BULLET_CHARS:
        return "bullet", 0

    return "para", 0


def line_with_inline_bold(top: float, row_chars: list) -> str:
    """
    Rebuild a line's text while wrapping runs of bold characters in **markdown
    bold**. This preserves inline emphasis (e.g. defined terms like
    **"data fiduciary"**) that block-level heading detection alone would lose.
    """
    out = []
    run = []
    run_bold = None
    for c in row_chars:
        b = "Bold" in (c.get("fontname") or "")
        if run_bold is None:
            run_bold = b
        if b != run_bold:
            seg = "".join(run)
            out.append(f"**{seg}**" if run_bold and seg.strip() else seg)
            run = [c["text"]]
            run_bold = b
        else:
            run.append(c["text"])
    if run:
        seg = "".join(run)
        out.append(f"**{seg}**" if run_bold and seg.strip() else seg)
    return "".join(out).strip()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — tables: extract separately, exclude their bbox from text lines
#           so table content is never duplicated as paragraph text
# ─────────────────────────────────────────────────────────────────────────────

def extract_tables_with_bbox(page):
    """Returns list of (bbox, rows) so we can exclude these regions from text."""
    out = []
    try:
        found = page.find_tables(
            table_settings={
                "vertical_strategy": "lines_strict",
                "horizontal_strategy": "lines_strict",
            }
        )
        if not found:
            found = page.find_tables()  # fallback to default heuristic
        for t in found:
            rows = t.extract()
            rows = [[(c or "").strip() for c in row] for row in rows]
            rows = [r for r in rows if any(cell for cell in r)]
            if rows:
                out.append((t.bbox, rows))
    except Exception:
        pass
    return out


def line_inside_any_bbox(ln: Line, page_height: float, bboxes) -> bool:
    # pdfplumber bbox: (x0, top, x1, bottom) in top-left origin, matches ln.top
    for (x0, top, x1, bottom) in bboxes:
        if top - 1 <= ln.top <= bottom + 1:
            return True
    return False


def table_to_markdown(rows: list[list[str]]) -> str:
    """Convert a raw table (list of rows) into a clean Markdown table."""
    if not rows:
        return ""
    # normalize row length
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]

    def clean_cell(c: str) -> str:
        c = c.replace("\n", " ").replace("|", "\\|").strip()
        return c if c else " "

    header = rows[0]
    body_rows = rows[1:] if len(rows) > 1 else []

    lines = []
    lines.append("| " + " | ".join(clean_cell(c) for c in header) + " |")
    lines.append("|" + "|".join(["---"] * width) + "|")
    for r in body_rows:
        lines.append("| " + " | ".join(clean_cell(c) for c in r) + " |")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — build ordered blocks per page (text blocks + table blocks interleaved
#           by vertical position, so reading order is preserved)
# ─────────────────────────────────────────────────────────────────────────────

def build_page_blocks(
    page, page_no: int, body_size: float, noise_norms: set[str]
) -> list[Block]:
    tables = extract_tables_with_bbox(page)
    table_bboxes = [bb for bb, _ in tables]

    raw_lines = merge_wrapped_lines(extract_lines(page, page_no))
    text_lines = [
        ln for ln in raw_lines
        if not line_inside_any_bbox(ln, page.height, table_bboxes)
        and not is_noise_line(ln, page.height, noise_norms)
    ]
    # Multi-column-aware ordering: read left column fully, then right column,
    # instead of pure top-to-bottom which would interleave unrelated sentences.
    text_lines = order_lines_by_columns(text_lines, page.width)

    # Build a synthetic vertical anchor that respects column reading order
    # (sequential index) rather than raw 'top', since column reordering may
    # have moved lines out of pure top-to-bottom sequence.
    anchored: list[tuple[float, str, object]] = []
    for seq, ln in enumerate(text_lines):
        anchored.append((seq, "line", ln))

    # Tables are inserted at the sequence position matching their original
    # vertical placement among the (now column-ordered) text lines.
    for (bbox, rows) in tables:
        table_top = bbox[1]
        insert_seq = len(text_lines)  # default: end of page
        for seq, ln in enumerate(text_lines):
            if ln.top >= table_top:
                insert_seq = seq - 0.5   # fractional so it sorts before that line
                break
        anchored.append((insert_seq, "table", rows))

    anchored.sort(key=lambda x: x[0])

    blocks: list[Block] = []
    para_buffer: list[str] = []
    buffer_kind = "para"
    buffer_depth = 0

    def flush_buffer():
        nonlocal para_buffer, buffer_kind, buffer_depth
        if para_buffer:
            blocks.append(Block(
                kind=buffer_kind,
                text=" ".join(para_buffer).strip(),
                page=page_no,
                list_depth=buffer_depth,
            ))
            para_buffer = []
            buffer_depth = 0

    for _, kind, payload in anchored:
        if kind == "table":
            flush_buffer()
            blocks.append(Block(kind="table", rows=payload, page=page_no))
            continue

        ln: Line = payload
        cls, depth = classify_line(ln, body_size)

        if cls in ("heading1", "heading2", "heading3"):
            flush_buffer()
            heading_text = (ln.plain_text or ln.text).strip()
            blocks.append(Block(kind=cls, text=heading_text, page=page_no))
            continue

        if cls in ("bullet", "numbered"):
            # A new clause/item marker always starts a fresh block.
            flush_buffer()
            buffer_kind = cls
            buffer_depth = depth
            para_buffer = [ln.text.strip()]
            continue  # do NOT flush yet — wrapped continuation lines below get appended

        # A plain-text line immediately following an open bullet/numbered buffer
        # is the wrapped continuation of that same clause (not a new paragraph).
        if buffer_kind in ("bullet", "numbered"):
            para_buffer.append(ln.text.strip())
            continue

        # plain paragraph text — accumulate, flush on kind change
        if buffer_kind != "para":
            flush_buffer()
            buffer_kind = "para"
        para_buffer.append(ln.text.strip())

    flush_buffer()
    return blocks


def merge_page_break_continuations(blocks: list[Block]) -> list[Block]:
    """
    If a numbered/bullet clause is the very last block on a page and the
    very first block on the next page is plain paragraph text starting in
    lowercase (i.e. mid-sentence), merge them — this is a clause that was
    split by a page boundary, not two separate items.
    """
    if not blocks:
        return blocks

    merged: list[Block] = [blocks[0]]
    for b in blocks[1:]:
        prev = merged[-1]
        crosses_page = b.page != prev.page
        prev_is_list_item = prev.kind in ("bullet", "numbered")
        continuation_shape = (
            b.kind == "para"
            and b.text
            and b.text[0].islower()
        )
        if crosses_page and prev_is_list_item and continuation_shape:
            prev.text = (prev.text + " " + b.text).strip()
            continue
        merged.append(b)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — render blocks to Markdown text
# ─────────────────────────────────────────────────────────────────────────────

def normalize_bullet_text(text: str) -> str:
    for ch in BULLET_CHARS:
        if text.startswith(ch):
            return text[len(ch):].strip()
    return text


def render_markdown(blocks: list[Block], doc_title: str | None) -> str:
    out: list[str] = []
    if doc_title:
        out.append(f"# {doc_title}\n")

    prev_kind = None
    list_buffer_open = False

    for b in blocks:
        if b.kind == "heading1":
            if list_buffer_open:
                out.append("")
                list_buffer_open = False
            out.append(f"\n## {b.text}\n")
        elif b.kind == "heading2":
            if list_buffer_open:
                out.append("")
                list_buffer_open = False
            out.append(f"\n### {b.text}\n")
        elif b.kind == "heading3":
            if list_buffer_open:
                out.append("")
                list_buffer_open = False
            out.append(f"\n#### {b.text}\n")
        elif b.kind == "bullet":
            indent = "  " * b.list_depth
            out.append(f"{indent}- {normalize_bullet_text(b.text)}")
            list_buffer_open = True
        elif b.kind == "numbered":
            indent = "  " * b.list_depth
            out.append(f"{indent}{b.text}")
            list_buffer_open = True
        elif b.kind == "table":
            if list_buffer_open:
                out.append("")
                list_buffer_open = False
            out.append("")
            out.append(table_to_markdown(b.rows))
            out.append("")
        elif b.kind == "para":
            if list_buffer_open:
                out.append("")
                list_buffer_open = False
            if b.text:
                out.append(b.text)
                out.append("")
        prev_kind = b.kind

    text = "\n".join(out)
    # collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — frontmatter
# ─────────────────────────────────────────────────────────────────────────────

def build_frontmatter(framework: str, version: str, applies_to: list[str]) -> str:
    applies_str = ", ".join(f'"{r}"' for r in applies_to)
    return (
        "---\n"
        f'framework: "{framework}"\n'
        f'version: "{version}"\n'
        f'applies_to: [{applies_str}]\n'
        "---\n\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONVERSION ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def convert(pdf_path: Path, framework: str | None, version: str, applies_to: list[str]) -> str:
    with pdfplumber.open(str(pdf_path)) as pdf:
        # Pass 1: extract raw lines per page (needed to detect repeating
        # headers/footers before we commit to including/excluding anything).
        pages_lines: list[list[Line]] = []
        page_heights: list[float] = []
        for i, page in enumerate(pdf.pages, start=1):
            lines = merge_wrapped_lines(extract_lines(page, i))
            pages_lines.append(lines)
            page_heights.append(page.height)

        all_lines = [ln for page_lines in pages_lines for ln in page_lines]
        body_size = body_font_size(all_lines)
        noise_norms = detect_running_headers_footers(pages_lines, page_heights)
        if noise_norms:
            print(f"  Stripped {len(noise_norms)} repeating header/footer line(s)")

        # Pass 2: build structured blocks per page, with noise stripped and
        # multi-column reading order applied.
        all_blocks: list[Block] = []
        for i, page in enumerate(pdf.pages, start=1):
            blocks = build_page_blocks(page, i, body_size, noise_norms)
            all_blocks.extend(blocks)

        # Pass 3: stitch numbered/bullet clauses that were split by a page break.
        all_blocks = merge_page_break_continuations(all_blocks)

    fw_name = framework or pdf_path.stem.replace("_", " ")
    md_body = render_markdown(all_blocks, doc_title=None)
    frontmatter = build_frontmatter(fw_name, version, applies_to)
    return frontmatter + f"# {fw_name}\n\n" + md_body


def main():
    ap = argparse.ArgumentParser(description="Convert a compliance PDF to structured Markdown.")
    ap.add_argument("input", type=Path, help="Path to input PDF")
    ap.add_argument("output", type=Path, help="Path to output .md file")
    ap.add_argument("--framework", default=None, help='Framework name, e.g. "DPDP Act 2023"')
    ap.add_argument("--version", default="2023", help="Version/year string")
    ap.add_argument(
        "--applies-to",
        default="compliance_officer,it_security,policy_approver,read_only_assessor",
        help="Comma-separated role keys",
    )
    args = ap.parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    applies_to = [r.strip() for r in args.applies_to.split(",") if r.strip()]
    md = convert(args.input, args.framework, args.version, applies_to)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md, encoding="utf-8")
    print(f"Converted: {args.input} → {args.output}")
    print(f"  {len(md.splitlines())} lines, {len(md)} chars")


if __name__ == "__main__":
    main()  