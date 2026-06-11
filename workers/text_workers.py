"""
extraction/worker.py
--------------------
Single worker for PDF extraction.
Designed to run alongside extraction_server.py.

This version is layout-aware:
- uses PyMuPDF
- handles basic single-column pages with natural reading order
- heuristically handles two-column research-paper pages
- keeps the same server contract as your current worker
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import time
import uuid
from typing import List, Tuple

import requests

try:
    # Newer PyMuPDF installs may expose pymupdf, older code often uses fitz.
    import pymupdf as fitz
except ImportError:
    import fitz

from parta.logger import time_it

SERVER_URL = "http://localhost:8004"
WORKER_ID = f"worker-{uuid.uuid4().hex[:6]}"

REQUEST_TIMEOUT = 30
WAIT_SLEEP = 2
ERROR_SLEEP = 5

Block = Tuple[float, float, float, float, str]


def _clean_block_text(text: str) -> str:
    """
    Normalize block text without destroying content structure too much.
    """
    if not text:
        return ""

    lines = [line.rstrip() for line in text.splitlines()]
    lines = [line for line in lines if line.strip()]
    return "\n".join(lines).strip()


def _collect_text_blocks(page) -> List[Block]:
    """
    Return only non-empty text blocks with coordinates.
    Each block is stored as:
        (x0, y0, x1, y1, text)
    """
    raw_blocks = page.get_text("blocks", sort=False)
    cleaned: List[Block] = []

    for block in raw_blocks:
        if len(block) < 5:
            continue

        x0, y0, x1, y1, text = block[:5]
        text = _clean_block_text(text)

        if not text:
            continue

        if x1 <= x0 or y1 <= y0:
            continue

        cleaned.append((float(x0), float(y0), float(x1), float(y1), text))

    return cleaned


def _is_probably_two_column(blocks: List[Block], page_width: float) -> bool:
    """
    Heuristic for detecting clear two-column pages.
    """
    if len(blocks) < 4:
        return False

    left = 0
    right = 0
    middle = 0

    for x0, y0, x1, y1, _text in blocks:
        center_x = (x0 + x1) / 2.0

        if center_x < page_width * 0.45:
            left += 1
        elif center_x > page_width * 0.55:
            right += 1
        else:
            middle += 1

    # Clear evidence on both sides, with at least some separation.
    if left >= 2 and right >= 2 and middle <= max(2, len(blocks) // 3):
        return True

    return False


def _extract_single_column_text(blocks: List[Block]) -> str:
    """
    Sort top-to-bottom, then left-to-right.
    This is good for single-column pages and many mixed-layout pages.
    """
    ordered = sorted(blocks, key=lambda b: (b[1], b[0]))
    return "\n\n".join(block[4] for block in ordered if block[4]).strip()


def _extract_two_column_text(
    blocks: List[Block], page_width: float, page_height: float
) -> str:
    """
    Heuristic two-column reconstruction.

    Strategy:
    - keep very wide blocks as full-width content
    - split the rest into left/right columns
    - place top full-width blocks first
    - then left column
    - then mid full-width blocks
    - then right column
    - then bottom full-width blocks

    This is not perfect, because PDFs are hostile artifacts pretending to be documents,
    but it is far better than raw stream order for research papers.
    """
    full_width: List[Block] = []
    left_col: List[Block] = []
    right_col: List[Block] = []

    for x0, y0, x1, y1, text in blocks:
        width = x1 - x0
        center_x = (x0 + x1) / 2.0

        # Blocks that span most of the page are usually title, abstract headings,
        # section headers, wide figures, or references headers.
        if width >= page_width * 0.65 or (
            x0 <= page_width * 0.12 and x1 >= page_width * 0.88
        ):
            full_width.append((x0, y0, x1, y1, text))
        elif center_x < page_width / 2.0:
            left_col.append((x0, y0, x1, y1, text))
        else:
            right_col.append((x0, y0, x1, y1, text))

    full_width = sorted(full_width, key=lambda b: (b[1], b[0]))
    left_col = sorted(left_col, key=lambda b: (b[1], b[0]))
    right_col = sorted(right_col, key=lambda b: (b[1], b[0]))

    top_band = page_height * 0.20
    bottom_band = page_height * 0.80

    top_full = [b for b in full_width if b[1] <= top_band]
    bottom_full = [b for b in full_width if b[1] >= bottom_band]
    middle_full = [b for b in full_width if b not in top_full and b not in bottom_full]

    parts: List[str] = []

    def add_group(group: List[Block]) -> None:
        if group:
            parts.append("\n\n".join(b[4] for b in group if b[4]).strip())

    add_group(top_full)
    add_group(left_col)
    add_group(middle_full)
    add_group(right_col)
    add_group(bottom_full)

    return "\n\n".join(part for part in parts if part).strip()


def _extract_page_text(page) -> str:
    """
    Extract one page using layout-aware heuristics.
    """
    blocks = _collect_text_blocks(page)
    if not blocks:
        return ""

    page_width = float(page.rect.width)
    page_height = float(page.rect.height)

    if not _is_probably_two_column(blocks, page_width):
        return _extract_single_column_text(blocks)

    return _extract_two_column_text(blocks, page_width, page_height)


@time_it
def process_chunk(pdf_bytes: bytes, start_offset: int) -> str:
    """
    Extract text from a PDF chunk.
    Keeps page numbering with start_offset so your downstream pipeline stays unchanged.
    """
    try:
        content_parts: List[str] = []

        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for i, page in enumerate(doc):
                page_text = _extract_page_text(page)

                if page_text:
                    content_parts.append(
                        f"\n\n## Page {start_offset + i + 1}\n\n{page_text}"
                    )

        return "".join(content_parts).strip()

    except Exception as exc:
        raise RuntimeError(f"PDF extraction failed: {exc}") from exc


def start_worker():
    print(f"[{WORKER_ID}] Starting layout-aware extraction worker...")

    while True:
        try:
            resp = requests.get(
                f"{SERVER_URL}/get_job",
                params={"worker_id": WORKER_ID},
                timeout=REQUEST_TIMEOUT,
            )

            if resp.status_code != 200:
                time.sleep(ERROR_SLEEP)
                continue

            data = resp.json()
            action = data.get("action")

            if action == "WAIT":
                time.sleep(WAIT_SLEEP)
                continue

            if action == "PROCESS":
                job_id = data["job_id"]
                book_id = data["book_id"]
                chunk_idx = data["chunk_idx"]
                start_offset = data.get("start_offset", 0)

                print(
                    f"[{WORKER_ID}] Got chunk {chunk_idx} for book {book_id}. Downloading..."
                )

                chunk_resp = requests.get(
                    f"{SERVER_URL}/chunk/{job_id}",
                    stream=True,
                    timeout=REQUEST_TIMEOUT,
                )

                if chunk_resp.status_code == 200:
                    try:
                        content = process_chunk(chunk_resp.content, start_offset)
                        print(
                            f"[{WORKER_ID}] Extraction success for chunk {chunk_idx}. Submitting."
                        )

                        requests.post(
                            f"{SERVER_URL}/submit_result",
                            json={
                                "job_id": job_id,
                                "worker_id": WORKER_ID,
                                "success": True,
                                "content": content,
                            },
                            timeout=REQUEST_TIMEOUT,
                        )

                    except Exception as e:
                        print(
                            f"[{WORKER_ID}] Extraction failed for chunk {chunk_idx}: {e}"
                        )
                        requests.post(
                            f"{SERVER_URL}/submit_result",
                            json={
                                "job_id": job_id,
                                "worker_id": WORKER_ID,
                                "success": False,
                                "content": "",
                            },
                            timeout=REQUEST_TIMEOUT,
                        )
                else:
                    print(f"[{WORKER_ID}] Failed to download chunk {job_id}")

        except requests.exceptions.ConnectionError:
            print(f"[{WORKER_ID}] Cannot reach server at {SERVER_URL}. Waiting...")
            time.sleep(ERROR_SLEEP)

        except Exception as e:
            print(f"[{WORKER_ID}] Error: {e}")
            time.sleep(ERROR_SLEEP)


if __name__ == "__main__":
    start_worker()
