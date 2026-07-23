# page_summary.py — PipeSight AI
#
# Human-readable companion to page_json.py. page_json.py stays exactly as
# it is — CSV/PDF/Excel/dashboard consumers (pdf_excel_export.py in
# particular) read coordinates straight off its top-level "location" key
# to draw overlay boxes, so that schema is not touched here.
#
# This module reads the SAME underlying data via build_page_json() and
# reshapes it into a second file per page for a human who just wants
# "what tags are on this page and what they mean" — tag name, its
# instrument-letter code, its loop number, and what it functionally is —
# with no coordinates, IDs, or confidence scores. It never writes to the
# DB; it's read-only and derived, same as page_json.py's relationship to
# the DB.
#
# Call export_summary_json(page_name) at the same points you'd call
# export_page_json(page_name) — after an extractor run and after every
# human-review action — so it stays in sync. It can also be called any
# time after the fact since it doesn't depend on DB state directly, only
# on page_json.py's output.

import json
import re
from pathlib import Path
from datetime import datetime

import db_builder
from page_json import build_page_json

OUT_DIR = Path("data") / "structured_output"


def _summary_output_path(page_name):
    """Same folder as page_json.py's *_tags.json, different suffix, so the
    two files for a page sit side by side."""
    if "_page_" in page_name:
        set_folder = page_name.split("_page_")[0]
    else:
        set_folder = "unsorted"
    folder = OUT_DIR / set_folder
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{page_name}_summary.json"


def _parse_tag(tag):
    """Split an ISA-format tag ('TT-101', 'PT2045') into its instrument
    letter code and loop/number portion. Falls back to (tag, None) when
    the text doesn't match the pattern (e.g. an unread placeholder like
    'UNREAD_1234') — callers should treat that as 'couldn't parse', not
    an error.
    """
    if not tag:
        return None, None
    m = re.match(r"^([A-Za-z]{1,4})[\s\-]?(\d[\w\-]*)$", tag.strip())
    if not m:
        return tag, None
    return m.group(1).upper(), m.group(2)


def build_summary_json(page_name, conn=None):
    """Builds the readable summary by calling build_page_json() and
    stripping/reshaping — never queries the DB directly, so it can't
    drift from what page_json.py considers current state.
    """
    page_data = build_page_json(page_name, conn=conn)
    if page_data is None:
        return None

    tags = []
    for tag_name, entry in page_data["tags"].items():
        abbreviation, loop_number = _parse_tag(tag_name)
        tags.append({
            "tag_name": tag_name,
            "abbreviation": abbreviation,
            "loop_number": loop_number,
            "instrument_type": entry["instrument_type"],
            "reading": entry["technical_details"]["raw_reading"],
            "status": entry["review_status"],
        })

    review_queue = []
    for r in page_data["review_queue"]:
        abbreviation, loop_number = _parse_tag(r["candidate_tag"])
        review_queue.append({
            "candidate_tag": r["candidate_tag"],
            "abbreviation": abbreviation,
            "loop_number": loop_number,
            "instrument_type": r["instrument_type"],
            "reading": r["raw_reading"],
            "status": r["review_status"],
        })

    return {
        "page": page_name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_tags": page_data["summary"]["total_tags"],
        "needs_review": page_data["summary"]["needs_review"],
        "tags": tags,
        "review_queue": review_queue,
    }


def export_summary_json(page_name, conn=None):
    """Build + write. Companion call to export_page_json(page_name)."""
    data = build_summary_json(page_name, conn=conn)
    if data is None:
        return None
    out_path = _summary_output_path(page_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return out_path


def export_summary_json_by_id(page_id, conn):
    """Same as export_summary_json but keyed by page_id — mirrors
    page_json.py's export_page_json_by_id, for callers (review_server.py's
    api_action) that track touched_pages as a set of page_id, not
    page_name.
    """
    row = conn.execute(
        "SELECT page_name FROM pages WHERE page_id = ?", (page_id,)
    ).fetchone()
    if not row:
        return None
    return export_summary_json(row["page_name"], conn=conn)


def export_all_summaries():
    conn = db_builder.get_connection()
    try:
        pages = conn.execute("SELECT page_name FROM pages").fetchall()
        written = []
        for p in pages:
            path = export_summary_json(p["page_name"], conn=conn)
            if path:
                written.append(path)
        return written
    finally:
        conn.close()


if __name__ == "__main__":
    paths = export_all_summaries()
    print(f"Wrote {len(paths)} summary JSON file(s) under {OUT_DIR}/")
    for p in paths:
        print(f"  {p}")
