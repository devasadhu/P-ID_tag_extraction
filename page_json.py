# page_json.py — PipeSight AI
#
# Canonical per-page JSON export. This is the single source of truth the
# supervisor asked for: CSV/PDF/Excel/dashboard should all read FROM this,
# not query the DB independently. It's built FROM the DB (not from the raw
# extractor rows) because the DB is what holds post-review state — a tag a
# human corrected or promoted only exists there, not in tags_v2.csv.
#
# Call export_page_json(page_name) any time the DB changes for that page —
# on extractor run AND after every human-review action — so every consumer
# is always reading current state. See wire-up in review_server.py's
# api_action() and tag_extractor_v2.py's _write_rows_to_db().
#
# Naming is model-agnostic per requirement: no "yolo"/"paddle"/"ocr-engine"
# anywhere in the filename or the top-level schema. Detection/reading
# numbers still exist (engineering needs them) but live inside
# technical_details, named by what they measure (detection_confidence,
# reading_confidence), not by which model produced them.

import json
from pathlib import Path
from datetime import datetime

import db_builder
from matching_engine import infer_instrument_function

OUT_DIR = Path("data") / "structured_output"

# Empirical finding from the confidence-scoring audit (July 13 session):
# OCR confidence is saturated near 1.0 on this document type regardless of
# whether the read is a real tag, so combined_confidence's 0.80 auto-accept
# threshold collapses in practice to yolo_conf >= ~0.53. That's already
# baked into tags.auto_accept for TAG rows. This constant applies the same
# lever to REVIEW_QUEUE rows, which currently have no auto-resolve at all —
# a MISROUTED_CANDIDATE sitting very close to a symbol with strong detection
# confidence is auto-marked "likely_valid" here (still shown, never
# silently dropped) so a human spends less time on obvious ones.
REVIEW_AUTO_RESOLVE_YOLO_CONF = 0.53
REVIEW_AUTO_RESOLVE_MAX_DIST_PX = 150  # mirrors review_server.triage_review_queue


def _page_output_path(page_name):
    """
    Mirrors the PDF folder structure (IMAGES_DIR / "SET 1" / "SET 1_page_1.png")
    so a user can find a page's structured output the same way they'd find
    its source page. Model-agnostic filename: page_tags.json, not
    yolo_output.json / paddleocr_output.json.
    """
    if "_page_" in page_name:
        set_folder = page_name.split("_page_")[0]
    else:
        set_folder = "unsorted"
    folder = OUT_DIR / set_folder
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{page_name}_tags.json"


def _tag_entry(row):
    tag_status = row["tag_status"]
    verification_method = row["verification_method"] or ""
    is_reviewed = verification_method in ("HUMAN_VERIFIED",)
    needs_review = verification_method == "HUMAN_REQUIRED" or (
        not row["auto_accept"] and tag_status != "SKIPPED"
    )

    if row["verified_by"]:
        review_state = "confirmed_by_human"
    elif row["auto_accept"]:
        review_state = "auto_confirmed"
    else:
        review_state = "needs_review"

    has_text_bbox = row["text_x1"] is not None
    return {
        "instrument_type": infer_instrument_function(row["tag"]),
        "location": {
            "x1": row["x1"], "y1": row["y1"], "x2": row["x2"], "y2": row["y2"],
        },
        # Where the OCR text itself sits, separate from the symbol's box
        # above — null when not available (RECOVERED_ROTATED/rescued rows
        # deliberately don't set this; see tag_extractor_v2.py). Consumers
        # that draw a highlight box for "what does this candidate tag
        # refer to" should prefer this over location when present.
        "text_location": {
            "x1": row["text_x1"], "y1": row["text_y1"],
            "x2": row["text_x2"], "y2": row["text_y2"],
        } if has_text_bbox else None,
        "review_status": review_state,
        "reviewed_by": row["verified_by"] or None,
        "reviewed_at": row["verified_at"] or None,
        "technical_details": {
            "tag_id": row["tag_id"],
            "det_id": row["det_id"],
            "class_name": row["class_name"],
            "class_id": row["class_id"],
            "tag_status": tag_status,
            "raw_reading": row["raw_ocr"],
            "detection_confidence": row["yolo_conf"],
            "reading_confidence": row["ocr_conf"],
            "combined_confidence": row["combined_conf"],
            "verification_method": verification_method or None,
        },
    }


def _review_candidate_entry(row):
    likely_valid = (
        row["kind"] == "MISROUTED_CANDIDATE"
        and row["nearest_dist_px"] is not None
        and row["nearest_dist_px"] <= REVIEW_AUTO_RESOLVE_MAX_DIST_PX
    )
    return {
        "candidate_tag": row["candidate_tag"] or None,
        "instrument_type": infer_instrument_function(row["candidate_tag"]),
        "raw_reading": row["raw_text"],
        "kind": row["kind"],
        "location": {
            "x1": row["x1"], "y1": row["y1"], "x2": row["x2"], "y2": row["y2"],
        },
        "review_status": "resolved" if row["status"] != "OPEN" else (
            "likely_valid" if likely_valid else "needs_review"
        ),
        "resolved_by": row["resolved_by"] or None,
        "resolved_at": row["resolved_at"] or None,
        "technical_details": {
            "review_id": row["review_id"],
            "reading_confidence": row["ocr_conf"],
            "nearest_class_name": row["nearest_class_name"],
            "nearest_routing": row["nearest_routing"],
            "nearest_distance_px": row["nearest_dist_px"],
            "status": row["status"],
        },
    }


def _skipped_entry(row):
    return {
        "class_name": row["class_name"],
        "routing": row["routing"],
        "location": {
            "x1": row["x1"], "y1": row["y1"], "x2": row["x2"], "y2": row["y2"],
        },
        "technical_details": {
            "det_id": row["det_id"],
            "detection_confidence": row["yolo_conf"],
        },
    }


def build_page_json(page_name, conn=None):
    own_conn = conn is None
    if own_conn:
        conn = db_builder.get_connection()
    try:
        page = conn.execute(
            "SELECT * FROM pages WHERE page_name = ?", (page_name,)
        ).fetchone()
        if not page:
            return None

        tag_rows = conn.execute(
            """SELECT t.tag_id, t.raw_ocr, t.ocr_conf, t.tag, t.tag_status,
                      t.combined_conf, t.auto_accept, t.verified_by, t.verified_at,
                      t.verification_method,
                      t.text_x1, t.text_y1, t.text_x2, t.text_y2,
                      s.det_id, s.class_name, s.class_id, s.yolo_conf, s.x1, s.y1, s.x2, s.y2,
                      s.routing
               FROM tags t JOIN symbols s ON s.symbol_id = t.symbol_id
               WHERE t.page_id = ?""",
            (page["page_id"],),
        ).fetchall()

        review_rows = conn.execute(
            """SELECT review_id, kind, candidate_tag, raw_text, ocr_conf,
                      x1, y1, x2, y2, nearest_class_name, nearest_routing,
                      nearest_dist_px, status, resolved_by, resolved_at
               FROM review_queue WHERE page_id = ?""",
            (page["page_id"],),
        ).fetchall()

        tags = {}
        non_instrument_symbols = []
        untagged_counter = 0
        for row in tag_rows:
            if row["tag_status"] == "SKIPPED":
                non_instrument_symbols.append(_skipped_entry(row))
                continue
            key = row["tag"]
            if not key:
                untagged_counter += 1
                key = f"UNREAD_{row['det_id']}"
            # Extremely rare (duplicate tag text on same page) — suffix so
            # nothing silently overwrites another entry in the dict.
            while key in tags:
                key = f"{key}__det{row['det_id']}"
            tags[key] = _tag_entry(row)

        review_queue = [_review_candidate_entry(r) for r in review_rows]

        return {
            "page": page_name,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "image": {
                "width": page["width_px"],
                "height": page["height_px"],
            },
            "summary": {
                "total_tags": len(tags),
                "needs_review": sum(1 for t in tags.values() if t["review_status"] == "needs_review")
                                 + sum(1 for r in review_queue if r["review_status"] == "needs_review"),
                "review_queue_open": sum(1 for r in review_rows if r["status"] == "OPEN"),
            },
            "tags": tags,
            "review_queue": review_queue,
            "non_instrument_symbols": non_instrument_symbols,
        }
    finally:
        if own_conn:
            conn.close()


def export_page_json(page_name, conn=None):
    """Build + write. Call this after any DB change for the page."""
    data = build_page_json(page_name, conn=conn)
    if data is None:
        return None
    out_path = _page_output_path(page_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return out_path


def export_page_json_by_id(page_id, conn):
    """Same as export_page_json but keyed by page_id — review_server.py's
    api_action tracks touched_pages as a set of page_id, not page_name."""
    row = conn.execute(
        "SELECT page_name FROM pages WHERE page_id = ?", (page_id,)
    ).fetchone()
    if not row:
        return None
    return export_page_json(row["page_name"], conn=conn)


def export_all_pages():
    conn = db_builder.get_connection()
    try:
        pages = conn.execute("SELECT page_name FROM pages").fetchall()
        written = []
        for p in pages:
            path = export_page_json(p["page_name"], conn=conn)
            if path:
                written.append(path)
        return written
    finally:
        conn.close()


if __name__ == "__main__":
    paths = export_all_pages()
    print(f"Wrote {len(paths)} page JSON file(s) under {OUT_DIR}/")
    for p in paths:
        print(f"  {p}")
