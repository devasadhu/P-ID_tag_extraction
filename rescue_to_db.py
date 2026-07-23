# rescue_to_db.py - PipeSight AI
# Bridges rescue_low_conf.py's output into the DB so it isn't a dead-end CSV.
#
# RESCUED rows  -> a real symbol was found at low-conf/loose-NMS that the
#                  production pass missed. Inserted as a new `symbols` row
#                  + a new `tags` row (tag_status=VALID, auto_accept=0,
#                  verification_method='AI_AUTO') so it shows up in the very
#                  next `human_review_export.py --export` for a human sanity
#                  check — these were found at conf as low as 0.11-0.17, they
#                  should NOT be trusted silently.
#
# STILL_MISSING rows -> confirmed genuine blind spots. Inserted as
#                  `review_queue` rows (kind=ORPHANED_TAG_CANDIDATE,
#                  status=OPEN) since the DB's review_queue table is
#                  currently empty (the original review_queue.csv from
#                  tag_extractor_v2.py was written before the DB schema
#                  existed on that run and never made it into the DB).
#
# Idempotent: safe to re-run on the same rescue_results.csv.
#   - RESCUED rows get a deterministic synthetic det_id derived from the
#     original orphaned-tag bbox, so re-running hits the existing
#     UNIQUE(page_id, det_id) constraint and inserts nothing new.
#   - STILL_MISSING rows are checked against existing OPEN review_queue
#     rows with the same page_id + bbox before inserting.
#   Known limitation: near-duplicate OCR reads of the SAME physical bubble
#   with slightly different bbox coordinates (this happens in your data —
#   e.g. BARG-57 rescued at (3805,1410,...) in one row and (3802,1412,...)
#   in another) will NOT be deduped against each other, since dedup is by
#   exact coordinate match. That's a spatial-clustering problem, not
#   something this script tries to solve — expect a handful of near-dup
#   rows in the review queue/tags table until that's addressed separately.
#
# Usage:
#   python rescue_to_db.py --rescue-csv data\tag_extraction_v2\rescue_results.csv
#   python rescue_to_db.py --dry-run    (preview counts, writes nothing)

import csv
import argparse
from pathlib import Path

import db_builder


def load_rescue_rows(rescue_csv_path):
    rows = []
    with open(rescue_csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def get_page_id(conn, page_name, cache, missing_pages_warned):
    if page_name in cache:
        return cache[page_name]
    row = conn.execute(
        "SELECT page_id FROM pages WHERE page_name = ?", (page_name,)
    ).fetchone()
    page_id = row["page_id"] if row else None
    cache[page_name] = page_id
    if page_id is None and page_name not in missing_pages_warned:
        print(f"  WARNING: page '{page_name}' not found in DB — all rows for "
              f"this page will be skipped. Run tag_extractor_v2.py on it first "
              f"(with DB writes enabled) before importing rescue results.")
        missing_pages_warned.add(page_name)
    return page_id


def get_class_id(conn, class_name, cache):
    if class_name in cache:
        return cache[class_name]
    row = conn.execute(
        "SELECT class_id FROM symbols WHERE class_name = ? LIMIT 1", (class_name,)
    ).fetchone()
    class_id = row["class_id"] if row else -1
    cache[class_name] = class_id
    if row is None:
        print(f"  NOTE: no existing symbol with class_name='{class_name}' found "
              f"to borrow a class_id from — storing class_id=-1 as a placeholder. "
              f"Fix manually in the DB if this matters downstream.")
    return class_id


def safe_float(value):
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser(description="Push rescue_low_conf.py results into the DB.")
    ap.add_argument("--rescue-csv", type=str,
                     default=str(Path("data") / "tag_extraction_v2" / "rescue_results.csv"))
    ap.add_argument("--dry-run", action="store_true",
                     help="Compute and print what would happen, but roll back "
                          "all DB changes instead of committing.")
    args = ap.parse_args()

    rescue_csv_path = Path(args.rescue_csv)
    if not rescue_csv_path.exists():
        print(f"rescue_results.csv not found at {rescue_csv_path}")
        return

    if not db_builder.DB_PATH.exists():
        print(f"Database not found at {db_builder.DB_PATH}. Run tag_extractor_v2.py "
              f"(with DB writes enabled) first so pages/symbols exist to attach to.")
        return

    rows = load_rescue_rows(rescue_csv_path)
    if not rows:
        print(f"No rows found in {rescue_csv_path}.")
        return

    conn = db_builder.get_connection()
    page_id_cache = {}
    class_id_cache = {}
    missing_pages_warned = set()
    touched_pages = set()

    symbols_inserted = 0
    symbols_skipped_dup = 0
    review_inserted = 0
    review_skipped_dup = 0
    skipped_missing_page = 0
    skipped_unknown_status = 0

    for row in rows:
        page_name = row["page"]
        page_id = get_page_id(conn, page_name, page_id_cache, missing_pages_warned)
        if page_id is None:
            skipped_missing_page += 1
            continue

        status = row.get("rescue_status", "")
        orig_x1, orig_y1 = int(row["orig_tag_x1"]), int(row["orig_tag_y1"])
        orig_x2, orig_y2 = int(row["orig_tag_x2"]), int(row["orig_tag_y2"])

        if status == "RESCUED":
            # Deterministic negative synthetic det_id so re-runs are no-ops
            # via the UNIQUE(page_id, det_id) constraint. Negative = "not a
            # real production detection index", easy to spot in queries.
            det_id = -(orig_x1 * 100000 + orig_y1)

            existing = conn.execute(
                "SELECT symbol_id FROM symbols WHERE page_id = ? AND det_id = ?",
                (page_id, det_id)
            ).fetchone()
            if existing:
                symbols_skipped_dup += 1
                continue

            class_name = row["rescued_class"]
            class_id = get_class_id(conn, class_name, class_id_cache)
            yolo_conf = safe_float(row["rescued_conf"]) or 0.0
            rx1, ry1 = int(row["rescued_x1"]), int(row["rescued_y1"])
            rx2, ry2 = int(row["rescued_x2"]), int(row["rescued_y2"])

            db_builder.insert_symbol(
                conn, page_id, det_id, class_name, class_id, yolo_conf,
                rx1, ry1, rx2, ry2
            )
            symbol_id = db_builder.get_symbol_id(conn, page_id, det_id)

            ocr_conf = safe_float(row.get("ocr_conf"))
            combined_conf = round(yolo_conf * ocr_conf, 4) if ocr_conf is not None else yolo_conf

            db_builder.insert_tag(
                conn, symbol_id, page_id,
                best_zone=None,
                raw_ocr=row.get("raw_text") or None,
                ocr_conf=ocr_conf,
                tag=row.get("candidate_tag") or None,
                tag_status="VALID",
                combined_conf=combined_conf,
                auto_accept=False,
                verified_by=None,
                verified_at=None,
                # Explicitly 'AI_AUTO', NOT None/NULL. If this were NULL,
                # human_review_export.py's filter
                # "t.verification_method != 'HUMAN_VERIFIED'" would silently
                # exclude the row — SQL's NULL != 'x' evaluates to NULL, not
                # TRUE, so NULL rows never match a WHERE clause. Worth
                # double-checking tag_extractor_v2.py sets this explicitly
                # too, or some RAW/MISSING tags may be invisibly skipping
                # human review right now.
                verification_method="AI_AUTO",
            )
            symbols_inserted += 1
            touched_pages.add(page_id)

        elif status == "STILL_MISSING":
            dup = conn.execute(
                """SELECT review_id FROM review_queue
                   WHERE page_id = ? AND kind = 'ORPHANED_TAG_CANDIDATE'
                     AND x1 = ? AND y1 = ? AND x2 = ? AND y2 = ?
                     AND status = 'OPEN'""",
                (page_id, orig_x1, orig_y1, orig_x2, orig_y2)
            ).fetchone()
            if dup:
                review_skipped_dup += 1
                continue

            ocr_conf = safe_float(row.get("ocr_conf"))
            db_builder.insert_review_item(
                conn, page_id,
                kind="ORPHANED_TAG_CANDIDATE",
                candidate_tag=row.get("candidate_tag") or None,
                raw_text=row.get("raw_text") or None,
                ocr_conf=ocr_conf,
                x1=orig_x1, y1=orig_y1, x2=orig_x2, y2=orig_y2,
                nearest_det_id=None, nearest_class_name=None,
                nearest_routing=None, nearest_dist_px=None,
            )
            review_inserted += 1
            touched_pages.add(page_id)

        else:
            skipped_unknown_status += 1

    for page_id in touched_pages:
        db_builder.update_page_stats(conn, page_id)

    if args.dry_run:
        conn.rollback()
    else:
        conn.commit()
    conn.close()

    print(f"\n{'='*60}")
    print(f"  rescue_to_db.py {'(DRY RUN — nothing written)' if args.dry_run else 'complete'}")
    print(f"{'='*60}")
    print(f"  New symbols+tags inserted (RESCUED)   : {symbols_inserted}")
    print(f"  Skipped as already-present (RESCUED)  : {symbols_skipped_dup}")
    print(f"  New review_queue rows (STILL_MISSING) : {review_inserted}")
    print(f"  Skipped as already-present (MISSING)  : {review_skipped_dup}")
    if skipped_missing_page:
        print(f"  Skipped, page not in DB               : {skipped_missing_page}")
    if skipped_unknown_status:
        print(f"  Skipped, unrecognized rescue_status    : {skipped_unknown_status}")
    print(f"{'='*60}")
    if symbols_inserted and not args.dry_run:
        print(f"\nNEXT STEP: run human_review_export.py --export --with-crops "
              f"— the {symbols_inserted} newly rescued tag(s) will show up "
              f"there, since auto_accept=0 for all of them regardless of "
              f"how confident the rescue detection was.")
    if review_inserted and not args.dry_run:
        print(f"\n{review_inserted} confirmed genuine blind spot(s) are now logged "
              f"in review_queue (status=OPEN) — no reviewer tool currently reads "
              f"this table, so they're findable via SQL for now "
              f"(SELECT * FROM review_queue WHERE status='OPEN') until a "
              f"review_queue export/import gets built alongside "
              f"human_review_export.py.")


if __name__ == "__main__":
    main()
