# human_review_export.py - PipeSight AI
#
# CSV-based review workflow — kept alongside review_server.py (the live
# browser UI) for reviewers who prefer working in a spreadsheet, or for
# batch review offline. Both write to the same DB through the same
# apply_tag_action() / apply_review_queue_action() functions, and both
# re-export page_json.py's canonical per-page JSON immediately after every
# write, so a spreadsheet edit and a browser click are equally "live" —
# nothing here is a separate source of truth.
#
# ── What changed (2026-07-16 rewrite) ──
# The old export dumped raw model fields (yolo_conf, ocr_conf, combined_conf,
# nearest_class_name, nearest_routing, nearest_dist_px) straight into the
# reviewer's CSV. That's the same jargon review_server.py already hides
# behind its "Show technical details" toggle — this file just hadn't been
# updated to match. Now:
#   - review_export.csv is the REVIEWER-FACING file: plain-language columns
#     only (what this is, the tag text, a High/Medium/Low confidence word,
#     a boxed image, a plain-language flag note). This is the file you fill
#     in and re-import.
#   - review_export_technical.csv is an optional companion (same row order,
#     joined by tag_id/review_id) with the full raw fields, for engineering
#     / audit use. Not meant to be filled in or re-imported.
#   - Obvious items are now pre-filled the same way review_server.py's
#     "quick confirm" bulk button would resolve them (ACCEPT for a
#     VALID-shaped tag just under the auto-accept bar; PROMOTE with the
#     OCR-read text + field-mounted default for a MISROUTED_CANDIDATE with
#     a tight spatial match). A reviewer skims those and only needs to
#     touch the rows that actually need a decision — same "reduce grey
#     volume" idea as review_server.py's triage, applied to the CSV path.
#   - Crops are now boxed by default (red box on the item, grey box on the
#     matching-engine's nearest symbol for MISROUTED_CANDIDATE rows) —
#     same convention review_server.py's /crop endpoint uses — so the
#     candidate tag text always lines up with what's actually highlighted
#     in the image, instead of a plain unboxed crop the reviewer had to
#     interpret blind. Use --no-crops to skip image generation (faster,
#     text-only review).
#
# reviewer_class (PROMOTE only) still takes free text. Current known-good
# values match review_server.py's CLASS_TYPE_OPTIONS: Instrument_Field,
# Instrument_Panel, Instrument_DCS — that list is itself flagged there as a
# placeholder pending confirmation against the real 61-class model, so
# don't treat it as exhaustive.
#
# Usage:
#   python human_review_export.py --export
#   python human_review_export.py --export --page "SET 1_page_1"
#   python human_review_export.py --export --no-crops
#   python human_review_export.py --export --technical
#   python human_review_export.py --import-csv data\human_review\review_export.csv --reviewer sadhana
#
# Round trip:
#   1. --export writes review_export.csv (and, with --technical, its
#      companion), one row per item needing review, with quick-confirm rows
#      already pre-filled — check them, don't retype them.
#   2. Fill in / correct reviewer_action per row:
#        For source=TAG:
#          ACCEPT   - existing tag value is correct as-is
#          REJECT   - not a real tag / not worth pursuing (stays RAW/MISSING,
#                     marked human-reviewed so it stops reappearing)
#          CORRECT  - fill reviewer_tag with the right value; promoted to VALID
#        For source=REVIEW_QUEUE:
#          DISMISS  - this orphaned text isn't a real instrument tag (OCR
#                     noise, a line tag, a false positive, etc.)
#          PROMOTE  - it IS a real tag with no YOLO-detected symbol. Fill
#                     reviewer_tag (final tag text) and reviewer_class
#                     (e.g. Instrument_Field) — a new symbol+tag row is
#                     created and marked HUMAN_VERIFIED immediately, since
#                     you just confirmed it by eye.
#      Leave reviewer_action blank on rows not yet reviewed — skipped on
#      import, reappear next --export.
#   3. --import-csv applies edits to the DB. TAG actions are logged to
#      validation_log as before. REVIEW_QUEUE actions update review_queue's
#      own status/resolved_by/resolved_at/resolution_note columns instead —
#      validation_log's schema requires a tag_id, so PROMOTE actions (which
#      DO create a tag_id) get logged there too; DISMISS actions don't have
#      one and are only recorded on the review_queue row itself. That's a
#      known gap (no unified audit trail entry for DISMISS yet) — flagging
#      rather than quietly leaving it undocumented.
#
# Retrain-pool flywheel (added 2026-07-13):
#   Every PROMOTE action is a human-confirmed real instrument bubble that
#   YOLO missed — exactly the recall-gap failure mode. Rather than let that
#   confirmation disappear into the DB and nothing else, each PROMOTE also
#   appends a YOLO-format label (page image + normalized bbox + class) to
#   data\retrain_pool\, so the human review process doubles as free, real-
#   world-verified training data for the next fine-tuning pass — no separate
#   data-collection effort needed. Re-running --import-csv over an
#   already-applied row is a no-op (status is no longer OPEN, so it's
#   skipped before this point) — manifest.csv is an audit trail, not a
#   de-dupe mechanism by itself. Disable with --no-retrain-pool if you just
#   want the DB update.

import csv
import argparse
from pathlib import Path
from datetime import datetime

from PIL import Image, ImageDraw

import db_builder  # reuses get_connection(), schema, and conventions
import page_json

OUT_DIR = Path("data") / "human_review"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR = OUT_DIR / "crops"

# ── Retrain-pool flywheel ──
# Standard YOLO training layout (images/ + labels/, one .txt per image,
# lines are "class_id x_center y_center width height" normalized 0-1) so
# this folder can be pointed at directly as a training data source later,
# or merged into an existing dataset.yaml.
RETRAIN_POOL_DIR    = Path("data") / "retrain_pool"
RETRAIN_IMAGES_DIR  = RETRAIN_POOL_DIR / "images"
RETRAIN_LABELS_DIR  = RETRAIN_POOL_DIR / "labels"
RETRAIN_MANIFEST    = RETRAIN_POOL_DIR / "manifest.csv"
RETRAIN_MANIFEST_FIELDS = [
    "added_at", "page_name", "tag_id", "review_id", "class_id", "class_name",
    "x1", "y1", "x2", "y2", "reviewer", "source_image_path",
]

# ── Reviewer-facing CSV — plain language only, this is what gets filled in ──
SIMPLE_CSV_FIELDS = [
    "source",             # TAG or REVIEW_QUEUE — needed so import knows which
                           # action set applies; not shown as a "confidence"
                           # or model field, just a routing tag.
    "tag_id", "review_id",
    "page_name",
    "what_is_this",       # plain-language description of the item + why it's
                           # here (pre-filled, informational — not edited)
    "current_tag",        # the tag/candidate text as read, whatever it is
    "confidence",         # High / Medium / Low / Unknown — no raw numbers
    "flag_note",          # plain-language heads-up (e.g. "looks like a pipe
                           # spec callout, not a tag") — blank if none
    "highlighted_image",  # crop with the item boxed in red (and, where
                           # relevant, the matched symbol boxed in grey) —
                           # what's boxed IS current_tag/candidate_tag
    "reviewer_action",
    "reviewer_tag",
    "reviewer_class",
    "reviewer_notes",
]

# ── Technical companion CSV — engineering/audit use, not for re-import ──
TECHNICAL_CSV_FIELDS = [
    "source", "tag_id", "review_id", "page_name", "det_id",
    "class_name", "kind",
    "x1", "y1", "x2", "y2",
    "detection_confidence", "reading_confidence", "combined_confidence",
    "raw_reading", "tag_status",
    "nearest_class_name", "nearest_routing", "nearest_distance_px",
]

TAG_ACTIONS = {"ACCEPT", "REJECT", "CORRECT"}
REVIEW_QUEUE_ACTIONS = {"DISMISS", "PROMOTE"}

# Orphaned-tag crops get no nearby symbol to anchor on, so give the reviewer
# more surrounding context than a normal symbol crop needs.
REVIEW_QUEUE_CROP_MARGIN = 200
TAG_CROP_MARGIN = 40

# Same empirical thresholds page_json.py uses for its own auto-resolve
# lever (see REVIEW_AUTO_RESOLVE_YOLO_CONF there) — kept in sync manually,
# not imported, since this is a display bucket, not a decision gate.
CONFIDENCE_HIGH = 0.80
CONFIDENCE_MEDIUM = 0.53

# Plain-language phrasing for reference_noise_reason() values from
# matching_engine.py, so the reviewer-facing note never shows a raw
# enum-style string like "pipe_or_line_spec_text". Mirrors
# review_server.py's REFERENCE_NOISE_LABELS — duplicated rather than
# imported, since review_server.py already imports THIS module (importing
# back would be circular). Keep both lists in sync if either changes.
REFERENCE_NOISE_LABELS = {
    "pipe_or_line_spec_text": "a pipe or line spec callout",
    "setpoint_annotation": "a pressure/temperature setpoint note",
    "non_instrument_abbreviation": "a known non-instrument abbreviation",
}

# Distance below which a MISROUTED_CANDIDATE is treated as "matching engine
# already found a tight spatial match" — same value review_server.py's
# triage_review_queue() and page_json.py's REVIEW_AUTO_RESOLVE_MAX_DIST_PX
# use. Kept in sync manually for the same reason as above.
QUICK_CONFIRM_MAX_DIST_PX = 150


# ─── plain-language helpers ─────────────────────────────────────────────────

def confidence_bucket(value):
    """Turn a raw 0-1 confidence float into a word a non-technical reviewer
    can act on without knowing what OCR/detection confidence even means."""
    if value is None or value == "":
        return "Unknown"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "Unknown"
    if v >= CONFIDENCE_HIGH:
        return "High"
    if v >= CONFIDENCE_MEDIUM:
        return "Medium"
    return "Low"


def triage_tag_row(row):
    """
    Same bucketing as review_server.py's triage_tag(): a VALID-shaped tag
    that just missed the auto-accept bar is a quick confirm, not a real
    unknown. RAW/MISSING genuinely needs a look. Duplicated here (not
    imported) for the same circular-import reason as REFERENCE_NOISE_LABELS.
    """
    return "quick_confirm" if row["tag_status"] == "VALID" else "needs_review"


def triage_review_queue_row(row, has_nearest):
    """Same bucketing as review_server.py's triage_review_queue()."""
    if row["kind"] == "MISROUTED_CANDIDATE" and has_nearest:
        dist = row["nearest_dist_px"] if "nearest_dist_px" in row.keys() else None
        if dist is not None and dist < QUICK_CONFIRM_MAX_DIST_PX:
            return "quick_confirm"
    return "needs_review"


def what_is_this_tag(row, priority):
    if priority == "quick_confirm":
        return "Existing tag, already read correctly — pre-checked below, only change it if it's wrong."
    if row["tag_status"] == "RAW":
        return "Existing symbol, but the text couldn't be read cleanly — needs a look."
    if row["tag_status"] == "MISSING":
        return "Existing symbol with no readable text at all — needs a look."
    return "Existing tag that needs a second look before it's accepted."


def what_is_this_review_queue(row, priority):
    if priority == "quick_confirm":
        return "Text found right next to a known symbol — pre-filled as a likely match below, only change it if it's wrong."
    if row["kind"] == "MISROUTED_CANDIDATE":
        return "Text found near a symbol, but not a confident match — needs a look."
    return "Text found with no nearby symbol at all — needs a look."


# ─── data access ────────────────────────────────────────────────────────────

def fetch_tag_rows(conn, page_name=None):
    """
    Anything that isn't auto-accepted needs a human look: RAW, MISSING, and
    any VALID tag that didn't clear the auto-accept confidence bar. SKIPPED
    rows (mechanical/structural/unknown symbols) are excluded.

    NOTE: verification_method uses IS NOT / OR IS NULL rather than plain
    != 'HUMAN_VERIFIED'. In SQL, NULL != 'x' evaluates to NULL (not TRUE),
    so a plain != silently drops every row where verification_method was
    never set, INCLUDING all normal RAW/MISSING tags if tag_extractor_v2.py
    doesn't explicitly set that column. That was a real bug in a previous
    version of this query — check tag_extractor_v2.py sets verification_method
    explicitly (e.g. 'AI_AUTO') when it inserts tags, or this fix is papering
    over a gap upstream rather than fixing it at the source.
    """
    query = """
        SELECT t.tag_id, pg.page_name, s.det_id, s.class_name,
               s.x1, s.y1, s.x2, s.y2, s.yolo_conf,
               t.raw_ocr, t.ocr_conf, t.tag, t.tag_status, t.combined_conf,
               pg.image_path
        FROM tags t
        JOIN symbols s ON s.symbol_id = t.symbol_id
        JOIN pages pg   ON pg.page_id  = t.page_id
        WHERE s.routing = 'instrument'
          AND t.auto_accept = 0
          AND (t.verification_method IS NULL
               OR t.verification_method != 'HUMAN_VERIFIED')
    """
    params = []
    if page_name:
        query += " AND pg.page_name = ?"
        params.append(page_name)
    query += " ORDER BY pg.page_name, s.det_id"
    return conn.execute(query, params).fetchall()


def fetch_review_queue_rows(conn, page_name=None):
    """
    OPEN review_queue rows: orphaned OCR text (or misrouted-symbol text)
    with no confirmed instrument tag yet. Includes STILL_MISSING rows
    inserted by rescue_to_db.py after a rescue attempt confirmed a genuine
    detection gap, not just a threshold artifact.

    Kept signature-stable (plain rows, no nearest_* columns) since
    review_server.py falls back to calling this directly when its own
    extended query fails. Use fetch_review_queue_rows_with_nearest() below
    for triage/export, which needs the nearest-match columns.
    """
    query = """
        SELECT rq.review_id, pg.page_name, rq.kind, rq.candidate_tag,
               rq.raw_text, rq.ocr_conf, rq.x1, rq.y1, rq.x2, rq.y2,
               pg.image_path
        FROM review_queue rq
        JOIN pages pg ON pg.page_id = rq.page_id
        WHERE rq.status = 'OPEN'
    """
    params = []
    if page_name:
        query += " AND pg.page_name = ?"
        params.append(page_name)
    query += " ORDER BY pg.page_name, rq.review_id"
    return conn.execute(query, params).fetchall()


def fetch_review_queue_rows_with_nearest(conn, page_name=None):
    """
    Same rows as fetch_review_queue_rows(), plus the nearest_* columns
    needed for quick-confirm triage and the technical CSV. Falls back
    cleanly if those columns don't exist on this DB — same pattern
    review_server.py's fetch_review_queue_rows_ext() uses, duplicated here
    rather than imported for the same circular-import reason as above.
    """
    import sqlite3
    ext_query = """
        SELECT rq.review_id, pg.page_name, rq.kind, rq.candidate_tag,
               rq.raw_text, rq.ocr_conf, rq.x1, rq.y1, rq.x2, rq.y2,
               rq.nearest_det_id, rq.nearest_class_name, rq.nearest_routing,
               rq.nearest_dist_px,
               pg.image_path
        FROM review_queue rq
        JOIN pages pg ON pg.page_id = rq.page_id
        WHERE rq.status = 'OPEN'
    """
    params = []
    if page_name:
        ext_query += " AND pg.page_name = ?"
        params.append(page_name)
    ext_query += " ORDER BY pg.page_name, rq.review_id"
    try:
        rows = conn.execute(ext_query, params).fetchall()
        return rows, True
    except sqlite3.OperationalError:
        return fetch_review_queue_rows(conn, page_name=page_name), False


# ─── crop / highlight generation ────────────────────────────────────────────

def _crop_bounds(img_w, img_h, x1, y1, x2, y2, margin, extra_box=None):
    all_x = [x1, x2]
    all_y = [y1, y2]
    if extra_box:
        ex1, ey1, ex2, ey2 = extra_box
        all_x += [ex1, ex2]
        all_y += [ey1, ey2]
    cx1 = max(0, min(all_x) - margin)
    cy1 = max(0, min(all_y) - margin)
    cx2 = min(img_w, max(all_x) + margin)
    cy2 = min(img_h, max(all_y) + margin)
    return cx1, cy1, cx2, cy2


def save_boxed_crop(image_path, x1, y1, x2, y2, out_path, margin, nearest_box=None):
    """
    Save a cropped PNG with the item itself boxed in red — same visual
    convention review_server.py's /crop endpoint uses — so what's boxed in
    the image always matches current_tag/candidate_tag in the row next to
    it. nearest_box (grey), when given, is the matching engine's nearest
    symbol for a MISROUTED_CANDIDATE, so the reviewer can see the actual
    match being proposed, not just the text alone.
    """
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size
    cx1, cy1, cx2, cy2 = _crop_bounds(img_w, img_h, x1, y1, x2, y2, margin, nearest_box)
    crop = img.crop((cx1, cy1, cx2, cy2))

    draw = ImageDraw.Draw(crop)
    if nearest_box:
        nx1, ny1, nx2, ny2 = nearest_box
        draw.rectangle((nx1 - cx1, ny1 - cy1, nx2 - cx1, ny2 - cy1),
                        outline=(120, 130, 140), width=2)
    draw.rectangle((x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1),
                    outline=(230, 30, 30), width=3)
    crop.save(out_path)


def append_retrain_sample(image_path, page_name, x1, y1, x2, y2, class_id, class_name,
                           tag_id, review_id, reviewer_name, now):
    """
    Turn one human-confirmed PROMOTE action into a YOLO training example.

    Copies the full source page image into the retrain pool once (subsequent
    promotions on the same page just append another label line — a page
    with several missed bubbles becomes one image + a growing label file,
    same as any normal YOLO dataset), then appends the normalized bbox as
    a new line in that page's label file, plus one manifest row for audit
    trail / traceability back to the exact tag_id or review_id it came from.

    Returns True if a new label line was written, False if this exact
    page+bbox was already in the pool (so re-running --import-csv on
    already-applied rows, or exporting/promoting the same spot twice,
    doesn't silently duplicate training examples).
    """
    RETRAIN_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    RETRAIN_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    dest_image = RETRAIN_IMAGES_DIR / f"{page_name}.png"
    label_file = RETRAIN_LABELS_DIR / f"{page_name}.txt"

    with Image.open(image_path) as img:
        img_w, img_h = img.size
        if not dest_image.exists():
            img.convert("RGB").save(dest_image)

    xc = (x1 + x2) / 2 / img_w
    yc = (y1 + y2) / 2 / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    new_line = f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"

    existing_lines = []
    if label_file.exists():
        existing_lines = label_file.read_text().splitlines()
    if new_line in existing_lines:
        return False  # already in the pool — don't duplicate

    with open(label_file, "a") as f:
        f.write(new_line + "\n")

    manifest_is_new = not RETRAIN_MANIFEST.exists()
    with open(RETRAIN_MANIFEST, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RETRAIN_MANIFEST_FIELDS)
        if manifest_is_new:
            writer.writeheader()
        writer.writerow({
            "added_at": now, "page_name": page_name,
            "tag_id": tag_id or "", "review_id": review_id or "",
            "class_id": class_id, "class_name": class_name,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "reviewer": reviewer_name, "source_image_path": str(image_path),
        })
    return True


# ─── EXPORT ─────────────────────────────────────────────────────────────────

def export_review_csv(page_name=None, with_crops=True, technical=False):
    conn = db_builder.get_connection()
    tag_rows = fetch_tag_rows(conn, page_name=page_name)
    rq_rows, has_nearest = fetch_review_queue_rows_with_nearest(conn, page_name=page_name)

    if with_crops:
        CROPS_DIR.mkdir(parents=True, exist_ok=True)

    simple_rows = []
    technical_rows = []
    crop_failures = 0
    quick_confirm_count = 0

    for r in tag_rows:
        priority = triage_tag_row(r)
        row_keys = r.keys()
        ref_noise = r["reference_noise"] if "reference_noise" in row_keys else ""

        highlighted_image = ""
        if with_crops and r["image_path"]:
            crop_file = CROPS_DIR / f"{r['page_name']}_det{r['det_id']}.png"
            if not crop_file.exists():
                try:
                    save_boxed_crop(r["image_path"], r["x1"], r["y1"], r["x2"], r["y2"],
                                     crop_file, margin=TAG_CROP_MARGIN)
                except Exception as e:
                    crop_failures += 1
                    print(f"  WARNING: crop failed for det {r['det_id']} on "
                          f"{r['page_name']}: {e}")
                    crop_file = None
            highlighted_image = str(crop_file) if crop_file else ""

        flag_note = ""
        if ref_noise:
            reason_text = REFERENCE_NOISE_LABELS.get(ref_noise, ref_noise.replace("_", " "))
            flag_note = f"Looks like {reason_text} near this symbol, not a confirmed instrument tag — check before accepting."

        # Pre-fill quick confirms — same resolution review_server.py's bulk
        # "confirm all" button would apply, just written in ahead of time
        # so a reviewer working the CSV doesn't have to type it.
        if priority == "quick_confirm":
            quick_confirm_count += 1
            action, r_tag, r_class = "ACCEPT", "", ""
        else:
            action, r_tag, r_class = "", "", ""

        simple_rows.append({
            "source": "TAG",
            "tag_id": r["tag_id"], "review_id": "",
            "page_name": r["page_name"],
            "what_is_this": what_is_this_tag(r, priority),
            "current_tag": r["tag"] or "(unreadable)",
            "confidence": confidence_bucket(r["combined_conf"]),
            "flag_note": flag_note,
            "highlighted_image": highlighted_image,
            "reviewer_action": action, "reviewer_tag": r_tag, "reviewer_class": r_class,
            "reviewer_notes": "",
        })

        if technical:
            technical_rows.append({
                "source": "TAG", "tag_id": r["tag_id"], "review_id": "",
                "page_name": r["page_name"], "det_id": r["det_id"],
                "class_name": r["class_name"], "kind": "",
                "x1": r["x1"], "y1": r["y1"], "x2": r["x2"], "y2": r["y2"],
                "detection_confidence": r["yolo_conf"],
                "reading_confidence": r["ocr_conf"],
                "combined_confidence": r["combined_conf"],
                "raw_reading": r["raw_ocr"] or "",
                "tag_status": r["tag_status"],
                "nearest_class_name": "", "nearest_routing": "", "nearest_distance_px": "",
            })

    for r in rq_rows:
        priority = triage_review_queue_row(r, has_nearest)

        # Reduce grey volume: MISROUTED_CANDIDATE rows with a tight spatial
        # match are pre-filled as PROMOTE (same logic review.html's bulk
        # quick-confirm button uses: candidate text as-is, field-mounted
        # default) rather than left for the reviewer to type from scratch.
        if priority == "quick_confirm":
            quick_confirm_count += 1
            action = "PROMOTE"
            r_tag = r["candidate_tag"] or ""
            r_class = "Instrument_Field"
        else:
            action, r_tag, r_class = "", "", ""

        nearest_box = None
        row_keys = r.keys()
        if (has_nearest and r["kind"] == "MISROUTED_CANDIDATE"
                and "nearest_det_id" in row_keys and r["nearest_det_id"] is not None):
            near_symbol = conn.execute(
                "SELECT x1, y1, x2, y2 FROM symbols WHERE page_id = ? AND det_id = ?",
                (conn.execute("SELECT page_id FROM pages WHERE page_name = ?",
                              (r["page_name"],)).fetchone()["page_id"], r["nearest_det_id"]),
            ).fetchone()
            if near_symbol:
                nearest_box = (near_symbol["x1"], near_symbol["y1"], near_symbol["x2"], near_symbol["y2"])

        highlighted_image = ""
        if with_crops and r["image_path"]:
            crop_file = CROPS_DIR / f"{r['page_name']}_rq{r['review_id']}.png"
            if not crop_file.exists():
                try:
                    save_boxed_crop(r["image_path"], r["x1"], r["y1"], r["x2"], r["y2"],
                                     crop_file, margin=REVIEW_QUEUE_CROP_MARGIN,
                                     nearest_box=nearest_box)
                except Exception as e:
                    crop_failures += 1
                    print(f"  WARNING: crop failed for review_id {r['review_id']} "
                          f"on {r['page_name']}: {e}")
                    crop_file = None
            highlighted_image = str(crop_file) if crop_file else ""

        simple_rows.append({
            "source": "REVIEW_QUEUE",
            "tag_id": "", "review_id": r["review_id"],
            "page_name": r["page_name"],
            "what_is_this": what_is_this_review_queue(r, priority),
            "current_tag": r["candidate_tag"] or r["raw_text"] or "(unreadable)",
            "confidence": confidence_bucket(r["ocr_conf"]),
            "flag_note": "",
            "highlighted_image": highlighted_image,
            "reviewer_action": action, "reviewer_tag": r_tag, "reviewer_class": r_class,
            "reviewer_notes": "",
        })

        if technical:
            technical_rows.append({
                "source": "REVIEW_QUEUE", "tag_id": "", "review_id": r["review_id"],
                "page_name": r["page_name"], "det_id": "",
                "class_name": "", "kind": r["kind"],
                "x1": r["x1"], "y1": r["y1"], "x2": r["x2"], "y2": r["y2"],
                "detection_confidence": "", "reading_confidence": r["ocr_conf"],
                "combined_confidence": "", "raw_reading": r["raw_text"] or "",
                "tag_status": "OPEN",
                "nearest_class_name": r["nearest_class_name"] if has_nearest else "",
                "nearest_routing": r["nearest_routing"] if has_nearest else "",
                "nearest_distance_px": r["nearest_dist_px"] if has_nearest else "",
            })

    conn.close()

    out_path = OUT_DIR / "review_export.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SIMPLE_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(simple_rows)

    print(f"\nExported {len(simple_rows)} item(s) needing review -> {out_path}")
    print(f"  {len(tag_rows)} from tags, {len(rq_rows)} from review_queue")
    print(f"  {quick_confirm_count} pre-filled as quick confirms — skim and "
          f"correct any that are wrong, everything else needs a real look")
    if with_crops:
        print(f"  Boxed images saved to {CROPS_DIR} ({crop_failures} failures)")
    if not simple_rows:
        print("  Nothing needs review right now — either the DB is empty, "
              "or everything is already auto-accepted / human-verified / resolved.")

    tech_path = None
    if technical:
        tech_path = OUT_DIR / "review_export_technical.csv"
        with open(tech_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TECHNICAL_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(technical_rows)
        print(f"  Technical companion (raw model fields, reference only, "
              f"do not re-import) -> {tech_path}")

    return out_path, tech_path

# ─── IMPORT ────────────────────────────────────────────────────────────────

def apply_tag_action(conn, row, action, reviewer_name, now):
    tag_id = int(row["tag_id"])
    current = conn.execute(
        "SELECT tag, tag_status, page_id FROM tags WHERE tag_id = ?", (tag_id,)
    ).fetchone()
    if not current:
        print(f"  WARNING: tag_id {tag_id} not found in DB (stale export?) — skipped.")
        return None, None

    old_tag = current["tag"]
    notes = row.get("reviewer_notes") or None

    if action == "ACCEPT":
        conn.execute(
            """UPDATE tags SET verified_by=?, verified_at=?,
                   verification_method='HUMAN_VERIFIED' WHERE tag_id=?""",
            (reviewer_name, now, tag_id)
        )
        log_action, new_val = "ACCEPTED", old_tag

    elif action == "REJECT":
        conn.execute(
            """UPDATE tags SET verified_by=?, verified_at=?,
                   verification_method='HUMAN_VERIFIED' WHERE tag_id=?""",
            (reviewer_name, now, tag_id)
        )
        log_action, new_val = "REJECTED", old_tag

    else:  # CORRECT
        new_tag = (row.get("reviewer_tag") or "").strip()
        if not new_tag:
            print(f"  WARNING: tag_id {tag_id} marked CORRECT but "
                  f"reviewer_tag is blank — skipped.")
            return None, None
        conn.execute(
            """UPDATE tags SET tag=?, tag_status='VALID',
                   verified_by=?, verified_at=?,
                   verification_method='HUMAN_VERIFIED' WHERE tag_id=?""",
            (new_tag, reviewer_name, now, tag_id)
        )
        log_action, new_val = "CORRECTED", new_tag

    conn.execute(
        """INSERT INTO validation_log
           (tag_id, action, old_value, new_value, performed_by, performed_at, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (tag_id, log_action, old_tag, new_val, reviewer_name, now, notes)
    )
    return log_action, current["page_id"]


def apply_review_queue_action(conn, row, action, reviewer_name, now, add_to_retrain_pool=True):
    review_id = int(row["review_id"])
    current = conn.execute(
        """SELECT rq.page_id, rq.x1, rq.y1, rq.x2, rq.y2, rq.candidate_tag,
                  rq.raw_text, rq.ocr_conf, rq.status, pg.page_name, pg.image_path
           FROM review_queue rq JOIN pages pg ON pg.page_id = rq.page_id
           WHERE rq.review_id = ?""", (review_id,)
    ).fetchone()
    if not current:
        print(f"  WARNING: review_id {review_id} not found in DB (stale export?) — skipped.")
        return None, None
    if current["status"] != "OPEN":
        print(f"  WARNING: review_id {review_id} is already '{current['status']}' "
              f"— skipped (stale export, someone else resolved it already).")
        return None, None

    notes = row.get("reviewer_notes") or None
    page_id = current["page_id"]

    if action == "DISMISS":
        conn.execute(
            """UPDATE review_queue SET status='DISMISSED', resolved_by=?,
                   resolved_at=?, resolution_note=? WHERE review_id=?""",
            (reviewer_name, now, notes, review_id)
        )
        return "DISMISSED", page_id

    # PROMOTE — human confirmed this orphaned text is a real tag with no
    # YOLO-detected symbol. Create the symbol + tag now, marked
    # HUMAN_VERIFIED immediately since a person looked at it directly.
    new_tag = (row.get("reviewer_tag") or "").strip()
    new_class = (row.get("reviewer_class") or "").strip()
    if not new_tag or not new_class:
        print(f"  WARNING: review_id {review_id} marked PROMOTE but "
              f"reviewer_tag/reviewer_class is blank — skipped.")
        return None, None

    x1, y1, x2, y2 = current["x1"], current["y1"], current["x2"], current["y2"]
    # Same deterministic-negative-det_id scheme as rescue_to_db.py, so a
    # tag already promoted/rescued at this exact bbox won't be duplicated.
    det_id = -(x1 * 100000 + y1)

    # BUG (found 2026-07-13 while verifying this script end-to-end): this
    # used to do its own existing-symbol SELECT, then call
    # db_builder.insert_symbol() followed by a separate
    # db_builder.get_symbol_id() call that doesn't exist anywhere in
    # db_builder.py — an immediate AttributeError on every PROMOTE action.
    # insert_symbol() already does exactly this "INSERT OR IGNORE, then
    # always look the row up" dance internally (see its own docstring
    # comment on why lastrowid isn't trustworthy here) and returns the
    # correct symbol_id either way — new insert or pre-existing row. No
    # need to duplicate that logic here.
    class_id_row = conn.execute(
        "SELECT class_id FROM symbols WHERE class_name = ? LIMIT 1", (new_class,)
    ).fetchone()
    class_id = class_id_row["class_id"] if class_id_row else -1
    # yolo_conf=0.0 is a placeholder meaning "not YOLO-detected, human-added"
    # — symbols.yolo_conf is NOT NULL so there's no clean way to say
    # "not applicable" without a schema change.
    symbol_id = db_builder.insert_symbol(conn, page_id, det_id, new_class, class_id,
                                          0.0, x1, y1, x2, y2)

    tag_id = db_builder.insert_tag(
        conn, symbol_id, page_id,
        best_zone=None,
        raw_ocr=current["raw_text"],
        ocr_conf=current["ocr_conf"],
        tag=new_tag,
        tag_status="VALID",
        combined_conf=current["ocr_conf"],
        auto_accept=False,
        verified_by=reviewer_name,
        verified_at=now,
        verification_method="HUMAN_VERIFIED",
    )
    conn.execute(
        """INSERT INTO validation_log
           (tag_id, action, old_value, new_value, performed_by, performed_at, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (tag_id, "PROMOTED_FROM_REVIEW_QUEUE", current["candidate_tag"], new_tag,
         reviewer_name, now, notes)
    )
    conn.execute(
        """UPDATE review_queue SET status='RESOLVED', resolved_by=?,
               resolved_at=?, resolution_note=? WHERE review_id=?""",
        (reviewer_name, now, f"Promoted to tag_id={tag_id}", review_id)
    )

    if add_to_retrain_pool and current["image_path"]:
        try:
            added = append_retrain_sample(
                current["image_path"], current["page_name"], x1, y1, x2, y2,
                class_id, new_class, tag_id, review_id, reviewer_name, now,
            )
            if added:
                print(f"  Retrain pool: added {new_class} label for "
                      f"{current['page_name']} (tag_id={tag_id})")
        except Exception as e:
            print(f"  WARNING: could not add review_id {review_id} to retrain "
                  f"pool ({current['page_name']}): {e} — DB update still succeeded.")

    return "PROMOTED", page_id


def apply_review_csv(csv_path, reviewer_name, add_to_retrain_pool=True):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    conn = db_builder.get_connection()
    now = datetime.now().isoformat()

    applied = {"ACCEPT": 0, "REJECT": 0, "CORRECT": 0, "DISMISS": 0, "PROMOTE": 0}
    skipped_blank = 0
    skipped_bad = 0
    touched_pages = set()

    for row in rows:
        source = (row.get("source") or "TAG").strip().upper()
        action = (row.get("reviewer_action") or "").strip().upper()
        if not action:
            skipped_blank += 1
            continue

        if source == "TAG":
            if action not in TAG_ACTIONS:
                print(f"  WARNING: tag_id {row.get('tag_id')} has unrecognized "
                      f"reviewer_action '{action}' (expected ACCEPT/REJECT/CORRECT) "
                      f"— skipped.")
                skipped_bad += 1
                continue
            log_action, page_id = apply_tag_action(conn, row, action, reviewer_name, now)

        elif source == "REVIEW_QUEUE":
            if action not in REVIEW_QUEUE_ACTIONS:
                print(f"  WARNING: review_id {row.get('review_id')} has unrecognized "
                      f"reviewer_action '{action}' (expected DISMISS/PROMOTE) "
                      f"— skipped.")
                skipped_bad += 1
                continue
            log_action, page_id = apply_review_queue_action(
                conn, row, action, reviewer_name, now,
                add_to_retrain_pool=add_to_retrain_pool)

        else:
            print(f"  WARNING: row has unrecognized source '{source}' — skipped.")
            skipped_bad += 1
            continue

        if log_action is None:
            skipped_bad += 1
            continue

        applied[action] += 1
        touched_pages.add(page_id)

    for page_id in touched_pages:
        db_builder.update_page_stats(conn, page_id)

    conn.commit()

    # Real-time propagation (same guarantee review_server.py's live /api/action
    # path already gives): a DB write from --import-csv is just as much a
    # "changed" event as a click in the browser, so every touched page's
    # canonical JSON is re-exported here too — otherwise CSV/PDF/Excel/
    # dashboard consumers of page_json go stale after a batch import even
    # though the DB itself is current. Best-effort: a JSON re-export failure
    # shouldn't undo an already-committed reviewer decision, so it's logged,
    # not raised.
    for page_id in touched_pages:
        try:
            page_json.export_page_json_by_id(page_id, conn)
        except Exception as json_err:
            print(f"  WARNING: page {page_id} JSON re-export failed: {json_err}")

    conn.close()

    print(f"\nApplied — tags: {applied['ACCEPT']} accepted, {applied['REJECT']} rejected, "
          f"{applied['CORRECT']} corrected")
    print(f"Applied — review_queue: {applied['DISMISS']} dismissed, "
          f"{applied['PROMOTE']} promoted to real tags")
    print(f"Skipped: {skipped_blank} left blank, {skipped_bad} invalid/stale rows")
    print(f"Logged {applied['ACCEPT'] + applied['REJECT'] + applied['CORRECT'] + applied['PROMOTE']} "
          f"actions to validation_log (DISMISS has no tag_id, so it's only "
          f"recorded on the review_queue row itself, not in validation_log)")

# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PipeSight AI — Human Review Export")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--export", action="store_true",
                        help="Export tags + review_queue items needing review to CSV")
    group.add_argument("--import-csv", dest="import_csv", type=str, metavar="PATH",
                        help="Apply a filled-in review CSV back to the DB")
    parser.add_argument("--page", type=str, default=None,
                         help="Limit export to a single page name")
    parser.add_argument("--no-crops", dest="with_crops", action="store_false",
                         help="Skip boxed image generation (faster, text-only "
                              "review). Crops are ON by default so the "
                              "reviewer always sees what's boxed, not just text.")
    parser.add_argument("--technical", action="store_true",
                         help="Also write review_export_technical.csv with "
                              "the raw model fields (confidence numbers, "
                              "nearest-match details) for engineering/audit "
                              "use. Not meant to be re-imported.")
    parser.add_argument("--reviewer", type=str, default="human",
                         help="Name to record as performed_by / verified_by / "
                              "resolved_by when importing (default: 'human')")
    parser.add_argument("--no-retrain-pool", action="store_true",
                         help="Skip writing PROMOTE actions to data\\retrain_pool\\ "
                              "(by default every PROMOTE also becomes a YOLO-format "
                              "training label — see module header)")
    args = parser.parse_args()

    if not db_builder.DB_PATH.exists():
        print(f"Database not found at {db_builder.DB_PATH}. Run "
              f"tag_extractor_v2.py without --no-db first, or "
              f"python db_builder.py to initialize an empty one.")
        return

    if args.export:
        export_review_csv(page_name=args.page, with_crops=args.with_crops,
                           technical=args.technical)
    else:
        apply_review_csv(args.import_csv, reviewer_name=args.reviewer,
                          add_to_retrain_pool=not args.no_retrain_pool)

if __name__ == "__main__":
    main()