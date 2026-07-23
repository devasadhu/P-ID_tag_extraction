# review_server.py - PipeSight AI
#
# Live human-review server. Replaces the old flow (export CSV -> edit by
# hand -> re-import CSV) with a single running app: every click writes
# straight to the DB and commits immediately, so there is exactly one
# source of truth and nothing can drift out of sync with what the CSV/PDF/
# Excel exporters or the dashboard read later.
#
# Reuses the exact same DB-writing functions human_review_export.py already
# has (apply_tag_action / apply_review_queue_action) instead of duplicating
# that logic — a click here does precisely what a CSV row with that action
# would have done, just applied the instant you make the call instead of in
# a later batch import.
#
# Run:
#   python review_server.py
#   (then open http://127.0.0.1:5000 in a browser)
#
# Requires: flask, Pillow (already a dependency via human_review_export.py)
#   pip install flask --break-system-packages   (if not already installed)
#
# ── Naming note (per supervisor feedback) ──
# Nothing in the JSON responses or file names below says "yolo" or "paddle"
# or "ocr" in reviewer-facing places — a reviewer switching between the
# simple and technical view should never need to know which model produced
# which number. The technical view still shows the real field names
# (ocr_conf, yolo_conf, nearest_dist_px, etc.) because that's the audience
# that needs them, but they're grouped under one "technical" block that's
# simply hidden, not relabeled with different data.
#
# ── Not yet verified against a live DB in this session ──
# db_builder.py wasn't available when this was written, so the extra
# review_queue columns (nearest_det_id / nearest_class_name / nearest_routing
# / nearest_dist_px) referenced below are inferred from gallery.html's saved
# payloads, not confirmed against the actual schema. The SELECT below tries
# them and falls back cleanly if they don't exist — see fetch_review_queue_rows_ext.
# Run once against the real DB and check the startup log line
# ("nearest-* columns: available / not found") before relying on it.

import io
import sqlite3
from pathlib import Path
from datetime import datetime

from flask import Flask, jsonify, request, render_template, send_file, abort
from PIL import Image, ImageDraw

import db_builder
import human_review_export as hre
import page_json
import page_summary

app = Flask(__name__)

CROP_CACHE_DIR = Path("data") / "human_review" / "crops_boxed"
CROP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Plain-language action vocabulary shown to reviewers, mapped back onto the
# exact action strings apply_tag_action / apply_review_queue_action expect.
# Changing labels here never requires touching the DB-writing code.
TAG_ACTION_LABELS = {
    "ACCEPT":  "Correct as shown",
    "REJECT":  "Not a real tag",
    "CORRECT": "Fix the text",
}
RQ_ACTION_LABELS = {
    "PROMOTE": "Yes, this is a real tag",
    "DISMISS": "Not a real tag",
}

# Plain-language phrasing for reference_noise_reason() values from
# matching_engine.py, so the reviewer-facing note never shows a raw
# enum-style string like "pipe_or_line_spec_text".
REFERENCE_NOISE_LABELS = {
    "pipe_or_line_spec_text": "a pipe or line spec callout",
    "setpoint_annotation": "a pressure/temperature setpoint note",
    "non_instrument_abbreviation": "a known non-instrument abbreviation",
}

# Short, plain-language options for the symbol type a PROMOTE needs
# (reviewer_class). Placeholder set — confirm the real class list from
# symbol_names.json / the merged 61-class model and adjust before relying
# on this for anything beyond the field-mounted default.
CLASS_TYPE_OPTIONS = [
    {"value": "Instrument_Field", "label": "Field-mounted instrument"},
    {"value": "Instrument_Panel", "label": "Panel-mounted instrument"},
    {"value": "Instrument_DCS",   "label": "DCS / computer function"},
]


# ─── data access ────────────────────────────────────────────────────────────

def fetch_review_queue_rows_ext(conn, page_name=None):
    """
    Same rows as human_review_export.fetch_review_queue_rows, plus the
    nearest_* columns gallery.html expects for the technical view, if that
    columns exist in review_queue. Falls back to the base query otherwise.
    """
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
        rows = hre.fetch_review_queue_rows(conn, page_name=page_name)
        return rows, False


def triage_tag(row):
    """
    Bucket TAG rows so obvious ones can be batch-confirmed instead of
    reviewed one at a time. RAW/MISSING got no ISA-shape match at all —
    genuinely needs a look. VALID-but-below-threshold already passed the
    shape check and just missed the confidence bar, so it's a quick
    confirm rather than a real unknown.
    """
    status = row["tag_status"]
    if status == "VALID":
        return "quick_confirm"
    return "needs_review"


def triage_review_queue(row, has_nearest):
    """
    ORPHANED_TAG_CANDIDATE has no nearby symbol at all — real unknown,
    needs a look. MISROUTED_CANDIDATE with a very close nearby symbol
    (matching engine already found a tight spatial match) is usually a
    genuine fix waiting to be confirmed, not a fresh judgment call.
    """
    kind = row["kind"]
    if kind == "MISROUTED_CANDIDATE" and has_nearest:
        dist = row["nearest_dist_px"]
        if dist is not None and dist < 150:
            return "quick_confirm"
    return "needs_review"


NEAR_DUPLICATE_PX = 60  # center-to-center distance below which two
                        # REVIEW_QUEUE items are treated as the same
                        # physical spot read differently by OCR, not two
                        # separate candidates.


def _bbox_centroid(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _bbox_dist(b1, b2):
    c1, c2 = _bbox_centroid(b1), _bbox_centroid(b2)
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


def cluster_near_duplicates(items, threshold=NEAR_DUPLICATE_PX):
    """
    Two review_queue candidates on the same page whose bboxes are within
    `threshold` px of each other are almost always the SAME physical
    symbol/text, read differently by two separate OCR passes (e.g. one
    pass picks up a stray mark as an extra letter, another pass doesn't) —
    not two distinct real candidates. Exact-bbox dedup (fix_db_integrity.py,
    the _write_rows_to_db guard) doesn't catch this because the boxes
    aren't identical, just close. Clustered so a reviewer resolves the
    spot once instead of twice with two different OCR guesses.
    Only clusters REVIEW_QUEUE items — TAG rows are already one-per-symbol.
    """
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    for i in range(len(items)):
        if items[i]["type"] != "REVIEW_QUEUE":
            continue
        # A MANUAL_ADDITION row is a reviewer deliberately flagging a
        # SECOND, distinct instrument next to something already on the
        # queue — proximity is the whole point, not a sign it's the same
        # spot read twice. Excluding it here (rather than just raising the
        # nudge past `threshold`) is the robust fix: it also covers the
        # case where a reviewer drags the cloned box close to another
        # existing candidate while positioning it, which a bigger nudge
        # wouldn't protect against.
        if items[i]["technical"].get("kind") == "MANUAL_ADDITION":
            continue
        for j in range(i + 1, len(items)):
            if items[j]["type"] != "REVIEW_QUEUE":
                continue
            if items[j]["technical"].get("kind") == "MANUAL_ADDITION":
                continue
            if items[i]["page"] != items[j]["page"]:
                continue
            if _bbox_dist(items[i]["technical"]["bbox"], items[j]["technical"]["bbox"]) <= threshold:
                union(i, j)

    groups = {}
    for i in range(len(items)):
        groups.setdefault(find(i), []).append(items[i])

    merged = []
    for group in groups.values():
        if len(group) == 1:
            merged.append(group[0])
            continue
        # Multiple conflicting OCR reads of the same spot — merge into one
        # card. No "correct as shown" quick action, since there's no single
        # unambiguous "as shown" when the readings disagree; reviewer must
        # either dismiss the whole cluster or pick/fix the correct text.
        first = group[0]
        merged.append({
            "id": first["id"],
            "merged_ids": [it["id"] for it in group],
            "type": "REVIEW_QUEUE",
            "page": first["page"],
            "tag": " / ".join(sorted(set(it["tag"] for it in group))),
            "raw_text": " / ".join(f'"{it["raw_text"]}"' for it in group),
            "priority": "needs_review",
            "crop_url": first["crop_url"],
            "actions": {"DISMISS": "Not a real tag"},
            "class_options": first.get("class_options", []),
            "note": f"OCR read this spot {len(group)} different ways — pick the correct tag or dismiss it.",
            "technical": {
                "readings": [it["tag"] for it in group],
                "bbox": first["technical"]["bbox"],
            },
        })
    return merged


def build_queue_payload(page_name=None):
    conn = db_builder.get_connection()
    tag_rows = hre.fetch_tag_rows(conn, page_name=page_name)
    rq_rows, has_nearest = fetch_review_queue_rows_ext(conn, page_name=page_name)
    conn.close()

    items = []

    for r in tag_rows:
        row_keys = r.keys()
        ref_noise = r["reference_noise"] if "reference_noise" in row_keys else ""
        item = {
            "id": f"tag_{r['tag_id']}",
            "type": "TAG",
            "page": r["page_name"],
            "tag": r["tag"] or "(unreadable)",
            "raw_text": r["raw_ocr"] or "",
            "priority": triage_tag(r),
            "crop_url": f"/crop/TAG/{r['tag_id']}",
            "actions": TAG_ACTION_LABELS,
            "technical": {
                "tag_status": r["tag_status"],
                "class_name": r["class_name"],
                "text_reading_confidence": r["ocr_conf"],
                "detection_confidence": r["yolo_conf"],
                "combined_confidence": r["combined_conf"],
                "det_id": r["det_id"],
                "bbox": [r["x1"], r["y1"], r["x2"], r["y2"]],
                "reference_noise": ref_noise,
            },
        }
        if ref_noise:
            # Plain-language flag for a VALID-shaped tag that also matched a
            # confirmed non-instrument pattern (pipe/line spec, setpoint
            # annotation, known abbreviation) — see reference_noise_reason()
            # in matching_engine.py. Shown unconditionally (not gated behind
            # the technical toggle) since this is exactly the kind of case
            # a reviewer needs flagged plainly, not buried in raw field names.
            reason_text = REFERENCE_NOISE_LABELS.get(ref_noise, ref_noise.replace("_", " "))
            item["note"] = f"Looks like {reason_text} near this symbol, not a confirmed instrument tag — check before accepting."
        items.append(item)

    for r in rq_rows:
        priority_val = triage_review_queue(r, has_nearest)

        # NOTE: previously skipped "quick_confirm" items entirely here,
        # which hid genuinely OPEN review_queue rows from the UI (they
        # still needed a human action, just an easy one) while the PDF/
        # Excel export kept showing them as pending — UI said "done",
        # export said otherwise. Now included like everything else;
        # "priority": "quick_confirm" lets the UI de-prioritize/sort them
        # into a fast-confirm bucket instead of making them vanish.

        technical = {
            "kind": r["kind"],
            "text_reading_confidence": r["ocr_conf"],
            "bbox": [r["x1"], r["y1"], r["x2"], r["y2"]],
        }
        if has_nearest:
            technical.update({
                "nearest_class_name": r["nearest_class_name"],
                "nearest_routing": r["nearest_routing"],
                "nearest_dist_px": r["nearest_dist_px"],
            })
        item = {
            "id": f"rq_{r['review_id']}",
            "type": "REVIEW_QUEUE",
            "page": r["page_name"],
            "tag": r["candidate_tag"] or "(unreadable)",
            "raw_text": r["raw_text"] or "",
            "priority": priority_val,
            "crop_url": f"/crop/REVIEW_QUEUE/{r['review_id']}",
            "actions": RQ_ACTION_LABELS,
            "class_options": CLASS_TYPE_OPTIONS,
            "technical": technical,
        }
        if r["kind"] == "MANUAL_ADDITION":
            item["note"] = ("You added this one — drag the box onto the instrument, "
                             "type its tag, and confirm.")
        items.append(item)

    items = cluster_near_duplicates(items)

    pages = {}
    for it in items:
        pages.setdefault(it["page"], []).append(it)

    return pages


# ─── crop-with-box generation ──────────────────────────────────────────────

def _crop_bounds(img_w, img_h, x1, y1, x2, y2, margin, text_box=None):
    """Shared by make_boxed_crop, the raw-crop endpoint, and crop_meta, so
    all three agree on the exact same crop rectangle — required for the
    box-edit overlay's coordinate math (crop_origin) to line up with what
    the reviewer actually sees."""
    all_x = [x1, x2]
    all_y = [y1, y2]
    if text_box:
        all_x += [text_box["x1"], text_box["x2"]]
        all_y += [text_box["y1"], text_box["y2"]]
    cx1 = max(0, min(all_x) - margin)
    cy1 = max(0, min(all_y) - margin)
    cx2 = min(img_w, max(all_x) + margin)
    cy2 = min(img_h, max(all_y) + margin)
    return cx1, cy1, cx2, cy2


def make_boxed_crop(image_path, x1, y1, x2, y2, margin, out_path, text_box=None):
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size
    cx1, cy1, cx2, cy2 = _crop_bounds(img_w, img_h, x1, y1, x2, y2, margin, text_box)
    crop = img.crop((cx1, cy1, cx2, cy2))

    draw = ImageDraw.Draw(crop)
    if text_box:
        sym_box = (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1)
        draw.rectangle(sym_box, outline=(120, 130, 140), width=2)
        txt_box = (text_box["x1"] - cx1, text_box["y1"] - cy1,
                   text_box["x2"] - cx1, text_box["y2"] - cy1)
        draw.rectangle(txt_box, outline=(230, 30, 30), width=3)
    else:
        box = (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1)
        draw.rectangle(box, outline=(230, 30, 30), width=3)

    crop.save(out_path)


def get_row_for_crop(row_type, item_id):
    conn = db_builder.get_connection()
    try:
        if row_type == "TAG":
            row = conn.execute(
                """SELECT t.tag_id, t.symbol_id, s.x1, s.y1, s.x2, s.y2,
                          t.text_x1, t.text_y1, t.text_x2, t.text_y2,
                          pg.image_path, pg.page_name
                   FROM tags t
                   JOIN symbols s ON s.symbol_id = t.symbol_id
                   JOIN pages pg  ON pg.page_id  = t.page_id
                   WHERE t.tag_id = ?""", (item_id,)
            ).fetchone()
        else:
            row = conn.execute(
                """SELECT rq.review_id, rq.kind, rq.x1, rq.y1, rq.x2, rq.y2,
                          rq.nearest_det_id,
                          pg.image_path, pg.page_name, pg.page_id,
                          ns.x1 AS near_x1, ns.y1 AS near_y1,
                          ns.x2 AS near_x2, ns.y2 AS near_y2
                   FROM review_queue rq
                   JOIN pages pg ON pg.page_id = rq.page_id
                   LEFT JOIN symbols ns
                     ON ns.page_id = rq.page_id AND ns.det_id = rq.nearest_det_id
                   WHERE rq.review_id = ?""", (item_id,)
            ).fetchone()
        return row
    finally:
        conn.close()


@app.route("/crop/<row_type>/<int:item_id>")
def crop_endpoint(row_type, item_id):
    if row_type not in ("TAG", "REVIEW_QUEUE"):
        abort(404)
    row = get_row_for_crop(row_type, item_id)
    if not row or not row["image_path"]:
        abort(404)

    margin = hre.TAG_CROP_MARGIN if row_type == "TAG" else hre.REVIEW_QUEUE_CROP_MARGIN
    cache_file = CROP_CACHE_DIR / f"{row_type}_{item_id}.png"
    if not cache_file.exists():
        text_box = None
        if row_type == "TAG" and row["text_x1"] is not None:
            text_box = {"x1": row["text_x1"], "y1": row["text_y1"],
                        "x2": row["text_x2"], "y2": row["text_y2"]}
            make_boxed_crop(row["image_path"], row["x1"], row["y1"], row["x2"], row["y2"],
                            margin, cache_file, text_box=text_box)
        elif (row_type == "REVIEW_QUEUE" and row["kind"] == "MISROUTED_CANDIDATE"
              and row["near_x1"] is not None):
            # Candidate text's own box drawn red (matches TAG rows' convention),
            # nearest symbol's box drawn grey — so the reviewer can see the
            # actual match the matching engine found, not just the text alone.
            candidate_box = {"x1": row["x1"], "y1": row["y1"],
                              "x2": row["x2"], "y2": row["y2"]}
            make_boxed_crop(row["image_path"], row["near_x1"], row["near_y1"],
                            row["near_x2"], row["near_y2"],
                            margin, cache_file, text_box=candidate_box)
        else:
            make_boxed_crop(row["image_path"], row["x1"], row["y1"], row["x2"], row["y2"],
                            margin, cache_file, text_box=None)
    return send_file(cache_file, mimetype="image/png")


def _editable_box(row, row_type):
    if row_type == "TAG":
        if row["text_x1"] is not None:
            return ([row["text_x1"], row["text_y1"], row["text_x2"], row["text_y2"]],
                     "tag_text")
        return ([row["x1"], row["y1"], row["x2"], row["y2"]], "symbol")
    return ([row["x1"], row["y1"], row["x2"], row["y2"]], "review_queue")


@app.route("/crop_raw/<row_type>/<int:item_id>")
def crop_raw_endpoint(row_type, item_id):
    if row_type not in ("TAG", "REVIEW_QUEUE"):
        abort(404)
    row = get_row_for_crop(row_type, item_id)
    if not row or not row["image_path"]:
        abort(404)

    margin = hre.TAG_CROP_MARGIN if row_type == "TAG" else hre.REVIEW_QUEUE_CROP_MARGIN
    text_box = None
    if row_type == "TAG" and row["text_x1"] is not None:
        text_box = {"x1": row["text_x1"], "y1": row["text_y1"],
                    "x2": row["text_x2"], "y2": row["text_y2"]}
    img = Image.open(row["image_path"]).convert("RGB")
    cx1, cy1, cx2, cy2 = _crop_bounds(img.width, img.height, row["x1"], row["y1"],
                                       row["x2"], row["y2"], margin, text_box)
    buf = io.BytesIO()
    img.crop((cx1, cy1, cx2, cy2)).save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


def _parse_item_id(id_str):
    if id_str.startswith("tag_"):
        return "TAG", int(id_str[len("tag_"):])
    if id_str.startswith("rq_"):
        return "REVIEW_QUEUE", int(id_str[len("rq_"):])
    return None, None


@app.route("/api/crop_meta/<id>")
def api_crop_meta(id):
    row_type, item_id = _parse_item_id(id)
    if row_type is None:
        return jsonify({"ok": False, "error": "unrecognized id"}), 400
    row = get_row_for_crop(row_type, item_id)
    if not row or not row["image_path"]:
        return jsonify({"ok": False, "error": "not found"}), 404

    margin = hre.TAG_CROP_MARGIN if row_type == "TAG" else hre.REVIEW_QUEUE_CROP_MARGIN
    text_box = None
    if row_type == "TAG" and row["text_x1"] is not None:
        text_box = {"x1": row["text_x1"], "y1": row["text_y1"],
                    "x2": row["text_x2"], "y2": row["text_y2"]}
    img = Image.open(row["image_path"])
    cx1, cy1, cx2, cy2 = _crop_bounds(img.width, img.height, row["x1"], row["y1"],
                                       row["x2"], row["y2"], margin, text_box)
    editable_box, _target = _editable_box(row, row_type)

    return jsonify({
        "ok": True,
        "editable_box": editable_box,
        "crop_origin": [cx1, cy1],
        "image_url": f"/crop_raw/{row_type}/{item_id}",
    })


@app.route("/api/bbox", methods=["POST"])
def api_bbox():
    body = request.get_json(force=True)
    id_str = body.get("id", "")
    row_type, item_id = _parse_item_id(id_str)
    if row_type is None:
        return jsonify({"ok": False, "error": "unrecognized id"}), 400
    try:
        x1, y1, x2, y2 = int(body["x1"]), int(body["y1"]), int(body["x2"]), int(body["y2"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"ok": False, "error": "x1/y1/x2/y2 must be integers"}), 400
    if x1 >= x2 or y1 >= y2:
        return jsonify({"ok": False, "error": "box has zero or negative area"}), 400

    conn = db_builder.get_connection()
    try:
        row = get_row_for_crop(row_type, item_id)
        if not row:
            return jsonify({"ok": False, "error": "not found"}), 404
        _box, target = _editable_box(row, row_type)

        if target == "tag_text":
            conn.execute(
                "UPDATE tags SET text_x1=?, text_y1=?, text_x2=?, text_y2=? WHERE tag_id=?",
                (x1, y1, x2, y2, item_id))
        elif target == "symbol":
            conn.execute(
                "UPDATE symbols SET x1=?, y1=?, x2=?, y2=? WHERE symbol_id=?",
                (x1, y1, x2, y2, row["symbol_id"]))
        else:  # review_queue
            conn.execute(
                "UPDATE review_queue SET x1=?, y1=?, x2=?, y2=? WHERE review_id=?",
                (x1, y1, x2, y2, item_id))
        conn.commit()

        # Cached boxed crop is now stale — drop it so /crop regenerates
        cache_file = CROP_CACHE_DIR / f"{row_type}_{item_id}.png"
        cache_file.unlink(missing_ok=True)

        # 🔥 FIX 2: Export page json after every layout box alteration
        try:
            page_json.export_page_json(row["page_name"], conn=conn)
            page_summary.export_summary_json(row["page_name"], conn=conn)
        except Exception as json_err:
            print(f"  WARNING: page JSON re-export failed after bbox save: {json_err}")

        return jsonify({"ok": True})
    except Exception as e:
        conn.rollback()
        return jsonify({"ok": False, "error": str(e)}), 500
    finally:
        conn.close()


@app.route("/api/duplicate_finding", methods=["POST"])
def api_duplicate_finding():
    """
    Lets a reviewer spawn a brand-new, independently-editable candidate in
    the SAME crop they're already looking at — for the case where fixing
    or confirming one candidate's box reveals a second, never-flagged
    instrument sitting right next to it in the same image (neither YOLO
    nor the OCR-driven review-queue scan caught it — a human just happened
    to notice it while working on something else). Rather than losing that
    finding, this clones the current item's page + starting bbox into a
    fresh OPEN review_queue row (kind='MANUAL_ADDITION', candidate_tag=None
    since nothing read it yet, nearest_det_id=None since it isn't tied to
    any detection at all) that the reviewer immediately drags onto the
    real second instrument's position and PROMOTEs — same downstream path
    (retrain pool, symbols/tags tables) as any other review-queue item, see
    apply_review_queue_action(): PROMOTE reads the bbox live off the row
    and the tag text from whatever the reviewer types at PROMOTE time, so
    neither field needs to be meaningful at creation time here.

    Works from either a TAG or REVIEW_QUEUE source card — a reviewer can
    notice a second missed instrument while looking at either kind.
    """
    body = request.get_json(force=True)
    id_str = body.get("id", "")
    row_type, item_id = _parse_item_id(id_str)
    if row_type is None:
        return jsonify({"ok": False, "error": "unrecognized id"}), 400

    row = get_row_for_crop(row_type, item_id)
    if not row or not row["image_path"]:
        return jsonify({"ok": False, "error": "not found"}), 404

    conn = db_builder.get_connection()
    try:
        page_id = conn.execute(
            "SELECT page_id FROM pages WHERE page_name = ?", (row["page_name"],)
        ).fetchone()["page_id"]

        # Starting box: same size as the source item, nudged +40px on both
        # axes so it doesn't render exactly on top of the original card's
        # box — the reviewer drags it onto the actual second instrument
        # from there using the same box-editor api_bbox already exposes.
        # Clamped so a source box near the page edge doesn't drift off it.
        img = Image.open(row["image_path"])
        NUDGE = 40
        x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
        w, h = x2 - x1, y2 - y1
        x1 = min(x1 + NUDGE, img.width - w)
        y1 = min(y1 + NUDGE, img.height - h)
        x2, y2 = x1 + w, y1 + h

        new_id = db_builder.insert_review_candidate(
            conn, page_id, kind="MANUAL_ADDITION",
            candidate_tag=None,
            raw_text="(manually added — reviewer spotted this, not yet identified)",
            ocr_conf=None,
            x1=x1, y1=y1, x2=x2, y2=y2,
            nearest_det_id=None, nearest_class_name=None,
            nearest_routing=None, nearest_dist_px=None,
        )
        conn.commit()

        # Same fix as api_bbox's "🔥 FIX 2" — the new row exists in the DB
        # now, but the page's JSON stays stale until something re-exports
        # it. Without this, the duplicate-finding button would silently
        # not show up for any consumer reading _tags.json until an
        # unrelated action on the same page happened to trigger a refresh.
        try:
            page_json.export_page_json(row["page_name"], conn=conn)
            page_summary.export_summary_json(row["page_name"], conn=conn)
        except Exception as json_err:
            print(f"  WARNING: page JSON re-export failed after duplicate_finding: {json_err}")

        return jsonify({"ok": True, "new_id": f"rq_{new_id}"})
    except Exception as e:
        conn.rollback()
        return jsonify({"ok": False, "error": str(e)}), 500
    finally:
        conn.close()


# ─── API ────────────────────────────────────────────────────────────────────

@app.route("/api/queue")
def api_queue():
    page = request.args.get("page")
    return jsonify(build_queue_payload(page_name=page))


@app.route("/api/stats")
def api_stats():
    page = request.args.get("page")
    pages = build_queue_payload(page_name=page)
    total = sum(len(v) for v in pages.values())
    quick = sum(1 for v in pages.values() for it in v if it["priority"] == "quick_confirm")
    return jsonify({
        "total_remaining": total,
        "quick_confirm_remaining": quick,
        "needs_review_remaining": total - quick,
        "pages": {p: len(v) for p, v in pages.items()},
    })


def _apply_single(conn, item_id, action, reviewer_name, now, reviewer_tag="", reviewer_class="", reviewer_notes=""):
    if item_id.startswith("tag_"):
        row_type, raw_id = "TAG", int(item_id[len("tag_"):])
    elif item_id.startswith("rq_"):
        row_type, raw_id = "REVIEW_QUEUE", int(item_id[len("rq_"):])
    else:
        return None, None

    fake_row = {
        "tag_id": raw_id, "review_id": raw_id,
        "reviewer_tag": reviewer_tag, "reviewer_class": reviewer_class,
        "reviewer_notes": reviewer_notes,
    }
    if row_type == "TAG":
        if action not in hre.TAG_ACTIONS:
            return None, None
        return hre.apply_tag_action(conn, fake_row, action, reviewer_name, now)
    else:
        if action not in hre.REVIEW_QUEUE_ACTIONS:
            return None, None
        return hre.apply_review_queue_action(conn, fake_row, action, reviewer_name, now,
                                              add_to_retrain_pool=True)


@app.route("/api/action", methods=["POST"])
def api_action():
    body = request.get_json(force=True)
    ids = body.get("ids") or ([body["id"]] if body.get("id") else [])
    action = (body.get("action") or "").strip().upper()
    reviewer_name = body.get("reviewer") or "reviewer"
    reviewer_tag = body.get("reviewer_tag", "")
    reviewer_class = body.get("reviewer_class", "")
    reviewer_notes = body.get("reviewer_notes", "")
    now = datetime.now().isoformat()

    if not ids:
        return jsonify({"ok": False, "error": "no id(s) provided"}), 400

    conn = db_builder.get_connection()
    try:
        results = []
        touched_pages = set()

        if action == "PROMOTE" and len(ids) > 1:
            plan = [(ids[0], "PROMOTE", reviewer_tag, reviewer_class, reviewer_notes)]
            plan += [(i, "DISMISS", "", "", "superseded — merged duplicate, corrected under a sibling id")
                     for i in ids[1:]]
        elif action == "DISMISS":
            plan = [(i, "DISMISS", "", "", reviewer_notes) for i in ids]
        else:
            plan = [(ids[0], action, reviewer_tag, reviewer_class, reviewer_notes)]

        for item_id, act, r_tag, r_class, r_notes in plan:
            log_action, page_id = _apply_single(conn, item_id, act, reviewer_name, now,
                                                 reviewer_tag=r_tag, reviewer_class=r_class,
                                                 reviewer_notes=r_notes)
            if log_action is None:
                conn.rollback()
                return jsonify({"ok": False, "error": f"action rejected for {item_id} "
                                                       f"(stale row, or missing required field)"}), 400
            results.append(log_action)
            if page_id is not None:
                touched_pages.add(page_id)

        for page_id in touched_pages:
            db_builder.update_page_stats(conn, page_id)
        conn.commit()

        # 🔥 FIX 2: Guarantee real-time propagation across page layout json trees
        for page_id in touched_pages:
            try:
                page_json.export_page_json_by_id(page_id, conn)
                page_summary.export_summary_json_by_id(page_id, conn)
            except Exception as json_err:
                print(f"  WARNING: page {page_id} JSON re-export failed: {json_err}")

        return jsonify({"ok": True, "applied": results})
    except Exception as e:
        conn.rollback()
        return jsonify({"ok": False, "error": str(e)}), 500
    finally:
        conn.close()


# ─── frontend ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("review.html")


if __name__ == "__main__":
    conn = db_builder.get_connection()
    _, has_nearest = fetch_review_queue_rows_ext(conn)
    conn.close()
    print(f"nearest-* columns on review_queue: {'available' if has_nearest else 'not found — technical view will hide them'}")
    print("Review server starting at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)