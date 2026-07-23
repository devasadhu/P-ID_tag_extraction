# rescue_low_conf.py - PipeSight AI
# Fast diagnostic: for every ORPHANED_TAG_CANDIDATE in review_queue.csv (OCR
# found tag-shaped text with NO YOLO detection anywhere nearby), crop a
# generous region around it and re-run YOLO on JUST that crop at a much
# lower confidence threshold and a looser NMS IoU. Two possible outcomes,
# and they point to two completely different fixes:
#
#   - A bubble shows up here that never showed up in the full-page pass
#     -> it was a confidence-threshold / NMS-suppression casualty, NOT a
#        model capability gap. Cheap fix: lower --conf and/or raise --iou
#        specifically for Instrument_Field in production inference.
#   - Still nothing, even near-zero confidence -> genuine model blind
#        spot on this exact spot. That's real evidence for the dense-
#        cluster training-data gap, not a guess.
#
# Usage:
#   python rescue_low_conf.py --review-queue data\tag_extraction_v2\review_queue.csv --images-dir data\haifa_real_pids\images --model models\best.pt
#
# Output: data\tag_extraction_v2\rescue_results.csv + printed summary.

import csv
import argparse
from pathlib import Path
from collections import defaultdict

INSTRUMENT_CLASS_NAMES = {"Instrument_Field", "Instrument_Panel", "Instrument_Aux_Panel"}

def resolve_image_path(page_name, images_dir):
    direct = images_dir / f"{page_name}.png"
    if direct.exists():
        return direct
    matches = list(images_dir.rglob(f"{page_name}.png"))
    return matches[0] if matches else None

def load_orphaned_rows(review_queue_path, page_filter=None):
    rows = []
    with open(review_queue_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["kind"] != "ORPHANED_TAG_CANDIDATE":
                continue
            if page_filter and row["page"] != page_filter:
                continue
            rows.append(row)
    return rows

def load_rescue_model(model_path):
    """
    Load a YOLO model for rescue scanning. Separate from tag_extractor_v2's
    own get_ocr()-style caching since this is a different library
    (ultralytics, not PaddleOCR) — callers that want caching across pages
    should hold onto the returned (model, names) tuple themselves, same as
    tag_extractor_v2.get_rescue_model() does.
    """
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    return model, model.names

def rescue_candidates(page_image, rows, model, names, page_name=None,
                       pad=350, conf=0.10, iou=0.75,
                       keep_missing_crops=False, missing_crops_dir=None):
    """
    Core rescue logic, extracted so it can run in-memory (called directly
    from tag_extractor_v2.py's auto-escalation, no review_queue.csv /
    rescue_results.csv round-trip needed) as well as from this script's own
    CLI (main(), below), which still reads/writes CSVs for standalone use.

    page_image: an already-open PIL Image for the page every row in `rows`
        belongs to (caller's responsibility — this function does NOT open
        images or resolve page paths, unlike the CLI path which handles
        multiple pages via image_cache).
    rows: list of dicts with at minimum candidate_tag, raw_text, ocr_conf,
        x1, y1, x2, y2 (string or numeric — both accepted). Same shape as
        review_queue.csv rows AND as the in-memory review_rows dicts
        find_review_queue_rows() in tag_extractor_v2.py produces.
    model, names: as returned by load_rescue_model().

    Returns a list of result dicts, same schema as rescue_results.csv
    (rescue_status is "RESCUED" or "STILL_MISSING").
    """
    tmp_dir = Path("data") / "_rescue_crops"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    if keep_missing_crops and missing_crops_dir:
        missing_crops_dir.mkdir(parents=True, exist_ok=True)

    results_out = []
    for i, row in enumerate(rows, 1):
        page = row.get("page", page_name or "page")
        x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
        cx1 = max(0, x1 - pad); cy1 = max(0, y1 - pad)
        cx2 = min(page_image.width, x2 + pad); cy2 = min(page_image.height, y2 + pad)
        crop = page_image.crop((cx1, cy1, cx2, cy2))
        candidate_tag = row.get("candidate_tag") or ""
        crop_path = tmp_dir / f"_rescue_{page}_{candidate_tag}_{i}.png"
        crop.save(crop_path)

        try:
            pred = model.predict(str(crop_path), conf=conf, iou=iou, verbose=False)
        except Exception as e:
            print(f"  [{i}/{len(rows)}] YOLO failed on {page} tag={candidate_tag!r}: {e}")
            crop_path.unlink(missing_ok=True)
            continue

        found_instrument = []
        if pred and len(pred) > 0:
            boxes = pred[0].boxes
            if boxes is not None:
                for b in boxes:
                    cls_id = int(b.cls.item())
                    cls_name = names.get(cls_id, f"class_{cls_id}")
                    if cls_name in INSTRUMENT_CLASS_NAMES:
                        bx1, by1, bx2, by2 = [float(v) for v in b.xyxy[0].tolist()]
                        found_instrument.append({
                            "class_name": cls_name,
                            "conf": round(float(b.conf.item()), 4),
                            "x1": round(cx1 + bx1), "y1": round(cy1 + by1),
                            "x2": round(cx1 + bx2), "y2": round(cy1 + by2),
                        })

        if not found_instrument and keep_missing_crops and missing_crops_dir:
            safe_tag = "".join(c if c.isalnum() or c in "-_" else "_" for c in candidate_tag)
            crop.save(missing_crops_dir / f"{page}_{safe_tag}_{i}.png")

        crop_path.unlink(missing_ok=True)

        ocr_conf_val = row.get("ocr_conf", "")
        base = {
            "page": page, "candidate_tag": candidate_tag,
            "raw_text": row.get("raw_text", ""), "ocr_conf": ocr_conf_val,
            "orig_tag_x1": x1, "orig_tag_y1": y1, "orig_tag_x2": x2, "orig_tag_y2": y2,
        }
        if found_instrument:
            best = max(found_instrument, key=lambda d: d["conf"])
            print(f"  [{i}/{len(rows)}] RESCUED  page={page}  tag={candidate_tag!r}  "
                  f"-> {best['class_name']} conf={best['conf']} at "
                  f"({best['x1']},{best['y1']},{best['x2']},{best['y2']})")
            results_out.append({
                **base, "rescue_status": "RESCUED",
                "rescued_class": best["class_name"], "rescued_conf": best["conf"],
                "rescued_x1": best["x1"], "rescued_y1": best["y1"],
                "rescued_x2": best["x2"], "rescued_y2": best["y2"],
                "all_candidates_found": len(found_instrument),
            })
        else:
            print(f"  [{i}/{len(rows)}] still nothing  page={page}  tag={candidate_tag!r} "
                  f"(genuine model blind spot at conf={conf})")
            results_out.append({
                **base, "rescue_status": "STILL_MISSING",
                "rescued_class": "", "rescued_conf": "",
                "rescued_x1": "", "rescued_y1": "", "rescued_x2": "", "rescued_y2": "",
                "all_candidates_found": 0,
            })
    return results_out

def main():
    ap = argparse.ArgumentParser(description="Rescue-scan: re-run YOLO at low conf / loose NMS "
                                              "around every OCR-flagged orphaned tag candidate.")
    ap.add_argument("--review-queue", type=str, default=str(Path("data") / "tag_extraction_v2" / "review_queue.csv"))
    ap.add_argument("--images-dir", type=str, required=True,
                    help="Root images folder, e.g. data\\haifa_real_pids\\images")
    ap.add_argument("--model", type=str, default=str(Path("models") / "best.pt"))
    ap.add_argument("--page", type=str, default=None, help="Only process rows for this page name")
    ap.add_argument("--pad", type=int, default=350,
                    help="Padding (px) around each orphaned tag's bbox for the rescue crop. "
                         "Generous by default since the bubble sits NEAR the tag, not on top of it.")
    ap.add_argument("--conf", type=float, default=0.10,
                    help="Rescue confidence threshold - deliberately much lower than production "
                         "default (usually 0.25+) to see everything the model even weakly considered.")
    ap.add_argument("--iou", type=float, default=0.75,
                    help="Rescue NMS IoU threshold - HIGHER than default (usually ~0.45-0.5) so "
                         "densely packed, genuinely-separate bubbles are LESS likely to be "
                         "suppressed as 'duplicates' of each other.")
    ap.add_argument("--out", type=str, default=str(Path("data") / "tag_extraction_v2" / "rescue_results.csv"))
    ap.add_argument("--keep-missing-crops", action="store_true",
                    help="Keep (don't delete) the crop image for every STILL_MISSING candidate, "
                         "saved under data\\_rescue_crops_missing\\, so you can visually inspect "
                         "them afterward to tell a real blind spot from an OCR false-positive "
                         "(text with no drawn symbol nearby at all).")
    args = ap.parse_args()

    review_queue_path = Path(args.review_queue)
    images_dir = Path(args.images_dir)
    if not review_queue_path.exists():
        print(f"review_queue.csv not found at {review_queue_path}")
        return

    rows = load_orphaned_rows(review_queue_path, page_filter=args.page)
    if not rows:
        print(f"No ORPHANED_TAG_CANDIDATE rows found"
              f"{f' for page {args.page}' if args.page else ''} in {review_queue_path}.")
        return

    print(f"Loaded {len(rows)} orphaned tag candidate(s) to re-check.")
    print(f"Loading YOLO model from {args.model} ...")
    model, names = load_rescue_model(args.model)

    from PIL import Image

    missing_crops_dir = Path("data") / "_rescue_crops_missing"

    # Group rows by page so each page image is opened/rescued once, same
    # as the old image_cache behavior, but delegating the actual per-row
    # crop/predict work to the shared rescue_candidates() function.
    rows_by_page = defaultdict(list)
    for row in rows:
        rows_by_page[row["page"]].append(row)

    results_out = []
    for page, page_rows in rows_by_page.items():
        img_path = resolve_image_path(page, images_dir)
        if img_path is None:
            print(f"  SKIP {page}: image not found under {images_dir} "
                  f"({len(page_rows)} candidate(s) skipped)")
            continue
        page_image = Image.open(img_path)
        results_out.extend(rescue_candidates(
            page_image, page_rows, model, names, page_name=page,
            pad=args.pad, conf=args.conf, iou=args.iou,
            keep_missing_crops=args.keep_missing_crops,
            missing_crops_dir=missing_crops_dir,
        ))

    rescued = sum(1 for r in results_out if r["rescue_status"] == "RESCUED")
    still_missing = sum(1 for r in results_out if r["rescue_status"] == "STILL_MISSING")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["page", "candidate_tag", "raw_text", "ocr_conf",
                  "orig_tag_x1", "orig_tag_y1", "orig_tag_x2", "orig_tag_y2",
                  "rescue_status", "rescued_class", "rescued_conf",
                  "rescued_x1", "rescued_y1", "rescued_x2", "rescued_y2",
                  "all_candidates_found"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_out)

    total = rescued + still_missing
    pct = round(rescued / total * 100, 1) if total else 0
    print(f"\n{'='*55}")
    print(f"  Rescue scan complete")
    print(f"{'='*55}")
    print(f"  Checked          : {total}")
    print(f"  RESCUED          : {rescued}  ({pct}%) -> confidence/NMS casualty, NOT a model gap")
    print(f"  STILL_MISSING    : {still_missing}  -> genuine model blind spot, real evidence "
          f"for the dense-cluster training-data hypothesis")
    print(f"  Output           : {out_path}")
    if args.keep_missing_crops:
        print(f"  Missing crops    : {missing_crops_dir}  (open these to check whether each "
              f"blind spot has a real drawn symbol or no symbol at all)")
    print(f"{'='*55}")
    if rescued > 0:
        print(f"\nNEXT STEP: {rescued} tags recovered purely by lowering conf/loosening NMS. "
              f"Try wiring conf={args.conf}/iou={args.iou} (or close to it) into your production "
              f"inference for the Instrument_Field class specifically, then re-run the full "
              f"pipeline and see how many of the original 7 detections become more.")
    if still_missing > 0:
        print(f"\n{still_missing} tags got NOTHING even at conf={args.conf} — those are real "
              f"candidates for the dense-cluster data-augmentation fix (copy-paste synthetic "
              f"clustering), not a quick inference-param tweak.")

if __name__ == "__main__":
    main()
