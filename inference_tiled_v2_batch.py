"""
PipeSight AI - inference_tiled_v2_batch.py

Same dual-scale (small-tile + large-tile) inference strategy as
inference_tiled_v2.py, extended to WALK the real Haifa P&ID data, which is
organized as one subfolder per PDF set:

    data\\haifa_real_pids\\images\\SET 1\\SET 1_page_1.png
    data\\haifa_real_pids\\images\\SET 1\\SET 1_page_2.png
    ...
    data\\haifa_real_pids\\images\\SET 10\\SET 10_page_3.png

For each set, results are written to their own output subfolder and their own
detections JSON, and a combined summary JSON/log is written across all sets.

Note: real pages are 9934x7017px (vs 5088x3312 for NexEra test pages), so tile
counts roughly double per page vs. what you saw during NexEra validation -
expect longer per-page runtime.

Usage:
    python inference_tiled_v2_batch.py

Setup:
    This script needs a class-name list matching best_merged.pt's 61 classes
    (see DATASET_YAML below / REPO_SETUP.md). It's intentionally NOT bundled
    in this repo — it's config, not code, and the repo doesn't ship datasets.
    Point DATASET_YAML at your own copy, or set the PIPESIGHT_DATASET_YAML
    environment variable to override it without editing this file.
"""

import cv2
import json
import os
import re
import sys
import time
import yaml
import numpy as np
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH       = r"models\best_merged.pt"

# Root folder containing one subfolder per PDF set (SET 1, SET 2, ... SET 10),
# as produced by batch_pdf_to_image.py
IMAGE_ROOT       = r"data\haifa_real_pids\images"

# Root for outputs - mirrors the same per-set subfolder structure as IMAGE_ROOT
OUTPUT_ROOT      = r"data\haifa_real_pids\results_tiled_v2"

# Combined summary across all sets (per-set JSONs are also written individually
# inside each set's own output subfolder)
COMBINED_JSON    = r"data\haifa_real_pids\results_tiled_v2\detections_all_sets.json"
SUMMARY_LOG      = r"data\haifa_real_pids\results_tiled_v2\run_summary.txt"

# Per-page/per-set/grand-total wall-clock timing (small-tile pass, large-tile
# pass, NMS, and total), written separately from detections so nothing about
# the existing detection JSON structure changes
TIMING_JSON      = r"data\haifa_real_pids\results_tiled_v2\timing_all_sets.json"

# ── Class-name source (NOT included in this repo — see "Setup" above) ─────────
# Pulling class names from dataset.yaml instead of symbol_names.json, since
# symbol_names.json may be stale (from the old 32-class model) and dataset.yaml
# is guaranteed to match best_merged.pt's actual 61-class order.
#
# This repo intentionally does not bundle dataset.yaml (it's produced by the
# dataset-building pipeline, which is out of scope here). To run this script:
#   1. Point DATASET_YAML below at your own copy, OR
#   2. Set an environment variable instead of editing this file:
#        set PIPESIGHT_DATASET_YAML=C:\path\to\your\dataset.yaml   (cmd)
#        $env:PIPESIGHT_DATASET_YAML="C:\path\to\your\dataset.yaml" (PowerShell)
DATASET_YAML = os.environ.get("PIPESIGHT_DATASET_YAML", r"data\unified_dataset\dataset.yaml")

# Small-tile pass — unchanged from original, tuned for symbols/instruments/valves
SMALL_TILE_SIZE  = 640
SMALL_OVERLAP    = 0.20

# Large-tile pass — NEW, tuned to give equipment enough context to be recognized
# 1536 chosen as a middle ground: big enough for equipment context, still small
# enough that inference stays reasonably fast per tile
LARGE_TILE_SIZE  = 1536
LARGE_OVERLAP    = 0.20

CONF_THRESHOLD   = 0.3
IOU_THRESHOLD    = 0.45          # NMS IoU threshold (applied across BOTH passes combined)

# Mask out non-diagram regions (notes column on right, title block at bottom)
MASK_RIGHT  = 0.82
MASK_BOTTOM = 0.92
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Load class names from dataset.yaml — fails with a clear, actionable message
# instead of a raw FileNotFoundError traceback, since this file is expected
# to be missing on a fresh clone of this repo (see "Setup" note above).
if not os.path.exists(DATASET_YAML):
    sys.exit(
        f"\n[SETUP REQUIRED] Class-name file not found: {DATASET_YAML}\n\n"
        f"This script needs a dataset.yaml listing best_merged.pt's 61 class\n"
        f"names in order. It isn't bundled in this repo (config, not code).\n\n"
        f"Fix by either:\n"
        f"  1. Editing DATASET_YAML near the top of this script to point at\n"
        f"     your own dataset.yaml, or\n"
        f"  2. Setting the PIPESIGHT_DATASET_YAML environment variable, e.g.\n"
        f"       set PIPESIGHT_DATASET_YAML=C:\\path\\to\\dataset.yaml\n"
    )

with open(DATASET_YAML, "r") as f:
    _cfg = yaml.safe_load(f)
_names = _cfg["names"]
if isinstance(_names, dict):
    SYMBOL_MAP = {str(k): v for k, v in _names.items()}
else:
    SYMBOL_MAP = {str(i): name for i, name in enumerate(_names)}

def get_symbol_name(class_id: int) -> str:
    return SYMBOL_MAP.get(str(class_id), f"symbol_{class_id}")

# Load model (same model for both passes)
model = YOLO(MODEL_PATH)


def run_single_pass(img, max_x, max_y, tile_size, overlap, conf, pass_name):
    """Run one tiled pass at a given tile_size over the (masked) image region.
    Returns (detections, elapsed_seconds, tile_count)."""
    t_start = time.perf_counter()
    step = int(tile_size * (1 - overlap))
    detections = []
    tile_count = 0

    for y in range(0, max_y, step):
        for x in range(0, max_x, step):
            x2 = min(x + tile_size, max_x)
            y2 = min(y + tile_size, max_y)
            tile = img[y:y2, x:x2]
            if tile.size == 0:
                continue
            tile_count += 1

            results = model(tile, conf=conf, imgsz=960, verbose=False)
            for r in results:
                for box in r.boxes:
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    detections.append({
                        "x1": bx1 + x,
                        "y1": by1 + y,
                        "x2": bx2 + x,
                        "y2": by2 + y,
                        "conf": round(confidence, 4),
                        "class_id": cls_id,
                        "class_name": get_symbol_name(cls_id),
                        "source_pass": pass_name,
                    })
    elapsed = time.perf_counter() - t_start
    return detections, elapsed, tile_count


def run_dual_scale_inference(image_path: str, conf: float = CONF_THRESHOLD):
    """
    Run TWO tiled passes (small tile for symbols, large tile for equipment)
    over a full P&ID image, then merge + NMS across both.
    Returns:
        combined detections list,
        n_small, n_large (raw pre-NMS counts),
        timing dict: {small_pass_sec, large_pass_sec, nms_sec, total_sec,
                      small_tile_count, large_tile_count}
    """
    t_page_start = time.perf_counter()
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [WARN] Could not read {image_path}")
        return [], 0, 0, {"small_pass_sec": 0, "large_pass_sec": 0, "nms_sec": 0,
                           "total_sec": 0, "small_tile_count": 0, "large_tile_count": 0}

    h, w = img.shape[:2]
    max_x = int(w * MASK_RIGHT)
    max_y = int(h * MASK_BOTTOM)

    small_dets, small_pass_sec, small_tile_count = run_single_pass(
        img, max_x, max_y, SMALL_TILE_SIZE, SMALL_OVERLAP, conf, "small")
    large_dets, large_pass_sec, large_tile_count = run_single_pass(
        img, max_x, max_y, LARGE_TILE_SIZE, LARGE_OVERLAP, conf, "large")

    t_nms_start = time.perf_counter()
    combined = small_dets + large_dets
    combined = nms(combined, iou_threshold=IOU_THRESHOLD)
    nms_sec = time.perf_counter() - t_nms_start

    total_sec = time.perf_counter() - t_page_start

    timing = {
        "small_pass_sec": round(small_pass_sec, 3),
        "large_pass_sec": round(large_pass_sec, 3),
        "nms_sec": round(nms_sec, 3),
        "total_sec": round(total_sec, 3),
        "small_tile_count": small_tile_count,
        "large_tile_count": large_tile_count,
    }
    return combined, len(small_dets), len(large_dets), timing


def iou(a, b):
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    return inter / (area_a + area_b - inter)


def nms(detections, iou_threshold=0.45):
    """Same NMS as original - works fine across mixed-source detections since
    it only compares box geometry + confidence, not which pass found them."""
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d["conf"], reverse=True)
    kept = []
    suppressed = set()
    for i, det in enumerate(detections):
        if i in suppressed:
            continue
        kept.append(det)
        for j in range(i + 1, len(detections)):
            if j not in suppressed and iou(det, detections[j]) > iou_threshold:
                suppressed.add(j)
    return kept


def draw_detections(image_path: str, detections: list, out_path: str):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (int(w * MASK_RIGHT), 0), (w, h), (0, 0, 80), -1)
    cv2.rectangle(overlay, (0, int(h * MASK_BOTTOM)), (w, h), (0, 0, 80), -1)
    cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        # color-code by source pass: green = small tile, orange = large tile
        color = (0, 255, 0) if det.get("source_pass") == "small" else (0, 140, 255)
        label = f'{det["class_name"]} {det["conf"]:.2f}'
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.imwrite(out_path, img)


def _natural_key(s):
    """Sort key so 'SET 2' sorts before 'SET 10' (plain string sort would not)."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    set_dirs = sorted(
        [d for d in os.listdir(IMAGE_ROOT) if os.path.isdir(os.path.join(IMAGE_ROOT, d))],
        key=_natural_key,
    )

    if not set_dirs:
        raise SystemExit(f"No subfolders found under {IMAGE_ROOT} - did the PDF->image conversion run?")

    combined_results = {}   # {"SET 1": {page_name: [detections]}, ...}
    combined_timing  = {}   # {"SET 1": {page_name: {timing dict}}, ...}
    set_summaries = []      # per-set totals for the run_summary.txt log

    grand_total_detections = 0
    grand_total_small_raw = 0
    grand_total_large_raw = 0
    grand_total_pages = 0
    grand_small_pass_sec = 0.0
    grand_large_pass_sec = 0.0
    grand_nms_sec = 0.0
    grand_total_sec = 0.0

    run_start = time.perf_counter()

    for set_name in set_dirs:
        set_image_dir = os.path.join(IMAGE_ROOT, set_name)
        set_output_dir = os.path.join(OUTPUT_ROOT, set_name)
        os.makedirs(set_output_dir, exist_ok=True)

        image_files = sorted([
            f for f in os.listdir(set_image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        print(f"\n=== {set_name} ({len(image_files)} page(s)) ===")

        set_results = {}
        set_timing = {}
        set_total_detections = 0
        set_total_small_raw = 0
        set_total_large_raw = 0
        set_total_sec = 0.0

        for fname in image_files:
            image_path = os.path.join(set_image_dir, fname)
            page_name  = os.path.splitext(fname)[0]
            out_path   = os.path.join(set_output_dir, f"{page_name}_detected.jpg")

            print(f"  Processing {fname} ...", end=" ", flush=True)
            detections, n_small, n_large, timing = run_dual_scale_inference(image_path)
            draw_detections(image_path, detections, out_path)

            set_results[page_name] = detections
            set_timing[page_name] = timing
            set_total_detections += len(detections)
            set_total_small_raw += n_small
            set_total_large_raw += n_large
            set_total_sec += timing["total_sec"]

            grand_small_pass_sec += timing["small_pass_sec"]
            grand_large_pass_sec += timing["large_pass_sec"]
            grand_nms_sec += timing["nms_sec"]
            grand_total_sec += timing["total_sec"]
            grand_total_pages += 1

            print(f"{len(detections)} final (raw: {n_small} small-tile + {n_large} large-tile before NMS) "
                  f"| {timing['total_sec']:.2f}s (small {timing['small_pass_sec']:.2f}s + "
                  f"large {timing['large_pass_sec']:.2f}s + nms {timing['nms_sec']:.2f}s)")

        set_json_path = os.path.join(set_output_dir, f"detections_{set_name.replace(' ', '_')}.json")
        with open(set_json_path, "w") as f:
            json.dump(set_results, f, indent=2)

        set_timing_path = os.path.join(set_output_dir, f"timing_{set_name.replace(' ', '_')}.json")
        with open(set_timing_path, "w") as f:
            json.dump(set_timing, f, indent=2)

        combined_results[set_name] = set_results
        combined_timing[set_name] = set_timing

        avg_sec = set_total_sec / len(image_files) if image_files else 0
        set_summaries.append(
            f"{set_name}: {set_total_detections} final detections "
            f"(raw: {set_total_small_raw} small-tile + {set_total_large_raw} large-tile), "
            f"{len(image_files)} page(s), {set_total_sec:.1f}s total ({avg_sec:.1f}s/page avg)"
        )

        grand_total_detections += set_total_detections
        grand_total_small_raw += set_total_small_raw
        grand_total_large_raw += set_total_large_raw

    run_elapsed = time.perf_counter() - run_start

    with open(COMBINED_JSON, "w") as f:
        json.dump(combined_results, f, indent=2)

    with open(TIMING_JSON, "w") as f:
        json.dump(combined_timing, f, indent=2)

    avg_page_sec = grand_total_sec / grand_total_pages if grand_total_pages else 0
    summary_lines = [
        "PipeSight AI - dual-scale tiled inference (best_merged.pt) on real Haifa P&ID data",
        f"Sets processed: {len(set_dirs)} | Pages processed: {grand_total_pages}",
        "",
        *set_summaries,
        "",
        f"GRAND TOTAL final detections (after cross-pass NMS): {grand_total_detections}",
        f"GRAND TOTAL raw small-tile detections (pre-NMS): {grand_total_small_raw}",
        f"GRAND TOTAL raw large-tile detections (pre-NMS): {grand_total_large_raw}",
        "",
        f"TIMING - small-tile pass total: {grand_small_pass_sec:.1f}s",
        f"TIMING - large-tile pass total: {grand_large_pass_sec:.1f}s",
        f"TIMING - NMS total: {grand_nms_sec:.1f}s",
        f"TIMING - summed per-page total: {grand_total_sec:.1f}s | avg/page: {avg_page_sec:.2f}s",
        f"TIMING - wall-clock for entire run (includes I/O, drawing, JSON writes): {run_elapsed:.1f}s",
    ]
    with open(SUMMARY_LOG, "w") as f:
        f.write("\n".join(summary_lines))

    print("\n" + "\n".join(summary_lines))
    print(f"\nPer-set results + JSON saved under: {OUTPUT_ROOT}\\<SET name>\\")
    print(f"Combined detections JSON saved to:  {COMBINED_JSON}")
    print(f"Combined timing JSON saved to:      {TIMING_JSON}")
    print(f"Summary log saved to:               {SUMMARY_LOG}")
    print("\nGreen boxes = found by small-tile (640) pass, Orange = found by large-tile (1536) pass")
