# tag_extractor_v2.py - PipeSight AI
# V2 redesign: full-page OCR first, then associate text blocks to symbol bboxes.
# Faster (1 OCR call per page vs N per symbol), more accurate (full context),
# and handles text outside small crop windows that v1 missed.
#
# Key differences from v1:
#   - PaddleOCR runs ONCE on full page image per page
#   - Text blocks associated to nearest instrument symbol by centroid distance
#   - Falls back to proximity search with configurable radius (default 200px)
#   - Writes results to DB via db_builder.py AND CSV for backward compatibility
#   - --model flag for swapping model weights. HARDCODED DEFAULT as of this
#     version: models\best_merged.pt, re-run via inference_tiled_v2_batch.py
#     (4-zone + resize tiling) — NOT the older single-pass inference_tiled.py.
#
# Usage (Haifa is the only supported dataset now — defaults point at it):
#   python tag_extractor_v2.py --page "SET 1_page_1"
#       (re-runs inference via inference_tiled_v2_batch.py --model best_merged.pt,
#        then extracts — this now happens by default, every run)
#   python tag_extractor_v2.py --page "SET 1_page_1" --no-model-rerun
#       (skip re-running inference, just load existing detections_all_sets.json)
#   python tag_extractor_v2.py --page "SET 1_page_1" --model models\best_61class.pt
#       (override to a different model / different inference run)
#   python tag_extractor_v2.py --all
#   python tag_extractor_v2.py --all --no-db   (CSV only, skip DB write)

import json
import csv
import re
import math
import argparse
import sqlite3
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from PIL import Image

# ─── PATHS ───────────────────────────────────────────────────────────────────
# Haifa real-data run is the only dataset in active use — this is now the
# hard default. Both are still overridable via --detections-json /
# --images-dir if you ever need to point at something else.
DETECTIONS_JSON = Path("data") / "haifa_real_pids" / "results_tiled_v2" / "detections_all_sets.json"
IMAGES_DIR      = Path("data") / "haifa_real_pids" / "images"
OUT_DIR         = Path("data") / "tag_extraction_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL   = Path(r"C:\Users\devas\Desktop\PipeSight AI\models\best_merged.pt")
MODEL_61CLASS   = Path("models") / "best_61class.pt"

# Inference script used to (re)generate detections when --model is passed.
# Hardcoded to the v2 batch/tiled script (4-zone + resize), NOT the older
# single-pass inference_tiled.py, which was silently under-detecting.
INFERENCE_SCRIPT = r"C:\Users\devas\Desktop\PipeSight AI\inference_tiled_v2_batch.py"

# Every Haifa page shares the same drawing template, including a left-margin
# border grid (row-letter tick marks, A-K) that YOLO occasionally misfires on
# as Flange_or_Nozzle at low confidence — confirmed via diagnose_flange_detail.py
# on SET 1_page_1: 62 near-identical 5-6px-wide slivers stacked at x1=0 down
# the full page height, spaced at ~62px multiples (the tick-mark row height),
# nowhere near real symbol width. This zone is applied to EVERY page
# automatically (not just SET 1) since the border template is shared across
# the whole Haifa dataset. Widened slightly (0-25px vs the 0-20px used in the
# one-off diagnosis run) to comfortably clear the strip on every page without
# needing per-page tuning. Extend via --exclude-zone for anything page-
# specific (title block, a legend table that isn't in the same spot on every
# page, etc.) — those stack on top of this, they don't replace it. Disable
# entirely with --no-default-exclude-zones if a page ever needs the raw
# margin content for some reason.
DEFAULT_EXCLUDE_ZONES = [
    (0, 0, 25, 10000),  # left-margin border/tick-mark strip, full page height
]

# OCR result cache — one JSON per page, written once, read forever after.
# Lets you rerun association/normalization logic (radius tuning, tag regex
# fixes, class routing changes) instantly without re-paying for OCR.
OCR_CACHE_DIR = Path("data") / "ocr_cache"
OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ─── MATCHING ENGINE (imported) ──────────────────────────────────────────────
# Spatial matching, ISA tag normalization, confidence scoring, and class
# routing all live in matching_engine.py now — see that module's docstring
# for the four signals it combines (distance, OCR confidence, ISA validity,
# class rules). This file owns orchestration only: OCR execution, page
# walking, CSV/DB writes, CLI.
from matching_engine import (
    INSTRUMENT_CLASSES_32, INSTRUMENT_CLASSES, REFERENCE_CLASSES,
    MECHANICAL_CLASSES, STRUCTURAL_CLASSES,
    load_canonical_class_names, canonical_class_name, get_routing,
    OCR_CORRECTIONS, ISA_PATTERNS, NON_INSTRUMENT_ABBREVIATIONS,
    apply_ocr_corrections, is_loop_prefix_only,
    load_excluded_tags, is_excluded_tag,
    MAX_ORPHAN_DIST_DEFAULT, PIPE_SPEC_PATTERN, is_pipe_spec_text,
    reference_noise_reason,
    normalize_tag, normalize_tag_strict,
    centroid, distance, box_area, associate_text_to_symbols,
    pick_best_text, nearest_symbol,
    combined_confidence, AUTO_ACCEPT_THRESHOLD,
    find_review_queue_rows,
    ISA_FIRST_LETTER, ISA_SUCCEEDING_LETTER, infer_instrument_function,
)

# ─── OCR LOADER ──────────────────────────────────────────────────────────────
# Run `nvidia-smi` / check `paddle.device.is_compiled_with_cuda()` once to confirm
# GPU is actually visible before assuming this helps. If PaddleOCR was never
# passed a device before, it may well have been silently running on CPU —
# that alone, times 4-6 tiles/page x 30 pages, plausibly explains most of the
# runtime. Override with --gpu-id -1 to force CPU if no GPU is available.
OCR_DEVICE = "gpu:0"
_ocr_engine = None

# CONFIRMED (checked against PaddleOCR's own documented parameter table):
# text_det_limit_side_len / text_det_limit_type ARE real, settable kwargs on
# both PaddleOCR() and .predict() — but max_side_limit (the thing actually
# printing "exceeds max_side_limit of 4000" and downscaling the page) is NOT
# a kwarg anywhere in the documented API. It only exists inside PaddleX's
# per-model YAML config (SubModules.TextDetection.max_side_limit). Passing
# text_det_limit_side_len=5120 was therefore never going to stop the
# downscale — the two are separate caps, and only the YAML-config one was
# ever actually clamping the image. This is bug #4 from the handoff, root-
# caused: it explains the 0/15 VALID run (page silently shrunk ~2.5x before
# OCR, blurring small ISA tags below what recognition could read).
#
# Fix: export the pipeline's own PaddleX config once, raise max_side_limit
# in it, cache the patched YAML, and load PaddleOCR from that config on
# every subsequent run. One-time cost (~one extra pipeline load), not paid
# per page.
PADDLEX_CONFIG_CACHE   = Path("data") / "_paddleocr_highres_config.yaml"
PADDLEX_MAX_SIDE_LIMIT = 20000  # comfortably above any Haifa page seen so far (9934px)

def _build_high_res_paddlex_config():
    """
    Returns the path to a cached PaddleX config with max_side_limit raised,
    or None if it couldn't be built (caller falls back to the default
    pipeline, which WILL downscale pages over 4000px on their longest side).
    """
    if PADDLEX_CONFIG_CACHE.exists():
        return PADDLEX_CONFIG_CACHE
    try:
        import yaml
        from paddleocr import PaddleOCR as _PaddleOCR
        print("  Building high-res PaddleX config (one-time)...")
        tmp = _PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                          use_textline_orientation=False, device=OCR_DEVICE)
        tmp.export_paddlex_config_to_yaml(str(PADDLEX_CONFIG_CACHE))
        del tmp

        with open(PADDLEX_CONFIG_CACHE, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        det_cfg = (cfg.get("SubModules") or {}).get("TextDetection")
        if det_cfg is None:
            print("  WARNING: SubModules.TextDetection not found in exported "
                  "PaddleX config — max_side_limit NOT patched. Pages over "
                  "4000px on their longest side will still be downscaled.")
            PADDLEX_CONFIG_CACHE.unlink(missing_ok=True)
            return None
        det_cfg["max_side_limit"] = PADDLEX_MAX_SIDE_LIMIT
        with open(PADDLEX_CONFIG_CACHE, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"  Patched max_side_limit -> {PADDLEX_MAX_SIDE_LIMIT}, cached at "
              f"{PADDLEX_CONFIG_CACHE}")
        return PADDLEX_CONFIG_CACHE
    except Exception as e:
        print(f"  WARNING: failed to build high-res PaddleX config ({e}) — "
              f"falling back to default pipeline (4000px downscale cap active, "
              f"tag OCR quality on large pages will suffer).")
        return None

def get_ocr():
    global _ocr_engine
    if _ocr_engine is None:
        print("Loading PaddleOCR (PP-OCRv6 medium)...")
        from paddleocr import PaddleOCR
        config_path = _build_high_res_paddlex_config()
        kwargs = dict(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=OCR_DEVICE,
        )
        if config_path:
            kwargs["paddlex_config"] = str(config_path)
        _ocr_engine = PaddleOCR(**kwargs)
        print("PaddleOCR loaded.")
    return _ocr_engine

def _ocr_cache_path(page_name):
    return OCR_CACHE_DIR / f"{page_name}.json"

RESIZE_OCR_TMP_DIR = Path("data") / "_ocr_resize_tmp"

def _run_ocr_resized(img, img_w, img_h, resize_target, page_name=None):
    """
    Shrink the page so its longest side is resize_target px (only if it's
    currently bigger — never upscales), OCR that single resized image in
    one pass, then scale every returned box back up to ORIGINAL resolution
    so downstream code never has to know a resize happened.

    This is the actual "resize to 4000-5000px, compare against tiling"
    experiment: unlike raising max_side_limit (already patched to 20000
    and NOT what caused the crash), this genuinely reduces the pixel count
    PaddleOCR has to hold in memory and process, at the cost of shrinking
    small instrument-tag text along with everything else. Whether that
    cost is acceptable is exactly what the sample-page comparison should
    tell you — don't assume, check the actual VALID/RAW/MISSING counts
    against the tiling baseline before switching production default.
    """
    RESIZE_OCR_TMP_DIR.mkdir(parents=True, exist_ok=True)
    long_side = max(img_w, img_h)
    if long_side <= resize_target:
        print(f"  Page {img_w}x{img_h}px already <= --resize-ocr target "
              f"{resize_target}px — running single pass at native size, no "
              f"actual resize needed.")
        if getattr(img, "filename", None):
            return _run_ocr_on_file(str(img.filename))
        return _run_ocr_on_file_from_image(img)

    scale = resize_target / long_side
    new_w, new_h = max(1, int(round(img_w * scale))), max(1, int(round(img_h * scale)))
    resized = img.convert("RGB").resize((new_w, new_h), Image.LANCZOS)
    tmp_path = RESIZE_OCR_TMP_DIR / f"_resized_{page_name or 'page'}.png"
    resized.save(tmp_path)
    print(f"  Page {img_w}x{img_h}px -> resized to {new_w}x{new_h}px "
          f"(--resize-ocr target {resize_target}px, scale={scale:.4f}) for "
          f"single-pass OCR...")

    # NOTE: deliberately NOT passing limit_side_len here. Forcing the
    # detector to run at (near) the resized image's full resolution was
    # the actual bug in the first version of this function — it defeated
    # the speed benefit of resizing entirely, since detection cost scales
    # roughly with the square of the target side length. Tiles never pass
    # limit_side_len either (see _run_ocr_on_file's default call site) and
    # that's a real part of why tiling was faster per-megapixel than the
    # first resize attempt. Let PaddleOCR use its own fast default target
    # here too, same as tiles get.
    try:
        blocks = _run_ocr_on_file(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)

    inv_scale = 1.0 / scale
    for b in blocks:
        b["x1"] = int(round(b["x1"] * inv_scale)); b["x2"] = int(round(b["x2"] * inv_scale))
        b["y1"] = int(round(b["y1"] * inv_scale)); b["y2"] = int(round(b["y2"] * inv_scale))

    print(f"  Resize-OCR: {len(blocks)} blocks found, coordinates rescaled "
          f"x{inv_scale:.4f} back to original {img_w}x{img_h}px space.")
    return blocks

def _run_ocr_on_file_from_image(img):
    """Fallback save-then-OCR for the (rare) case an opened Image has no
    backing file path (e.g. was itself produced in-memory)."""
    tmp_path = RESIZE_OCR_TMP_DIR / "_native_size_tmp.png"
    img.convert("RGB").save(tmp_path)
    try:
        return _run_ocr_on_file(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)

def run_ocr_full_page(image_path, page_name=None, force_recompute=False, force_tiling=False,
                       force_single_pass=False, resize_target=None):
    """
    Run PaddleOCR on the page image, or load from cache if already computed.
    Pages under OCR_TILE_MAX_SIDE run in a single pass, no options needed.
    Pages over that (like the 9934x7017px Haifa scans) now TILE BY DEFAULT
    — see note below on why the old single-pass-by-default behavior was a
    crash bug, not just a style choice. Pass --single-pass-ocr to opt back
    into the old behavior if you have the RAM/GPU for it.

    resize_target (--resize-ocr): a THIRD strategy, opt-in, highest priority
    of the three. Instead of tiling (many OCR calls + tile-overlap dedup) or
    single-pass-at-full-res (one OCR call, but the actual crash cause — the
    ~70MP array itself exhausted RAM, not a PaddleOCR internal downscale
    limit, since PADDLEX_MAX_SIDE_LIMIT is already patched to 20000), this
    actually shrinks the pixel data before OCR ever sees it: one OCR call,
    real memory reduction, no tile-overlap dedup needed. The tradeoff is
    legibility — small instrument tag text shrinks too, so this is a real
    accuracy/speed tradeoff to test on a sample page, not a free win.
    Returned block coordinates are rescaled back to ORIGINAL image
    resolution before returning, so every downstream consumer (symbol
    association, zone crops, review-queue) works unchanged either way.
    Returns list of dicts: {text, conf, x1, y1, x2, y2}
    """
    cache_path = _ocr_cache_path(page_name) if page_name else None
    if cache_path and cache_path.exists() and not force_recompute:
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        if isinstance(cached, dict) and cached.get("_complete"):
            import time
            age_hrs = (time.time() - cache_path.stat().st_mtime) / 3600
            print(f"  ⚠ Using CACHED OCR for {page_name}: {len(cached['blocks'])} "
                  f"blocks, written {age_hrs:.1f}h ago ({cache_path}). If the "
                  f"pipeline code (crop logic, corrections, association) has "
                  f"changed since then, this cache is STALE — pass "
                  f"--recompute-ocr to force a fresh OCR pass before trusting "
                  f"this run's numbers.")
            return cached["blocks"]
        # Partial/interrupted cache from a previous run (e.g. laptop closed
        # mid-tile) — don't trust it as final, recompute this page cleanly.
        print(f"  Found incomplete cache for {page_name}, recomputing...")

    from PIL import Image
    img = Image.open(image_path)
    img_w, img_h = img.size

    if resize_target:
        blocks = _run_ocr_resized(img, img_w, img_h, resize_target, page_name)
    elif max(img_w, img_h) <= OCR_TILE_MAX_SIDE:
        blocks = _run_ocr_on_file(str(image_path))
    elif force_single_pass and not force_tiling:
        # Explicit opt-in only. Was previously the silent default for every
        # page over OCR_TILE_MAX_SIDE — that was the actual crash cause on
        # this machine: a single PaddleOCR call at ~limit_side_len=9984 on a
        # 9934x7017px (~70-megapixel) page, CPU-only ("device not available!
        # Switching to CPU"), silently exhausts RAM and gets killed by
        # Windows with no traceback — exactly the "output stops after
        # 'PaddleOCR loaded.', back to prompt" behavior seen today. Kept as
        # an override for machines with enough RAM (or GPU) to do it in one
        # shot, since single-pass avoids tile-overlap dedup entirely.
        limit = int(math.ceil(max(img_w, img_h) / 32) * 32) + 32  # round up, PaddleOCR wants multiples of 32
        print(f"  Page {img_w}x{img_h}px — single-pass OCR with limit_side_len={limit}"
              f" (--single-pass-ocr forced; this is high-RAM/GPU territory)...")
        blocks = _run_ocr_on_file(str(image_path), limit_side_len=limit)
    else:
        # Default path for any page over OCR_TILE_MAX_SIDE. Each tile stays
        # well under PaddleOCR's max_side_limit, so nothing ever downscales
        # and nothing risks the single-pass RAM spike above.
        print(f"  Page {img_w}x{img_h}px exceeds {OCR_TILE_MAX_SIDE}px — tiling for OCR...")
        tmp_dir = Path("data") / "_ocr_tiles"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        all_blocks = []
        tile_count = 0
        for x0, y0, x1, y1 in _iter_ocr_tiles(img_w, img_h):
            tile_count += 1
            tile_path = tmp_dir / f"_tile_{x0}_{y0}.png"
            img.crop((x0, y0, x1, y1)).save(tile_path)
            try:
                tile_blocks = _run_ocr_on_file(str(tile_path))
            finally:
                tile_path.unlink(missing_ok=True)
            # Translate tile-local coordinates back to full-page coordinates
            for b in tile_blocks:
                b["x1"] += x0; b["x2"] += x0
                b["y1"] += y0; b["y2"] += y0
            all_blocks.extend(tile_blocks)
            # Flush partial cache after every tile so a crash/interrupt mid-page
            # doesn't lose tiles already OCR'd — only the current tile is at risk.
            # Marked _complete=False; only the final write below sets it True.
            if cache_path:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump({"_complete": False, "blocks": all_blocks}, f)

        blocks = _dedupe_overlap_blocks(all_blocks)
        print(f"  OCR tiling: {tile_count} tiles, {len(all_blocks)} raw blocks -> "
              f"{len(blocks)} after overlap dedup")

    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"_complete": True, "blocks": blocks}, f)
    return blocks

# Kept comfortably under PaddleOCR's 4000px max_side_limit so no internal
# downscale ever triggers on a tile.
OCR_TILE_MAX_SIDE = 3600
OCR_TILE_OVERLAP  = 250  # generous margin so no tag gets cut across a tile edge

# Default target for --resize-ocr — midpoint of the 4000-5000px range asked
# for. Override per-run with --resize-ocr 4000 / --resize-ocr 5000 etc. to
# find the sweet spot on your actual page density before committing to one.
RESIZE_OCR_TARGET_DEFAULT = 4500

def _iter_ocr_tiles(img_w, img_h, tile_size=OCR_TILE_MAX_SIDE, overlap=OCR_TILE_OVERLAP):
    step = tile_size - overlap
    xs = list(range(0, img_w, step)) if img_w > tile_size else [0]
    ys = list(range(0, img_h, step)) if img_h > tile_size else [0]
    for y0 in ys:
        for x0 in xs:
            x1 = min(x0 + tile_size, img_w)
            y1 = min(y0 + tile_size, img_h)
            yield x0, y0, x1, y1

def _iou(a, b):
    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, a["x2"] - a["x1"]) * max(0, a["y2"] - a["y1"])
    area_b = max(0, b["x2"] - b["x1"]) * max(0, b["y2"] - b["y1"])
    return inter / (area_a + area_b - inter)

def _dedupe_overlap_blocks(blocks, iou_thresh=0.3):
    """
    Tile overlap margins cause the same text to be read twice. Keep the
    highest-confidence copy of each; drop lower-confidence boxes that
    substantially overlap something already kept.
    """
    blocks_sorted = sorted(blocks, key=lambda b: -b["conf"])
    kept = []
    for b in blocks_sorted:
        if not any(_iou(b, k) > iou_thresh for k in kept):
            kept.append(b)
    return kept

def _run_ocr_on_file(path_str, limit_side_len=None):
    ocr = get_ocr()
    if limit_side_len:
        # Raises the detection-stage TARGET side length (limit_side_len/
        # limit_type) so PaddleOCR aims to run at full page resolution
        # instead of a small default (64px min-side by default). This is
        # separate from max_side_limit (the hard downscale ceiling, default
        # 4000px) — that one is patched once in get_ocr() via a custom
        # PaddleX config, since it's not exposed as a kwarg here. Both need
        # to be right: this controls what size PaddleOCR TARGETS, the config
        # patch controls whether it's ALLOWED to actually reach that size.
        try:
            result = ocr.predict(path_str, text_det_limit_side_len=limit_side_len,
                                  text_det_limit_type="max")
        except TypeError:
            print("  NOTE: this PaddleOCR install doesn't accept "
                  "text_det_limit_side_len/text_det_limit_type — falling back "
                  "to default call.")
            result = ocr.predict(path_str)
    else:
        result = ocr.predict(path_str)

    text_blocks = []
    if not result:
        return text_blocks

    # PaddleOCR v3 returns a list of result dicts per image.
    # IMPORTANT: rec_texts/rec_scores must be paired with rec_polys, NOT dt_polys.
    # dt_polys is the raw text-DETECTION-stage output (pre-recognition); rec_polys
    # is the array PaddleOCR guarantees is index-aligned with rec_texts/rec_scores
    # post-recognition. Recognition can merge/drop/reorder boxes relative to
    # detection, so zipping against dt_polys silently pairs text with the wrong
    # bounding box whenever that happens — this was breaking symbol association
    # for tags that were read correctly but placed at the wrong coordinates.
    for page_result in result:
        rec_texts  = page_result.get("rec_texts", [])
        rec_scores = page_result.get("rec_scores", [])
        rec_polys  = page_result.get("rec_polys", [])

        if len(rec_polys) != len(rec_texts):
            print(f"  WARNING: rec_polys ({len(rec_polys)}) and rec_texts "
                  f"({len(rec_texts)}) length mismatch — some text blocks "
                  f"will be dropped for this page.")

        for text, conf, poly in zip(rec_texts, rec_scores, rec_polys):
            if not text or not text.strip():
                continue
            # poly is [[x1,y1],[x2,y1],[x2,y2],[x1,y2]] or similar
            xs = [pt[0] for pt in poly]
            ys = [pt[1] for pt in poly]
            text_blocks.append({
                "text": text.strip(),
                "conf": float(conf),
                "x1": int(min(xs)),
                "y1": int(min(ys)),
                "x2": int(max(xs)),
                "y2": int(max(ys)),
            })

    return text_blocks

# ─── TARGETED ROTATED RE-CHECK ────────────────────────────────────────────────
# Implements the "targeted rotated re-check" design goal from the handoff:
# the full-page pass runs at 0° only (use_textline_orientation=False), so any
# vertically-set or rotated tag is invisible to it — not misread, never
# attempted. Blanket 4-direction OCR on the whole page was already ruled out
# as too slow (the old v1 approach). Instead: only crop-and-rotate the small
# region around a symbol that got ZERO associated text from the full-page
# pass. Keeps single-pass speed for the common case (most symbols get text
# on the first pass); only pays the extra 3x-rotation OCR cost on the
# minority that came up empty.
ROTATION_CROP_TMP_DIR = Path("data") / "_ocr_rotation_tmp"

def targeted_rotated_recheck(image, det, radius, page_name):
    """
    For a single instrument symbol that got no associated text from the
    full-page OCR pass, crop a margin-padded region around it and try OCR
    at 90/180/270 degrees (0 was already covered by the full-page pass).
    Returns the best VALID/RAW candidate found, or None.

    NOTE on coordinates: the recovered text's bbox is reported as the
    symbol's own bbox (not reprojected through the rotation) — precise
    pixel geometry on a rotated crop isn't needed for this to be useful;
    what matters is "this tag belongs near this symbol," which the crop
    itself already guarantees. Treat recovered rows as needing human
    verification regardless of confidence (see RECOVERED_ROTATED handling
    in process_page) — don't rely on their bbox for further spatial logic.
    """
    ROTATION_CROP_TMP_DIR.mkdir(parents=True, exist_ok=True)
    margin = radius
    x1 = max(0, det["x1"] - margin)
    y1 = max(0, det["y1"] - margin)
    x2 = min(image.width,  det["x2"] + margin)
    y2 = min(image.height, det["y2"] + margin)
    crop = image.crop((x1, y1, x2, y2))

    best = None
    for angle in (90, 180, 270):
        rotated = crop.rotate(angle, expand=True)
        tile_path = ROTATION_CROP_TMP_DIR / f"_rot_{page_name}_{det['_det_id']}_{angle}.png"
        rotated.save(tile_path)
        try:
            blocks = _run_ocr_on_file(str(tile_path))
        except Exception as e:
            print(f"    Rotated re-check OCR failed (det {det['_det_id']}, {angle}°): {e}")
            blocks = []
        finally:
            tile_path.unlink(missing_ok=True)

        for blk in blocks:
            if not blk.get("text", "").strip():
                continue
            tag, status = normalize_tag(blk["text"])
            if status != "VALID":
                continue
            if best is None or blk["conf"] > best["conf"]:
                best = {
                    "text": blk["text"], "conf": blk["conf"],
                    "_angle": angle, "_tag": tag, "_status": status,
                }
    return best

# ─── ZONE-DIRECTIONAL OCR ──────────────────────────────────────────────────
# Implements both supervisor suggestions together, since they're the same
# idea: only OCR near an instrument symbol (skip pipe bends/reducers/valves
# entirely — they're never OCR'd at all in zone mode), and within that,
# check the zone tag text most commonly sits in first. On this drawing
# standard (and P&IDs generally) the tag bubble's text sits above or to the
# right of the symbol far more often than left/below, so those are tried
# first; left/below are only checked if both come up empty. This means the
# common case (~most symbols) costs exactly ONE small OCR call, not four.
#
# CROP SIZING — tightened after real-data testing on SET 1_page_1 showed the
# original radius*1.5 boxes (up to ~670px) were large enough to sweep in a
# NEIGHBORING instrument's own tag on this densely-packed drawing (det3's
# "right" zone grabbed FIT 1003's text instead of its own PIT 1003 — both
# real instruments sit within ~300px of each other in that cluster). Zone
# depth (how far to look in the search direction) and margin (how far to
# pad perpendicular to it) are now independent, small, fixed values instead
# of scaled off search_radius (which was tuned for full-page's different
# "nearest text block" distance metric, not crop sizing).
ZONE_CROP_TMP_DIR   = Path("data") / "_ocr_zone_tmp"
ZONE_PRIORITY_ORDER = ("above", "right", "left", "below")
ZONE_DEPTH  = 150  # how far to look in the search direction
ZONE_MARGIN = 40   # padding perpendicular to the search direction

# How far PAST zone_depth to retry, PER-SYMBOL, when a symbol's zones come
# back with zone_used=None at the default depth. Applied only to that one
# symbol's crop, never globally — blanket-widening zone_depth for every
# symbol is what caused det124's false positive (Open Issue #1). See
# process_page()'s zone-ocr branch for the escalation logic.
ZONE_DEPTH_ESCALATE = 120

def _zone_crop_box(det, zone, img_w, img_h, depth=ZONE_DEPTH, margin=ZONE_MARGIN):
    """
    Crop box for one directional zone around a symbol's bbox, clipped to
    image bounds. Returns None if the resulting box is degenerate (e.g.
    symbol sits flush against the page edge).
    """
    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
    if zone == "above":
        bx1, by1, bx2, by2 = x1 - margin, y1 - depth, x2 + margin, y1
    elif zone == "right":
        bx1, by1, bx2, by2 = x2, y1 - margin, x2 + depth, y2 + margin
    elif zone == "left":
        bx1, by1, bx2, by2 = x1 - depth, y1 - margin, x1, y2 + margin
    elif zone == "below":
        bx1, by1, bx2, by2 = x1 - margin, y2, x2 + margin, y2 + depth
    else:
        return None
    bx1 = max(0, int(bx1)); by1 = max(0, int(by1))
    bx2 = min(img_w, int(bx2)); by2 = min(img_h, int(by2))
    if bx2 <= bx1 or by2 <= by1:
        return None
    return (bx1, by1, bx2, by2)

def extract_tag_zoned(image, det, page_name, zones=ZONE_PRIORITY_ORDER,
                       depth=ZONE_DEPTH, margin=ZONE_MARGIN):
    """
    Per-symbol targeted OCR: try zones in priority order (above, right,
    left, below — see ZONE_PRIORITY_ORDER), stop at the first zone that
    yields a VALID tag. Returns (raw_ocr, ocr_conf, zone_used, text_bbox) —
    zone_used is None if nothing matched in any zone (raw_ocr will hold
    the best RAW attempt found, if any, for human review). text_bbox is
    the picked text's own bbox in PAGE coordinates (converted from the
    zone crop's local coordinates using that crop's page-space origin) —
    this is where the text actually sits, separate from the symbol's own
    bbox, so the reviewer UI can highlight the text itself rather than
    just the symbol it got matched to.

    Within a zone's crop, blocks are ranked by distance from the symbol's
    own center (not raster/reading order) and only the closest 2 are
    combined — same proximity discipline as pick_best_text() uses for
    full-page mode. Using ALL blocks in reading order was the bug that
    caused zone mode to grab a neighboring instrument's text on a densely
    packed page (see module comment above): reading order has no idea
    which block is actually this symbol's, proximity does.

    Uses normalize_tag() (the full/permissive matcher, not the strict
    review-queue one) — a crop taken directly from a confirmed instrument
    symbol's neighborhood has real spatial justification for trusting a
    single-letter tag that a whole-page scan doesn't have.
    """
    ZONE_CROP_TMP_DIR.mkdir(parents=True, exist_ok=True)
    sym_cx, sym_cy = centroid(det["x1"], det["y1"], det["x2"], det["y2"])
    best_raw, best_conf, best_bbox = None, 0.0, None

    for zone in zones:
        box = _zone_crop_box(det, zone, image.width, image.height, depth=depth, margin=margin)
        if box is None:
            continue
        crop = image.crop(box)
        tile_path = ZONE_CROP_TMP_DIR / f"_zone_{page_name}_{det['_det_id']}_{zone}.png"
        crop.save(tile_path)
        try:
            blocks = _run_ocr_on_file(str(tile_path))
        except Exception as e:
            print(f"    Zone OCR failed (det {det['_det_id']}, {zone}): {e}")
            blocks = []
        finally:
            tile_path.unlink(missing_ok=True)

        blocks = [b for b in blocks if b.get("text", "").strip() and len(b["text"].strip()) > 1]
        if not blocks:
            continue

        # Convert each block's crop-local centroid back to page coordinates
        # so distance-to-symbol is measured in the same space as the
        # symbol's own center, then rank by that distance — not reading
        # order — and keep only the closest 2.
        for b in blocks:
            bcx, bcy = centroid(b["x1"], b["y1"], b["x2"], b["y2"])
            b["_dist_to_symbol"] = distance(box[0] + bcx, box[1] + bcy, sym_cx, sym_cy)
        blocks.sort(key=lambda b: b["_dist_to_symbol"])
        top = blocks[:2]

        combined = " ".join(b["text"].strip() for b in top)
        avg_conf = sum(b["conf"] for b in top) / len(top)
        # top blocks are still in crop-local coords — shift by the zone
        # crop's own top-left (box[0], box[1]) to get page-space, then
        # take the union bbox of whichever blocks were actually combined.
        xs = [box[0] + b["x1"] for b in top] + [box[0] + b["x2"] for b in top]
        ys = [box[1] + b["y1"] for b in top] + [box[1] + b["y2"] for b in top]
        text_bbox = {"x1": min(xs), "y1": min(ys), "x2": max(xs), "y2": max(ys)}
        tag, status = normalize_tag(combined)
        if status == "VALID":
            return combined, avg_conf, zone, text_bbox
        if best_raw is None:
            best_raw, best_conf, best_bbox = combined, avg_conf, text_bbox

    return best_raw, best_conf, None, best_bbox

# ─── MAIN PROCESSING ─────────────────────────────────────────────────────────

def resolve_image_path(page_name):
    """
    Look for {page_name}.png directly under IMAGES_DIR first (original flat
    layout). If not found, search one level of subfolders (Haifa layout:
    IMAGES_DIR / "SET 1" / "SET 1_page_1.png", etc.).
    """
    direct = IMAGES_DIR / f"{page_name}.png"
    if direct.exists():
        return direct
    matches = list(IMAGES_DIR.rglob(f"{page_name}.png"))
    return matches[0] if matches else direct

def _run_zone_ocr_pass(page_image, instrument_syms, page_name,
                        zone_depth=ZONE_DEPTH, zone_margin=ZONE_MARGIN,
                        zone_rotated_recheck=True):
    """
    Runs the per-symbol zone-OCR pass (above/right/left/below, with
    per-symbol depth escalation and targeted rotated re-check) over every
    instrument symbol given. This is the primary "which tag belongs to
    which symbol" method — it never touches full-page OCR at all, so its
    cost scales with instrument count, not page pixel count.

    Returns (rows_by_det_id, valid_count, raw_count, missing_count,
    recovered_count) — rows_by_det_id maps det_id -> the row dict for that
    symbol, letting a caller (process_page's zone-ocr branch, or the
    auto-escalation flow) look up/override individual symbols' results
    without re-running the whole pass.
    """
    rows_by_det_id = {}
    valid_count = raw_count = missing_count = recovered_count = 0
    for det in instrument_syms:
        raw_ocr, ocr_conf, zone_used, text_bbox = extract_tag_zoned(
            page_image, det, page_name, depth=zone_depth, margin=zone_margin
        )

        # Per-symbol adaptive depth escalation (Open Issue #1, step 2).
        # Only retries THIS symbol's crop, at a wider depth, and only
        # when the default depth found nothing VALID — every other
        # symbol's crop is untouched, so this can't reintroduce the
        # cross-symbol contamination that a blanket --zone-depth widen
        # caused (det124). Never runs when zone_used is already set.
        escalated = False
        if zone_used is None:
            wide_depth = zone_depth + ZONE_DEPTH_ESCALATE
            raw_ocr2, ocr_conf2, zone_used2, text_bbox2 = extract_tag_zoned(
                page_image, det, page_name, depth=wide_depth, margin=zone_margin
            )
            if zone_used2 is not None:
                raw_ocr, ocr_conf, zone_used, text_bbox = raw_ocr2, ocr_conf2, zone_used2, text_bbox2
                escalated = True
            elif raw_ocr2 and not raw_ocr:
                raw_ocr, ocr_conf, text_bbox = raw_ocr2, ocr_conf2, text_bbox2

        tag, tag_status = normalize_tag(raw_ocr) if raw_ocr else (None, "MISSING")

        # Still nothing VALID after depth escalation — try a rotated
        # re-check on just this symbol (Open Issue #2), same targeted
        # per-symbol scope as full-page mode's targeted_rotated_recheck.
        rotated = False
        if tag_status != "VALID" and zone_rotated_recheck:
            recovered = targeted_rotated_recheck(
                page_image, det, zone_depth + ZONE_DEPTH_ESCALATE, page_name
            )
            if recovered:
                raw_ocr    = recovered["text"]
                ocr_conf   = recovered["conf"]
                tag        = recovered["_tag"]
                tag_status = "RECOVERED_ROTATED"
                rotated    = True
                recovered_count += 1
                # targeted_rotated_recheck's own docstring: its bbox isn't
                # reprojected through the rotation and isn't trustworthy
                # for spatial use — drop it so the reviewer UI falls back
                # to the symbol's own box instead of a misleading one.
                text_bbox = None

        cconf = combined_confidence(det["conf"], ocr_conf, tag_status)

        # Low-trust VALID matches: single-letter-only (bug #4), loop-prefix-
        # only (Open Issue #1, step 3), or matching a confirmed non-
        # instrument reference/spec pattern (pipe-spec, setpoint annotation,
        # known abbreviation) — all structurally-permissive matches that
        # have each produced a confirmed false positive on real data. Force
        # human sign-off, never silently auto-accept, regardless of
        # confidence score.
        single_letter_only = False
        loop_prefix_only   = False
        noise_reason        = None
        if tag_status == "VALID":
            _, strict_status = normalize_tag_strict(raw_ocr)
            single_letter_only = (strict_status != "VALID")
            loop_prefix_only   = is_loop_prefix_only(tag)
            noise_reason       = reference_noise_reason(raw_ocr)
        low_trust = single_letter_only or loop_prefix_only or bool(noise_reason)

        auto_accept = (cconf >= AUTO_ACCEPT_THRESHOLD and tag_status == "VALID"
                       and not low_trust)
        verification_method = "AI_AUTO" if auto_accept else (
            "HUMAN_REQUIRED" if tag_status in ("RAW", "MISSING", "RECOVERED_ROTATED")
            or low_trust else ""
        )
        if tag_status == "VALID":   valid_count   += 1
        elif tag_status == "RAW":   raw_count     += 1
        elif tag_status == "MISSING": missing_count += 1
        # Always print — not just on a zone match. This is the fastest way
        # to see WHY a RAW/MISSING happened (genuine OCR misread vs. a
        # pattern the regex doesn't cover vs. nothing found in any zone)
        # without re-running with different flags to guess.
        zone_note = f"'{zone_used}' zone" if zone_used else "no zone matched"
        if escalated:
            zone_note += f" (escalated depth={zone_depth + ZONE_DEPTH_ESCALATE})"
        if rotated:
            zone_note += " (rotated recheck)"
        flag_note = ""
        if low_trust:
            reason = ("single-letter" if single_letter_only
                       else "loop-prefix" if loop_prefix_only
                       else noise_reason)
            flag_note = f" [LOW-TRUST:{reason}]"
        print(f"    det {det['_det_id']} ({det['class_name']}): {zone_note} -> "
              f"raw={raw_ocr!r} tag={tag!r} status={tag_status}{flag_note}")
        rows_by_det_id[det["_det_id"]] = {
            "page": page_name, "det_id": det["_det_id"],
            "class_name": det["class_name"], "class_id": det["class_id"],
            "yolo_conf": round(det["conf"], 4), "x1": det["x1"], "y1": det["y1"],
            "x2": det["x2"], "y2": det["y2"], "routing": "instrument",
            "raw_ocr": raw_ocr or "", "ocr_conf": round(ocr_conf, 4),
            "tag": tag or "", "tag_status": tag_status,
            "inferred_function": infer_instrument_function(tag) or "",
            "combined_conf": cconf, "auto_accept": auto_accept,
            "verified_by": "", "verified_at": "",
            "verification_method": verification_method,
            "reference_noise": noise_reason or "",
            "text_x1": text_bbox["x1"] if text_bbox else None,
            "text_y1": text_bbox["y1"] if text_bbox else None,
            "text_x2": text_bbox["x2"] if text_bbox else None,
            "text_y2": text_bbox["y2"] if text_bbox else None,
        }
    return rows_by_det_id, valid_count, raw_count, missing_count, recovered_count

def _skipped_rows(page_name, other_syms):
    """Row dicts for non-instrument (mechanical/structural/unknown) symbols
    — identical shape/content regardless of which OCR strategy ran, so
    every process_page branch can call this instead of repeating it."""
    return [{
        "page": page_name, "det_id": det["_det_id"],
        "class_name": det["class_name"], "class_id": det["class_id"],
        "yolo_conf": round(det["conf"], 4), "x1": det["x1"], "y1": det["y1"],
        "x2": det["x2"], "y2": det["y2"], "routing": det["_routing"],
        "raw_ocr": "", "ocr_conf": 0.0, "tag": "", "tag_status": "SKIPPED",
        "combined_conf": 0.0, "auto_accept": False,
        "verified_by": "", "verified_at": "", "verification_method": "",
        "text_x1": None, "text_y1": None, "text_x2": None, "text_y2": None,
    } for det in other_syms]

def _rescued_row(page_name, rescued, det_id, ocr_conf):
    """
    Row dict for a symbol that YOLO never detected in production but
    rescue_low_conf's low-conf/loose-NMS crop scan found. Marked
    routing="instrument", tag_status="VALID", auto_accept=False,
    verification_method="HUMAN_REQUIRED" regardless of confidence — same
    "found at conf as low as 0.10-0.17, never trust silently" convention
    rescue_to_db.py uses when pushing these to the DB. class_id=-1 (no
    canonical class_id available from a bare YOLO predict() call; DB
    layer already knows how to handle this — see rescue_to_db.get_class_id).
    """
    tag, tag_status = normalize_tag(rescued["candidate_tag"])
    cconf = combined_confidence(rescued["rescued_conf"], ocr_conf, tag_status)
    return {
        "page": page_name, "det_id": det_id,
        "class_name": rescued["rescued_class"], "class_id": -1,
        "yolo_conf": rescued["rescued_conf"],
        "x1": rescued["rescued_x1"], "y1": rescued["rescued_y1"],
        "x2": rescued["rescued_x2"], "y2": rescued["rescued_y2"],
        "routing": "instrument",
        "raw_ocr": rescued["raw_text"], "ocr_conf": round(ocr_conf, 4),
        "tag": tag or rescued["candidate_tag"], "tag_status": tag_status or "VALID",
        "inferred_function": infer_instrument_function(tag) or "",
        "combined_conf": cconf, "auto_accept": False,
        "verified_by": "", "verified_at": "",
        "verification_method": "HUMAN_REQUIRED",
        "_rescued": True,
        "text_x1": None, "text_y1": None, "text_x2": None, "text_y2": None,
    }

_rescue_model_cache = {}

def get_rescue_model(model_path):
    """Lazy-load + cache a YOLO rescue model by path, mirroring get_ocr()'s
    load-once pattern above — avoids reloading ultralytics weights once per
    page when processing --all runs."""
    key = str(model_path)
    if key not in _rescue_model_cache:
        import rescue_low_conf
        print(f"  Loading rescue model from {model_path}...")
        _rescue_model_cache[key] = rescue_low_conf.load_rescue_model(model_path)
    return _rescue_model_cache[key]

def process_page(page_name, all_detections, use_db=True, search_radius=250,
                  force_recompute_ocr=False, ocr_only=False, force_tiling=False,
                  force_single_pass=False, resize_target=None,
                  export_all_text=False, rotated_recheck=True, review_queue=True,
                  zone_ocr=False, zone_depth=ZONE_DEPTH, zone_margin=ZONE_MARGIN,
                  zone_rotated_recheck=True, exclude_zones=None,
                  max_orphan_dist=MAX_ORPHAN_DIST_DEFAULT, filter_pipe_spec=True,
                  auto=False, auto_resize_target=None, escalate_tile=False,
                  rescue_on_escalate=True,
                  rescue_model_path=None, rescue_pad=350, rescue_conf=0.10,
                  rescue_iou=0.75):
    image_path = resolve_image_path(page_name)
    if not image_path.exists():
        print(f"  Image not found: {image_path} (searched {IMAGES_DIR} and its subfolders)")
        return [], [], [], []

    detections = all_detections.get(page_name, [])
    if not detections and not ocr_only:
        print(f"  No detections for {page_name}")
        return [], [], [], []

    page_image = Image.open(image_path)
    img_w, img_h = page_image.width, page_image.height

    print(f"Processing {page_name} ({len(detections)} detections, {img_w}×{img_h}px)...")

    # 0. Drop detections sitting inside excluded regions (title block, BOM/
    # legend table, general-notes box) BEFORE anything else touches them.
    # This is the single highest-leverage fix reported in the P&ID
    # digitization literature for tag-extraction false positives — a
    # Springer study (Tag classification and detection for P&IDs) found
    # most tag false positives came from "the diagram description on the
    # right pane" and raised F1 from 85.9% to 89.6% by cropping that region
    # out before detection/OCR, rather than trying to filter it out after
    # the fact with text-shape heuristics. Coordinates are page-pixel
    # (x1,y1,x2,y2), same space as detections — find them once per drawing
    # template by eyeballing a page, then reuse via --exclude-zone.
    if exclude_zones:
        before = len(detections)
        def _in_excluded(det):
            cx, cy = centroid(det["x1"], det["y1"], det["x2"], det["y2"])
            return any(ex1 <= cx <= ex2 and ey1 <= cy <= ey2
                       for ex1, ey1, ex2, ey2 in exclude_zones)
        detections = [d for d in detections if not _in_excluded(d)]
        dropped = before - len(detections)
        if dropped:
            print(f"  Excluded {dropped}/{before} detections inside "
                  f"{len(exclude_zones)} exclude-zone(s) (title block/legend/notes)")

    # 1. Classify detections FIRST — zone-OCR mode needs to know which
    # detections are instruments before deciding what to OCR at all; doing
    # this before the OCR step (rather than after, as previously) is what
    # makes "only OCR near instrument symbols, skip everything else"
    # actually possible instead of OCR-ing the whole page regardless.
    instrument_syms = []
    other_syms      = []
    corrections     = 0
    for i, det in enumerate(detections):
        # Detections JSON's own class_name can be stale — always trust class_id
        # against dataset.yaml instead (see canonical_class_name() above).
        corrected_name = canonical_class_name(det["class_id"], det["class_name"])
        if corrected_name != det["class_name"]:
            corrections += 1
            det["class_name"] = corrected_name
        routing = get_routing(det["class_name"])
        det["_routing"] = routing
        det["_det_id"]  = i + 1
        if routing == "instrument":
            instrument_syms.append(det)
        else:
            other_syms.append(det)

    if corrections:
        print(f"  Corrected {corrections}/{len(detections)} stale class names using dataset.yaml")
    print(f"  Instruments: {len(instrument_syms)}  |  Skipped: {len(other_syms)}")

    all_text_rows = []
    review_rows   = []
    dropped_rows  = []

    if auto:
        # ── AUTO MODE (new default) ──────────────────────────────────────
        # Streamlined pipeline per supervisor's ordering: 4-zone OCR decides
        # which tag belongs to which symbol FIRST (cheap — cost scales with
        # instrument count, not page pixels). Only if that's incomplete
        # (fewer VALID tags than instruments) does this escalate to
        # resize-OCR, which reads the whole page in one pass and — because
        # it already has whole-page context — also builds the review queue
        # needed to run the rescue (low-conf/loose-NMS YOLO) pass on
        # whatever's still unresolved. If zone-ocr alone gets everything
        # (e.g. 7/7), escalation is skipped entirely and this costs exactly
        # what --zone-ocr alone costs — that's the "cut short if things are
        # in favour" behavior asked for.
        print(f"  [auto mode] Pass 1/2: zone-OCR ({len(instrument_syms)} instrument symbol(s))...")
        zone_rows, valid_count, raw_count, missing_count, recovered_count = _run_zone_ocr_pass(
            page_image, instrument_syms, page_name,
            zone_depth=zone_depth, zone_margin=zone_margin,
            zone_rotated_recheck=zone_rotated_recheck,
        )
        inst_count = len(instrument_syms)
        complete = (inst_count == 0) or (valid_count == inst_count)

        # Escalation to Pass 2 (resize-OCR + review-queue) now runs whenever
        # review_queue is wanted (the default) — NOT only when zone-ocr left
        # something unresolved. Reasoning: zone-ocr can only confirm/fix tags
        # for instruments YOLO already boxed; it has zero visibility into
        # instruments YOLO missed detecting entirely (confirmed real case:
        # SET 1_page_2 — YOLO found 1/5 real instrument bubbles, zone-ocr
        # read that 1 correctly, "complete"=True under the old rule, and the
        # other 4 never got a chance to surface anywhere, not even as review
        # candidates). Only find_review_queue_rows()'s whole-page, symbol-
        # agnostic scan (Pass 2's job) can catch that class of miss — so
        # "zone-ocr succeeded on what it found" is the wrong signal for
        # whether to skip it. --no-review-queue is the correct way to get
        # the old fast-exit behavior back for a given run. (The full-page
        # association loop below already skips any symbol zone-ocr already
        # resolved to VALID — see the `continue` on tag_status == "VALID" —
        # so always running Pass 2 never overwrites a good zone-ocr result,
        # it only adds coverage zone-ocr structurally couldn't have had.)
        skip_escalation = complete and not export_all_text and not review_queue

        if skip_escalation:
            rows = list(zone_rows.values()) + _skipped_rows(page_name, other_syms)
            valid_pct = round(valid_count / inst_count * 100, 1) if inst_count else 0
            print(f"  → VALID:{valid_count}  RAW:{raw_count}  MISSING:{missing_count}  "
                  f"RECOVERED(rotated):{recovered_count}  SKIPPED:{len(other_syms)}  "
                  f"({valid_pct}%)  [auto mode — zone-ocr sufficed, --no-review-queue set, "
                  f"no escalation needed]")
            if use_db:
                _write_rows_to_db(page_name, img_w, img_h, image_path, rows, [])
            return rows, all_text_rows, review_rows, dropped_rows

        reason = ("zone-ocr sufficed but review-queue needs whole-page context"
                   if complete and not export_all_text
                   else "--export-all-text requested (needs whole-page context)"
                   if complete else f"zone-ocr incomplete ({valid_count}/{inst_count} VALID)")

        # Escalation strategy: resize-OCR by default (2026-07-16, reversed
        # from the previous tiling-default — see --tile-fallback below for
        # how to get the old behavior back).
        #
        # KNOWN RISK, carried forward from before this flip: resize-OCR
        # died silently on this machine twice in earlier testing (no
        # Python traceback at all, straight back to the shell prompt) at
        # the "resized to 4500x3179px..." point — only ~14MP, far under
        # the ~70MP that caused the original documented single-pass crash.
        # That signature (hard crash, no exception) points at an unstable
        # Paddle CPU inference call, not a page-specific data problem, so
        # it can plausibly recur — especially on pages bigger than the
        # ~14MP where it happened before (this page is 9934x7017 ≈ 70MP).
        # Made the default anyway per explicit instruction; if a run dies
        # silently at that resize point, that's this known failure mode,
        # not a new bug — re-run the same page with --tile-fallback to
        # fall back to the previously-default, proven-stable tiling path.
        if escalate_tile:
            print(f"  [auto mode] Pass 2/2: escalating to tiled OCR "
                  f"(--tile-fallback) — {reason}...")
            text_blocks = run_ocr_full_page(image_path, page_name=page_name,
                                             force_recompute=force_recompute_ocr,
                                             force_tiling=True)
        else:
            target = auto_resize_target or resize_target or RESIZE_OCR_TARGET_DEFAULT
            print(f"  [auto mode] Pass 2/2: escalating to resize-OCR "
                  f"(target={target}px) — {reason}...")
            text_blocks = run_ocr_full_page(image_path, page_name=page_name,
                                             force_recompute=force_recompute_ocr,
                                             resize_target=target)
        print(f"  Full-page OCR complete: {len(text_blocks)} text blocks found")

        if exclude_zones:
            before_tb = len(text_blocks)
            def _blk_excluded(b):
                bcx, bcy = centroid(b["x1"], b["y1"], b["x2"], b["y2"])
                return any(ex1 <= bcx <= ex2 and ey1 <= bcy <= ey2
                           for ex1, ey1, ex2, ey2 in exclude_zones)
            text_blocks = [b for b in text_blocks if not _blk_excluded(b)]
            dropped_tb = before_tb - len(text_blocks)
            if dropped_tb:
                print(f"  Excluded {dropped_tb}/{before_tb} OCR text blocks inside "
                      f"exclude-zone(s) (title block/legend/notes)")

        if export_all_text:
            all_text_rows = build_all_text_rows(
                page_name, text_blocks, instrument_syms + other_syms, search_radius
            )

        # review_rows doubles as rescue's input (orphaned candidates), so
        # build it whenever either is wanted — but respect an explicit
        # --no-review-queue + --no-rescue combination and skip entirely.
        review_rows_for_rescue = []
        if review_queue or rescue_on_escalate:
            review_rows, dropped_rows = find_review_queue_rows(
                page_name, text_blocks, instrument_syms + other_syms, search_radius,
                max_orphan_dist=max_orphan_dist, filter_pipe_spec=filter_pipe_spec
            )
            if dropped_rows:
                reason_counts = Counter(r["drop_reason"] for r in dropped_rows)
                print(f"  Filtered {len(dropped_rows)} candidate(s) before review queue "
                      f"({', '.join(f'{v} {k}' for k, v in reason_counts.items())})")
            if review_rows:
                misrouted = sum(1 for r in review_rows if r["kind"] == "MISROUTED_CANDIDATE")
                orphaned_tags = len(review_rows) - misrouted
                print(f"  Review queue: {len(review_rows)} flagged "
                      f"({misrouted} misrouted-candidate, {orphaned_tags} orphaned-tag-candidate)")
            if not review_queue:
                # Built only for rescue's sake — don't report/return/write it
                # as if the user asked for the review-queue output itself.
                review_rows_for_rescue, review_rows = review_rows, []

        # Full-page association, but ONLY overrides symbols zone-ocr didn't
        # already resolve to VALID — zone's match is more spatially
        # precise (crop taken directly from the symbol's own neighborhood;
        # see extract_tag_zoned docstring), so a symbol zone already got
        # right is never re-decided by the coarser whole-page nearest-block
        # method. This is the "resize fills what zone couldn't" merge.
        associations = associate_text_to_symbols(
            text_blocks, instrument_syms, search_radius=search_radius
        )
        merged_rows = dict(zone_rows)
        still_unresolved = []
        for sym_idx, det in enumerate(instrument_syms):
            det_id = det["_det_id"]
            if merged_rows.get(det_id, {}).get("tag_status") == "VALID":
                continue
            nearby = associations.get(sym_idx, [])
            raw_ocr, ocr_conf, text_bbox = pick_best_text(nearby)
            tag, tag_status = normalize_tag(raw_ocr)
            if tag_status == "MISSING" and rotated_recheck:
                recovered = targeted_rotated_recheck(page_image, det, search_radius, page_name)
                if recovered:
                    raw_ocr, ocr_conf = recovered["text"], recovered["conf"]
                    tag, tag_status = recovered["_tag"], "RECOVERED_ROTATED"
                    recovered_count += 1
                    # See targeted_rotated_recheck's docstring — its text
                    # isn't tied to a reliable bbox, so don't carry the
                    # old (now-stale) pick_best_text bbox forward either.
                    text_bbox = None
            cconf = combined_confidence(det["conf"], ocr_conf, tag_status)
            single_letter_only = loop_prefix_only = False
            noise_reason = None
            if tag_status == "VALID":
                _, strict_status = normalize_tag_strict(raw_ocr)
                single_letter_only = (strict_status != "VALID")
                loop_prefix_only = is_loop_prefix_only(tag)
                noise_reason = reference_noise_reason(raw_ocr)
            low_trust = single_letter_only or loop_prefix_only or bool(noise_reason)
            auto_accept = (cconf >= AUTO_ACCEPT_THRESHOLD and tag_status == "VALID"
                           and not low_trust)
            verification_method = "AI_AUTO" if auto_accept else (
                "HUMAN_REQUIRED" if tag_status in ("RAW", "MISSING", "RECOVERED_ROTATED")
                or low_trust else ""
            )
            new_row = {
                "page": page_name, "det_id": det_id,
                "class_name": det["class_name"], "class_id": det["class_id"],
                "yolo_conf": round(det["conf"], 4), "x1": det["x1"], "y1": det["y1"],
                "x2": det["x2"], "y2": det["y2"], "routing": "instrument",
                "raw_ocr": raw_ocr or "", "ocr_conf": round(ocr_conf, 4),
                "tag": tag or "", "tag_status": tag_status,
                "inferred_function": infer_instrument_function(tag) or "",
                "combined_conf": cconf, "auto_accept": auto_accept,
                "verified_by": "", "verified_at": "",
                "verification_method": verification_method,
                "reference_noise": noise_reason or "",
                "text_x1": text_bbox["x1"] if text_bbox else None,
                "text_y1": text_bbox["y1"] if text_bbox else None,
                "text_x2": text_bbox["x2"] if text_bbox else None,
                "text_y2": text_bbox["y2"] if text_bbox else None,
            }
            # Prefer whichever pass actually found something — a resize
            # RAW beats a zone MISSING even though neither is VALID, so
            # human review at least has real OCR text to look at.
            old_row = merged_rows.get(det_id)
            if old_row is None or tag_status == "VALID" or (
                old_row["tag_status"] == "MISSING" and tag_status != "MISSING"
            ):
                merged_rows[det_id] = new_row
            if merged_rows[det_id]["tag_status"] != "VALID":
                still_unresolved.append(det)

        rescued_rows = []
        rescue_source_rows = review_rows if review_queue else review_rows_for_rescue
        if rescue_on_escalate and rescue_source_rows:
            orphaned = [r for r in rescue_source_rows if r["kind"] == "ORPHANED_TAG_CANDIDATE"]
            if orphaned:
                print(f"  [auto mode] Running rescue on {len(orphaned)} orphaned "
                      f"candidate(s) (conf={rescue_conf}, iou={rescue_iou})...")
                import rescue_low_conf
                model, names = get_rescue_model(rescue_model_path or DEFAULT_MODEL)
                results = rescue_low_conf.rescue_candidates(
                    page_image, orphaned, model, names, page_name=page_name,
                    pad=rescue_pad, conf=rescue_conf, iou=rescue_iou,
                )
                rescued = [r for r in results if r["rescue_status"] == "RESCUED"]
                still_missing_n = len(results) - len(rescued)
                print(f"  Rescue: {len(rescued)} recovered, {still_missing_n} "
                      f"still a genuine blind spot")
                next_det_id = max([d["_det_id"] for d in detections], default=0) + 1

                # Guard against re-detecting a symbol that's already known —
                # either from the production pass (instrument_syms) or from
                # an earlier result in THIS rescue batch. rescue_low_conf's
                # own NMS (rescue_iou, default 0.75 — deliberately loose so
                # it doesn't merge away real neighbors) only dedupes within
                # its own single call; it has no visibility into symbols
                # found elsewhere. Without this, an orphaned text block that
                # sits near an already-detected instrument (associated or
                # not — association can fail for reasons unrelated to
                # whether the symbol itself was found) gets "rescued" as if
                # it were brand-new, given a fresh det_id, and the same
                # physical bubble ends up with two or three symbol+tag rows,
                # each OCR'd differently. IoU 0.3 mirrors the threshold
                # _dedupe_overlap_blocks already uses above for the same
                # class of problem in raw OCR blocks.
                RESCUE_DEDUPE_IOU = 0.3
                known_boxes = [
                    {"x1": d["x1"], "y1": d["y1"], "x2": d["x2"], "y2": d["y2"]}
                    for d in instrument_syms
                ]
                skipped_as_dup = 0
                for r in rescued:
                    rbox = {"x1": r["rescued_x1"], "y1": r["rescued_y1"],
                            "x2": r["rescued_x2"], "y2": r["rescued_y2"]}
                    if any(_iou(rbox, b) > RESCUE_DEDUPE_IOU for b in known_boxes):
                        skipped_as_dup += 1
                        continue
                    ocr_conf_val = float(r["ocr_conf"]) if r.get("ocr_conf") not in (None, "") else 0.0
                    row = _rescued_row(page_name, r, next_det_id, ocr_conf_val)
                    rescued_rows.append(row)
                    known_boxes.append(rbox)  # also guards against two rescue results colliding with each other
                    next_det_id += 1
                if skipped_as_dup:
                    print(f"  Rescue: discarded {skipped_as_dup} rescued detection(s) as "
                          f"near-duplicates of an already-known symbol (IoU > {RESCUE_DEDUPE_IOU})")

        # Recount after merge + rescue for the final summary line.
        valid_count = sum(1 for r in merged_rows.values() if r["tag_status"] == "VALID")
        raw_count = sum(1 for r in merged_rows.values() if r["tag_status"] == "RAW")
        missing_count = sum(1 for r in merged_rows.values() if r["tag_status"] == "MISSING")
        rows = list(merged_rows.values()) + rescued_rows + _skipped_rows(page_name, other_syms)
        inst_count_final = inst_count + len(rescued_rows)
        valid_pct = round((valid_count + len(rescued_rows)) / inst_count_final * 100, 1) if inst_count_final else 0
        print(f"  → VALID:{valid_count + len(rescued_rows)}  RAW:{raw_count}  MISSING:{missing_count}  "
              f"RECOVERED(rotated):{recovered_count}  RESCUED:{len(rescued_rows)}  "
              f"SKIPPED:{len(other_syms)}  ({valid_pct}%)  [auto mode — escalated]")

        if use_db:
            _write_rows_to_db(page_name, img_w, img_h, image_path, rows, review_rows)
        return rows, all_text_rows, review_rows, dropped_rows

    if zone_ocr:
        # ── ZONE-OCR MODE (explicit, forced — no escalation) ─────────────
        # No full-page OCR at all, ever, regardless of completeness. Only
        # crop-and-OCR the small region around each instrument symbol.
        # Trades away all_text_export and review_queue (both need
        # whole-page OCR context) for a large runtime cut. For the
        # streamlined "zone first, escalate only if needed" behavior, use
        # the default auto mode instead of this explicit flag.
        if export_all_text or review_queue:
            print("  NOTE: --zone-ocr skips full-page OCR, so all-text-export "
                  "and review-queue are unavailable this run (they need "
                  "whole-page text). Use the default auto mode (no "
                  "--zone-ocr flag) to get escalation + these outputs "
                  "only when needed.")
        if zone_depth < search_radius:
            print(f"  NOTE: --zone-depth ({zone_depth}px) is smaller than "
                  f"--radius ({search_radius}px) — a tag that full-page mode "
                  f"would find (within radius) can still be missed by zone "
                  f"mode if it sits further than {zone_depth}px from the "
                  f"bubble in its zone. If RAW/MISSING tags turn out to have "
                  f"no legible text at all in the printed zone crops, try "
                  f"raising --zone-depth first before assuming it's a "
                  f"genuine OCR misread.")
        zone_rows, valid_count, raw_count, missing_count, recovered_count = _run_zone_ocr_pass(
            page_image, instrument_syms, page_name,
            zone_depth=zone_depth, zone_margin=zone_margin,
            zone_rotated_recheck=zone_rotated_recheck,
        )
        rows = list(zone_rows.values()) + _skipped_rows(page_name, other_syms)
        inst_count = len(instrument_syms)
        valid_pct  = round(valid_count / inst_count * 100, 1) if inst_count else 0
        print(f"  → VALID:{valid_count}  RAW:{raw_count}  MISSING:{missing_count}  "
              f"RECOVERED(rotated):{recovered_count}  SKIPPED:{len(other_syms)}  "
              f"({valid_pct}%)  [zone-ocr mode]")
        if use_db:
            _write_rows_to_db(page_name, img_w, img_h, image_path, rows, [])
        return rows, all_text_rows, review_rows, dropped_rows

    # ── FULL-PAGE MODE (explicit — --resize-ocr / --tile-ocr / --single-pass-ocr) ──
    # 2. Full-page OCR — cached after first run, so re-running this script to
    # tweak association/normalization logic never re-pays the OCR cost.
    text_blocks = run_ocr_full_page(image_path, page_name=page_name,

                                     force_recompute=force_recompute_ocr,
                                     force_tiling=force_tiling,
                                     force_single_pass=force_single_pass,
                                     resize_target=resize_target)
    print(f"  OCR complete: {len(text_blocks)} text blocks found")

    if exclude_zones:
        before_tb = len(text_blocks)
        def _blk_excluded(b):
            bcx, bcy = centroid(b["x1"], b["y1"], b["x2"], b["y2"])
            return any(ex1 <= bcx <= ex2 and ey1 <= bcy <= ey2
                       for ex1, ey1, ex2, ey2 in exclude_zones)
        text_blocks = [b for b in text_blocks if not _blk_excluded(b)]
        dropped_tb = before_tb - len(text_blocks)
        if dropped_tb:
            print(f"  Excluded {dropped_tb}/{before_tb} OCR text blocks inside "
                  f"exclude-zone(s) (title block/legend/notes)")

    if ocr_only:
        return [], [], [], []

    # 2b. Full-page text export — every OCR block, nearest-symbol context
    # regardless of class. Independent of the instrument-only pipeline below.
    if export_all_text:
        all_text_rows = build_all_text_rows(
            page_name, text_blocks, instrument_syms + other_syms, search_radius
        )
        orphaned = sum(1 for r in all_text_rows if not r["within_radius"])
        print(f"  All-text export: {len(all_text_rows)} OCR blocks total "
              f"({orphaned} not within {search_radius}px of any symbol)")

    # 2c. Review queue — tag-shaped text near a non-instrument symbol
    # (MISROUTED_CANDIDATE) or near no symbol at all (ORPHANED_TAG_CANDIDATE).
    # On by default: this is the difference between "silently discarded" and
    # "flagged for a human," not an optional extra.
    if review_queue:
        review_rows, dropped_rows = find_review_queue_rows(
            page_name, text_blocks, instrument_syms + other_syms, search_radius,
            max_orphan_dist=max_orphan_dist, filter_pipe_spec=filter_pipe_spec
        )
        if dropped_rows:
            reason_counts = Counter(r["drop_reason"] for r in dropped_rows)
            print(f"  Filtered {len(dropped_rows)} candidate(s) before review queue "
                  f"({', '.join(f'{v} {k}' for k, v in reason_counts.items())})")
        if review_rows:
            misrouted = sum(1 for r in review_rows if r["kind"] == "MISROUTED_CANDIDATE")
            orphaned_tags = len(review_rows) - misrouted
            print(f"  Review queue: {len(review_rows)} flagged "
                  f"({misrouted} misrouted-candidate, {orphaned_tags} orphaned-tag-candidate)")
            # OCR-only insight: these candidates have no confirmed YOLO
            # instrument detection nearby (that's what put them here), but
            # the tag text itself still tells you what KIND of instrument
            # it likely is (see infer_instrument_function). This is the
            # "what did OCR see that YOLO missed" view — a supplementary
            # signal for human review, not an auto-added instrument count.
            func_counts = Counter(
                r["inferred_function"] for r in review_rows if r["inferred_function"]
            )
            if func_counts:
                print(f"  OCR-inferred instrument types among flagged candidates "
                      f"(YOLO did not confirm these — verify before counting):")
                for func, cnt in func_counts.most_common():
                    print(f"      {cnt:>3}x  {func}")

    # 3. Associate text blocks to instrument symbols
    associations = associate_text_to_symbols(
        text_blocks, instrument_syms, search_radius=search_radius
    )

    # 4. Build result rows
    rows = []

    # Skipped symbols (mechanical/structural/unknown)
    for det in other_syms:
        rows.append({
            "page":        page_name,
            "det_id":      det["_det_id"],
            "class_name":  det["class_name"],
            "class_id":    det["class_id"],
            "yolo_conf":   round(det["conf"], 4),
            "x1": det["x1"], "y1": det["y1"], "x2": det["x2"], "y2": det["y2"],
            "routing":     det["_routing"],
            "raw_ocr":     "",
            "ocr_conf":    0.0,
            "tag":         "",
            "tag_status":  "SKIPPED",
            "combined_conf": 0.0,
            "auto_accept": False,
            "verified_by": "",
            "verified_at": "",
            "verification_method": "",
        })

    # Instrument symbols — extract and normalize tags
    valid_count    = 0
    raw_count      = 0
    missing_count  = 0
    recovered_count = 0

    for sym_idx, det in enumerate(instrument_syms):
        nearby = associations.get(sym_idx, [])
        raw_ocr, ocr_conf, text_bbox = pick_best_text(nearby)
        tag, tag_status   = normalize_tag(raw_ocr)

        # No text found on the 0°-only full-page pass — try a targeted
        # rotated re-check on just this symbol's neighborhood before
        # giving up. See targeted_rotated_recheck() docstring for scope.
        if tag_status == "MISSING" and rotated_recheck:
            recovered = targeted_rotated_recheck(page_image, det, search_radius, page_name)
            if recovered:
                raw_ocr    = recovered["text"]
                ocr_conf   = recovered["conf"]
                tag        = recovered["_tag"]
                tag_status = "RECOVERED_ROTATED"
                recovered_count += 1
                text_bbox  = None  # see targeted_rotated_recheck docstring

        cconf             = combined_confidence(det["conf"], ocr_conf, tag_status)

        # A VALID tag that ONLY matches via the single-letter catch-all
        # pattern (not the strict multi-letter patterns) is low-trust —
        # confirmed against real data: det103 on SET 1_page_1 matched "B-2"
        # this way from "10111 B2 H10-CHV-0226", a spec/zone reference, not
        # a tag. Real single-letter ISA tags do exist, so this doesn't
        # reject the match — it just forces human sign-off instead of
        # silent auto-accept, same treatment as RECOVERED_ROTATED.
        # A VALID tag that ONLY matches via the loop-prefix pattern
        # (digit-LETTERS-digit) is equally low-trust — same failure class,
        # confirmed on det124 in zone-ocr mode (see is_loop_prefix_only()).
        # A VALID tag matching a confirmed non-instrument reference/spec
        # pattern (pipe-spec, setpoint annotation, known abbreviation) is
        # the same failure class again — confirmed on this same drawing:
        # a line-spec diamond symbol normalized to B-10/B-13/B-80, and a
        # 4" 600#RF flange spec normalized to N-10131/W-10131.
        single_letter_only = False
        loop_prefix_only   = False
        noise_reason        = None
        if tag_status == "VALID":
            _, strict_status = normalize_tag_strict(raw_ocr)
            single_letter_only = (strict_status != "VALID")
            loop_prefix_only   = is_loop_prefix_only(tag)
            noise_reason       = reference_noise_reason(raw_ocr)
        low_trust = single_letter_only or loop_prefix_only or bool(noise_reason)

        # Recovered-via-rotation and low-trust tags always need human
        # sign-off — both are lower-trust cases regardless of confidence.
        auto_accept       = (cconf >= AUTO_ACCEPT_THRESHOLD and tag_status == "VALID"
                              and not low_trust)
        verification_method = "AI_AUTO" if auto_accept else (
            "HUMAN_REQUIRED" if tag_status in ("RAW", "MISSING", "RECOVERED_ROTATED")
            or low_trust else ""
        )

        if tag_status == "VALID":     valid_count     += 1
        elif tag_status == "RAW":     raw_count       += 1
        elif tag_status == "MISSING": missing_count   += 1

        rows.append({
            "page":        page_name,
            "det_id":      det["_det_id"],
            "class_name":  det["class_name"],
            "class_id":    det["class_id"],
            "yolo_conf":   round(det["conf"], 4),
            "x1": det["x1"], "y1": det["y1"], "x2": det["x2"], "y2": det["y2"],
            "routing":     "instrument",
            "raw_ocr":     raw_ocr or "",
            "ocr_conf":    round(ocr_conf, 4),
            "tag":         tag or "",
            "tag_status":  tag_status,
            "inferred_function": infer_instrument_function(tag) or "",
            "combined_conf": cconf,
            "auto_accept": auto_accept,
            "verified_by": "",
            "verified_at": "",
            "verification_method": verification_method,
            "reference_noise": noise_reason or "",
            "text_x1": text_bbox["x1"] if text_bbox else None,
            "text_y1": text_bbox["y1"] if text_bbox else None,
            "text_x2": text_bbox["x2"] if text_bbox else None,
            "text_y2": text_bbox["y2"] if text_bbox else None,
        })

    inst_count = len(instrument_syms)
    valid_pct  = round(valid_count / inst_count * 100, 1) if inst_count else 0
    print(f"  → VALID:{valid_count}  RAW:{raw_count}  "
          f"MISSING:{missing_count}  RECOVERED(rotated):{recovered_count}  "
          f"SKIPPED:{len(other_syms)}  ({valid_pct}%)")

    # 5. Write to DB
    if use_db:
        _write_rows_to_db(page_name, img_w, img_h, image_path, rows, review_rows)

    return rows, all_text_rows, review_rows, dropped_rows

def _write_rows_to_db(page_name, img_w, img_h, image_path, rows, review_rows):
    """
    Shared DB-write path for both zone-OCR and full-page modes.

    BUG FIXED (2026-07-14): insert_tag() and insert_review_candidate() have no
    uniqueness protection the way insert_symbol() does (UNIQUE(page_id,
    det_id) + INSERT OR IGNORE) — every re-run of the extractor on an
    already-processed page used to append a brand-new tags/review_queue
    row per symbol instead of replacing the old one. Across repeated dev
    re-runs this silently multiplied every symbol's tag row (and every
    review-queue candidate) N times, which is exactly what showed up as
    identical repeated cards in the review UI. Now each write:
      - skips re-inserting a tag for any symbol a human has already
        HUMAN_VERIFIED (never overwrite real review work)
      - otherwise deletes prior (non-human-verified) tag rows for that
        symbol_id before inserting the fresh one, so a re-run REPLACES
        rather than ACCUMULATES
      - skips inserting a review_queue candidate that already exists for
        this exact (page, kind, candidate_tag, bbox) and is still OPEN
        (avoid growing duplicates) or already RESOLVED/DISMISSED (a human
        already decided — never resurrect it)
    """
    try:
        from db_builder import (get_connection, get_or_create_pid,
                                 get_or_create_page, insert_symbol,
                                 insert_tag, insert_review_candidate,
                                 update_page_stats)
        import page_json
        conn   = get_connection()
        # BUG FIXED (2026-07-21): this was hardcoded to the literal string
        # "pid_pdf.pdf" for every single page ever processed, regardless of
        # which actual PDF/SET it came from. Every page across every SET
        # collapsed into the same p_ids row, which is why the dashboard
        # needed a page_name-derived _source_label() workaround instead of
        # trusting p_ids.file_name directly. Derive the real stem the same
        # way page_json.py's _page_output_path() and the dashboard's
        # _source_label() already do — split on "_page_" — so all three
        # stay consistent with each other.
        pdf_stem = page_name.split("_page_")[0] if "_page_" in page_name else page_name
        pid_id = get_or_create_pid(
            conn, f"{pdf_stem}.pdf", file_path=str(Path(image_path).parent)
        )
        page_id = get_or_create_page(
            conn, pid_id, page_name,
            page_number=int(page_name.split("_")[-1]) if page_name[-2:].isdigit() else None,
            image_path=str(image_path),
            width_px=img_w, height_px=img_h,
        )

        tags_written = 0
        tags_skipped_verified = 0
        for row in rows:
            sym_id = insert_symbol(
                conn, page_id,
                det_id=row["det_id"], class_name=row["class_name"],
                class_id=row["class_id"], yolo_conf=row["yolo_conf"],
                x1=row["x1"], y1=row["y1"], x2=row["x2"], y2=row["y2"],
            )

            existing = conn.execute(
                "SELECT tag_id, verification_method FROM tags WHERE symbol_id = ?",
                (sym_id,)
            ).fetchall()
            if any(e["verification_method"] == "HUMAN_VERIFIED" for e in existing):
                # A person already reviewed this exact symbol — don't
                # silently overwrite their decision with a fresh pipeline
                # guess just because the page was re-processed.
                tags_skipped_verified += 1
                continue

            if existing:
                existing_ids = [e["tag_id"] for e in existing]
                conn.executemany("DELETE FROM validation_log WHERE tag_id = ?",
                                  [(tid,) for tid in existing_ids])
                conn.executemany("DELETE FROM tags WHERE tag_id = ?",
                                  [(tid,) for tid in existing_ids])

            insert_tag(
                conn, sym_id, page_id,
                best_zone=None,
                raw_ocr=row["raw_ocr"] or None,
                ocr_conf=row["ocr_conf"] or None,
                tag=row["tag"] or None,
                tag_status=row["tag_status"],
                combined_conf=row["combined_conf"],
                auto_accept=row["auto_accept"],
                verification_method=row["verification_method"] or None,
                text_x1=row.get("text_x1"), text_y1=row.get("text_y1"),
                text_x2=row.get("text_x2"), text_y2=row.get("text_y2"),
            )
            tags_written += 1

        review_written = 0
        review_skipped = 0
        for rrow in review_rows:
            dup = conn.execute(
                """SELECT review_id, status FROM review_queue
                   WHERE page_id = ? AND kind = ? AND candidate_tag = ?
                     AND x1 = ? AND y1 = ? AND x2 = ? AND y2 = ?""",
                (page_id, rrow["kind"], rrow["candidate_tag"],
                 rrow["x1"], rrow["y1"], rrow["x2"], rrow["y2"])
            ).fetchone()
            if dup is not None:
                # Already OPEN (would just be a duplicate) or already
                # RESOLVED/DISMISSED (a human decided — don't resurrect).
                review_skipped += 1
                continue
            insert_review_candidate(
                conn, page_id,
                kind=rrow["kind"], candidate_tag=rrow["candidate_tag"],
                raw_text=rrow["raw_text"], ocr_conf=rrow["ocr_conf"],
                x1=rrow["x1"], y1=rrow["y1"], x2=rrow["x2"], y2=rrow["y2"],
                nearest_det_id=rrow["nearest_det_id"],
                nearest_class_name=rrow["nearest_class_name"],
                nearest_routing=rrow["nearest_routing"],
                nearest_dist_px=rrow["nearest_dist_px"],
            )
            review_written += 1

        update_page_stats(conn, page_id)
        conn.commit()

        try:
            json_path = page_json.export_page_json(page_name, conn=conn)
        except Exception as json_err:
            json_path = None
            print(f"  WARNING: page JSON export failed: {json_err}")

        conn.close()
        print(f"  DB updated ({tags_written} tags written, "
              f"{tags_skipped_verified} skipped [human-verified], "
              f"{review_written} review items written, "
              f"{review_skipped} review items skipped [duplicate/already resolved]).")
        if json_path:
            print(f"  Structured output: {json_path}")
    except Exception as e:
        print(f"  DB write failed: {e}")

# ─── FULL-PAGE TEXT EXPORT (every OCR block, not just instrument-associated) ──
# The OCR cache already holds every text block PaddleOCR found on the page —
# title block, general notes, line numbers, valve tags (GAV-xxxx), everything.
# The main pipeline only ever surfaces the subset within `search_radius` of an
# *instrument* symbol (Instrument_Field/Panel/etc). This exports the full set,
# with nearest-detection context (any class, not just instrument), so nothing
# on the page is silently discarded. This is the "orphaned text capture" step
# from the handoff — run this first, before building any new
# association/audit logic on top of it.

ALL_TEXT_CSV_FIELDS = [
    "page", "block_idx", "text", "ocr_conf",
    "x1", "y1", "x2", "y2", "cx", "cy",
    "nearest_det_id", "nearest_class_name", "nearest_routing", "nearest_dist_px",
    "within_radius",
]


def build_all_text_rows(page_name, text_blocks, detections, radius):
    """
    One row per raw OCR block on the page, regardless of whether it ever
    gets used by the instrument-tag pipeline. Flags whether it fell within
    `radius` of ANY detected symbol (not just instrument classes) so you can
    separate "near a valve/vessel but not an instrument" (title block, notes,
    line specs, etc — expected, not a bug) from text that's near an
    instrument symbol but got missed upstream (actual gap worth
    investigating).
    """
    rows = []
    for i, blk in enumerate(text_blocks):
        bcx, bcy = centroid(blk["x1"], blk["y1"], blk["x2"], blk["y2"])
        det, dist = nearest_symbol(blk, detections)
        rows.append({
            "page": page_name,
            "block_idx": i,
            "text": blk["text"],
            "ocr_conf": round(blk["conf"], 4),
            "x1": blk["x1"], "y1": blk["y1"], "x2": blk["x2"], "y2": blk["y2"],
            "cx": round(bcx, 1), "cy": round(bcy, 1),
            "nearest_det_id": det.get("_det_id") if det else None,
            "nearest_class_name": det.get("class_name") if det else None,
            "nearest_routing": det.get("_routing") if det else None,
            "nearest_dist_px": round(dist, 1) if dist is not None else None,
            "within_radius": bool(dist is not None and dist <= radius),
        })
    return rows

def write_all_text_csv(all_text_rows, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_TEXT_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_text_rows)
    print(f"CSV written: {out_path}")

# ─── REVIEW QUEUE (misrouted symbols + orphaned tags) ─────────────────────────
# Two distinct failure modes that the instrument-only pipeline structurally
# cannot see, both surfaced here instead of silently dropped:
#
#   MISROUTED_CANDIDATE     — text that reads as an ISA tag sits near a
#                              symbol, but that symbol's routing isn't
#                              "instrument" (mechanical/structural/unknown).
#                              Points at a classification/routing bug
#                              upstream (dataset.yaml mapping, model
#                              confusion), not an OCR/extraction bug.
#   ORPHANED_TAG_CANDIDATE  — text that reads as an ISA tag has NO detected
#                              symbol within search_radius at all. Points at
#                              a YOLO detection miss.
#
# "Reads as an ISA tag" reuses normalize_tag() — the same merge-then-match
# logic already validated against real instrument-associated tags — applied
# to (a) each individual OCR block, and (b) every pair of blocks within
# `pair_proximity` px of each other (catches the common split-token case:
# loop number + function code as two separate OCR blocks with no symbol
# between them).

REVIEW_QUEUE_CSV_FIELDS = [
    "page", "kind", "candidate_tag", "inferred_function", "raw_text", "ocr_conf",
    "x1", "y1", "x2", "y2",
    "nearest_det_id", "nearest_class_name", "nearest_routing", "nearest_dist_px",
]


def write_review_queue_csv(rows, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_QUEUE_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV written: {out_path}")

DROPPED_CANDIDATES_CSV_FIELDS = REVIEW_QUEUE_CSV_FIELDS + ["drop_reason"]

def write_dropped_candidates_csv(rows, out_path):
    """
    Audit trail for candidates find_review_queue_rows() rejected (pipe-spec
    text or exceeds max_orphan_dist) instead of flagging as ORPHANED_TAG_
    CANDIDATE. Exists so the filter's false-negative rate (real tags
    accidentally dropped) is checkable, not just its false-positive
    reduction — spot-check this file after tuning --max-orphan-dist.
    """
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DROPPED_CANDIDATES_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV written: {out_path}")

# ─── OUTPUT ──────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "page", "det_id", "class_name", "class_id", "yolo_conf",
    "x1", "y1", "x2", "y2", "routing", "raw_ocr", "ocr_conf",
    "tag", "tag_status", "inferred_function", "combined_conf", "auto_accept",
    "verified_by", "verified_at", "verification_method", "reference_noise",
    "text_x1", "text_y1", "text_x2", "text_y2",
]

def write_csv(all_rows, out_path):
    # Row dicts may carry internal bookkeeping keys prefixed with "_"
    # (e.g. _rescued_row()'s "_rescued", used only for the total_rescued
    # stat before this point) that were never meant to become CSV columns
    # — same convention as det["_det_id"]/det["_routing"] getting renamed
    # to clean "det_id"/"routing" keys elsewhere. Strip them here instead
    # of hardcoding each one into CSV_FIELDS, so a future internal flag
    # doesn't cause this exact crash again.
    clean_rows = [{k: v for k, v in row.items() if not k.startswith("_")}
                  for row in all_rows]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(clean_rows)
    print(f"CSV written: {out_path}")

# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PipeSight AI — Tag Extractor v2")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--page",  type=str, help='Single page name, e.g. "SET 1_page_1"')
    group.add_argument("--set",   type=str, help='Every page in one Haifa set, e.g. "SET 1" '
                                                   '(matches "SET 1_page_1", "SET 1_page_2", ... '
                                                   'in natural page-number order — writes them all '
                                                   'into one tags_v2.csv/review_queue.csv/'
                                                   'dropped_candidates.csv run, so pdf_excel_export.py '
                                                   'can combine them into one PDF+Excel afterward)')
    group.add_argument("--all",   action="store_true", help="Process all pages in detections JSON")
    parser.add_argument("--model",  type=str, default=str(DEFAULT_MODEL),
                        help="Path to YOLO model weights (for re-running inference). "
                             f"Hardcoded default: {DEFAULT_MODEL}. Pass an explicit "
                             "--model path to override, or --no-model-rerun to skip "
                             "re-running inference and just load existing detections_tiled.json.")
    parser.add_argument("--no-model-rerun", action="store_true",
                        help="Skip re-running inference even though --model has a hardcoded "
                             "default; just load whatever is already in the detections JSON.")
    parser.add_argument("--radius", type=int, default=250,
                        help="Text association search radius in pixels (default: 250)")
    parser.add_argument("--no-db", action="store_true",
                        help="Skip DB write, output CSV only")
    parser.add_argument("--no-rotated-recheck", action="store_true",
                        help="Disable the targeted rotated re-check (90/180/270 "
                             "degree crop-and-retry) for instrument symbols that "
                             "got zero associated text from the full-page pass. "
                             "On by default. Disable for a faster run once you've "
                             "confirmed it isn't finding anything on a given page.")
    parser.add_argument("--no-review-queue", action="store_true",
                        help="Disable review-queue detection (tag-shaped text "
                             "near a misrouted or missing symbol). On by default.")
    parser.add_argument("--max-orphan-dist", type=int, default=MAX_ORPHAN_DIST_DEFAULT,
                        help=f"Drop ORPHANED_TAG_CANDIDATE review-queue rows whose "
                             f"nearest_dist_px exceeds this (default: "
                             f"{MAX_ORPHAN_DIST_DEFAULT}px). Confirmed false-positive "
                             f"source (revision-table/general-notes text) sits at "
                             f"880-5400px on real data; genuine candidates sit at "
                             f"280-620px. Dropped rows are written to "
                             f"dropped_candidates.csv for audit — check that file "
                             f"after tuning this value.")
    parser.add_argument("--no-pipe-spec-filter", action="store_true",
                        help="Disable the pipe/line-spec text filter (flange "
                             "ratings like 600#RF, fractional/bare inch sizes) "
                             "for ORPHANED_TAG_CANDIDATE and MISROUTED_CANDIDATE "
                             "rows. On by default.")
    parser.add_argument("--zone-ocr", action="store_true",
                        help="Force zone-OCR ONLY, no escalation, ever — even if "
                             "some instruments end up MISSING. Skips full-page "
                             "OCR entirely: for each instrument symbol, crop and "
                             "OCR only the region directly around it "
                             "(above/right first, then left/below as fallback); "
                             "non-instrument symbols are never OCR'd at all. "
                             "--export-all-text and the review queue are "
                             "unavailable in this mode. For the normal "
                             "'zone first, escalate only if incomplete' "
                             "behavior, don't pass this — that's the default "
                             "(auto mode).")
    parser.add_argument("--no-auto", action="store_true",
                        help="Disable auto mode (zone-OCR first, auto-escalate "
                             "to resize-OCR + rescue only if incomplete), which "
                             "is the default whenever none of --zone-ocr/"
                             "--resize-ocr/--tile-ocr/--single-pass-ocr is "
                             "explicitly passed. With --no-auto, plain full-page "
                             "OCR (tiled by default for large pages) runs "
                             "unconditionally instead, same as the old default "
                             "behavior before auto mode existed.")
    parser.add_argument("--no-rescue", action="store_true",
                        help="Auto mode only: skip the rescue (low-conf/loose-NMS "
                             "YOLO re-check) pass on orphaned candidates during "
                             "escalation. On by default — rescue only costs "
                             "anything when zone-ocr was incomplete AND the "
                             "resize-ocr pass actually found orphaned tag-shaped "
                             "text with no nearby YOLO detection.")
    parser.add_argument("--rescue-model", type=str, default=None,
                        help="Model to use for the rescue pass during auto-mode "
                             "escalation. Defaults to --model if given, else "
                             "models/best.pt. Independent of the main detection "
                             "model so you can rescue-scan with a different "
                             "checkpoint if needed.")
    parser.add_argument("--rescue-conf", type=float, default=0.10,
                        help="Auto mode only: rescue confidence threshold "
                             "(default 0.10 — deliberately low, same default as "
                             "rescue_low_conf.py's standalone CLI).")
    parser.add_argument("--rescue-iou", type=float, default=0.75,
                        help="Auto mode only: rescue NMS IoU threshold (default "
                             "0.75 — deliberately high so densely packed, "
                             "genuinely-separate bubbles aren't suppressed as "
                             "'duplicates').")
    parser.add_argument("--rescue-pad", type=int, default=350,
                        help="Auto mode only: padding (px) around each orphaned "
                             "tag's bbox for the rescue crop (default 350).")
    parser.add_argument("--auto-resize-target", type=int, default=None,
                        metavar="TARGET_PX",
                        help="Auto mode only: resize-OCR target px to use during "
                             "escalation (default: RESIZE_OCR_TARGET_DEFAULT, "
                             f"currently {RESIZE_OCR_TARGET_DEFAULT}). Same "
                             "parameter as --resize-ocr but scoped to the "
                             "auto-escalation step specifically.")
    parser.add_argument("--tile-fallback", action="store_true",
                        help="Auto mode only: use tiled OCR as the Pass-2 "
                             "escalation strategy instead of the default "
                             "resize-OCR. This is the OLD default (before "
                             "2026-07-16) — use this if resize-OCR crashes "
                             "silently on a page (known risk, see comment "
                             "in process_page()) and you need a proven-"
                             "stable fallback instead. Does not affect "
                             "--tile-ocr, which still forces tiling for "
                             "the whole page unconditionally, bypassing "
                             "auto mode / zone-OCR entirely.")
    parser.add_argument("--zone-depth", type=int, default=ZONE_DEPTH,
                        help=f"--zone-ocr/auto only: how far (px) to search in each "
                             f"direction (above/right/left/below) from the "
                             f"instrument bbox (default: {ZONE_DEPTH}). Raise "
                             f"this if RAW/MISSING tags are actually just "
                             f"sitting further from the bubble than the "
                             f"current depth reaches.")
    parser.add_argument("--zone-margin", type=int, default=ZONE_MARGIN,
                        help=f"--zone-ocr/auto only: padding (px) perpendicular to "
                             f"the search direction (default: {ZONE_MARGIN}). "
                             f"Keep this small on densely-packed pages — too "
                             f"large and a zone crop starts sweeping in a "
                             f"neighboring symbol's tag instead of this one's.")
    parser.add_argument("--no-zone-rotated-recheck", action="store_true",
                        help="--zone-ocr/auto only: disable the per-symbol rotated "
                             "re-check (90/180/270 degree crop-and-retry) that "
                             "runs after depth escalation still finds nothing "
                             "VALID. On by default.")
    parser.add_argument("--exclude-zone", action="append", default=None,
                        metavar="X1,Y1,X2,Y2",
                        help="Page-pixel rectangle to exclude entirely from "
                             "detections AND OCR text (title block, BOM/legend "
                             "table, general-notes box). Repeatable for "
                             "multiple regions. Stacks on top of the built-in "
                             "left-margin default (see --no-default-exclude-"
                             "zones to turn that off). Coordinates are in the "
                             "same space as detections_*.json. This is the "
                             "biggest single lever for cutting tag false-positives per "
                             "the P&ID digitization literature — find the box "
                             "once per drawing template by eyeballing a page.")
    parser.add_argument("--no-default-exclude-zones", action="store_true",
                        help="Disable the built-in default exclude-zone(s) "
                             "(currently: the Haifa template's left-margin "
                             "border/tick-mark strip). On by default so you "
                             "don't have to pass --exclude-zone 0,0,25,10000 "
                             "by hand on every run.")
    parser.add_argument("--exclude-tags-file", type=str, default=None,
                        help="Path to a text file, one tag/prefix per line "
                             "(e.g. 'H-10'), of confirmed non-instrument "
                             "strings that are structurally indistinguishable "
                             "from real ISA tags (drawing-own unit/area "
                             "prefixes, etc. — see bug history). Only applied "
                             "to the review-queue's whole-page scan, never to "
                             "tags read near a confirmed instrument symbol. "
                             "Grow this file from each run's flagged output "
                             "instead of re-guessing regex rules.")
    parser.add_argument("--export-all-text", action="store_true",
                        help="Also export EVERY OCR text block on the page (not just "
                             "those associated to an instrument symbol) to "
                             "all_text_v1.csv, with nearest-symbol context and an "
                             "orphaned flag. Use this to audit what the pipeline is "
                             "currently discarding (title block, notes, line numbers, "
                             "valve tags, etc). Reads from the OCR cache — no extra "
                             "OCR cost if the page's cache already exists.")
    parser.add_argument("--ocr-only", action="store_true",
                        help="Run/cache OCR only, skip association+tagging+DB. "
                             "Use this as step 1 (e.g. overnight) to build the "
                             "OCR cache for all pages without waiting on the "
                             "rest of the pipeline.")
    parser.add_argument("--recompute-ocr", action="store_true",
                        help="Ignore any cached OCR result and re-run PaddleOCR "
                             "for these pages (writes a fresh cache entry).")
    parser.add_argument("--tile-ocr", action="store_true",
                        help="Force tile-based OCR (this is now the default for "
                             "pages over the single-pass safe limit — this flag "
                             "is kept for explicitness/back-compat and to force "
                             "tiling even if you also pass --single-pass-ocr).")
    parser.add_argument("--single-pass-ocr", action="store_true",
                        help="Force one PaddleOCR call on the full page (raised "
                             "limit_side_len) instead of tiling. This was the "
                             "old silent default and IS THE CRASH CAUSE on "
                             "CPU-only machines for large pages (e.g. 9934x7017 "
                             "~70MP silently exhausts RAM, killed with no "
                             "traceback). Only use this if you have a GPU or "
                             "confirmed spare RAM headroom.")
    parser.add_argument("--resize-ocr", type=int, nargs="?", const=RESIZE_OCR_TARGET_DEFAULT,
                        default=None, metavar="TARGET_PX",
                        help=f"THIRD OCR strategy, opt-in, takes priority over --tile-ocr/"
                             f"--single-pass-ocr if also passed. Shrinks the page so its "
                             f"longest side is TARGET_PX (default {RESIZE_OCR_TARGET_DEFAULT} "
                             f"if you pass --resize-ocr with no number), OCRs it in one pass, "
                             f"then rescales all text-block coordinates back to original "
                             f"resolution. Real memory/runtime win over tiling (one OCR call, "
                             f"no tile-overlap dedup) at the cost of shrinking small tag text "
                             f"too — test on a sample page and compare VALID/RAW/MISSING "
                             f"counts against tiling before making this the production default.")
    parser.add_argument("--detections-json", type=str, default=None,
                        help="Path to detections JSON (e.g. data/haifa_real_pids/"
                             "results_tiled_v2/detections_all_sets.json for Haifa real data)")
    parser.add_argument("--images-dir", type=str, default=None,
                        help="Root images folder (e.g. data/haifa_real_pids/images "
                             "for Haifa real data - SET subfolders are searched automatically)")
    parser.add_argument("--dataset-yaml", type=str,
                        default=str(Path("data") / "unified_dataset" / "dataset.yaml"),
                        help="Path to dataset.yaml — used as the canonical class_id->name "
                             "mapping, since the detections JSON's own class_name field can "
                             "be stale. Default: data/unified_dataset/dataset.yaml")
    args = parser.parse_args()

    global DETECTIONS_JSON, IMAGES_DIR
    if args.detections_json:
        DETECTIONS_JSON = Path(args.detections_json)
    if args.images_dir:
        IMAGES_DIR = Path(args.images_dir)

    if args.exclude_tags_file:
        load_excluded_tags(args.exclude_tags_file)

    # Load canonical class_id -> name mapping (dataset.yaml is ground truth;
    # never trust the detections JSON's own class_name field, it can be stale).
    yaml_path = Path(args.dataset_yaml)
    if yaml_path.exists():
        load_canonical_class_names(yaml_path)
    else:
        print(f"\n\u26a0\ufe0f  dataset.yaml not found at {yaml_path} — class names will NOT be "
              f"corrected, routing may be wrong if the detections JSON has stale class_name "
              f"values. Pass --dataset-yaml to point at the right file.\n")

    # If --model specified (hardcoded default: best_merged.pt), re-run inference
    # first via the v2 batch/tiled script (4-zone + resize), NOT the older
    # single-pass inference_tiled.py — that mismatch was silently causing far
    # fewer detections than expected.
    if args.model and not args.no_model_rerun:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            print("Available models:")
            for m in Path("models").glob("*.pt"):
                print(f"  {m}")
            return
        print(f"Re-running inference with {model_path} via {INFERENCE_SCRIPT}...")
        import subprocess, sys
        subprocess.run([sys.executable, INFERENCE_SCRIPT,
                        "--model", str(model_path)], check=True)
        print("Inference complete. Loading fresh detections...")

    # Load detections
    if not DETECTIONS_JSON.exists():
        print(f"Detections JSON not found: {DETECTIONS_JSON}")
        print("Run inference_tiled.py first.")
        return

    with open(DETECTIONS_JSON, encoding="utf-8") as f:
        raw_detections = json.load(f)

    # Haifa's combined detections_all_sets.json is nested one level deeper
    # ({"SET 1": {"SET 1_page_1": [...], ...}, "SET 2": {...}}) vs. the
    # original flat layout ({"pid_pdf_page_01": [...]}). Detect and flatten.
    is_nested = bool(raw_detections) and all(
        isinstance(v, dict) for v in raw_detections.values()
    )
    if is_nested:
        all_detections = {}
        for set_name, pages in raw_detections.items():
            all_detections.update(pages)
        print(f"Detected nested SET-level JSON — flattened {len(raw_detections)} "
              f"sets into {len(all_detections)} pages.")
    else:
        all_detections = raw_detections

    # Resolve page list
    if args.page:
        page_names = [args.page]
    elif args.set:
        import re as _re
        pattern = _re.compile(rf"^{_re.escape(args.set)}_page_(\d+)$")
        matched = []
        for name in all_detections.keys():
            m = pattern.match(name)
            if m:
                matched.append((int(m.group(1)), name))
        if not matched:
            print(f"No pages matched --set \"{args.set}\" (expected names like "
                  f"\"{args.set}_page_1\") — check the exact set name, e.g. \"SET 1\" not \"Set 1\".")
            return
        page_names = [name for _, name in sorted(matched)]
    elif args.all:
        page_names = sorted(all_detections.keys())
    else:
        parser.print_help()
        return

    use_db = not args.no_db

    all_rows      = []
    all_text_rows = []
    all_review_rows = []
    all_dropped_rows = []
    total_valid  = 0
    total_raw    = 0
    total_miss   = 0
    total_skip   = 0
    total_inst   = 0
    total_recovered = 0
    total_rescued = 0
    start        = datetime.now()

    print(f"\n{'='*55}")
    print(f"  PipeSight AI — Tag Extractor v2")
    print(f"  Pages: {len(page_names)}  |  Search radius: {args.radius}px")
    print(f"{'='*55}\n")

    # Auto mode is the default: zone-OCR first, auto-escalate to
    # resize-OCR + rescue only if incomplete (--tile-fallback swaps that
    # escalation step to tiled OCR instead, without leaving auto mode —
    # see process_page()'s escalate_tile comment for why resize-OCR is
    # the default despite its known silent-crash history on this machine).
    # Any explicit single-strategy flag (--zone-ocr forced-only,
    # --resize-ocr, --tile-ocr, --single-pass-ocr) or --no-auto opts back
    # into the old "always run exactly the strategy I named, no
    # escalation" behavior.
    explicit_strategy = (args.zone_ocr or args.resize_ocr is not None
                          or args.tile_ocr or args.single_pass_ocr)
    use_auto = (not args.no_auto) and (not explicit_strategy)
    if use_auto:
        escalation_desc = "tiled OCR (--tile-fallback)" if args.tile_fallback else "resize-OCR"
        print(f"  Strategy: AUTO (zone-OCR first, then always {escalation_desc} for "
              f"review-queue coverage — rescue only if incomplete; pass "
              f"--no-review-queue for the old zone-ocr-only fast exit) — pass "
              f"--zone-ocr/--resize-ocr/--tile-ocr/--single-pass-ocr/--no-auto "
              f"to override.\n")

    exclude_zones = [] if args.no_default_exclude_zones else list(DEFAULT_EXCLUDE_ZONES)
    if args.exclude_zone:
        for z in args.exclude_zone:
            try:
                x1, y1, x2, y2 = (int(v.strip()) for v in z.split(","))
                exclude_zones.append((x1, y1, x2, y2))
            except ValueError:
                print(f"  WARNING: could not parse --exclude-zone '{z}' "
                      f"(expected X1,Y1,X2,Y2) — skipping it")
    if exclude_zones:
        print(f"  {len(exclude_zones)} exclude-zone(s) active: {exclude_zones}\n")

    for page_name in page_names:
        rows, text_rows, review_rows, dropped_rows = process_page(page_name, all_detections,
                            use_db=use_db, search_radius=args.radius,
                            force_recompute_ocr=args.recompute_ocr,
                            ocr_only=args.ocr_only,
                            force_tiling=args.tile_ocr,
                            force_single_pass=args.single_pass_ocr,
                            resize_target=args.resize_ocr,
                            export_all_text=args.export_all_text,
                            rotated_recheck=not args.no_rotated_recheck,
                            review_queue=not args.no_review_queue,
                            zone_ocr=args.zone_ocr,
                            zone_depth=args.zone_depth,
                            zone_margin=args.zone_margin,
                            zone_rotated_recheck=not args.no_zone_rotated_recheck,
                            exclude_zones=exclude_zones,
                            max_orphan_dist=args.max_orphan_dist,
                            filter_pipe_spec=not args.no_pipe_spec_filter,
                            auto=use_auto,
                            auto_resize_target=args.auto_resize_target,
                            escalate_tile=args.tile_fallback,
                            rescue_on_escalate=not args.no_rescue,
                            rescue_model_path=args.rescue_model or args.model,
                            rescue_pad=args.rescue_pad,
                            rescue_conf=args.rescue_conf,
                            rescue_iou=args.rescue_iou)
        all_rows.extend(rows)
        all_text_rows.extend(text_rows)
        all_review_rows.extend(review_rows)
        all_dropped_rows.extend(dropped_rows)
        for r in rows:
            if r["tag_status"] == "VALID":   total_valid += 1
            elif r["tag_status"] == "RAW":   total_raw   += 1
            elif r["tag_status"] == "MISSING": total_miss += 1
            elif r["tag_status"] == "SKIPPED": total_skip += 1
            elif r["tag_status"] == "RECOVERED_ROTATED": total_recovered += 1
            if r.get("_rescued"):                        total_rescued += 1
            if r["routing"] == "instrument":   total_inst += 1

    elapsed = (datetime.now() - start).total_seconds()

    if args.ocr_only:
        print(f"\nOCR cache build complete for {len(page_names)} pages "
              f"in {elapsed:.1f}s. Cache dir: {OCR_CACHE_DIR}")
        print("Re-run without --ocr-only to do association+tagging — "
              "it will read from cache and skip PaddleOCR entirely.")
        return

    # Write combined CSV
    csv_out = OUT_DIR / "tags_v2.csv"
    write_csv(all_rows, csv_out)

    all_text_csv_out = None
    if args.export_all_text:
        all_text_csv_out = OUT_DIR / "all_text_v1.csv"
        write_all_text_csv(all_text_rows, all_text_csv_out)
        orphaned_total = sum(1 for r in all_text_rows if not r["within_radius"])
        print(f"  All-text export : {len(all_text_rows)} OCR blocks "
              f"({orphaned_total} orphaned, i.e. not within {args.radius}px of "
              f"any symbol) -> {all_text_csv_out}")

    review_csv_out = None
    if all_review_rows:
        review_csv_out = OUT_DIR / "review_queue.csv"
        write_review_queue_csv(all_review_rows, review_csv_out)

    dropped_csv_out = None
    if all_dropped_rows:
        dropped_csv_out = OUT_DIR / "dropped_candidates.csv"
        write_dropped_candidates_csv(all_dropped_rows, dropped_csv_out)

    valid_pct = round(total_valid / total_inst * 100, 1) if total_inst else 0

    print(f"\n{'='*55}")
    print(f"  PipeSight AI — Tag Extraction v2 Complete")
    print(f"{'='*55}")
    print(f"  Pages processed      : {len(page_names)}")
    print(f"  Instrument symbols   : {total_inst}")
    print(f"  VALID tags           : {total_valid}  ({valid_pct}% of instruments)")
    print(f"  RAW  (not ISA)       : {total_raw}")
    print(f"  MISSING              : {total_miss}")
    print(f"  RECOVERED (rotated)  : {total_recovered}")
    print(f"  RESCUED (auto mode)  : {total_rescued}")
    print(f"  SKIPPED              : {total_skip}")
    if all_review_rows:
        misrouted_n = sum(1 for r in all_review_rows if r["kind"] == "MISROUTED_CANDIDATE")
        orphaned_n  = len(all_review_rows) - misrouted_n
        print(f"  Review queue         : {len(all_review_rows)} rows "
              f"({misrouted_n} misrouted, {orphaned_n} orphaned-tag) -> {review_csv_out}")
    if all_dropped_rows:
        print(f"  Filtered candidates  : {len(all_dropped_rows)} rows "
              f"(pipe-spec text / exceeds --max-orphan-dist {args.max_orphan_dist}px) "
              f"-> {dropped_csv_out}")
    print(f"  Time                 : {elapsed:.1f}s")
    print(f"  Output               : {csv_out}")
    print(f"{'='*55}")

    # Note on model switching
    print(f"\nTo use 61-class model when training completes:")
    print(f'  python tag_extractor_v2.py --page "SET 1_page_1" --model models\\best_61class.pt')


if __name__ == "__main__":
    main()