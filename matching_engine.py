# matching_engine.py - PipeSight AI
#
# The "spatial matching engine" named in the supervisor's task list — pulled
# out of tag_extractor_v2.py into its own module so it's a visible, separately
# testable component instead of logic embedded in a 2000+ line CLI script.
#
# This module owns everything that decides, for a given OCR text candidate,
# (a) which symbol (if any) it belongs to, (b) whether the resulting tag is
# ISA-valid, (c) what class of symbol it's near and whether that's even a
# sane pairing, and (d) how much to trust the result overall. Concretely,
# that's four signals combining into one score:
#
#   1. Distance     — centroid distance between text and symbol (zone/radius
#                      search, see associate_text_to_symbols / nearest_symbol)
#   2. OCR confidence — from PaddleOCR, see pick_best_text
#   3. ISA validity  — regex-pattern match against ISA S5.1 tag shapes,
#                      see normalize_tag / normalize_tag_strict
#   4. Class rules   — is the nearby symbol even an instrument class, see
#                      get_routing / find_review_queue_rows
#
# combined_confidence() is where signals 1+2 converge into a single number;
# find_review_queue_rows() is where signal 4 (class rules) and signal 1
# (distance) together decide MISROUTED vs ORPHANED vs "normal pipeline, skip".
#
# Everything here is pure logic — no file I/O, no OCR execution, no DB
# writes, no CLI. tag_extractor_v2.py imports from this module and stays
# responsible for orchestration (running OCR, walking pages, writing
# CSVs/DB rows).

import re
import math
from pathlib import Path
from collections import defaultdict

# ─── CLASS ROUTING ───────────────────────────────────────────────────────────
# 32-class model class names (original)
# NOTE: "Box" was previously (wrongly) included here. Real-data check against
# SET 1_page_1 showed every single false-positive tag match (N-22 from a
# nitrogen-purge note, NOTE-71/NOTE-718 from note callouts, TO-1363 from an
# equipment-list ref, CSO-18/CSO-10131 from valve-status labels + nearby WBS
# numbers, H-10 from the drawing's own unit prefix) came from a "Box"-class
# detection — a generic label/reference box, not an instrument bubble. Every
# correct match (PDIT-1002, PIT-1003, TIT-1003, PDI-1002, PDI-1002A, PG-1002)
# came from "Instrument_Field". Moved to REFERENCE_CLASSES below.
INSTRUMENT_CLASSES_32 = {"Instrument_Field", "Instrument_Panel", "Instrument_Aux_Panel"}

# 61-class model adds no new instrument classes — same 3 instrument types,
# just more mechanical/process classes that are all skipped for tag extraction.
# Update this set if new instrument-type classes are added in future.
INSTRUMENT_CLASSES = INSTRUMENT_CLASSES_32

# Generic label/reference boxes — equipment-list refs, WBS numbers, unit
# prefixes, note callouts. Routed separately (not "instrument", not
# "mechanical"/"structural" either — it's not a physical symbol) so it's
# visible in per-page counts and the all-text/review-queue audits rather
# than silently lumped in with either bucket.
REFERENCE_CLASSES = {"Box"}

MECHANICAL_CLASSES = {
    "Gate_Valve", "Ball_Valve", "Globe_Valve", "Butterfly_Valve", "Check_Valve",
    "Diaphragm_Valve", "Needle_Valve", "Plug_Valve", "Angle_Valve", "Knife_Valve",
    "Bleeder_Valve", "Rotary_Valve", "Pressure_Regulator", "Safety_Relief_Valve",
    "Control_Valve", "Hand_Operated_Valve", "Motor_Operated_Valve", "Solenoid_Valve",
    "Hydraulic_Valve", "Float_Operated_Valve",
    "Centrifugal_Pump", "Gear_Pump", "Screw_Pump", "Positive_Displacement_Pump", "Sump_Pump",
    "Centrifugal_Compressor", "Reciprocating_Compressor", "Rotary_Compressor",
    "Heat_Exchanger", "Air_Cooler", "Condenser", "Furnace",
    "Vessel", "Tank", "Column", "Drum", "Motor", "Turbine",
    "Filter", "Strainer", "Mixer", "Agitator", "Boiler",
    "Rupture_Disk", "Sight_Glass", "Paddle_Blind", "Spectacle_Blind",
    "Expansion_Joint", "Vacuum_Pump", "Ejector", "Cooling_Tower",
    # 32-class originals
    "Gate_Valve_NO", "Globe_Valve_NO",
}
STRUCTURAL_CLASSES = {
    "Reducer", "Flange_or_Nozzle", "Pipe_Insulation_or_Tracing",
    "Flow_Arrow", "Orifice_Plate", "Rotameter",
}

# ─── CANONICAL CLASS NAMES (dataset.yaml is ground truth) ────────────────────
# The detections JSON's own "class_name" field can be stale (confirmed: values
# like "symbol_57", "Not_used", "Globe_Valve_NC" turned out to be leftover
# placeholder/old-mapping names once cross-checked against dataset.yaml's
# actual class order by class_id — e.g. class_id=57 is really Instrument_Field,
# not "symbol_57"). Never trust the JSON's class_name string for routing;
# always re-derive it from class_id against dataset.yaml.
_CLASS_ID_TO_NAME = None

def load_canonical_class_names(dataset_yaml_path):
    global _CLASS_ID_TO_NAME
    import yaml
    with open(dataset_yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg["names"]
    _CLASS_ID_TO_NAME = {i: n for i, n in enumerate(names)}
    print(f"Loaded {len(_CLASS_ID_TO_NAME)} canonical class names from {dataset_yaml_path}")
    return _CLASS_ID_TO_NAME

def canonical_class_name(class_id, fallback_name):
    if _CLASS_ID_TO_NAME is None:
        return fallback_name  # dataset.yaml not loaded — caller should have loaded it
    return _CLASS_ID_TO_NAME.get(class_id, fallback_name)

def get_routing(class_name):
    if class_name in INSTRUMENT_CLASSES:  return "instrument"
    if class_name in REFERENCE_CLASSES:   return "reference"
    if class_name in MECHANICAL_CLASSES:  return "mechanical"
    if class_name in STRUCTURAL_CLASSES:  return "structural"
    return "unknown_class"

# ─── ISA TAG NORMALIZATION ────────────────────────────────────────────────────
# Mirrors v1 normalization logic exactly.
OCR_CORRECTIONS = {
    'O': '0', 'I': '1', 'B': '8', 'S': '5',
    'o': '0', 'i': '1', 'b': '8', 's': '5',
}

ISA_PATTERNS = [
    # Standard: FT-101, PT-203A
    re.compile(r'^([A-Z]{1,4})-(\d{2,5}[A-Z]?)$'),
    # No dash: FT101, PT203A
    re.compile(r'^([A-Z]{1,4})(\d{2,5}[A-Z]?)$'),
    # Loop prefix: 1-FT-101
    re.compile(r'^(\d+)-([A-Z]{1,4})-(\d{2,5}[A-Z]?)$'),
    # Single letter + number: V-32, V32
    re.compile(r'^([A-Z])-?(\d{1,5}[A-Z]?)$'),
]

def apply_ocr_corrections(text):
    result = []
    for ch in text:
        result.append(OCR_CORRECTIONS.get(ch, ch))
    return ''.join(result)

def _try_isa_patterns(candidate):
    for pat in ISA_PATTERNS:
        m = pat.match(candidate)
        if m:
            groups = m.groups()
            return '-'.join(g for g in groups if g)
    return None

# Known P&ID abbreviations that match the ISA letter-prefix shape but are
# NOT instrument function codes — confirmed from real false-positive matches
# on SET 1_page_1 (CSO/CSC = Car Seal Open/Closed valve status, NOTE = note
# callout, TO/FROM = flow-direction labels, H10 = this drawing's own unit
# prefix). Extend this set as new false-positive words turn up in review.
NON_INSTRUMENT_ABBREVIATIONS = {
    "CSO", "CSC", "NC", "NO", "TO", "FROM", "NOTE", "MIN", "MAX", "TYP",
    "LOCAL", "REV", "SHT", "IFC", "GD", "WBS", "REF",
    # Added 2026-07-13, review-queue audit on SET 1 (122-row post-fix queue):
    # "MOC" (Motor-Operated-Closed, same family as CSO/CSC) was producing
    # MOCC-0011 / MOCA-0062 / MOCV-11 by merging with adjacent digits/letters
    # (e.g. "MOC C0011" -> MOCC-0011). "FC" (Fail-Closed) and "TSO" (valve
    # status callout) were producing FCT-50 / TSOV-11 / FCV-11 the same way —
    # FC/TSO sit next to a valve as a status label, not a tag, but "TSO"
    # survives the OCR-correction pass (S->5, O->0 => "T50") and merges with
    # an adjacent word into something ISA-shaped.
    "MOC", "FC", "TSO",
}

def _try_isa_patterns_strict(candidate):
    """
    Same as _try_isa_patterns but excludes the single-letter catch-all
    pattern (^([A-Z])-?(\\d{1,5}[A-Z]?)$). That pattern matches valve/spec/
    grid references (V16, A24, B24...) just as readily as real single-
    letter instrument tags, and without spatial context to disambiguate
    (see _tag_shaped_candidates, which scans the WHOLE page rather than
    only text near an instrument symbol) it's the dominant source of
    false-positive review-queue flags. Used for review-queue detection
    only — normalize_tag() itself still uses all patterns, since there
    real single-letter tags near a confirmed instrument symbol are legit.

    Also rejects any match whose letter-prefix is a known non-instrument
    P&ID abbreviation (NON_INSTRUMENT_ABBREVIATIONS) — checked against
    every all-letter run in the candidate, not just the first regex group,
    since group order differs across the loop-prefix pattern.
    """
    for pat in ISA_PATTERNS[:3]:
        m = pat.match(candidate)
        if not m:
            continue
        letter_runs = re.findall(r'[A-Z]+', candidate)
        if any(run in NON_INSTRUMENT_ABBREVIATIONS for run in letter_runs):
            continue
        groups = m.groups()
        # BUG (found via smoke test on 'S1 R1'): slicing to ISA_PATTERNS[:3]
        # only removes the *dedicated* single-letter catch-all (pattern
        # index 3) — but patterns 0-2 all declare their letter-prefix group
        # as [A-Z]{1,4}, which already accepts a single letter on its own.
        # So a single-letter tag can still slip through patterns 0-2.
        # Confirmed concretely: the OCR-correction pass turns 'S1' into
        # '51' (S->5), which then concatenates with the adjacent word 'R1'
        # into 'R151' — matching the no-dash pattern (letters='R',
        # digits='151') even though 'R' is a single letter. Same class of
        # false positive as the V16/A24/B24 case above, just reached via
        # the correction pass instead of the raw pass. Reject explicitly
        # here rather than relying on which pattern produced the match.
        letters_group = next((g for g in groups if g and g.isalpha()), None)
        if letters_group is not None and len(letters_group) < 2:
            continue
        return '-'.join(g for g in groups if g)
    return None

def is_loop_prefix_only(tag):
    """
    True if a VALID tag's shape could only have come from the loop-prefix
    ISA pattern (digit-LETTERS-digit, e.g. 1-FT-101) — identifiable by the
    tag string having exactly 2 dashes, since every other ISA_PATTERNS
    entry produces exactly 1 dash in its joined output (see
    _try_isa_patterns' `'-'.join(...)`). Confirmed real false positive:
    det124 on SET 1_page_1 matched '10010-GAV-3607' this way from a zone
    crop that swept in a neighboring Gate_Valve reference ('...GAV-3607')
    merged with an unrelated '100' fragment — coincidentally loop-prefix
    shaped, not a real tag. Same treatment as single-letter-only matches
    (see normalize_tag_strict): forced to HUMAN_REQUIRED, never
    auto-accepted, regardless of confidence score.
    """
    return bool(tag) and tag.count('-') == 2

# Drawing-specific confirmed non-instrument strings that are structurally
# identical to a real ISA tag (e.g. 'H-10', this drawing's own unit/area
# prefix — see Bug History #1) — no regex can distinguish these from a
# genuine tag by shape alone, since they ARE that shape. Populated from
# --exclude-tags-file; only applied to the review-queue's free whole-page
# scan (find_review_queue_rows), never to tags read near a confirmed
# instrument symbol, where the same string is much more likely genuine.
EXCLUDED_TAG_PREFIXES = set()

def load_excluded_tags(path):
    global EXCLUDED_TAG_PREFIXES
    p = Path(path)
    if not p.exists():
        print(f"  WARNING: --exclude-tags-file '{path}' not found, ignoring")
        return
    with open(p, encoding="utf-8") as f:
        EXCLUDED_TAG_PREFIXES = {
            line.strip().upper() for line in f if line.strip() and not line.startswith("#")
        }
    print(f"  Loaded {len(EXCLUDED_TAG_PREFIXES)} excluded tag prefix(es) "
          f"from {path}")

def is_excluded_tag(tag):
    if not tag or not EXCLUDED_TAG_PREFIXES:
        return False
    tag = tag.upper()
    return any(tag == p or tag.startswith(p + "-") for p in EXCLUDED_TAG_PREFIXES)

# Distance and pipe-spec filters for the review queue's free whole-page scan
# (find_review_queue_rows only — never touches tags read near a confirmed
# instrument symbol). Added after auditing SET 1_page_1's 49 "STILL_MISSING"
# rescue results against review_queue.csv: ~35-39 of the 49 turned out to be
# false positives from three sources — (a) OCR reading straight across the
# revision-table/general-notes text sitting 880-5400px from the nearest
# symbol, (b) pipe/line-spec callouts (flange ratings, nominal pipe sizes)
# sitting close to a symbol but never meant to BE a tag, (c) legend/setpoint
# table entries. This filter targets (a) and (b), which are cleanly
# separable by distance and text shape respectively; (c) is not fully
# resolved here (BARG-57/C-153 legend rows sit at 670-840px, straddling any
# reasonable single cutoff) and may need a follow-up heuristic — e.g.
# detecting a repeating tabular legend region — if it shows up as a large
# share of a future page's flagged rows.
MAX_ORPHAN_DIST_DEFAULT = 800

PIPE_SPEC_PATTERN = re.compile(
    r'#\s*(RF|FF|RTJ)\b'          # flange rating: 600#RF, 150# FF, #RTJ
    r'|\d+\s*-\s*\d+/\d+\s*"'     # fractional inch size: 1-1/2"
    r'|(?<!\w)\d+"'               # bare inch size: 4"
    , re.I
)

def is_pipe_spec_text(raw_text):
    """
    True if raw_text looks like a pipe/line specification callout (flange
    rating, nominal pipe size) rather than an instrument tag. Confirmed
    false-positive source on SET 1_page_1: P-1, P-13A, N-2, N-24, N-380,
    S-14, W-10131 etc. all normalized from text like "P1 600#RF",
    "N2 600#RF", "S1 4\" 600# RF" — line/pipe spec text sitting near a
    flange, reducer, or purge connection, not an ISA instrument tag.
    """
    return bool(raw_text) and bool(PIPE_SPEC_PATTERN.search(raw_text))

# Added 2026-07-13, review-queue audit on SET 1: PSV/control-valve setpoint
# annotations ("SET @ 57 barg", "SET @ 3.5 barg") were repeatedly
# normalizing to a spurious tag "BARG-57" / "BARG-1078" — the unit word
# "barg" (bar-gauge) matches the single-letter-prefix-plus-digits shape
# just like a real tag does, and it's the dominant term in the phrase, so
# no amount of word-pair-order/dash tuning avoids it. Distinct from
# is_pipe_spec_text: this is a pressure/temperature setpoint callout, not a
# line spec. Confirmed on SET 1_page_1/page_3, 7 occurrences across both
# pages, always co-located with a Control_Valve/Gate_Valve (the PSV itself).
SETPOINT_ANNOTATION_PATTERN = re.compile(
    r'SET\s*@\s*[\d.]+\s*(BARG|PSIG|PSI|BAR|KPA|MPA)\b', re.I
)

def is_setpoint_annotation(raw_text):
    """
    True if raw_text looks like a pressure/temperature setpoint callout
    (e.g. "SET @ 57 barg") rather than an instrument tag.
    """
    return bool(raw_text) and bool(SETPOINT_ANNOTATION_PATTERN.search(raw_text))

def reference_noise_reason(raw_text):
    """
    Returns a short reason string if raw_text matches one of the three
    confirmed non-instrument patterns already used by
    find_review_queue_rows() (pipe/line spec, setpoint annotation, known
    non-instrument abbreviation), else None.

    Previously these three checks only ran inside find_review_queue_rows()
    — the whole-page free scan. The near-symbol extraction paths in
    tag_extractor_v2.py (extract_tag_zoned / _run_zone_ocr_pass /
    process_page's full-page association loop) call the permissive
    normalize_tag() directly and never ran raw_ocr through any of these
    three checks, even though the exact same false-positive text can sit
    right next to a confirmed instrument symbol — confirmed on real data:
    a diamond line-spec symbol ('15/13' with '-10', '2"-H10-NI-...',
    'H10-HCX-036', 'N2' purge note nearby) normalized to B-10/B-13/B-80,
    and a 4" 600#RF flange spec with a line number + flow-direction
    letter normalized to N-10131/W-10131 — both near a real symbol, both
    single-letter-prefix "tags" that are actually spec/reference noise,
    not ambiguous OCR. This function lets the near-symbol paths apply the
    same "not a tag" definition instead of only ever seeing shape-based
    single-letter/loop-prefix low-trust signals.
    """
    if not raw_text:
        return None
    if is_pipe_spec_text(raw_text):
        return "pipe_or_line_spec_text"
    if is_setpoint_annotation(raw_text):
        return "setpoint_annotation"
    letter_runs = re.findall(r'[A-Z]+', raw_text.upper())
    if any(run in NON_INSTRUMENT_ABBREVIATIONS for run in letter_runs):
        return "non_instrument_abbreviation"
    return None

def _clean_candidate(c):
    return re.sub(r'[*\'\"\/\\]', '', c).strip()

def _build_merge_candidates(words):
    """
    Adjacent pairs FIRST, both orders, dash and no-dash — this is where a
    real split tag (loop number + function code) lives. Single words are
    tried LAST, as a fallback only: trying singles first lets an unrelated
    swept-in token (e.g. a nearby valve ref like "V16", which itself
    matches the permissive single-letter-plus-number pattern) win before
    the real pair ever gets checked.
    """
    cands = []
    for i in range(len(words) - 1):
        a, b = words[i], words[i + 1]
        cands += [f"{a}-{b}", f"{b}-{a}", f"{a}{b}", f"{b}{a}"]
    cands += list(words)
    return cands

def _build_pair_candidates(words):
    """Adjacent pairs only, both orders, dash and no-dash — no bare singles."""
    cands = []
    for i in range(len(words) - 1):
        a, b = words[i], words[i + 1]
        cands += [f"{a}-{b}", f"{b}-{a}", f"{a}{b}", f"{b}{a}"]
    return cands

def normalize_tag(raw_ocr):
    """
    Try to extract an ISA-conformant tag from raw OCR text.
    Returns (tag, status) where status is VALID, RAW, or MISSING.

    Tries RAW text against ISA patterns first, across every word and
    adjacent word-pair — only falls back to O/I/B/S OCR-correction if
    nothing raw matches. Applying corrections before matching (the
    previous behavior) silently destroyed legitimate all-letter function
    codes like PDIS/PSV/TIS before they ever reached the pattern check.

    The correction-fallback pass is further restricted to word PAIRS (plus
    single words that already contain a digit) — never a bare all-letter
    single word. A standalone function code with no digits anywhere isn't
    a complete tag; "correcting" it into one (e.g. PDIS -> PD-15) doesn't
    fix an OCR misread, it fabricates a loop number that was never read.
    """
    if not raw_ocr or not raw_ocr.strip():
        return None, "MISSING"

    words = [w for w in raw_ocr.upper().split() if w]
    if not words:
        return None, "MISSING"

    # Pass 1: raw text, no corrections. Pairs first, singles as fallback.
    for cand in _build_merge_candidates(words):
        tag = _try_isa_patterns(_clean_candidate(cand))
        if tag:
            return tag, "VALID"

    # Pass 2: OCR-correct, then retry — pairs only, plus single words that
    # already contain a digit (so there's actually something to correct).
    corrected_words = [apply_ocr_corrections(w) for w in words]
    pass2_candidates = _build_pair_candidates(corrected_words)
    if len(words) == 1 and any(c.isdigit() for c in words[0]):
        pass2_candidates.append(corrected_words[0])
    for cand in pass2_candidates:
        tag = _try_isa_patterns(_clean_candidate(cand))
        if tag:
            return tag, "VALID"

    # Didn't match ISA pattern — return cleaned raw text for human review
    cleaned = re.sub(r'\s+', '', raw_ocr.upper())
    cleaned = apply_ocr_corrections(cleaned)
    return cleaned, "RAW"

def normalize_tag_strict(raw_ocr):
    """
    Same algorithm as normalize_tag() but matches against
    _try_isa_patterns_strict (excludes the noisy single-letter catch-all)
    and drops known non-instrument abbreviations before candidate
    generation.

    Words in NON_INSTRUMENT_ABBREVIATIONS are removed from the word list
    up front, not checked post-hoc against the final candidate string —
    OCR correction can reshape a denylisted word enough to escape a
    string-level check (confirmed: "CSO" -> corrected "C50" -> merges
    with an adjacent number into "C5018", which no longer contains the
    substring "CSO" for a post-hoc check to catch). Removing it before
    any merging/correction happens closes that gap.
    """
    if not raw_ocr or not raw_ocr.strip():
        return None, "MISSING"
    words = [w for w in raw_ocr.upper().split() if w]
    if not words:
        return None, "MISSING"

    words = [w for w in words if w not in NON_INSTRUMENT_ABBREVIATIONS]
    if not words:
        cleaned = re.sub(r'\s+', '', raw_ocr.upper())
        return cleaned, "RAW"

    for cand in _build_merge_candidates(words):
        tag = _try_isa_patterns_strict(_clean_candidate(cand))
        if tag:
            return tag, "VALID"

    corrected_words = [apply_ocr_corrections(w) for w in words]
    pass2_candidates = _build_pair_candidates(corrected_words)
    if len(words) == 1 and any(c.isdigit() for c in words[0]):
        pass2_candidates.append(corrected_words[0])
    for cand in pass2_candidates:
        tag = _try_isa_patterns_strict(_clean_candidate(cand))
        if tag:
            return tag, "VALID"

    cleaned = re.sub(r'\s+', '', raw_ocr.upper())
    cleaned = apply_ocr_corrections(cleaned)
    return cleaned, "RAW"

# ─── TEXT-TO-SYMBOL ASSOCIATION ───────────────────────────────────────────────

def centroid(x1, y1, x2, y2):
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def distance(cx1, cy1, cx2, cy2):
    return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

def box_area(x1, y1, x2, y2):
    return max(0, x2 - x1) * max(0, y2 - y1)

def associate_text_to_symbols(text_blocks, instrument_symbols, search_radius=250):
    """
    For each instrument symbol, find all text blocks whose centroid falls within
    search_radius pixels of the symbol centroid. Return the closest block(s) per symbol.

    text_blocks: list of dicts with keys: text, conf, x1, y1, x2, y2
    instrument_symbols: list of detection dicts (x1, y1, x2, y2, conf, class_name, ...)
    Returns: dict mapping symbol index → list of associated text blocks, sorted by distance
    """
    associations = defaultdict(list)

    for sym_idx, sym in enumerate(instrument_symbols):
        scx, scy = centroid(sym["x1"], sym["y1"], sym["x2"], sym["y2"])

        for blk in text_blocks:
            bcx, bcy = centroid(blk["x1"], blk["y1"], blk["x2"], blk["y2"])
            dist = distance(scx, scy, bcx, bcy)
            if dist <= search_radius:
                associations[sym_idx].append({**blk, "_dist": dist})

        # Sort by distance
        associations[sym_idx].sort(key=lambda b: b["_dist"])

    return associations

def text_blocks_bbox(blocks):
    """
    Union bbox (page-space coords) covering a set of OCR text blocks —
    i.e. where the picked text actually sits on the page, as opposed to
    the symbol it got associated with. Returns None for an empty list.
    """
    if not blocks:
        return None
    xs = [b["x1"] for b in blocks] + [b["x2"] for b in blocks]
    ys = [b["y1"] for b in blocks] + [b["y2"] for b in blocks]
    return {"x1": min(xs), "y1": min(ys), "x2": max(xs), "y2": max(ys)}

def pick_best_text(text_blocks_near_symbol):
    """
    From candidate text blocks near a symbol, pick the best one for tag extraction.
    Strategy: prefer blocks that contain alphanumeric content, weighted by OCR conf
    and inverse distance. Returns (combined_text, avg_conf, text_bbox) — text_bbox
    is the union bbox (page-space) of the blocks actually picked, or None if
    nothing was picked. Callers should store this separately from the
    symbol's own bbox: it's what a reviewer needs highlighted to see WHICH
    text produced this candidate tag, not just which symbol it's near.
    """
    if not text_blocks_near_symbol:
        return None, 0.0, None

    # Filter out purely whitespace or single-char blocks
    candidates = [b for b in text_blocks_near_symbol
                  if b.get("text", "").strip() and len(b["text"].strip()) > 1]
    if not candidates:
        return None, 0.0, None

    # Take up to 3 nearest blocks and concatenate (handles split tags)
    top = candidates[:3]
    combined_text = " ".join(b["text"].strip() for b in top)
    avg_conf = sum(b["conf"] for b in top) / len(top)
    return combined_text, avg_conf, text_blocks_bbox(top)

def nearest_symbol(blk, detections):
    """
    Find the closest detection (any class/routing) to a text block's centroid.
    Returns (det_or_None, dist_or_None).
    """
    bcx, bcy = centroid(blk["x1"], blk["y1"], blk["x2"], blk["y2"])
    best_det, best_dist = None, None
    for det in detections:
        scx, scy = centroid(det["x1"], det["y1"], det["x2"], det["y2"])
        d = distance(scx, scy, bcx, bcy)
        if best_dist is None or d < best_dist:
            best_det, best_dist = det, d
    return best_det, best_dist

# ─── COMBINED CONFIDENCE ──────────────────────────────────────────────────────

def combined_confidence(yolo_conf, ocr_conf, tag_status):
    if tag_status == "SKIPPED":
        return 0.0
    if tag_status == "MISSING":
        return round(yolo_conf * 0.4, 4)
    return round(yolo_conf * 0.4 + ocr_conf * 0.6, 4)

AUTO_ACCEPT_THRESHOLD = 0.80

# ─── REVIEW QUEUE CLASS RULES (misrouted symbols + orphaned tags) ────────────
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

def _tag_shaped_candidates(text_blocks, pair_proximity=150):
    """
    Yields (tag, status, source_blocks) for every text_blocks entry (single
    block or adjacent pair) that normalize_tag_strict() rates VALID.
    source_blocks is a list of 1 or 2 blocks — used to derive a combined
    bbox/conf. Uses the strict (multi-letter-only) matcher deliberately —
    see normalize_tag_strict docstring for why.

    Blocks that are ALREADY independently VALID on their own are excluded
    from pairing (see BUG note below) — they're still yielded as singles.
    """
    n = len(text_blocks)
    already_valid_ids = set()
    # Single blocks
    for blk in text_blocks:
        tag, status = normalize_tag_strict(blk.get("text", ""))
        if status == "VALID":
            yield tag, status, [blk]
            already_valid_ids.add(id(blk))
    # Adjacent pairs (spatial proximity, not the same as instrument radius)
    for i in range(n):
        bi = text_blocks[i]
        if id(bi) in already_valid_ids:
            # BUG (found via SET 1_page_3 review_queue.csv audit): a block
            # that's already a complete, independently-valid tag (e.g.
            # "410-ALR-002") was still being paired with its neighbors.
            # Gluing it to an unrelated adjacent number ("1076") produces
            # "1076410-ALR-002", which happens to re-match the loop-prefix
            # ISA_PATTERN (digit-run + letters + digits) even though the
            # "1076" and "410-ALR-002" have nothing to do with each other —
            # two different pieces of drawing text that just happened to
            # land within pair_proximity. A block that's already a complete
            # tag on its own never legitimately needs to be merged with a
            # neighbor; skipping it here closes this false-positive class
            # without touching the genuine split-tag case (loop number +
            # function code as two separate blocks, e.g. "TI" + "1076"),
            # since neither half of a real split tag is independently VALID
            # under the strict matcher on its own.
            continue
        bicx, bicy = centroid(bi["x1"], bi["y1"], bi["x2"], bi["y2"])
        for j in range(i + 1, n):
            bj = text_blocks[j]
            if id(bj) in already_valid_ids:
                continue
            bjcx, bjcy = centroid(bj["x1"], bj["y1"], bj["x2"], bj["y2"])
            if distance(bicx, bicy, bjcx, bjcy) > pair_proximity:
                continue
            combined = f"{bi['text']} {bj['text']}"
            # BUG (found via SET 1_page_1 review_queue.csv audit): this used
            # to call normalize_tag() — the permissive matcher — instead of
            # normalize_tag_strict(). Single blocks below already use the
            # strict matcher, but this pair path didn't, so it let through
            # exactly the single-letter/non-instrument-abbreviation strings
            # normalize_tag_strict() exists to reject (e.g. "S1 R1" -> S-1,
            # "A1 REV" -> A-1, both revision-table leakage). Confirmed this
            # was the mechanism behind ~7 of the 49 "STILL_MISSING" rows.
            tag, status = normalize_tag_strict(combined)
            if status == "VALID":
                yield tag, status, [bi, bj]

def find_review_queue_rows(page_name, text_blocks, detections, radius,
                            max_orphan_dist=MAX_ORPHAN_DIST_DEFAULT,
                            filter_pipe_spec=True):
    """
    Builds MISROUTED_CANDIDATE and ORPHANED_TAG_CANDIDATE rows. Dedupes so
    the same underlying text doesn't get flagged twice (once as a single
    block, once as part of a pair).

    Returns (rows, dropped_rows). dropped_rows carries candidates that
    would otherwise have been ORPHANED_TAG_CANDIDATE rows but got rejected
    by either filter below, tagged with why — so they're visible in an
    audit CSV instead of silently vanishing, and the filter itself stays
    inspectable/tunable rather than a black box.

    max_orphan_dist: ORPHANED_TAG_CANDIDATE rows whose nearest_dist_px
        exceeds this are dropped (revision-table/general-notes leakage —
        confirmed on SET 1_page_1 to sit at 880-5400px, far past any real
        tag-to-symbol distance). Only applied to ORPHANED, never MISROUTED
        (misrouted candidates are by definition already close to a symbol).
    filter_pipe_spec: drop candidates whose raw_text matches
        is_pipe_spec_text() (flange ratings, nominal pipe sizes) OR
        is_setpoint_annotation() (PSV/control-valve "SET @ X barg" callouts)
        — confirmed false-positive sources distinct from the distance
        issue, since these sit close to a symbol (287-641px for pipe-spec,
        150-280px for setpoint annotations) but aren't tags.
    """
    rows = []
    dropped_rows = []
    seen_block_ids = set()  # id(blk) of blocks already claimed by a flagged row

    for tag, status, blocks in _tag_shaped_candidates(text_blocks):
        if is_excluded_tag(tag):
            continue
        key = tuple(sorted(id(b) for b in blocks))
        if key in seen_block_ids:
            continue
        xs = [b["x1"] for b in blocks] + [b["x2"] for b in blocks]
        ys = [b["y1"] for b in blocks] + [b["y2"] for b in blocks]
        bbox = {"x1": min(xs), "y1": min(ys), "x2": max(xs), "y2": max(ys)}
        raw_text = " ".join(b["text"] for b in blocks)
        avg_conf = sum(b["conf"] for b in blocks) / len(blocks)

        det, dist = nearest_symbol(bbox, detections)
        within = dist is not None and dist <= radius
        if not within:
            kind = "ORPHANED_TAG_CANDIDATE"
        elif det.get("_routing") != "instrument":
            kind = "MISROUTED_CANDIDATE"
        else:
            # Within radius of an instrument symbol — this is the normal
            # pipeline's territory already, not a gap. Skip.
            continue

        row = {
            "page": page_name, "kind": kind, "candidate_tag": tag,
            "inferred_function": infer_instrument_function(tag),
            "raw_text": raw_text, "ocr_conf": round(avg_conf, 4),
            "x1": bbox["x1"], "y1": bbox["y1"], "x2": bbox["x2"], "y2": bbox["y2"],
            "nearest_det_id": det.get("_det_id") if det else None,
            "nearest_class_name": det.get("class_name") if det else None,
            "nearest_routing": det.get("_routing") if det else None,
            "nearest_dist_px": round(dist, 1) if dist is not None else None,
        }

        if filter_pipe_spec and is_pipe_spec_text(raw_text):
            dropped_rows.append({**row, "drop_reason": "pipe_or_line_spec_text"})
            seen_block_ids.add(key)
            continue
        if filter_pipe_spec and is_setpoint_annotation(raw_text):
            # Reuses the filter_pipe_spec flag rather than adding a new CLI
            # switch — both are "text near a symbol that isn't actually a
            # tag" filters, and there's no case seen so far where you'd want
            # one on and the other off.
            dropped_rows.append({**row, "drop_reason": "setpoint_annotation"})
            seen_block_ids.add(key)
            continue
        if (kind == "ORPHANED_TAG_CANDIDATE" and dist is not None
                and dist > max_orphan_dist):
            dropped_rows.append({**row, "drop_reason": "exceeds_max_orphan_dist"})
            seen_block_ids.add(key)
            continue

        seen_block_ids.add(key)
        rows.append(row)
    return rows, dropped_rows

# ─── OCR-ONLY INSTRUMENT-TYPE INFERENCE ─────────────────────────────────────
# The ISA tag's own letter prefix encodes what KIND of instrument it is
# (ISA S5.1 first-letter/succeeding-letter convention), independent of
# whatever YOLO bounding box (if any) sits near it. This means a tag-shaped
# piece of OCR text carries real instrument-class information even when
# YOLO never drew a bubble there at all — this is what lets review-queue
# rows (which by definition have no confirmed instrument detection nearby)
# still report a meaningful instrument type instead of just "unknown."
# Not exhaustive — covers the common P&ID measured-variable + function
# letters. Extend as new prefixes turn up in review.
ISA_FIRST_LETTER = {
    "P": "Pressure", "D": "Differential", "T": "Temperature", "F": "Flow",
    "L": "Level", "A": "Analysis", "S": "Speed/Frequency", "V": "Vibration",
    "W": "Weight/Force", "E": "Voltage", "I": "Current", "J": "Power",
    "Q": "Quantity/Event", "R": "Radiation", "M": "Moisture/Humidity",
    "H": "Hand (manual)", "K": "Time/Schedule", "Z": "Position",
}
ISA_SUCCEEDING_LETTER = {
    "I": "Indicator", "T": "Transmitter", "C": "Controller", "S": "Switch",
    "R": "Recorder", "G": "Glass/Gauge (local)", "A": "Alarm", "V": "Valve",
    "E": "Element (primary sensing)", "Y": "Relay/Compute", "Z": "Actuator",
    "L": "Low", "H": "High",
}

def infer_instrument_function(tag):
    """
    Best-effort human-readable description of an ISA tag's function,
    derived purely from its letter-prefix — no YOLO bounding box required.
    E.g. 'PDI-1002' -> 'Differential Pressure Indicator'. Returns None if
    the prefix isn't recognized (still a valid tag shape, just an
    uncommon/unmapped function code — not an error).
    """
    if not tag:
        return None
    prefix_match = re.match(r'^(?:\d+-)?([A-Z]{1,4})-', tag)
    if not prefix_match:
        return None
    letters = prefix_match.group(1)
    parts = []
    if letters and letters[0] in ISA_FIRST_LETTER:
        parts.append(ISA_FIRST_LETTER[letters[0]])
    for ch in letters[1:]:
        if ch in ISA_SUCCEEDING_LETTER:
            parts.append(ISA_SUCCEEDING_LETTER[ch])
    return " ".join(parts) if parts else None
