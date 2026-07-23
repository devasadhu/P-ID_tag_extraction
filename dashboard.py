# dashboard.py - PipeSight AI
#
# Restructured 2026-07-21. Still read-only over pipsight.db — never writes
# to the DB, corrections still happen only in review_server.py (now
# embedded live, see "Review this page" below, instead of a separate app).
#
# Two design calls worth flagging explicitly rather than burying in a diff:
#
# 1. SOURCE OF TRUTH: page_json.py's own docstring says CSV/PDF/Excel/
#    dashboard should all read FROM the JSON exports, not query the DB
#    independently. This file still queries the DB directly. Reason: the
#    two things reviewer-productivity reporting actually needs —
#    validation_log, and review_queue.resolved_by/resolved_at — aren't
#    part of the JSON schema at all (they're cross-page audit history, not
#    per-page state), and re-reading a JSON file per page on every
#    dashboard load adds a staleness risk (forgot to re-export?) the DB
#    doesn't have. Kept DB-direct on purpose — flagging so it's a
#    conscious choice, not an accidental contract violation.
#
# 2. DERIVED FIELDS: instrument_type / confidence_level / found_by /
#    caution_note were added to page_json.py's output. To get them here
#    without reintroducing the db_builder import fragility this file's
#    original header explicitly avoided ("db_builder pulls in
#    matching_engine.py for routing logic this dashboard never needs"):
#      - infer_instrument_function is imported directly from
#        matching_engine.py (lightweight, doesn't pull in db_builder) —
#        real logic, not a reinvented ISA lookup table.
#      - The confidence-tier / found-by / reference-noise-label logic is
#        SMALL (each <10 lines) and is duplicated locally rather than
#        imported via `import page_json` (which would drag db_builder
#        back in transitively). If you change those thresholds in
#        page_json.py, change them here too — marked below.
#
# 3. RESTRUCTURE (2026-07-21) — two problems this fixes:
#      a) Selecting a specific page in the sidebar used to change almost
#         nothing: every chart/tab above the fold always showed the whole
#         DB, and the only page-scoped content was a small "Detail"
#         section appended after everything else. Now the sidebar
#         selection is a real mode switch: "All pages" shows the
#         dashboard as before; picking a page replaces the main body with
#         a page-scoped view instead of appending to the bottom of it.
#      b) Two separate reviewer-productivity views existed (Operational
#         Breakdown > Reviewer Productivity, and Drill-Downs > By
#         Reviewer) — merged into one tab.
#    Also: review_server.py's Flask app is now launched as a guarded
#    background process (checked via a TCP probe + session_state so it's
#    only attempted once, not on every Streamlit rerun — a previous
#    attempt without that guard broke rendering) and embedded via iframe,
#    scoped to whichever page is selected, so review happens inside the
#    dashboard instead of a separate app at a separate URL.
#
# Run:
#   pip install streamlit pandas plotly --break-system-packages   (if not already installed)
#   streamlit run dashboard.py

import socket
import subprocess
import sqlite3
import time
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import quote

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from matching_engine import infer_instrument_function

DEFAULT_DB_PATH = str(Path("data") / "pipsight.db")

st.set_page_config(page_title="PipeSight AI — Dashboard", layout="wide")

# ─── shared color palette — mirrors pdf_excel_export.py's COLOR_* constants
# so a status means the same color everywhere a reviewer looks (dashboard,
# PDF overlay, Excel). Converted from those RGB tuples to hex for Plotly.
TAG_STATUS_COLORS = {
    "VALID":              "#228B22",  # forest green   — COLOR_VALID_AUTO
    "RAW":                "#FF8C00",  # orange         — COLOR_RAW
    "MISSING":            "#DC143C",  # crimson        — COLOR_MISSING
    "RECOVERED_ROTATED":  "#1E90FF",  # dodger blue    — COLOR_RECOVERED_ROT
    "RESCUED":            "#9370DB",  # medium purple — new tier, no PDF/Excel precedent yet
    "SKIPPED":            "#969696",  # mid gray       — non-instrument rows
}
RQ_KIND_COLORS = {
    "MISROUTED_CANDIDATE":     "#696969",  # dim gray  — COLOR_REVIEW_QUEUE
    "ORPHANED_TAG_CANDIDATE":  "#DAA520",  # goldenrod — COLOR_VALID_REVIEW (needs-a-look tone)
    "MANUAL_ADDITION":         "#8B4513",  # saddle brown — human-flagged, distinct from auto-found
}
# Confidence tiers reused everywhere (KPI cards, charts, tables) so green/
# amber/red means the same thing across the whole dashboard, not just
# within one chart.
CONFIDENCE_COLORS = {"high": "#228B22", "medium": "#FF8C00", "low": "#DC143C"}
FALLBACK_COLOR = "#404040"

# Mirrors page_json.py's REVIEW_AUTO_RESOLVE_YOLO_CONF — the empirical
# finding that combined_confidence saturates near 1.0 on this document
# type, so yolo_conf is the value that actually discriminates. KEEP THIS
# IN SYNC with page_json.py if that threshold changes.
REVIEW_AUTO_RESOLVE_YOLO_CONF = 0.53

# Mirrors page_json.py's REFERENCE_NOISE_LABELS. KEEP IN SYNC.
REFERENCE_NOISE_LABELS = {
    "pipe_or_line_spec_text": "a pipe or line spec callout",
    "setpoint_annotation": "a pressure/temperature setpoint note",
    "non_instrument_abbreviation": "a known non-instrument abbreviation",
}

# ─── review-server embed settings ───────────────────────────────────────────
REVIEW_SERVER_HOST = "127.0.0.1"
REVIEW_SERVER_PORT = 5000


# ─── derived-field helpers — mirrors page_json.py's _parse_tag /
# _confidence_level / _found_by. Kept here as small, pure, easily-audited
# duplicates rather than an import, per the design note above.

def _parse_tag(tag):
    if not tag or not isinstance(tag, str):
        return None, None
    import re
    m = re.match(r"^([A-Za-z]{1,4})[\s\-]?(\d[\w\-]*)$", tag.strip())
    if not m:
        return tag, None
    return m.group(1).upper(), m.group(2)


def _confidence_level(auto_accept, yolo_conf):
    if auto_accept:
        return "high"
    if pd.notna(yolo_conf) and yolo_conf >= REVIEW_AUTO_RESOLVE_YOLO_CONF:
        return "medium"
    return "low"


def _found_by(tag_status, verification_method):
    if tag_status == "RESCUED" or (verification_method or "") == "AI_AUTO":
        return "rescue_pass"
    return "standard_detection"


def _source_label(page_name):
    """Historically p_ids.file_name was a hardcoded placeholder for every
    row (fixed 2026-07-21 in tag_extractor_v2.py — see backfill_pid_filenames.py
    for existing DBs). Derive from page_name's "SET N" prefix as a display
    fallback that still works even on a DB that hasn't been backfilled yet.
    """
    if not isinstance(page_name, str):
        return "Unknown source"
    if "_page_" in page_name:
        set_part = page_name.split("_page_")[0]
        return f"Haifa — {set_part}"
    return "Haifa"


def _bar(df, x_col, y_col, color_map, height=320):
    """One consistent status-colored bar chart helper so every simple bar
    chart in this file looks/behaves the same way instead of ad hoc
    styling."""
    fig = px.bar(
        df, x=x_col, y=y_col,
        color=x_col,
        color_discrete_map=color_map,
        text=y_col,
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=10, b=0),
        height=height,
        xaxis_title=None,
        yaxis_title=None,
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ─── review-server launch guard ─────────────────────────────────────────────
#
# A previous attempt at this (removed after it "broke rendering") almost
# certainly launched subprocess.Popen(["python", "review_server.py"]) with
# no guard. Streamlit reruns the ENTIRE script on every widget interaction
# — not just on page load — so an unguarded Popen() either tries to re-bind
# an already-used port (throws) or spawns duplicate server processes, and
# an uncaught exception there halts the whole page render, which is why it
# looked like it broke everything rather than just the review section.
#
# Fixed here with two layers: a session_state flag so we don't even probe
# the port on every rerun once we know it's up, and a TCP connect check
# (not a blind sleep) so we don't fight an already-running manually-started
# server either.

def _review_server_port_open(host=REVIEW_SERVER_HOST, port=REVIEW_SERVER_PORT, timeout=0.3):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def ensure_review_server_running():
    """Returns True once something is listening on the review-server port
    (whether we launched it or it was already running). Never raises —
    launch failures are recorded in session_state and surfaced by the
    caller, not thrown, so a failure here can't take down the rest of the
    dashboard."""
    if st.session_state.get("review_server_launched"):
        return True
    if _review_server_port_open():
        st.session_state.review_server_launched = True
        return True
    try:
        proc = subprocess.Popen(
            ["python", "review_server.py"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        st.session_state.review_server_process = proc
    except Exception as e:
        st.session_state.review_server_launch_error = str(e)
        return False

    for _ in range(20):  # up to ~5s, polled rather than a blind sleep
        if _review_server_port_open():
            st.session_state.review_server_launched = True
            return True
        time.sleep(0.25)

    st.session_state.review_server_launch_error = (
        "review_server.py didn't start listening within 5s — check the "
        "terminal running Streamlit for its error output."
    )
    return False


# ─── connection / cached reads ──────────────────────────────────────────────

def get_conn(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _fetch_tags(conn):
    """Selects tags including reference_noise when that column exists —
    same defensive pattern page_json.py uses, for DBs built before that
    column was added."""
    base = """SELECT t.tag_id, t.page_id, pg.page_name, t.tag, t.tag_status,
                     t.raw_ocr, t.ocr_conf, t.combined_conf, t.auto_accept,
                     t.verified_by, t.verification_method, t.created_at,
                     s.class_name, s.yolo_conf{ref_col}
              FROM tags t
              JOIN pages pg ON pg.page_id = t.page_id
              JOIN symbols s ON s.symbol_id = t.symbol_id"""
    try:
        return pd.read_sql_query(base.format(ref_col=", t.reference_noise"), conn)
    except Exception:
        return pd.read_sql_query(base.format(ref_col=""), conn)


@st.cache_data(ttl=30)
def load_tables(db_path, _cache_bust):
    """Pulls everything the dashboard needs in one pass. _cache_bust is an
    unused int that only exists so the manual Refresh button can force a
    re-read (Streamlit's cache keys on args, not on file mtime)."""
    conn = get_conn(db_path)

    pages = pd.read_sql_query(
        """SELECT pg.page_id, pg.page_name, pg.page_number, pid.file_name AS raw_file_name,
                  pg.total_detections, pg.instrument_count, pg.valid_count,
                  pg.raw_count, pg.missing_count, pg.skipped_count
           FROM pages pg JOIN p_ids pid ON pid.pid_id = pg.pid_id
           ORDER BY pg.page_name, pg.page_number""",
        conn,
    )

    tags = _fetch_tags(conn)

    review_queue = pd.read_sql_query(
        """SELECT rq.review_id, rq.page_id, pg.page_name, rq.kind,
                  rq.candidate_tag, rq.raw_text, rq.ocr_conf,
                  rq.nearest_class_name, rq.nearest_routing, rq.nearest_dist_px,
                  rq.status, rq.resolved_by, rq.resolved_at, rq.created_at
           FROM review_queue rq
           JOIN pages pg ON pg.page_id = rq.page_id""",
        conn,
    )

    validation_log = pd.read_sql_query(
        """SELECT vl.log_id, vl.tag_id, t.tag, vl.action, vl.old_value,
                  vl.new_value, vl.performed_by, vl.performed_at, vl.notes
           FROM validation_log vl
           JOIN tags t ON t.tag_id = vl.tag_id
           ORDER BY vl.performed_at DESC
           LIMIT 500""",
        conn,
    )

    conn.close()
    return pages, tags, review_queue, validation_log


def _enrich(tags, review_queue):
    """Adds the same derived fields page_json.py computes — instrument
    type, loop/abbreviation split, confidence tier, discovery method,
    caution note — onto the raw DB dataframes. Vectorized where practical."""
    if len(tags):
        tags = tags.copy()
        tags["instrument_type"] = tags["tag"].map(infer_instrument_function)
        parsed = tags["tag"].map(_parse_tag)
        tags["abbreviation"] = parsed.map(lambda p: p[0])
        tags["loop_number"] = parsed.map(lambda p: p[1])
        tags["confidence_level"] = tags.apply(
            lambda r: _confidence_level(r["auto_accept"], r["yolo_conf"]), axis=1
        )
        tags["found_by"] = tags.apply(
            lambda r: _found_by(r["tag_status"], r["verification_method"]), axis=1
        )
        if "reference_noise" in tags.columns:
            tags["caution_note"] = tags["reference_noise"].map(
                lambda rn: REFERENCE_NOISE_LABELS.get(rn, rn.replace("_", " ")) if rn else None
            )
        else:
            tags["caution_note"] = None
        tags["created_at"] = pd.to_datetime(tags["created_at"], errors="coerce")

    if len(review_queue):
        review_queue = review_queue.copy()
        review_queue["instrument_type"] = review_queue["candidate_tag"].map(infer_instrument_function)
        parsed_rq = review_queue["candidate_tag"].map(_parse_tag)
        review_queue["abbreviation"] = parsed_rq.map(lambda p: p[0])
        review_queue["loop_number"] = parsed_rq.map(lambda p: p[1])
        review_queue["found_by"] = review_queue["kind"].map(
            lambda k: "human_added" if k == "MANUAL_ADDITION" else "standard_detection"
        )
        review_queue["created_at"] = pd.to_datetime(review_queue["created_at"], errors="coerce")
        review_queue["resolved_at"] = pd.to_datetime(review_queue["resolved_at"], errors="coerce")

    return tags, review_queue


def _build_reviewer_activity(validation_log, review_queue):
    """validation_log alone undercounts reviewer work — it only logs tag
    corrections (ACCEPT/REJECT/CORRECT), never PROMOTE/DISMISS actions on
    review_queue, since those write resolved_by/resolved_at directly on
    the review_queue row instead of a log table. This merges both into
    one activity stream so 'reviewer productivity' reflects everything a
    reviewer actually did, not just half of it.
    """
    parts = []
    if len(validation_log):
        vl = validation_log.copy()
        vl["reviewer"] = vl["performed_by"]
        vl["when"] = pd.to_datetime(vl["performed_at"], errors="coerce")
        vl["source"] = "tag_correction"
        vl["item"] = vl["tag"]
        parts.append(vl[["reviewer", "when", "action", "item", "source"]])

    if len(review_queue):
        rq = review_queue[review_queue["resolved_by"].notna()].copy()
        if len(rq):
            rq["reviewer"] = rq["resolved_by"]
            rq["when"] = pd.to_datetime(rq["resolved_at"], errors="coerce")
            rq["source"] = "review_queue"
            rq["item"] = rq["candidate_tag"]
            rq["action"] = rq["status"]
            parts.append(rq[["reviewer", "when", "action", "item", "source"]])

    if not parts:
        return pd.DataFrame(columns=["reviewer", "when", "action", "item", "source"])
    return pd.concat(parts, ignore_index=True).sort_values("when", ascending=False)


# ─── sidebar ─────────────────────────────────────────────────────────────────

st.sidebar.title("PipeSight AI")
db_path = st.sidebar.text_input("Database path", value=DEFAULT_DB_PATH)

if "cache_bust" not in st.session_state:
    st.session_state.cache_bust = 0
if st.sidebar.button("🔄 Refresh"):
    st.session_state.cache_bust += 1

if not Path(db_path).exists():
    st.error(
        f"No database found at `{db_path}`. Run `python tag_extractor_v2.py` "
        f"(or `python db_builder.py`) first, or fix the path in the sidebar."
    )
    st.stop()

pages, tags, review_queue, validation_log = load_tables(db_path, st.session_state.cache_bust)
pages["source"] = pages["page_name"].map(_source_label)
tags, review_queue = _enrich(tags, review_queue)
reviewer_activity = _build_reviewer_activity(validation_log, review_queue)

st.sidebar.divider()
view_mode = st.sidebar.radio(
    "View", ["Simple", "Technical"], horizontal=True,
    help="Simple hides raw confidence numbers and internal IDs — same "
         "split page_summary.json vs page_tags.json uses.",
)
activity_window_hours = st.sidebar.slider(
    "\"What's new\" window (hours)", min_value=1, max_value=168, value=24,
)

page_filter = st.sidebar.selectbox(
    "Drill into a page", options=["All pages"] + list(pages["page_name"]),
)

st.sidebar.caption(f"Loaded from `{db_path}` · auto-refreshes every 30s, or use the button above.")

st.title("PipeSight AI — Extraction Dashboard")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE DETAIL MODE — a specific page is selected. Replaces the Overview
# body entirely rather than appending below it, so picking a page in the
# sidebar visibly changes the main view instead of only adding a footnote.
# ═══════════════════════════════════════════════════════════════════════════

if page_filter != "All pages":
    page_row = pages[pages["page_name"] == page_filter].iloc[0]
    page_id = page_row["page_id"]
    page_tags = tags[tags["page_id"] == page_id] if len(tags) else tags
    page_rq_open = (
        review_queue[(review_queue["page_id"] == page_id) & (review_queue["status"] == "OPEN")]
        if len(review_queue) else review_queue
    )
    page_valid_pct = (
        (page_row["valid_count"] / page_row["instrument_count"] * 100)
        if page_row["instrument_count"] else 0.0
    )

    st.subheader(f"📄 {page_filter}")
    st.caption(f"Source: {page_row['source']}" + (
        f" · raw file_name: `{page_row['raw_file_name']}`" if view_mode == "Technical" else ""
    ))

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Instrument tags", int(page_row["instrument_count"]))
    p2.metric("Valid", int(page_row["valid_count"]), f"{page_valid_pct:.0f}%")
    p3.metric("Needs OCR/manual fix", int(page_row["raw_count"] + page_row["missing_count"]))
    p4.metric("Open review items", len(page_rq_open))

    st.divider()

    # ─── live review, embedded ──────────────────────────────────────────────
    st.markdown("### Review this page")
    if ensure_review_server_running():
        review_url = f"http://{REVIEW_SERVER_HOST}:{REVIEW_SERVER_PORT}/?page={quote(page_filter)}"
        st.components.v1.iframe(review_url, height=900, scrolling=True)
    else:
        st.error(
            "Couldn't start the review server: "
            f"{st.session_state.get('review_server_launch_error', 'unknown error')}\n\n"
            "Try running `python review_server.py` manually in another terminal, then hit Refresh."
        )
        if st.button("Retry launch"):
            st.session_state.review_server_launched = False
            st.session_state.pop("review_server_launch_error", None)
            st.rerun()

    st.divider()

    # ─── read-only tables (kept for a quick scan without touching review UI) ──
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Tags on this page**")
        cols = ["tag", "tag_status", "instrument_type", "confidence_level", "found_by", "caution_note"]
        if view_mode == "Technical":
            cols += ["raw_ocr", "ocr_conf", "combined_conf", "class_name", "verification_method"]
        st.dataframe(
            page_tags[[c for c in cols if c in page_tags.columns]],
            hide_index=True, use_container_width=True,
        )
    with d2:
        st.markdown("**Open review-queue items on this page**")
        if len(page_rq_open):
            cols = ["kind", "candidate_tag", "instrument_type", "found_by"]
            if view_mode == "Technical":
                cols += ["raw_text", "ocr_conf", "nearest_class_name", "nearest_dist_px"]
            st.dataframe(
                page_rq_open[[c for c in cols if c in page_rq_open.columns]],
                hide_index=True, use_container_width=True,
            )
        else:
            st.caption("Nothing open for this page.")

# ═══════════════════════════════════════════════════════════════════════════
# OVERVIEW MODE — "All pages". Everything below is unchanged in substance
# from before, just no longer sharing the screen with a half-rendered
# page-detail footnote.
# ═══════════════════════════════════════════════════════════════════════════

else:
    total_pages = len(pages)
    total_detections = int(pages["total_detections"].sum()) if total_pages else 0
    instrument_count = int(pages["instrument_count"].sum()) if total_pages else 0
    valid_count = int(pages["valid_count"].sum()) if total_pages else 0
    raw_count = int(pages["raw_count"].sum()) if total_pages else 0
    missing_count = int(pages["missing_count"].sum()) if total_pages else 0
    open_review = int((review_queue["status"] == "OPEN").sum()) if len(review_queue) else 0
    valid_pct = (valid_count / instrument_count * 100) if instrument_count else 0.0
    high_conf_pct = (
        (tags["confidence_level"] == "high").mean() * 100 if len(tags) else 0.0
    )
    caution_count = int(tags["caution_note"].notna().sum()) if len(tags) and "caution_note" in tags else 0
    caution_rate = (caution_count / len(tags) * 100) if len(tags) else 0.0

    # ─── health score — a single composite number, not a data fact. Weights
    # are a starting proposal (VALID rate matters most, then backlog, then
    # confidence mix, then caution flags) — validate these with your
    # supervisors before treating the number as authoritative; it's meant to
    # start a conversation, not settle one.
    backlog_rate = (open_review / instrument_count) if instrument_count else 0.0
    health_score = (
        0.4 * (valid_pct / 100)
        + 0.3 * max(0.0, 1 - min(backlog_rate, 1.0))
        + 0.2 * (high_conf_pct / 100)
        + 0.1 * max(0.0, 1 - min(caution_rate / 100, 1.0))
    ) * 100

    # ─── "What's new since last review" + anomaly check, both windowed on
    # activity_window_hours rather than a persisted "last viewed" timestamp
    # (Streamlit has no cross-session state without adding new storage) —
    # simpler, and "what changed in the last N hours" is honest about what's
    # actually knowable from created_at/performed_at alone.
    cutoff = datetime.now() - timedelta(hours=activity_window_hours)
    new_tags = int((tags["created_at"] >= cutoff).sum()) if len(tags) and tags["created_at"].notna().any() else 0
    new_rq = int((review_queue["created_at"] >= cutoff).sum()) if len(review_queue) and review_queue["created_at"].notna().any() else 0
    resolved_recent = int((reviewer_activity["when"] >= cutoff).sum()) if len(reviewer_activity) else 0

    with st.container(border=True):
        wc1, wc2 = st.columns([3, 1])
        with wc1:
            st.markdown(f"**What's new in the last {activity_window_hours}h**")
            st.markdown(
                f"- {new_tags} new tag(s) extracted\n"
                f"- {new_rq} new review-queue item(s) opened\n"
                f"- {resolved_recent} item(s) resolved by reviewers\n"
                f"- Net backlog change: {new_rq - resolved_recent:+d}"
            )
        with wc2:
            st.metric("Health score", f"{health_score:.0f}/100")

    # Rule-based anomaly banner — deliberately not a model/score, so the
    # reason it fired is always stated in plain language.
    anomaly_msgs = []
    if new_rq - resolved_recent > 20:
        anomaly_msgs.append(f"Backlog grew by {new_rq - resolved_recent} in the last {activity_window_hours}h.")
    if instrument_count and valid_pct < 50:
        anomaly_msgs.append(f"Overall VALID rate is {valid_pct:.0f}% — below half of instrument detections.")
    if caution_rate > 25:
        anomaly_msgs.append(f"{caution_rate:.0f}% of tags carry a reference-noise caution flag.")
    if anomaly_msgs:
        st.warning("⚠ " + "  ·  ".join(anomaly_msgs))

    st.divider()

    # ─── executive KPI row ──────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Pages processed", total_pages)
    k2.metric("Total detections", total_detections)
    k3.metric("Instrument tags", instrument_count)
    k4.metric("Valid tags", valid_count, f"{valid_pct:.0f}%")
    k5.metric("Needs OCR/manual fix", raw_count + missing_count)
    k6.metric("Open review queue", open_review)

    if view_mode == "Technical":
        k7, k8, k9 = st.columns(3)
        k7.metric("High-confidence tags", f"{high_conf_pct:.0f}%")
        k8.metric("Caution-flagged tags", caution_count, f"{caution_rate:.0f}%")
        rescued_count = int((tags["found_by"] == "rescue_pass").sum()) if len(tags) else 0
        k9.metric("Rescue-pass recoveries", rescued_count)

    st.divider()

    # ─── overview charts ─────────────────────────────────────────────────────
    oc1, oc2 = st.columns(2)
    with oc1:
        st.subheader("Tag status breakdown")
        if len(tags):
            status_counts = tags["tag_status"].value_counts().rename_axis("tag_status").reset_index(name="count")
            _bar(status_counts, "tag_status", "count", TAG_STATUS_COLORS)
        else:
            st.caption("No tags in the database yet.")

    with oc2:
        st.subheader("Open review queue by kind")
        open_rq = review_queue[review_queue["status"] == "OPEN"] if len(review_queue) else pd.DataFrame()
        if len(open_rq):
            kind_counts = open_rq["kind"].value_counts().rename_axis("kind").reset_index(name="count")
            _bar(kind_counts, "kind", "count", RQ_KIND_COLORS)
        else:
            st.caption("Review queue is empty — nothing open.")

    st.divider()

    # ─── main tabs — 6, down from 7: Reviewer Productivity (Operational
    # Breakdown) and By Reviewer (Drill-Downs) covered overlapping ground
    # (both were "what has each reviewer done") and are now one tab.
    st.header("Breakdown")
    tab_page, tab_instrument, tab_loop, tab_reviewer, tab_hotspots, tab_confidence = st.tabs(
        ["By Page", "By Instrument Type", "By Loop", "By Reviewer", "Error Hotspots", "Confidence Analysis"]
    )

    with tab_page:
        st.caption("Tag status by page (stacked) — also your index for picking a page to drill into via the sidebar.")
        if len(tags):
            stacked = tags.groupby(["page_name", "tag_status"]).size().reset_index(name="count")
            fig = px.bar(
                stacked, x="page_name", y="count", color="tag_status",
                color_discrete_map=TAG_STATUS_COLORS, barmode="stack",
            )
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=340, xaxis_title=None, yaxis_title=None)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        if len(pages):
            per_page = pages.copy()
            per_page["valid_pct"] = per_page.apply(
                lambda r: (r["valid_count"] / r["instrument_count"] * 100) if r["instrument_count"] else 0.0,
                axis=1,
            )
            if len(review_queue):
                open_by_page = (
                    review_queue[review_queue["status"] == "OPEN"]
                    .groupby("page_id").size().rename("open_review_items")
                )
                per_page = per_page.join(open_by_page, on="page_id")
            per_page["open_review_items"] = per_page.get("open_review_items", 0)
            per_page["open_review_items"] = per_page["open_review_items"].fillna(0).astype(int)

            if len(tags):
                caution_by_page = tags[tags["caution_note"].notna()].groupby("page_name").size().rename("caution_flags") if "caution_note" in tags else None
                high_conf_by_page = tags[tags["confidence_level"] == "high"].groupby("page_name").size().rename("high_conf_tags")
                if caution_by_page is not None:
                    per_page = per_page.join(caution_by_page, on="page_name")
                per_page = per_page.join(high_conf_by_page, on="page_name")
                per_page["caution_flags"] = per_page.get("caution_flags", 0)
                per_page["caution_flags"] = per_page["caution_flags"].fillna(0).astype(int)
                per_page["high_conf_tags"] = per_page.get("high_conf_tags", 0)
                per_page["high_conf_tags"] = per_page["high_conf_tags"].fillna(0).astype(int)

            display_cols = [
                "source", "page_name", "total_detections", "instrument_count",
                "valid_count", "raw_count", "missing_count", "valid_pct", "open_review_items",
            ]
            if view_mode == "Technical":
                display_cols += ["high_conf_tags", "caution_flags", "raw_file_name"]

            st.dataframe(
                per_page[[c for c in display_cols if c in per_page.columns]],
                column_config={
                    "valid_pct": st.column_config.ProgressColumn(
                        "Valid %", min_value=0, max_value=100, format="%.0f%%"
                    ),
                },
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.caption("No pages in the database yet.")

    with tab_instrument:
        st.caption("Tag counts by instrument function (from matching_engine.infer_instrument_function)")
        if len(tags):
            by_type = tags["instrument_type"].value_counts().rename_axis("instrument_type").reset_index(name="count")
            fig = px.bar(by_type, x="instrument_type", y="count", text="count")
            fig.update_traces(marker_color="#0F62B0", textposition="outside")
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=360, xaxis_title=None, yaxis_title=None)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("No tags yet.")

    with tab_loop:
        st.caption("Instruments grouped by ISA loop number — e.g. TE-1076/TI-1076/TIT-1076/HS-1076 all belong "
                   "to loop 1076. Only loops with 2+ members shown; includes open review-queue candidates so "
                   "you can see which loops still have something unresolved.")
        if len(tags) or len(review_queue):
            combined = pd.concat([
                tags[["tag", "loop_number", "instrument_type"]].rename(columns={"tag": "item"}).assign(source="tag") if len(tags) else pd.DataFrame(),
                review_queue[["candidate_tag", "loop_number", "instrument_type"]].rename(columns={"candidate_tag": "item"}).assign(source="review_queue_candidate") if len(review_queue) else pd.DataFrame(),
            ], ignore_index=True)
            combined = combined.dropna(subset=["loop_number"])
            loop_sizes = combined.groupby("loop_number").size()
            multi_loops = loop_sizes[loop_sizes > 1].index
            combined = combined[combined["loop_number"].isin(multi_loops)].sort_values("loop_number")
            if len(combined):
                for loop_num, group in combined.groupby("loop_number"):
                    open_in_loop = (group["source"] == "review_queue_candidate").sum()
                    label = f"Loop {loop_num} — {len(group)} member(s)"
                    if open_in_loop:
                        label += f" ⚠ {open_in_loop} unresolved"
                    with st.expander(label):
                        st.dataframe(group[["item", "instrument_type", "source"]], hide_index=True, use_container_width=True)
            else:
                st.caption("No multi-member loops found on the current data.")
        else:
            st.caption("No tags or review-queue candidates yet.")

    with tab_reviewer:
        st.caption("Combines tag corrections (validation_log) + review-queue resolutions "
                   "(review_queue.resolved_by/resolved_at) — validation_log alone misses PROMOTE/DISMISS actions.")
        if len(reviewer_activity) and reviewer_activity["reviewer"].notna().any():
            by_reviewer = (
                reviewer_activity.dropna(subset=["reviewer"])
                .groupby("reviewer").size().rename("items_resolved")
                .sort_values(ascending=False).reset_index()
            )
            rc1, rc2 = st.columns([1, 1])
            with rc1:
                st.caption("Items resolved per reviewer")
                fig = px.bar(by_reviewer, x="reviewer", y="items_resolved", text="items_resolved")
                fig.update_traces(marker_color="#0F62B0", textposition="outside")
                fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=320, xaxis_title=None, yaxis_title=None)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            with rc2:
                st.caption("Activity over time")
                daily = reviewer_activity.dropna(subset=["when"]).copy()
                if len(daily):
                    daily["day"] = daily["when"].dt.date
                    daily_counts = daily.groupby(["day", "reviewer"]).size().reset_index(name="count")
                    fig = px.line(daily_counts, x="day", y="count", color="reviewer", markers=True)
                    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=320, xaxis_title=None, yaxis_title=None)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                else:
                    st.caption("No timestamped activity yet.")

            rq_resolved = review_queue[review_queue["resolved_at"].notna() & review_queue["created_at"].notna()] if len(review_queue) else pd.DataFrame()
            if len(rq_resolved):
                rq_resolved = rq_resolved.copy()
                rq_resolved["hours_to_resolve"] = (
                    (rq_resolved["resolved_at"] - rq_resolved["created_at"]).dt.total_seconds() / 3600
                )
                st.caption(f"Median time-to-resolve (review queue): {rq_resolved['hours_to_resolve'].median():.1f}h")

            st.divider()
            reviewer_options = ["All"] + sorted(reviewer_activity["reviewer"].dropna().unique().tolist())
            chosen_reviewer = st.selectbox("Filter activity log", reviewer_options)
            filtered = reviewer_activity if chosen_reviewer == "All" else reviewer_activity[reviewer_activity["reviewer"] == chosen_reviewer]
            st.dataframe(filtered, hide_index=True, use_container_width=True)
        else:
            st.caption("No reviewer activity recorded yet.")

    with tab_hotspots:
        st.caption("Caution flags by page — where reference-noise (pipe specs, setpoints, known non-instrument text) concentrates")
        if len(tags) and "caution_note" in tags and tags["caution_note"].notna().any():
            hotspot = (
                tags[tags["caution_note"].notna()]
                .groupby("page_name").size().rename("caution_count")
                .sort_values(ascending=False).reset_index()
            )
            fig = px.bar(hotspot, x="page_name", y="caution_count", text="caution_count")
            fig.update_traces(marker_color="#DC143C", textposition="outside")
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=300, xaxis_title=None, yaxis_title=None)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("No caution-flagged tags — either genuinely clean, or this DB predates the reference_noise column.")

        st.caption("Detection blind spots — dropped/rescue-failed candidates aren't in this DB (they live in "
                   "dropped_candidates.csv and the rescue-pass console log). Import that CSV here if you want "
                   "it in this view — flag it and I'll wire it in.")

    with tab_confidence:
        st.caption("Move the threshold to see how many tags would flip tiers — mirrors the actual "
                   "REVIEW_AUTO_RESOLVE_YOLO_CONF lever in page_json.py.")
        if len(tags):
            threshold = st.slider("yolo_conf threshold", 0.0, 1.0, REVIEW_AUTO_RESOLVE_YOLO_CONF, 0.01)
            above = int((tags["yolo_conf"] >= threshold).sum())
            below = len(tags) - above
            tc1, tc2 = st.columns(2)
            tc1.metric(f"At/above {threshold:.2f}", above)
            tc2.metric(f"Below {threshold:.2f}", below)

            fig = px.scatter(
                tags, x="yolo_conf", y="ocr_conf", color="tag_status",
                color_discrete_map=TAG_STATUS_COLORS,
                hover_data=["tag", "page_name"] if view_mode == "Technical" else None,
            )
            fig.add_vline(x=threshold, line_dash="dash", line_color="#404040")
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=380)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("No tags yet.")

    st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# EXPORT — always visible in both modes, applies to the whole DB either way
# (pdf_excel_export.py doesn't currently support a single-page export).
# ═══════════════════════════════════════════════════════════════════════════

st.divider()
st.subheader("Export")
ec1, ec2 = st.columns([1, 3])
with ec1:
    if st.button("📄 Regenerate PDF + Excel"):
        with st.spinner("Running pdf_excel_export.py..."):
            try:
                result = subprocess.run(
                    ["python", "pdf_excel_export.py", "--images-dir", "data\\haifa_real_pids\\images"],
                    capture_output=True, text=True, timeout=300,
                )
                if result.returncode == 0:
                    st.success("Export complete — check data\\tag_extraction_v2\\ for pipesight_review.pdf/.xlsx")
                else:
                    st.error(f"Export failed:\n{result.stderr[-800:]}")
            except FileNotFoundError:
                st.error("pdf_excel_export.py not found — run this from the project root.")
            except Exception as e:
                st.error(f"Export failed: {e}")
with ec2:
    st.caption("Re-runs the existing pdf_excel_export.py against current DB state — doesn't build a second "
               "export pipeline. Files land in data\\tag_extraction_v2\\ as usual.")
