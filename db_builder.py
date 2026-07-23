# db_builder.py - PipeSight AI
# Creates and manages the SQLite database for P&ID digitization results.
# Import this module from tag_extractor_v2.py to insert results directly.
# Run standalone to initialize (or reset) the database.
#
# Usage:
#   python db_builder.py              — create DB (skips if already exists)
#   python db_builder.py --reset      — drop and recreate all tables
#   python db_builder.py --summary    — print row counts per table
#   python db_builder.py --ingest     — ingest existing tags_raw.csv into DB
#
# 2026-07 migration: added review_queue table (used by page_json.py,
# human_review_export.py, review_server.py) and tags.text_x1/y1/x2/y2
# (used by page_json.py's text_location and review_server.py's api_bbox
# for editing where the OCR text itself sits, separate from the symbol
# box). Both are added via ALTER TABLE / CREATE TABLE IF NOT EXISTS in
# init_db() so re-running against an existing DB never drops data —
# only --reset does that, same as before.

import sqlite3
import csv
import argparse
from pathlib import Path
from datetime import datetime

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DB_PATH  = Path("data") / "pipsight.db"
CSV_PATH = Path("data") / "tag_extraction" / "tags_raw.csv"

# ─── SCHEMA ──────────────────────────────────────────────────────────────────
SCHEMA = """
-- P&ID documents (one row per source PDF/file)
CREATE TABLE IF NOT EXISTS p_ids (
    pid_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name       TEXT NOT NULL UNIQUE,        -- e.g. pid_pdf.pdf
    file_path       TEXT,
    total_pages     INTEGER,
    created_at      TEXT DEFAULT (datetime('now')),
    notes           TEXT
);

-- Pages within a P&ID
CREATE TABLE IF NOT EXISTS pages (
    page_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    pid_id          INTEGER NOT NULL REFERENCES p_ids(pid_id),
    page_name       TEXT NOT NULL UNIQUE,        -- e.g. pid_pdf_page_01
    page_number     INTEGER,
    image_path      TEXT,
    width_px        INTEGER,
    height_px       INTEGER,
    processed_at    TEXT,
    total_detections INTEGER DEFAULT 0,
    instrument_count INTEGER DEFAULT 0,
    valid_count     INTEGER DEFAULT 0,
    raw_count       INTEGER DEFAULT 0,
    missing_count   INTEGER DEFAULT 0,
    skipped_count   INTEGER DEFAULT 0
);

-- Detected symbols (one row per YOLO detection)
CREATE TABLE IF NOT EXISTS symbols (
    symbol_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id         INTEGER NOT NULL REFERENCES pages(page_id),
    det_id          INTEGER NOT NULL,            -- original detection index on the page
    class_name      TEXT NOT NULL,
    class_id        INTEGER NOT NULL,
    yolo_conf       REAL NOT NULL,
    x1              INTEGER NOT NULL,
    y1              INTEGER NOT NULL,
    x2              INTEGER NOT NULL,
    y2              INTEGER NOT NULL,
    routing         TEXT,                        -- instrument / mechanical / structural / unknown_class
    UNIQUE(page_id, det_id)
);

-- Extracted tags (one row per instrument symbol that was OCR'd)
CREATE TABLE IF NOT EXISTS tags (
    tag_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol_id       INTEGER NOT NULL REFERENCES symbols(symbol_id),
    page_id         INTEGER NOT NULL REFERENCES pages(page_id),
    best_zone       TEXT,                        -- above/below/left/right/none
    raw_ocr         TEXT,
    ocr_conf        REAL,
    tag             TEXT,                        -- normalized ISA tag
    tag_status      TEXT NOT NULL,               -- VALID / RAW / MISSING / SKIPPED
    combined_conf   REAL,
    auto_accept     INTEGER DEFAULT 0,           -- 0=False, 1=True
    verified_by     TEXT,
    verified_at     TEXT,
    verification_method TEXT,                    -- AI_AUTO / HUMAN_REQUIRED / HUMAN_VERIFIED
    text_x1         INTEGER,                     -- where the OCR text itself sits,
    text_y1         INTEGER,                     -- separate from the symbol box above
    text_x2         INTEGER,                     -- (x1/y1/x2/y2). NULL when not available
    text_y2         INTEGER,                     -- (RECOVERED_ROTATED/rescued rows).
    created_at      TEXT DEFAULT (datetime('now'))
);

-- Human-review candidates: orphaned OCR text and misrouted-symbol text
-- that didn't cleanly resolve into a tag during extraction.
CREATE TABLE IF NOT EXISTS review_queue (
    review_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id              INTEGER NOT NULL REFERENCES pages(page_id),
    kind                  TEXT NOT NULL,          -- MISROUTED_CANDIDATE / ORPHANED_TAG_CANDIDATE
    candidate_tag         TEXT,
    raw_text              TEXT,
    ocr_conf              REAL,
    x1                    INTEGER,
    y1                    INTEGER,
    x2                    INTEGER,
    y2                    INTEGER,
    nearest_det_id        INTEGER,
    nearest_class_name    TEXT,
    nearest_routing       TEXT,
    nearest_dist_px       REAL,
    status                TEXT NOT NULL DEFAULT 'OPEN',  -- OPEN / DISMISSED / RESOLVED
    resolved_by           TEXT,
    resolved_at           TEXT,
    resolution_note       TEXT,
    created_at            TEXT DEFAULT (datetime('now'))
);

-- Audit log for all manual verifications and corrections
CREATE TABLE IF NOT EXISTS validation_log (
    log_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tag_id          INTEGER NOT NULL REFERENCES tags(tag_id),
    action          TEXT NOT NULL,               -- ACCEPTED / REJECTED / CORRECTED
    old_value       TEXT,
    new_value       TEXT,
    performed_by    TEXT DEFAULT 'human',
    performed_at    TEXT DEFAULT (datetime('now')),
    notes           TEXT
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_pages_pid          ON pages(pid_id);
CREATE INDEX IF NOT EXISTS idx_symbols_page       ON symbols(page_id);
CREATE INDEX IF NOT EXISTS idx_tags_page          ON tags(page_id);
CREATE INDEX IF NOT EXISTS idx_tags_status        ON tags(tag_status);
CREATE INDEX IF NOT EXISTS idx_tags_symbol        ON tags(symbol_id);
CREATE INDEX IF NOT EXISTS idx_vallog_tag         ON validation_log(tag_id);
CREATE INDEX IF NOT EXISTS idx_reviewq_page       ON review_queue(page_id);
CREATE INDEX IF NOT EXISTS idx_reviewq_status     ON review_queue(status);
"""

# Columns added after the original tags table shipped. Listed here so
# init_db() can ALTER TABLE them onto a pre-existing DB that was created
# before this migration — CREATE TABLE IF NOT EXISTS alone won't add
# columns to a table that already exists.
TAGS_MIGRATION_COLUMNS = [
    ("text_x1", "INTEGER"),
    ("text_y1", "INTEGER"),
    ("text_x2", "INTEGER"),
    ("text_y2", "INTEGER"),
]

# ─── CLASS ROUTING — mirrors tag_extractor.py logic ──────────────────────────
INSTRUMENT_CLASSES = {
    "Instrument_Field", "Instrument_Panel", "Instrument_Aux_Panel", "Box"
}
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
}
STRUCTURAL_CLASSES = {
    "Reducer", "Flange_or_Nozzle", "Pipe_Insulation_or_Tracing",
    "Flow_Arrow", "Orifice_Plate", "Rotameter",
}

def get_routing(class_name):
    if class_name in INSTRUMENT_CLASSES:
        return "instrument"
    if class_name in MECHANICAL_CLASSES:
        return "mechanical"
    if class_name in STRUCTURAL_CLASSES:
        return "structural"
    return "unknown_class"

# ─── DB HELPERS ──────────────────────────────────────────────────────────────

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn

def _migrate_tags_columns(conn):
    """Add any TAGS_MIGRATION_COLUMNS missing from an existing tags table.
    No-op on a fresh DB, since CREATE TABLE already includes them."""
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(tags)")}
    for col_name, col_type in TAGS_MIGRATION_COLUMNS:
        if col_name not in existing:
            conn.execute(f"ALTER TABLE tags ADD COLUMN {col_name} {col_type}")
            print(f"  Migrated: added tags.{col_name} ({col_type})")

def init_db(reset=False):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection()
    if reset:
        print("Dropping all tables...")
        conn.executescript("""
            DROP TABLE IF EXISTS validation_log;
            DROP TABLE IF EXISTS review_queue;
            DROP TABLE IF EXISTS tags;
            DROP TABLE IF EXISTS symbols;
            DROP TABLE IF EXISTS pages;
            DROP TABLE IF EXISTS p_ids;
        """)
        print("Tables dropped.")
    conn.executescript(SCHEMA)
    _migrate_tags_columns(conn)
    conn.commit()
    conn.close()
    print(f"Database initialized: {DB_PATH.resolve()}")

# ─── INSERT HELPERS (used by tag_extractor_v2.py) ────────────────────────────

def get_or_create_pid(conn, file_name, file_path=None, total_pages=None):
    row = conn.execute(
        "SELECT pid_id FROM p_ids WHERE file_name = ?", (file_name,)
    ).fetchone()
    if row:
        return row["pid_id"]
    cur = conn.execute(
        "INSERT INTO p_ids (file_name, file_path, total_pages) VALUES (?, ?, ?)",
        (file_name, file_path, total_pages)
    )
    conn.commit()
    return cur.lastrowid

def get_or_create_page(conn, pid_id, page_name, page_number=None,
                        image_path=None, width_px=None, height_px=None):
    row = conn.execute(
        "SELECT page_id FROM pages WHERE page_name = ?", (page_name,)
    ).fetchone()
    if row:
        return row["page_id"]
    cur = conn.execute(
        """INSERT INTO pages
           (pid_id, page_name, page_number, image_path, width_px, height_px, processed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (pid_id, page_name, page_number, image_path, width_px, height_px,
         datetime.now().isoformat())
    )
    conn.commit()
    return cur.lastrowid

def insert_symbol(conn, page_id, det_id, class_name, class_id, yolo_conf,
                  x1, y1, x2, y2):
    routing = get_routing(class_name)
    conn.execute(
        """INSERT OR IGNORE INTO symbols
           (page_id, det_id, class_name, class_id, yolo_conf,
            x1, y1, x2, y2, routing)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (page_id, det_id, class_name, class_id, yolo_conf,
         x1, y1, x2, y2, routing)
    )
    # INSERT OR IGNORE silently no-ops on a UNIQUE(page_id, det_id) conflict
    # without raising IntegrityError, so cur.lastrowid is NOT reliable here —
    # it can return a stale id from a previous insert on the connection,
    # which then gets passed into insert_tag() and fails its FK check.
    # Always explicitly look the row up instead of trusting lastrowid.
    row = conn.execute(
        "SELECT symbol_id FROM symbols WHERE page_id=? AND det_id=?",
        (page_id, det_id)
    ).fetchone()
    return row["symbol_id"] if row else None

def insert_tag(conn, symbol_id, page_id, best_zone, raw_ocr, ocr_conf,
               tag, tag_status, combined_conf, auto_accept,
               verified_by=None, verified_at=None, verification_method=None,
               text_x1=None, text_y1=None, text_x2=None, text_y2=None):
    cur = conn.execute(
        """INSERT INTO tags
           (symbol_id, page_id, best_zone, raw_ocr, ocr_conf, tag, tag_status,
            combined_conf, auto_accept, verified_by, verified_at, verification_method,
            text_x1, text_y1, text_x2, text_y2)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (symbol_id, page_id, best_zone, raw_ocr, ocr_conf, tag, tag_status,
         combined_conf, 1 if auto_accept else 0,
         verified_by, verified_at, verification_method,
         text_x1, text_y1, text_x2, text_y2)
    )
    return cur.lastrowid

def insert_review_candidate(conn, page_id, kind, candidate_tag, raw_text, ocr_conf,
                             x1, y1, x2, y2, nearest_det_id=None,
                             nearest_class_name=None, nearest_routing=None,
                             nearest_dist_px=None):
    cur = conn.execute(
        """INSERT INTO review_queue
           (page_id, kind, candidate_tag, raw_text, ocr_conf, x1, y1, x2, y2,
            nearest_det_id, nearest_class_name, nearest_routing, nearest_dist_px)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (page_id, kind, candidate_tag, raw_text, ocr_conf, x1, y1, x2, y2,
         nearest_det_id, nearest_class_name, nearest_routing, nearest_dist_px)
    )
    return cur.lastrowid

def update_page_stats(conn, page_id):
    """Recompute and update page-level summary counts from tags + symbols."""
    stats = conn.execute("""
        SELECT
            COUNT(DISTINCT s.symbol_id)                              AS total_det,
            SUM(CASE WHEN s.routing='instrument' THEN 1 ELSE 0 END) AS inst_count,
            SUM(CASE WHEN t.tag_status='VALID'   THEN 1 ELSE 0 END) AS valid_c,
            SUM(CASE WHEN t.tag_status='RAW'     THEN 1 ELSE 0 END) AS raw_c,
            SUM(CASE WHEN t.tag_status='MISSING' THEN 1 ELSE 0 END) AS miss_c,
            SUM(CASE WHEN t.tag_status='SKIPPED' THEN 1 ELSE 0 END) AS skip_c
        FROM symbols s
        LEFT JOIN tags t ON t.symbol_id = s.symbol_id
        WHERE s.page_id = ?
    """, (page_id,)).fetchone()
    conn.execute("""
        UPDATE pages SET
            total_detections = ?,
            instrument_count = ?,
            valid_count      = ?,
            raw_count        = ?,
            missing_count    = ?,
            skipped_count    = ?
        WHERE page_id = ?
    """, (stats["total_det"], stats["inst_count"],
          stats["valid_c"], stats["raw_c"],
          stats["miss_c"], stats["skip_c"], page_id))
    conn.commit()

# ─── INGEST EXISTING CSV ─────────────────────────────────────────────────────

def ingest_csv(csv_path=CSV_PATH):
    """Load an existing tags_raw.csv into the database."""
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return

    conn = get_connection()
    # BUG FIXED (2026-07-21): was hardcoded to "pid_pdf.pdf" for every row
    # regardless of which SET the row's page actually belongs to — same bug
    # as tag_extractor_v2.py's _write_rows_to_db, fixed the same way: derive
    # the real PDF stem from the page name (split on "_page_") instead of a
    # placeholder. A CSV can contain rows from multiple SETs, so this is
    # cached per-stem here rather than resolved once for the whole file.
    pid_ids  = {}
    page_ids = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows   = list(reader)

    inserted_symbols = 0
    inserted_tags    = 0

    for row in rows:
        page_name = row["page"]
        pdf_stem = page_name.split("_page_")[0] if "_page_" in page_name else page_name
        if pdf_stem not in pid_ids:
            pid_ids[pdf_stem] = get_or_create_pid(
                conn, f"{pdf_stem}.pdf", str(csv_path.parent)
            )
        pid_id = pid_ids[pdf_stem]
        if page_name not in page_ids:
            page_ids[page_name] = get_or_create_page(
                conn, pid_id, page_name,
                page_number=int(page_name.split("_")[-1]) if page_name[-2:].isdigit() else None
            )
        page_id = page_ids[page_name]

        symbol_id = insert_symbol(
            conn, page_id,
            det_id     = int(row["det_id"]),
            class_name = row["class_name"],
            class_id   = int(row["class_id"]),
            yolo_conf  = float(row["yolo_conf"]),
            x1=int(row["x1"]), y1=int(row["y1"]),
            x2=int(row["x2"]), y2=int(row["y2"]),
        )
        if symbol_id:
            inserted_symbols += 1

        tag_status = row.get("tag_status", "SKIPPED")
        insert_tag(
            conn, symbol_id, page_id,
            best_zone          = row.get("best_zone") or None,
            raw_ocr            = row.get("raw_ocr") or None,
            ocr_conf           = float(row["ocr_conf"]) if row.get("ocr_conf") else None,
            tag                = row.get("tag") or None,
            tag_status         = tag_status,
            combined_conf      = float(row["combined_conf"]) if row.get("combined_conf") else None,
            auto_accept        = row.get("auto_accept", "False") == "True",
            verified_by        = row.get("verified_by") or None,
            verified_at        = row.get("verified_at") or None,
            verification_method= row.get("verification_method") or None,
        )
        inserted_tags += 1

    for page_name, page_id in page_ids.items():
        update_page_stats(conn, page_id)

    conn.commit()
    conn.close()
    print(f"Ingested {inserted_symbols} symbols, {inserted_tags} tags from {csv_path}")

# ─── SUMMARY ─────────────────────────────────────────────────────────────────

def print_summary():
    if not DB_PATH.exists():
        print("Database does not exist yet. Run: python db_builder.py")
        return
    conn = get_connection()
    tables = ["p_ids", "pages", "symbols", "tags", "review_queue", "validation_log"]
    print(f"\nDatabase: {DB_PATH.resolve()}")
    print(f"{'Table':<20} {'Rows':>8}")
    print("-" * 30)
    for t in tables:
        cnt = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t:<18} {cnt:>8}")

    print("\nTag status breakdown:")
    for row in conn.execute(
        "SELECT tag_status, COUNT(*) as cnt FROM tags GROUP BY tag_status ORDER BY cnt DESC"
    ):
        print(f"  {row['tag_status']:<12} {row['cnt']:>6}")

    print("\nReview queue status breakdown:")
    for row in conn.execute(
        "SELECT status, COUNT(*) as cnt FROM review_queue GROUP BY status ORDER BY cnt DESC"
    ):
        print(f"  {row['status']:<12} {row['cnt']:>6}")

    print("\nPer-page summary:")
    for row in conn.execute(
        """SELECT pg.page_name, pg.total_detections, pg.instrument_count,
                  pg.valid_count, pg.raw_count, pg.missing_count
           FROM pages pg JOIN p_ids pid ON pid.pid_id = pg.pid_id
           ORDER BY pg.page_number"""
    ):
        valid_pct = (row["valid_count"] / row["instrument_count"] * 100
                     if row["instrument_count"] else 0)
        print(f"  {row['page_name']}  det={row['total_detections']}  "
              f"inst={row['instrument_count']}  valid={row['valid_count']} "
              f"({valid_pct:.0f}%)")
    conn.close()

# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PipeSight AI — Database Builder")
    parser.add_argument("--reset",   action="store_true", help="Drop and recreate all tables")
    parser.add_argument("--summary", action="store_true", help="Print row counts and stats")
    parser.add_argument("--ingest",  action="store_true", help="Ingest existing tags_raw.csv")
    args = parser.parse_args()

    init_db(reset=args.reset)

    if args.ingest:
        ingest_csv()

    if args.summary:
        print_summary()
        return

    if not args.reset and not args.ingest:
        print("Database ready. Use --ingest to load existing CSV, --summary to inspect.")

if __name__ == "__main__":
    main()