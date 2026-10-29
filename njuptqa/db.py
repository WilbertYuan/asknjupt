from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from . import config
from .util import chinese_ngrams, utc_now

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    org TEXT,
    topics TEXT,
    audience TEXT,
    access_scope TEXT,
    year INTEGER,
    academic_year TEXT,
    discovered_from TEXT,
    status TEXT,
    notes TEXT,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS crawl_manifest (
    id INTEGER PRIMARY KEY,
    url TEXT NOT NULL,
    final_url TEXT,
    fetched_at TEXT,
    http_status INTEGER,
    content_type TEXT,
    content_hash TEXT,
    byte_size INTEGER,
    ext TEXT,
    raw_path TEXT,
    parent_url TEXT,
    attempt INTEGER,
    error TEXT,
    is_login_page INTEGER DEFAULT 0,
    is_empty INTEGER DEFAULT 0,
    UNIQUE(url, content_hash)
);

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    source_id INTEGER,
    url TEXT UNIQUE NOT NULL,
    title TEXT,
    doc_type TEXT,
    org TEXT,
    topics TEXT,
    published_at TEXT,
    updated_at TEXT,
    fetched_at TEXT,
    content_hash TEXT,
    raw_path TEXT,
    parse_status TEXT,
    quality_flags TEXT,
    FOREIGN KEY(source_id) REFERENCES sources(id)
);

CREATE TABLE IF NOT EXISTS document_versions (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    fetched_at TEXT,
    raw_path TEXT,
    http_status INTEGER,
    UNIQUE(document_id, content_hash),
    FOREIGN KEY(document_id) REFERENCES documents(id)
);

CREATE TABLE IF NOT EXISTS blocks (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL,
    version_id INTEGER,
    parent_id INTEGER,
    ordinal INTEGER,
    block_type TEXT,
    heading TEXT,
    text TEXT,
    locator TEXT,
    print_page INTEGER,
    pdf_page INTEGER,
    token_count INTEGER,
    FOREIGN KEY(document_id) REFERENCES documents(id)
);

CREATE TABLE IF NOT EXISTS tables (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL,
    version_id INTEGER,
    title TEXT,
    caption TEXT,
    locator TEXT,
    raw_json TEXT,
    header_json TEXT,
    n_rows INTEGER,
    n_cols INTEGER,
    quality_flags TEXT,
    pdf_page INTEGER,
    FOREIGN KEY(document_id) REFERENCES documents(id)
);

CREATE TABLE IF NOT EXISTS table_rows (
    id INTEGER PRIMARY KEY,
    table_id INTEGER NOT NULL,
    row_index INTEGER,
    role TEXT,
    cells_json TEXT,
    text TEXT,
    FOREIGN KEY(table_id) REFERENCES tables(id)
);

CREATE TABLE IF NOT EXISTS table_cells (
    id INTEGER PRIMARY KEY,
    table_id INTEGER NOT NULL,
    row_index INTEGER,
    col_index INTEGER,
    header_path TEXT,
    value TEXT,
    numeric REAL,
    unit TEXT,
    FOREIGN KEY(table_id) REFERENCES tables(id)
);

CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT,
    year INTEGER,
    attrs_json TEXT,
    UNIQUE(name, kind, year)
);

CREATE TABLE IF NOT EXISTS aliases (
    id INTEGER PRIMARY KEY,
    entity_id INTEGER NOT NULL,
    alias TEXT NOT NULL,
    note TEXT,
    FOREIGN KEY(entity_id) REFERENCES entities(id)
);

CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY,
    entity TEXT,
    metric TEXT,
    value TEXT,
    numeric REAL,
    unit TEXT,
    year INTEGER,
    academic_year TEXT,
    audience TEXT,
    scope TEXT,
    source_document_id INTEGER,
    source_locator TEXT,
    origin TEXT,
    verified TEXT,
    extra_json TEXT
);

CREATE TABLE IF NOT EXISTS relations (
    id INTEGER PRIMARY KEY,
    rel_type TEXT,
    src_kind TEXT,
    src_id INTEGER,
    dst_kind TEXT,
    dst_id INTEGER,
    note TEXT
);

CREATE TABLE IF NOT EXISTS retrieval_units (
    id INTEGER PRIMARY KEY,
    document_id INTEGER,
    unit_type TEXT,
    title TEXT,
    text TEXT,
    search_text TEXT,
    year INTEGER,
    academic_year TEXT,
    audience TEXT,
    locator TEXT,
    token_count INTEGER,
    parent_locator TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS retrieval_fts USING fts5(
    title, search_text, text,
    content='retrieval_units',
    content_rowid='id'
);

CREATE TABLE IF NOT EXISTS admission_plans (
    id INTEGER PRIMARY KEY,
    document_id INTEGER,
    year INTEGER,
    admission_type TEXT,
    province TEXT,
    category TEXT,
    group_code TEXT,
    first_subject TEXT,
    reselect TEXT,
    major TEXT,
    plan INTEGER,
    row_role TEXT,
    notes TEXT,
    table_id INTEGER,
    row_index INTEGER,
    locator TEXT
);

CREATE TABLE IF NOT EXISTS fee_rules (
    id INTEGER PRIMARY KEY,
    document_id INTEGER,
    year INTEGER,
    category TEXT,
    item TEXT,
    amount REAL,
    unit TEXT,
    uplift_ratio REAL,
    notes TEXT,
    locator TEXT
);

CREATE TABLE IF NOT EXISTS lab_centers (
    id INTEGER PRIMARY KEY,
    document_id INTEGER,
    academic_year TEXT,
    seq TEXT,
    name TEXT,
    college TEXT,
    area_m2 REAL,
    table_id INTEGER,
    row_index INTEGER,
    locator TEXT
);

CREATE TABLE IF NOT EXISTS calculation_traces (
    id INTEGER PRIMARY KEY,
    question TEXT,
    op TEXT,
    inputs_json TEXT,
    formula TEXT,
    result TEXT,
    numeric REAL,
    unit TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS evaluation_cases (
    id INTEGER PRIMARY KEY,
    split TEXT,
    question TEXT,
    gold_answer TEXT,
    gold_numeric REAL,
    must_evidence TEXT,
    year INTEGER,
    topic TEXT,
    scoring TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS run_traces (
    id INTEGER PRIMARY KEY,
    question TEXT,
    plan_json TEXT,
    evidence_json TEXT,
    answer TEXT,
    refs TEXT,
    latency_ms INTEGER,
    peak_mb REAL,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS quality_queue (
    id INTEGER PRIMARY KEY,
    kind TEXT,
    document_id INTEGER,
    locator TEXT,
    detail TEXT,
    status TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS experiment_log (
    id INTEGER PRIMARY KEY,
    created_at TEXT,
    stage TEXT,
    config_json TEXT,
    metrics_json TEXT,
    notes TEXT
);
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    config.ensure_dirs()
    db_path = path or config.DB_PATH
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(conn: sqlite3.Connection | None = None) -> sqlite3.Connection:
    own = conn is None
    conn = conn or connect()
    conn.executescript(SCHEMA)
    conn.commit()
    if own:
        return conn
    return conn


@contextmanager
def session() -> Iterator[sqlite3.Connection]:
    conn = init_db()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def upsert_source(conn: sqlite3.Connection, rec: dict[str, Any]) -> int:
    now = utc_now()
    conn.execute(
        """
        INSERT INTO sources(url, org, topics, audience, access_scope, year, academic_year,
                            discovered_from, status, notes, created_at, updated_at)
        VALUES(:url,:org,:topics,:audience,:access_scope,:year,:academic_year,
               :discovered_from,:status,:notes,:created_at,:updated_at)
        ON CONFLICT(url) DO UPDATE SET
            org=COALESCE(excluded.org, sources.org),
            topics=COALESCE(excluded.topics, sources.topics),
            audience=COALESCE(excluded.audience, sources.audience),
            access_scope=COALESCE(excluded.access_scope, sources.access_scope),
            status=COALESCE(excluded.status, sources.status),
            notes=COALESCE(excluded.notes, sources.notes),
            updated_at=excluded.updated_at
        """,
        {
            "url": rec["url"],
            "org": rec.get("org"),
            "topics": rec.get("topics"),
            "audience": rec.get("audience"),
            "access_scope": rec.get("access") or rec.get("access_scope"),
            "year": rec.get("year"),
            "academic_year": rec.get("academic_year"),
            "discovered_from": rec.get("discovered_from"),
            "status": rec.get("status", "discovered"),
            "notes": rec.get("notes"),
            "created_at": now,
            "updated_at": now,
        },
    )
    row = conn.execute("SELECT id FROM sources WHERE url=?", (rec["url"],)).fetchone()
    return int(row["id"])


def get_source_id(conn: sqlite3.Connection, url: str) -> int | None:
    row = conn.execute("SELECT id FROM sources WHERE url=?", (url,)).fetchone()
    return int(row["id"]) if row else None


def insert_retrieval_unit(conn: sqlite3.Connection, rec: dict[str, Any]) -> int:
    search = rec.get("search_text") or chinese_ngrams(
        f"{rec.get('title') or ''} {rec.get('text') or ''}"
    )
    cur = conn.execute(
        """
        INSERT INTO retrieval_units(document_id, unit_type, title, text, search_text,
            year, academic_year, audience, locator, token_count, parent_locator)
        VALUES(?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            rec.get("document_id"),
            rec.get("unit_type"),
            rec.get("title"),
            rec.get("text"),
            search,
            rec.get("year"),
            rec.get("academic_year"),
            rec.get("audience"),
            rec.get("locator"),
            rec.get("token_count"),
            rec.get("parent_locator"),
        ),
    )
    rid = int(cur.lastrowid)
    conn.execute(
        "INSERT INTO retrieval_fts(rowid, title, search_text, text) VALUES(?,?,?,?)",
        (rid, rec.get("title") or "", search, rec.get("text") or ""),
    )
    return rid


def rebuild_fts(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM retrieval_fts")
    rows = conn.execute(
        "SELECT id, title, search_text, text FROM retrieval_units"
    ).fetchall()
    for r in rows:
        conn.execute(
            "INSERT INTO retrieval_fts(rowid, title, search_text, text) VALUES(?,?,?,?)",
            (r["id"], r["title"] or "", r["search_text"] or "", r["text"] or ""),
        )


def log_experiment(conn: sqlite3.Connection, stage: str, config_json: str, metrics_json: str, notes: str = "") -> None:
    conn.execute(
        "INSERT INTO experiment_log(created_at, stage, config_json, metrics_json, notes) VALUES(?,?,?,?,?)",
        (utc_now(), stage, config_json, metrics_json, notes),
    )


def add_quality(conn: sqlite3.Connection, kind: str, detail: str, document_id: int | None = None, locator: str | None = None) -> None:
    conn.execute(
        "INSERT INTO quality_queue(kind, document_id, locator, detail, status, created_at) VALUES(?,?,?,?,?,?)",
        (kind, document_id, locator, detail, "open", utc_now()),
    )
