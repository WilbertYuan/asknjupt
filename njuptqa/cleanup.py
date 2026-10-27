"""Drop garbled or empty records from the retrieval index.

Discarded documents keep their row and content hash so a later parse does not
put the same unreadable file back. Structured sources used by admission plans,
fees, labs, and extracted facts are never discarded.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict

from .db import rebuild_fts, session
from .util import chinese_ngrams

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
PUNCT = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
ASSET_SQL = """
    lower(url) LIKE '%.jpg%' OR lower(url) LIKE '%.jpeg%' OR lower(url) LIKE '%.png%'
    OR lower(url) LIKE '%.gif%' OR lower(url) LIKE '%.webp%' OR lower(url) LIKE '%.bmp%'
    OR lower(url) LIKE '%.css%' OR lower(url) LIKE '%.js%' OR lower(url) LIKE '%.mp4%'
    OR lower(url) LIKE '%.swf%' OR lower(url) LIKE '%.ppt%'
    OR lower(url) LIKE '%/_upload/article/images/%'
    OR lower(url) LIKE '%/_upload/article/videos/%'
    OR lower(url) LIKE '%/_upload/column/%'
    OR lower(url) LIKE '%visitcount%'
"""


def cjk_count(text: str | None) -> int:
    return len(CJK_RE.findall(text or ""))


def repair_text(text: str | None) -> str | None:
    """Restore UTF-8 that was decoded as Latin-1 / cp1252, when that yields Chinese."""
    if not text or cjk_count(text) >= 2:
        return text
    if all(ord(ch) < 128 for ch in text):
        return text
    for enc in ("latin1", "cp1252"):
        try:
            fixed = text.encode(enc).decode("utf-8")
        except Exception:
            continue
        if fixed != text and cjk_count(fixed) >= 2:
            return fixed
    return text


def repair_json(text: str | None) -> str | None:
    if not text or cjk_count(text) >= 2 and not any(ord(ch) > 127 and not ("\u4e00" <= ch <= "\u9fff") for ch in text[:200]):
        # Still walk JSON when the blob itself may mix mojibake cells with Chinese keys.
        if not text or "å" not in text and "Ã" not in text and "æ" not in text:
            return text
    try:
        data = json.loads(text)
    except Exception:
        return repair_text(text)

    changed = False

    def walk(value):
        nonlocal changed
        if isinstance(value, str):
            fixed = repair_text(value)
            if fixed != value:
                changed = True
            return fixed
        if isinstance(value, list):
            return [walk(item) for item in value]
        if isinstance(value, dict):
            return {key: walk(item) for key, item in value.items()}
        return value

    updated = walk(data)
    if not changed:
        return text
    return json.dumps(updated, ensure_ascii=False)


def is_garbage(text: str | None) -> bool:
    raw = (text or "").strip()
    if len(raw) < 40:
        return False
    cjk = cjk_count(raw)
    ratio = cjk / len(raw)
    letters = sum(ch.isascii() and ch.isalpha() for ch in raw)
    if letters / len(raw) > 0.45 and ratio < 0.05:
        return False
    weird = sum(1 for ch in raw if ord(ch) > 127 and not ("\u4e00" <= ch <= "\u9fff"))
    punct = sum(ch in PUNCT for ch in raw)
    if ratio < 0.12 and punct / len(raw) > 0.22:
        return True
    if ratio < 0.08 and weird / len(raw) > 0.12:
        return True
    if len(raw) >= 80 and ratio < 0.02 and letters / len(raw) < 0.2:
        return True
    return False


def useful_cjk(text: str | None) -> int:
    if is_garbage(text):
        return 0
    return cjk_count(text)


def _protected_ids(conn) -> set[int]:
    ids: set[int] = set()
    for sql in (
        "SELECT DISTINCT document_id FROM admission_plans WHERE document_id IS NOT NULL",
        "SELECT DISTINCT document_id FROM fee_rules WHERE document_id IS NOT NULL",
        "SELECT DISTINCT document_id FROM lab_centers WHERE document_id IS NOT NULL",
        "SELECT DISTINCT source_document_id FROM facts WHERE source_document_id IS NOT NULL",
    ):
        ids.update(int(row[0]) for row in conn.execute(sql) if row[0] is not None)
    ids.update(
        int(row[0])
        for row in conn.execute(
            """
            SELECT d.id FROM documents d
            JOIN sources s ON s.id = d.source_id
            WHERE s.url LIKE 'local://%'
            """
        )
    )
    return ids


def _repair(conn) -> dict[str, int]:
    counts = {"titles": 0, "blocks": 0, "cells": 0, "rows": 0, "tables": 0, "units": 0, "notes": 0}
    title_updates = []
    for row in conn.execute("SELECT id, title FROM documents"):
        fixed = repair_text(row["title"])
        if fixed != row["title"]:
            title_updates.append((fixed, row["id"]))
    conn.executemany("UPDATE documents SET title=? WHERE id=?", title_updates)
    counts["titles"] = len(title_updates)

    block_updates = []
    for row in conn.execute("SELECT id, text FROM blocks"):
        fixed = repair_text(row["text"])
        if fixed != row["text"]:
            block_updates.append((fixed, row["id"]))
    conn.executemany("UPDATE blocks SET text=? WHERE id=?", block_updates)
    counts["blocks"] = len(block_updates)

    cell_updates = []
    for row in conn.execute("SELECT id, value FROM table_cells"):
        fixed = repair_text(row["value"])
        if fixed != row["value"]:
            cell_updates.append((fixed, row["id"]))
    conn.executemany("UPDATE table_cells SET value=? WHERE id=?", cell_updates)
    counts["cells"] = len(cell_updates)

    row_updates = []
    for row in conn.execute("SELECT id, text FROM table_rows"):
        fixed = repair_text(row["text"])
        if fixed != row["text"]:
            row_updates.append((fixed, row["id"]))
    conn.executemany("UPDATE table_rows SET text=? WHERE id=?", row_updates)
    counts["rows"] = len(row_updates)

    table_updates = []
    for row in conn.execute("SELECT id, raw_json, header_json, title FROM tables"):
        raw = repair_json(row["raw_json"])
        header = repair_json(row["header_json"])
        title = repair_text(row["title"])
        if raw != row["raw_json"] or header != row["header_json"] or title != row["title"]:
            table_updates.append((raw, header, title, row["id"]))
    conn.executemany("UPDATE tables SET raw_json=?, header_json=?, title=? WHERE id=?", table_updates)
    counts["tables"] = len(table_updates)

    unit_updates = []
    for row in conn.execute("SELECT id, title, text FROM retrieval_units"):
        title = repair_text(row["title"])
        text = repair_text(row["text"])
        if title != row["title"] or text != row["text"]:
            search = chinese_ngrams(f"{title or ''} {text or ''}")
            unit_updates.append((title, text, search, row["id"]))
    conn.executemany("UPDATE retrieval_units SET title=?, text=?, search_text=? WHERE id=?", unit_updates)
    counts["units"] = len(unit_updates)

    note_updates = []
    for row in conn.execute("SELECT id, notes FROM sources WHERE notes IS NOT NULL"):
        fixed = repair_text(row["notes"])
        if fixed != row["notes"]:
            note_updates.append((fixed, row["id"]))
    conn.executemany("UPDATE sources SET notes=? WHERE id=?", note_updates)
    counts["notes"] = len(note_updates)
    return counts


def _wipe(conn, doc_ids: list[int]) -> None:
    if not doc_ids:
        return
    conn.execute("DROP TABLE IF EXISTS _gone")
    conn.execute("CREATE TEMP TABLE _gone(id INTEGER PRIMARY KEY)")
    conn.executemany("INSERT INTO _gone(id) VALUES(?)", [(doc_id,) for doc_id in doc_ids])
    conn.execute("DELETE FROM retrieval_units WHERE document_id IN (SELECT id FROM _gone)")
    conn.execute(
        """
        DELETE FROM table_cells WHERE table_id IN (
            SELECT id FROM tables WHERE document_id IN (SELECT id FROM _gone)
        )
        """
    )
    conn.execute(
        """
        DELETE FROM table_rows WHERE table_id IN (
            SELECT id FROM tables WHERE document_id IN (SELECT id FROM _gone)
        )
        """
    )
    conn.execute("DELETE FROM tables WHERE document_id IN (SELECT id FROM _gone)")
    conn.execute("DELETE FROM blocks WHERE document_id IN (SELECT id FROM _gone)")
    conn.execute("DELETE FROM quality_queue WHERE document_id IN (SELECT id FROM _gone)")
    conn.execute(
        """
        UPDATE documents
        SET parse_status='discarded', quality_flags='garbled_or_empty'
        WHERE id IN (SELECT id FROM _gone)
        """
    )


def cleanup() -> dict:
    with session() as conn:
        repaired = _repair(conn)
        protected = _protected_ids(conn)
        block_cjk: dict[int, int] = defaultdict(int)
        garbage_blocks: list[int] = []
        for row in conn.execute("SELECT id, document_id, text FROM blocks"):
            if is_garbage(row["text"]):
                garbage_blocks.append(int(row["id"]))
            else:
                block_cjk[int(row["document_id"])] += cjk_count(row["text"])
        cell_cjk: dict[int, int] = defaultdict(int)
        for row in conn.execute(
            """
            SELECT t.document_id AS document_id, c.value AS value
            FROM table_cells c JOIN tables t ON t.id = c.table_id
            """
        ):
            if row["document_id"] is None or is_garbage(row["value"]):
                continue
            cell_cjk[int(row["document_id"])] += cjk_count(row["value"])

        discard: list[int] = []
        examples: list[str] = []
        by_type: dict[str, int] = defaultdict(int)
        docs = conn.execute("SELECT id, title, doc_type, parse_status FROM documents").fetchall()
        for doc in docs:
            doc_id = int(doc["id"])
            if doc_id in protected or doc["parse_status"] == "discarded":
                continue
            useful = block_cjk[doc_id] + cell_cjk[doc_id]
            kind = doc["doc_type"] or ""
            drop = False
            if kind in {"bin", "zip"}:
                drop = True
            elif kind == "image":
                drop = useful < 80
            elif useful < 12:
                drop = True
            if not drop:
                continue
            discard.append(doc_id)
            by_type[kind or "unknown"] += 1
            if len(examples) < 12:
                examples.append(f"{doc_id} {(doc['title'] or '')[:40]}")

        _wipe(conn, discard)
        conn.execute("DROP TABLE IF EXISTS _gblocks")
        conn.execute("CREATE TEMP TABLE _gblocks(id INTEGER PRIMARY KEY)")
        if garbage_blocks:
            conn.executemany("INSERT OR IGNORE INTO _gblocks(id) VALUES(?)", [(block_id,) for block_id in garbage_blocks])
        kept_garbage_blocks = [
            int(row[0])
            for row in conn.execute("SELECT id FROM blocks WHERE id IN (SELECT id FROM _gblocks)")
        ]
        conn.execute("DELETE FROM blocks WHERE id IN (SELECT id FROM _gblocks)")

        garbage_units = [
            int(row["id"])
            for row in conn.execute("SELECT id, document_id, text FROM retrieval_units")
            if int(row["document_id"] or 0) not in set(discard) and is_garbage(row["text"])
        ]
        # Discarded documents already lost their units. Skip protected bodies only when the
        # unit itself is unreadable; structured rows with Chinese headers are kept.
        if garbage_units:
            conn.executemany("DELETE FROM retrieval_units WHERE id=?", [(unit_id,) for unit_id in garbage_units])

        conn.execute(
            f"""
            DELETE FROM sources
            WHERE id NOT IN (SELECT source_id FROM documents WHERE source_id IS NOT NULL)
              AND ({ASSET_SQL})
            """
        )
        removed_sources = conn.execute("SELECT changes()").fetchone()[0]
        rebuild_fts(conn)
        stats = {
            "repaired": repaired,
            "discarded_documents": len(discard),
            "discarded_by_type": dict(by_type),
            "discarded_examples": examples,
            "garbage_blocks_removed": len(kept_garbage_blocks),
            "garbage_units_removed": len(garbage_units),
            "asset_sources_removed": int(removed_sources or 0),
            "protected_documents": len(protected),
            "documents_left": conn.execute("SELECT COUNT(*) FROM documents WHERE parse_status!='discarded'").fetchone()[0],
            "retrieval_units_left": conn.execute("SELECT COUNT(*) FROM retrieval_units").fetchone()[0],
        }
    return stats
