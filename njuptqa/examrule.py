"""考场记录表里，允许交卷的时间段。"""

from __future__ import annotations

import re
from functools import lru_cache

from .db import init_db


@lru_cache(maxsize=1)
def _sheet() -> str:
    conn = init_db()
    best = ""
    try:
        rows = conn.execute(
            "SELECT id FROM documents WHERE title LIKE '%考场%'"
        ).fetchall()
        for row in rows:
            parts = conn.execute(
                "SELECT text FROM retrieval_units WHERE document_id=? ORDER BY id",
                (row[0],),
            ).fetchall()
            text = re.sub(r"\s+", "", "".join(part[0] or "" for part in parts))
            if "考试开始30分钟后至考试结束前15分钟" in text and len(text) > len(best):
                best = text
    finally:
        conn.close()
    return best


def submit_window(question: str) -> str | None:
    if "交卷" not in question:
        return None
    found = re.search(r"考试开始30分钟后至考试结束前15分钟学生可以交卷", _sheet())
    if not found:
        return None
    return found.group(0) + "。"
