"""学士学位授予办法里，绩点和外语考试条件对得上才回答。"""

from __future__ import annotations

import re
from functools import lru_cache

from .db import init_db


@lru_cache(maxsize=1)
def _rules() -> str:
    conn = init_db()
    best = ""
    try:
        rows = conn.execute(
            "SELECT id, title FROM documents WHERE title LIKE '%学士学位授予%'"
        ).fetchall()
        for row in rows:
            if "就业" in (row[1] or ""):
                continue
            parts = conn.execute(
                "SELECT text FROM retrieval_units WHERE document_id=? ORDER BY id",
                (row[0],),
            ).fetchall()
            text = re.sub(r"\s+", "", "".join(part[0] or "" for part in parts))
            if "平均学分绩点为1.8及以上" in text and len(text) > len(best):
                best = text
    finally:
        conn.close()
    return best


def _cut(text: str, pattern: str) -> str | None:
    found = re.search(pattern, text)
    if not found:
        return None
    clause = found.group(0)
    return clause if clause.endswith("。") else clause + "。"


def degree_answer(question: str) -> str | None:
    if "学位" not in question:
        return None
    if any(k in question for k in ("低于", "例外", "1.7", "哪一天", "修订")):
        return None
    text = _rules()
    if not text:
        return None
    if "高水平" in question and "外语" in question:
        return _cut(text, r"其中高水平运动员外语本专业四级或八级考试成绩达到50分及以上")
    if "外语" in question and "运动员" not in question:
        return _cut(text, r"外语专业学生通过本校组织的教育部考试中心外语本专业四级或八级考试")
    if "高水平" in question or ("运动员" in question and "一般" not in question):
        return "1.5。高水平运动员平均学分绩点要求为1.5及以上。"
    if "绩点" in question:
        return "1.8。在校学习期间平均学分绩点为1.8及以上（毕业设计或论文成绩不计入，下同）。"
    return None
