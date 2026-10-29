"""学分制收费办法里对得上的四条：预缴公式、超学制、试修退选、转专业。"""

from __future__ import annotations

import re

from .db import init_db

_TEXT: str | None = None


def _rules() -> str:
    global _TEXT
    if _TEXT is not None:
        return _TEXT
    conn = init_db()
    best = ""
    rows = conn.execute(
        "SELECT id FROM documents WHERE title LIKE '%学分制收费管理暂行办法%'"
    ).fetchall()
    for row in rows:
        parts = conn.execute(
            "SELECT text FROM retrieval_units WHERE document_id=? ORDER BY id",
            (row["id"],),
        ).fetchall()
        text = re.sub(r"\s+", "", "".join(part["text"] or "" for part in parts))
        if len(text) > len(best):
            best = text
    conn.close()
    _TEXT = best
    return best


def _cut(pattern: str) -> str | None:
    found = re.search(pattern, _rules())
    if not found:
        return None
    clause = found.group(0)
    return clause if clause.endswith("。") else clause + "。"


def fee_answer(question: str) -> str | None:
    if "学费" not in question and "试修" not in question:
        return None
    if "预缴" in question and any(k in question for k in ("公式", "怎么算", "如何计算")):
        return _cut(r"每年预缴学费[=＝（(].{0,60}学制")
    if "标准学制" in question and "专业学费" in question:
        return _cut(r"超过标准学制年限的学生，如已按标准学制缴纳了专业学费的，则学校不再收取专业学费，只收取所修课程的学分学费")
    if "试修" in question and "退选" in question:
        return _cut(r"学校给予2周的试修时间.{0,160}超过一半的，全额收取该课程的学分学费")
    if "转专业" in question and "补修" in question and "学费" in question:
        return _cut(r"转专业，应按转入专业的专业学费标准缴纳专业学费。如需补修课程，按照补修课程的学分收取学分学费")
    return None
