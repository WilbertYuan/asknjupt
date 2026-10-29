"""选读课程细则里对得上的两条：加修审批和补改选时间。"""

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
        "SELECT id FROM documents WHERE title LIKE '%选读课程实施细则%'"
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


def course_answer(question: str) -> str | None:
    text = _rules()
    if not text:
        return None
    if "加修" in question and "学费" not in question and "收费" not in question and any(k in question for k in ("审批", "申请", "手续")):
        found = re.search(r"加（选）修其他专业课程的学生，必须在规定选课时间内上网选课、填写《学生加（选）修课程申请表》，经学生所在学院分管教学院长批准，报教务处备案。", text)
        if found:
            return found.group(0)
    if "补改选" in question and "开学后" in question:
        found = re.search(r"补改选一般在开学后2周内进行", text)
        if found:
            return found.group(0) + "。"
    return None
