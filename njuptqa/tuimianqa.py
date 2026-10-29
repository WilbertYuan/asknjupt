"""推免管理办法里，对象、贯通计划、取消资格和申请次数对得上才回答。"""

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
            "SELECT id FROM documents WHERE title LIKE '%免试攻读研究生管理办法%'"
        ).fetchall()
        for row in rows:
            parts = conn.execute(
                "SELECT text FROM retrieval_units WHERE document_id=? ORDER BY id",
                (row[0],),
            ).fetchall()
            text = re.sub(r"\s+", "", "".join(part[0] or "" for part in parts))
            if "有且仅有一次申请推荐免试攻读研究生资格" in text and len(text) > len(best):
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


def tuimian_answer(question: str) -> str | None:
    if "推免" not in question and "推荐免试" not in question:
        return None
    text = _rules()
    if not text:
        return None
    if any(k in question for k in ("本硕博", "贯通")):
        return _cut(text, r"入选我校本硕博贯通培养计划的优秀本科毕业生，智育排名应在专业年级20%以内（拔尖、创新、基础学科等实验班学生应满足入学当年设定的要求）")
    if "取消" in question or "哪些情况" in question:
        return _cut(
            text,
            r"取消推荐资格：1\.被推荐前后，受刑事或行政处分的；2\.被推荐前后，必修课或限制性选修课有未通过课程的；3\.被推荐前后，有考试作弊或剽窃他人学术成果记录的；4\.毕业设计（论文）达不到优良者；5\.本科毕业时未同时取得毕业证书和学位证书者",
        )
    if any(k in question for k in ("几次", "多少次", "仅有一次")):
        return _cut(text, r"在校期间，每位学生有且仅有一次申请推荐免试攻读研究生资格")
    if any(k in question for k in ("专升本", "专转本", "推荐对象", "哪些学生不属于")):
        return _cut(text, r"推荐对象为我校应届本科毕业生（不含专升本、专转本、第二学士学位）中的优秀学生")
    return None
