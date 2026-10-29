"""文档实体解析：点名文件时先锁定文档族，再在族内找证据。

高置信度才返回文档 id。问句没有点名文件、或年份对不上时返回空，
调用方继续全库召回，避免误锁。
"""

from __future__ import annotations

import re
from typing import Any

from .db import init_db

# 试点族来自交接里的高失败来源，再加上招生章程（地址题点名它）。
# cues 按长度优先匹配，避免「综合评价」抢走「综合评价招生简章」。
_FAMILIES: list[dict[str, Any]] = [
    {"id": "quality_report", "cues": ("本科教学质量报告", "教学质量报告"), "like": "%教学质量报告%"},
    {"id": "status_rules", "cues": ("学籍管理办法",), "like": "%学籍管理办法%"},
    {"id": "loan_rules", "cues": ("借阅规则", "读者借阅"), "like": "%借阅%"},
    {"id": "comp_eval", "cues": ("综合评价招生简章",), "like": "%综合评价招生简章%"},
    {"id": "retest_notice", "cues": ("复试考生须知", "复试须知"), "like": "%复试%须知%"},
    {"id": "admission_charter", "cues": ("招生章程",), "like": "%招生章程%"},
]


def _years(text: str) -> list[int]:
    return [int(y) for y in re.findall(r"20\d{2}", text or "")]


def resolve_documents(question: str) -> dict[str, Any]:
    """返回 {family, year, document_ids, confidence}。

    confidence 为 high 时，document_ids 是「问句点名的那一版」；
    否则 document_ids 为空。
    """
    q = question or ""
    family = None
    for fam in _FAMILIES:
        if any(cue in q for cue in fam["cues"]):
            family = fam
            break
    if family is None:
        return {"family": None, "year": None, "document_ids": [], "confidence": "none"}
    asked = _years(q)
    year = asked[-1] if asked else None
    conn = init_db()
    try:
        rows = conn.execute(
            "SELECT id, title FROM documents WHERE title LIKE ?",
            (family["like"],),
        ).fetchall()
        titled = [(int(i), t or "") for i, t in rows if t]

        def lengths(ids: list[int]) -> dict[int, int]:
            if not ids:
                return {}
            marks = ",".join("?" * len(ids))
            found = conn.execute(
                f"""SELECT document_id, COALESCE(SUM(length(text)),0)
                    FROM retrieval_units WHERE document_id IN ({marks})
                    GROUP BY document_id""",
                ids,
            ).fetchall()
            return {int(i): int(n) for i, n in found}

        def substantial(ids: list[int]) -> list[int]:
            lens = lengths(ids)
            long = [i for i in ids if lens.get(i, 0) >= 2000]
            return long or ids

        if year is not None:
            matched = [i for i, t in titled if year in _years(t)]
            # 栏目页标题带年份、正文只是页脚。正文在标题不含年份的 PDF 里。
            # 标题命中的文档都太短时，改锁正文里写了该年份、且足够长的同族文档。
            if matched and all(n < 2000 for n in lengths(matched).values()):
                body = conn.execute(
                    """
                    SELECT d.id FROM documents d
                    JOIN retrieval_units ru ON ru.document_id = d.id
                    WHERE d.title LIKE ? AND ru.text LIKE ?
                    GROUP BY d.id
                    HAVING COALESCE(SUM(length(ru.text)),0) >= 2000
                    """,
                    (family["like"], f"%{year}%"),
                ).fetchall()
                if body:
                    matched = [int(i) for (i,) in body]
            if matched:
                return {
                    "family": family["id"],
                    "year": year,
                    "document_ids": substantial(matched),
                    "confidence": "high",
                }
            return {
                "family": family["id"],
                "year": year,
                "document_ids": [],
                "confidence": "miss",
            }
        # 没点年份：只在族内标题唯一时锁定，多版本不猜。
        if len(titled) == 1:
            return {
                "family": family["id"],
                "year": None,
                "document_ids": [titled[0][0]],
                "confidence": "high",
            }
        return {"family": family["id"], "year": None, "document_ids": [], "confidence": "ambiguous"}
    finally:
        conn.close()


def focus_hits(question: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """问句点了文件名，而且检索结果里已经有这份文件时，丢掉其他文件。

    保持原来的先后顺序，不按重合度重排，也不把文件里没被检索到的段落补进来。
    """
    if not hits:
        return hits
    cue = ""
    for fam in _FAMILIES:
        for name in fam["cues"]:
            if name in (question or "") and len(name) > len(cue):
                cue = name
    if len(cue) < 4:
        return hits
    kept = [h for h in hits if cue in (h.get("doc_title") or "")]
    return kept or hits
