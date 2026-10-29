"""证据集合组装：多槽位问题按「每个槽位都有适用证据」作答。

和「哪句话最像答案」不同。这里先列出必需槽位，再在同一适用
范围内把值和槽位绑上。允许：

- 一个句子覆盖多个槽位（教职员工、离退休人员借期 90 天）；
- 不同槽位取同一个值；
- 证据来自同一文档族里的不同段落。

不允许：同一槽位在该范围内出现两个未消解的值。缺任一槽位、
或范围里有冲突，就不提交候选，原答案路径继续。
"""

from __future__ import annotations

import re
from typing import Any

from .db import init_db
from .docresolve import resolve_documents

_CN = {"一": "1", "二": "2", "三": "3", "四": "4", "五": "5", "六": "6"}
_SPLIT = re.compile(r"[。；;\n]")
_COMMA = re.compile(r"[，,]")


def _slot_groups(question: str) -> list[list[str]]:
    """「A、B和C分别」里的槽位。

    枚举在「分别」前面的最后一个逗号之后。最后一项常粘着
    「学生申请…」「的图书借期」，先切掉再拆。第一项可能粘着
    「列出的」，所以再按后缀生成几组候选，由取值是否全部成功来选。
    """
    idx = question.find("分别")
    if idx < 0:
        return []
    left = question[:idx]
    enum_at = max(left.rfind("、"), left.rfind("和"), left.rfind("与"))
    if enum_at < 0:
        return []
    # 逗号可能在枚举之后（「三年级学生申请跳级时，允许的…分别」），
    # 不能用「最后一个逗号」把枚举切掉。
    cut = max(left.rfind("，", 0, enum_at), left.rfind(",", 0, enum_at), left.rfind("。", 0, enum_at))
    window = left[cut + 1 :]
    if "、" not in window and "和" not in window and "与" not in window:
        return []
    raw = [p.strip() for p in re.split(r"[、和与]", window) if p.strip()]
    if len(raw) < 2:
        return []
    cleaned: list[str] = []
    for i, part in enumerate(raw):
        if i == len(raw) - 1:
            part = re.split(r"的|申请|时|允许|对应|，|,", part)[0]
            part = re.sub(r"学生$", "", part)
        part = re.sub(r"(地址|电话|学费|借期|时间|标准|上限|范围)$", "", part)
        cleaned.append(part.strip())
    if any(len(p) < 2 for p in cleaned[1:]):
        return []
    groups: list[list[str]] = []
    first = cleaned[0]
    for length in range(min(8, len(first)), 1, -1):
        head = first[-length:]
        group = [head, *cleaned[1:]]
        if all(2 <= len(p) <= 8 for p in group) and group not in groups:
            groups.append(group)
    return groups


def _aliases(slot: str) -> list[str]:
    names = [slot]
    m = re.fullmatch(r"([一二三四五六])年级", slot)
    if m:
        names.append(_CN[m.group(1)] + "年级")
    if "教职" in slot:
        names.extend(["教职员工", "教职工"])
    if slot.endswith("校区"):
        names.append(slot + "地址")
    # 去重保序
    out: list[str] = []
    for n in names:
        if n not in out:
            out.append(n)
    return out


def _value_spec(question: str) -> tuple[str, re.Pattern[str]]:
    if re.search(r"多少天|几天|借期|期限", question):
        return "day", re.compile(r"(?<!\d)(\d{1,3})\s*天")
    if "学分" in question:
        return "credit", re.compile(r"(?<!\d)(\d{1,3})\s*学分")
    if re.search(r"地址|在哪|哪里", question):
        return "address", re.compile(r"([一-鿿]{2,16}(?:路|街|道|巷)\s*\d+\s*号)")
    if "电话" in question:
        return "phone", re.compile(r"(?<!\d)(\d{7,8})(?!\d)")
    if re.search(r"多少元|学费|费用|收费", question):
        return "money", re.compile(r"(?<!\d)(\d{2,6}(?:\.\d+)?)\s*元")
    # 「多少人」不能把「17 个」当成人数。
    if re.search(r"多少人|几人|多少名|入选多少|多少毕业生", question):
        return "person", re.compile(r"(?<!\d)(\d{1,6})\s*人")
    return "number", re.compile(r"(?<!\d)(\d+(?:\.\d+)?)(?!\d)")


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _values(kind: str, pattern: re.Pattern[str], text: str) -> list[str]:
    found = [_norm(m.group(1) if m.lastindex else m.group(0)) for m in pattern.finditer(text)]
    if kind == "number":
        found = [v for v in found if not re.fullmatch(r"20\d{2}|19\d{2}", v)]
    # 保序去重
    out: list[str] = []
    for v in found:
        if v not in out:
            out.append(v)
    return out


def _units(conn: Any, question: str, doc_ids: list[int]) -> list[dict[str, Any]]:
    """优先在锁定文档里取块。没锁定时只用够具体的属性词，不用「元」这种单字。"""
    kind, _ = _value_spec(question)
    hint = ""
    for word in ("跳级", "借期", "学校地址", "地址", "电话"):
        if word in question:
            hint = word
            break
    if not hint:
        hint = {"day": "借期", "credit": "学分", "phone": "电话"}.get(kind, "")
    if doc_ids:
        marks = ",".join("?" * len(doc_ids))
        sql = f"""
            SELECT ru.id, ru.document_id, ru.text, d.title, d.url
            FROM retrieval_units ru JOIN documents d ON d.id = ru.document_id
            WHERE ru.document_id IN ({marks})
        """
        params: tuple[Any, ...] = tuple(doc_ids)
        if hint:
            sql += " AND ru.text LIKE ?"
            params = (*params, f"%{hint}%")
        sql += " LIMIT 40"
        rows = conn.execute(sql, params).fetchall()
        if not rows and hint:
            rows = conn.execute(
                f"""
                SELECT ru.id, ru.document_id, ru.text, d.title, d.url
                FROM retrieval_units ru JOIN documents d ON d.id = ru.document_id
                WHERE ru.document_id IN ({marks}) LIMIT 40
                """,
                tuple(doc_ids),
            ).fetchall()
    else:
        if not hint or hint in {"学分", "地址", "元"}:
            return []
        rows = conn.execute(
            """
            SELECT ru.id, ru.document_id, ru.text, d.title, d.url
            FROM retrieval_units ru JOIN documents d ON d.id = ru.document_id
            WHERE ru.text LIKE ? LIMIT 40
            """,
            (f"%{hint}%",),
        ).fetchall()
    out = []
    for rid, did, text, title, url in rows:
        out.append({
            "id": int(rid),
            "document_id": int(did),
            "text": text or "",
            "title": title or "",
            "url": url or "",
        })
    return out


def _bind(question: str, slots: list[str], units: list[dict[str, Any]]) -> dict[str, Any] | None:
    kind, pattern = _value_spec(question)
    # slot -> list of (value, quote, unit)
    bound: dict[str, list[tuple[str, str, dict[str, Any]]]] = {s: [] for s in slots}
    alias_map = {s: _aliases(s) for s in slots}
    for unit in units:
        for seg in _SPLIT.split(unit["text"]):
            if len(seg) > 220:
                continue
            pieces = _COMMA.split(seg) if kind != "address" else [seg]
            # 地址常常用分号隔开两个校区，上面已经按分号切开了。
            for piece in pieces:
                if not piece or len(piece) > 180:
                    continue
                flat = _norm(piece)
                present = [s for s in slots if any(_norm(a) in flat for a in alias_map[s])]
                vals = _values(kind, pattern, piece)
                if not present or len(vals) != 1:
                    continue
                # 年级数字本身不能当成学分或天数。
                if kind in {"credit", "day", "number"} and any(
                    _norm(a).startswith(vals[0]) for s in present for a in alias_map[s]
                ):
                    continue
                for s in present:
                    bound[s].append((vals[0], piece.strip(), unit))
    conflicts: list[str] = []
    missing: list[str] = []
    chosen: list[tuple[str, str, str, dict[str, Any]]] = []
    for s in slots:
        vals = {v for v, _, _ in bound[s]}
        if not vals:
            missing.append(s)
            continue
        if len(vals) > 1:
            conflicts.append(s)
            continue
        val, quote, unit = bound[s][0]
        chosen.append((s, val, quote, unit))
    if missing or conflicts or len(chosen) != len(slots):
        return {
            "validation_status": "incomplete",
            "covered_slots": [s for s, _, _, _ in chosen],
            "missing_slots": missing,
            "conflicts": conflicts,
            "answer": "",
        }
    # 按证据原文顺序输出，同一句只出现一次。
    quotes: list[str] = []
    for _, _, quote, _ in chosen:
        if quote not in quotes:
            quotes.append(quote)
    unit = chosen[0][3]
    return {
        "validation_status": "verified",
        "covered_slots": [s for s, _, _, _ in chosen],
        "missing_slots": [],
        "conflicts": [],
        "values": {s: v for s, v, _, _ in chosen},
        "evidence_ids": sorted({u["id"] for *_, u in chosen}),
        "answer": "；".join(quotes),
        "doc_title": unit["title"],
        "url": unit["url"],
    }


def assemble_answer(question: str) -> dict[str, Any] | None:
    """多槽位组装入口。不是「分别」题，或没有任何一组槽位能验证，返回 None。"""
    if "分别" not in (question or ""):
        return None
    groups = _slot_groups(question)
    if not groups:
        return None
    scope = resolve_documents(question)
    doc_ids = scope["document_ids"] if scope["confidence"] == "high" else []
    conn = init_db()
    try:
        units = _units(conn, question, doc_ids)
    finally:
        conn.close()
    if not units:
        return None
    best = None
    for slots in groups:
        got = _bind(question, slots, units)
        if not got:
            continue
        if got["validation_status"] == "verified":
            got["scope"] = scope
            return got
        if best is None:
            best = got
    return best
