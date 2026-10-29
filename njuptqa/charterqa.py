"""普通本科招生章程里能对上原句的条款。对不上就返回空，交给后面的计算和原文。"""

from __future__ import annotations

import re

from .db import init_db

_CACHE: dict[int, str] = {}


def _compact(year: int) -> str:
    if year in _CACHE:
        return _CACHE[year]
    conn = init_db()
    rows = conn.execute(
        "SELECT id, title FROM documents WHERE title LIKE ?",
        (f"%{year}年%招生章程%",),
    ).fetchall()
    best = ""
    for row in rows:
        title = row["title"] or ""
        if "就业" in title or "硕士" in title:
            continue
        parts = conn.execute(
            "SELECT text FROM retrieval_units WHERE document_id=? ORDER BY id",
            (row["id"],),
        ).fetchall()
        text = re.sub(r"\s+", "", "".join(part["text"] or "" for part in parts))
        if len(text) > len(best):
            best = text
    conn.close()
    _CACHE[year] = best
    return best


def _year(question: str) -> int | None:
    found = []
    for item in re.findall(r"20\d{2}", question):
        if item not in found:
            found.append(item)
    if len(found) == 1:
        return int(found[0])
    return None


def _clause(question: str, pattern: str, years: tuple[int, ...] = (2024, 2025, 2026)) -> str | None:
    """问句写了年份就只用那一年。没写年份时，列出的章程里这句话必须完全一样。"""
    year = _year(question)
    if year:
        found = re.search(pattern, _compact(year))
        return found.group(0) if found else None
    versions = []
    for item in years:
        found = re.search(pattern, _compact(item))
        versions.append(found.group(0) if found else "")
    if versions[0] and all(item == versions[0] for item in versions):
        return versions[0]
    return None


def charter_answer(question: str) -> str | None:
    if "硕士" in question or "研究生" in question:
        return None
    reserve = _clause(
        question,
        r"预留计划数不超过年度本科招生计划总数的1%，用于调节各地统考上线生源的不平衡[^。]*。",
    )
    if reserve and "预留" in question and any(k in question for k in ("比例", "用途", "百分之", "多少")):
        if "人数" not in question:
            return reserve
    languages = _clause(
        question,
        r"英语专业招生语种要求为英语，其他专业不限制考生应试的外语语种，但学校的公共外语课程只开设英语课程。",
    )
    if languages and "语种" in question and "英语" in question:
        return languages
    physical = _clause(
        question,
        r"除体检不合格考生外，我校相关专业没有体检要求限制。对录取新生的男女比例无要求。",
    )
    if physical and "体检" in question and "男女" in question:
        return physical
    tuition = _clause(
        question,
        r"普通本科新生学费预收标准：文科类专业每生每年\d+元，理科类专业每生每年\d+元，工科类专业每生每年\d+元，艺术类专业每生每年\d+元",
    )
    if (
        tuition
        and "预收" in question
        and "优势" not in question
        and all(k in question for k in ("文科", "理科", "工科"))
    ):
        return tuition + "。"
    # 这条调剂规则 2024 年章程还没有，2025 和 2026 的原句一致。
    obey = _clause(
        question,
        r"服从调剂的考生按照.?分数优先.?的原则，调剂到相同专业组（科类）内符合录取条件且未录满的专业",
        (2025, 2026),
    )
    denies = _clause(question, r"不服从专业调剂的考生，作退档处理", (2025, 2026))
    # 「服从」是「不服从」的子串，先拿掉否定再判断有没有正面的服从。
    bare = question.replace("不服从", "")
    both = "服从" in bare and "不服从" in question
    only_deny = "不服从" in question and "服从" not in bare
    only_obey = "服从" in bare and "不服从" not in question
    if "调剂" in question and "无法满足" in question:
        parts = []
        if (both or only_obey) and obey:
            parts.append(obey + "。")
        if (both or only_deny) and denies:
            parts.append(denies + "。")
        if parts:
            return "".join(parts)
    phones = _phones(question)
    if phones:
        return phones
    agency = _clause(
        question,
        r"不委托任何中介机构或个人进行招生录取工作，不收取国家规定以外的任何费用。",
    )
    if agency and "中介" in question:
        if any(k in question for k in ("是否", "有没有", "授权")):
            return "没有。" + agency
        return agency
    diploma = _diploma(question)
    if diploma:
        return diploma
    if "域名" in question:
        domain = _clause(question, r"学校域名：www\.njupt\.edu\.cn")
        if domain:
            return domain
    return None


def _phones(question: str) -> str | None:
    if "咨询电话" not in question or "监督" not in question:
        return None
    year = _year(question)
    if year is None:
        return None
    text = _compact(year)
    found = re.search(r"（0?25）(\d{8})(\d{8})", text)
    watch = re.search(r"监督电话：?（0?25）(\d{8})", text)
    if not found or not watch:
        return None
    line = f"咨询电话025-{found.group(1)}、025-{found.group(2)}；监督电话025-{watch.group(1)}"
    if "邮箱" in question:
        mail = re.search(r"监督邮箱：?([A-Za-z0-9._-]+@[A-Za-z0-9.-]+)", text)
        if mail:
            line += f"；监督邮箱{mail.group(1)}"
    return line + "。"


def _diploma(question: str) -> str | None:
    if not ("毕业" in question and "学位" in question and "证书" in question):
        return None
    grad = _clause(question, r"颁发南京邮电大学本科毕业证书。")
    degree = _clause(question, r"符合学士学位授予条件者，授予南京邮电大学颁发的学士学位证书。")
    if grad and degree:
        return grad + degree
    return None
