"""本科教学质量报告的结构化读取。

PDF 抽文本没有落进 tables，数字仍在 retrieval_units 里。
这里只在问句点名这份报告、并且能把栏目和数字对齐时才回答。
对不上就返回 None，后面的路径照旧。
"""

from __future__ import annotations

import re
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache

from .db import init_db

_DISCIPLINES = ("工学", "理学", "管理学", "文学", "教育学", "经济学", "法学", "艺术学")


def _q(value: Decimal | str | float, places: int) -> str:
    quantum = Decimal("1") if places <= 0 else Decimal("0." + "0" * places)
    return format(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP), "f")


def _is_teaching_report(question: str) -> bool:
    if "就业质量报告" in question:
        return False
    return "教学质量报告" in question or "质量报告" in question


@lru_cache(maxsize=1)
def _report_text() -> str:
    conn = init_db()
    try:
        docs = conn.execute(
            "SELECT id FROM documents WHERE title LIKE '%本科教学质量报告%'"
        ).fetchall()
        best = ""
        for (doc_id,) in docs:
            parts = [
                text or ""
                for (text,) in conn.execute(
                    "SELECT text FROM retrieval_units WHERE document_id=? ORDER BY id",
                    (doc_id,),
                )
            ]
            blob = "\n".join(parts)
            if len(blob) > len(best):
                best = blob
    finally:
        conn.close()
    return best


def _slice(text: str, start: str, end: str | None, span: int = 800, last: bool = False) -> str:
    i = text.rfind(start) if last else text.find(start)
    if i < 0:
        return ""
    if end:
        j = text.find(end, i + len(start))
        if j > i:
            return text[i:j]
    return text[i : i + span]


def _count_percent_pairs(block: str) -> list[tuple[int, str]]:
    return [(int(n), pct) for n, pct in re.findall(r"(?<![\d.])(\d+)\s+(\d+\.\d+)%", block)]


def _staff() -> dict[str, list[tuple[str, int, str]]]:
    text = _report_text()
    groups = {
        "职称": (["正高", "副高", "中级及以下"], "职称结构", "学位结构"),
        "学位": (["博士", "硕士", "本科及以下"], "学位结构", "年龄结构"),
        "年龄": (["35岁及以下", "36-45岁", "46-55岁", "56岁及以上"], "年龄结构", "学缘结构"),  # 正文里先出现一次「年龄结构」
        "学缘": (["本校", "境内", "境外"], "学缘结构", "3.1.2"),
    }
    out: dict[str, list[tuple[str, int, str]]] = {}
    for name, (labels, start, end) in groups.items():
        pairs = _count_percent_pairs(_slice(text, start, end, 500, last=(name == "年龄")))
        if len(pairs) >= len(labels):
            out[name] = [(label, num, pct) for label, (num, pct) in zip(labels, pairs)]
    return out


def _disciplines() -> list[tuple[str, int, str, str]]:
    text = _report_text()
    block = _slice(text, "表 2-1", "2.3", 2500) or _slice(text, "表2-1", "2.3", 2500)
    if not block:
        block = text
    found: list[tuple[str, int, str, str]] = []
    for name in _DISCIPLINES:
        matched = re.search(
            rf"{name}\s+(\d+)\s+(\d+\.\d+)%\s*([\s\S]*?)(?={'|'.join(_DISCIPLINES)}|\n2\.3|$)",
            block,
        )
        if not matched:
            continue
        majors = re.sub(r"\s+", "", matched.group(3))
        majors = re.split(r"[。\n]", majors)[0]
        found.append((name, int(matched.group(1)), matched.group(2), majors))
    return found


def _mentioned(question: str, name: str) -> bool:
    """「理学」是「管理学」的后缀，不能按子串算进问句。"""
    masked = question
    for other in _DISCIPLINES:
        if len(other) > len(name) and name in other:
            masked = masked.replace(other, " ")
    return name in masked


def _labs() -> list[tuple[str, str, int]]:
    text = _report_text()
    # 正文会先写「见表 3-2」，表题在后面。
    block = _slice(text, "表 3-2", "表 3-3", 4000, last=True)
    rows: list[tuple[str, str, int]] = []
    for line in block.splitlines():
        matched = re.match(r"\s*\d+\s+(.+?)\s+(\d+)\s*$", line)
        if not matched:
            continue
        body, area = matched.group(1), int(matched.group(2))
        name_match = re.match(r"(.+?(?:中心|实验室|教学部))\s*(.*)", body)
        if not name_match:
            continue
        rows.append((name_match.group(1), name_match.group(2).strip(), area))
    return rows


def _platforms() -> list[tuple[str, str, str, int]]:
    text = _report_text()
    block = _slice(text, "表 3-3", "表 3-4", 2500, last=True)
    return [
        (year, name, level, int(fee))
        for year, name, level, fee in re.findall(
            r"(\d{4})年\s+(\S+)\s+(国家级|省级)\s+(\d+)",
            block,
        )
    ]


def _dachuang() -> dict[str, list[int]] | None:
    text = _report_text()
    block = _slice(text, "表 4-2", "4.8.2", 600) or _slice(text, "表4-2", "4.8.2", 600)
    matched = re.search(
        r"年份\s+2024\s+2023\s+2022\s+合计\s+"
        r"校级\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+"
        r"省级\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+"
        r"国家级\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+"
        r"合计\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)",
        block,
    )
    if not matched:
        return None
    nums = [int(x) for x in matched.groups()]
    keys = ("校级", "省级", "国家级", "合计")
    return {key: nums[i * 4 : (i + 1) * 4] for i, key in enumerate(keys)}


def _norm_lab(name: str) -> str:
    return name.replace("教学", "")


def _find_lab(side: str, labs: list[tuple[str, str, int]]) -> tuple[str, str, int] | None:
    side_n = _norm_lab(re.sub(r"\s+", "", side))
    best: tuple[int, tuple[str, str, int]] | None = None
    for row in labs:
        name_n = _norm_lab(row[0])
        if name_n and (name_n in side_n or side_n in name_n):
            if best is None or len(name_n) > best[0]:
                best = (len(name_n), row)
    return None if best is None else best[1]


def _years(question: str) -> list[str]:
    """去掉「2023—2024学年」这种学年跨度，避免每道题都被当成在问 2024。"""
    plain = re.sub(r"20\d\d\s*[—–－\-~～至到]\s*20\d\d", "", question)
    return re.findall(r"20\d\d", plain)


def _slots(question: str) -> list[str]:
    matched = re.search(
        r"([\u4e00-\u9fffA-Za-z0-9≤≥\-]{2,16}(?:、[\u4e00-\u9fffA-Za-z0-9≤≥\-]{2,16}){1,6})(?:教师|专任教师)?分别",
        question,
    )
    if not matched:
        return []
    slots = []
    for part in matched.group(1).split("、"):
        part = re.sub(r"(专任)?教师$", "", part)
        if part:
            slots.append(part)
    return slots


def _bind_slots(slots: list[str], columns: list[tuple[str, int]]) -> list[tuple[str, int]] | None:
    used: set[str] = set()
    bound: list[tuple[str, int]] = []
    for slot in slots:
        best: tuple[int, int, str, int] | None = None
        for label, value in columns:
            if label in slot or slot in label:
                pos = slot.find(label) if label in slot else 0
                cand = (pos, -len(label), label, value)
                if best is None or cand < best:
                    best = cand
        if best is None or best[2] in used:
            return None
        used.add(best[2])
        bound.append((slot, best[3]))
    return bound


def _money_after(text: str, label: str) -> Decimal | None:
    compact = re.sub(r"\s+", "", text)
    # 间隔不能吃掉数字，否则「25499.04」会被当成「9.04」。
    matched = re.search(rf"{re.escape(label)}[^\d]{{0,12}}(\d+\.\d+)", compact)
    if not matched:
        return None
    return Decimal(matched.group(1))


def _stated_numbers(question: str) -> str | None:
    """运算数已经写在问句里时直接算。同时在问系数的题交给原来的教师公式。"""
    if any(k in question for k in ("折算系数", "反推")):
        return None
    students = re.search(r"折合在校生\s*(\d+(?:\.\d+)?)", question)
    teachers = re.search(r"折合教师(?:总数)?\s*(\d+(?:\.\d+)?)", question)
    if students and teachers and "生师比" in question:
        ratio = Decimal(students.group(1)) / Decimal(teachers.group(1))
        return f"{_q(ratio, 2)}:1"
    books = re.search(r"(\d+(?:\.\d+)?)万册", question)
    if books and "换成册" in question:
        return f"{int(Decimal(books.group(1)) * 10000)}册"
    share = re.search(r"(\d+)名[^，。；]{0,20}中\s*(\d+)名", question)
    if share and "比例" in question:
        pct = Decimal(share.group(2)) / Decimal(share.group(1)) * 100
        return f"{_q(pct, 2)}%"
    surveyed = re.search(r"调查人数为?\s*(\d+)\s*人", question)
    valid = re.search(r"有效问卷\s*(\d+)\s*份", question)
    if surveyed and valid and "回收" in question:
        pct = Decimal(valid.group(1)) / Decimal(surveyed.group(1)) * 100
        return f"{_q(pct, 2)}%"
    senior = re.search(r"正高(?:级)?\s*(\d+)\s*人", question)
    vice = re.search(r"副高(?:级)?\s*(\d+)\s*人", question)
    base = re.search(r"专任教师\s*(\d+)\s*人", question)
    if senior and vice and base and any(k in question for k in ("比例", "占比")):
        pct = (Decimal(senior.group(1)) + Decimal(vice.group(1))) / Decimal(base.group(1)) * 100
        return f"{_q(pct, 2)}%"
    return None


def _from_tables(question: str) -> str | None:
    text = _report_text()
    if not text:
        return None
    staff = _staff()
    if "分别" in question and staff.get("职称"):
        bound = _bind_slots(question and _slots(question), [(n, v) for n, v, _ in staff["职称"]])
        if bound and any("正高" in slot or "副高" in slot or "中级" in slot for slot, _ in bound):
            return "、".join(str(value) for _, value in bound) + "人"
    if staff.get("学位") and all(k in question for k in ("博士", "硕士", "合计", "本科及以下")):
        degree = {name: num for name, num, _ in staff["学位"]}
        total = degree.get("博士", 0) + degree.get("硕士", 0)
        rest = degree.get("本科及以下")
        if total and rest is not None:
            return f"合计{total}人，本科及以下{rest}人"
    if "最多" in question and "年龄" in question and staff.get("年龄"):
        name, num, pct = max(staff["年龄"], key=lambda item: item[1])
        label = "36-45岁" if "36-45" in name or "36" in name else name
        if num == max(item[1] for item in staff["年龄"]):
            return f"{label}组人数最多，{num}人，占{pct}%"
    if "分别" in question and staff.get("学缘") and "本校" in question and "境外" in question:
        slots = _slots(question)
        if slots:
            for label, _num, _pct in staff["学缘"]:
                if slots[0].endswith(label) and len(slots[0]) > len(label):
                    slots[0] = label
                    break
        bound = _bind_slots(slots, [(n, v) for n, v, _ in staff["学缘"]])
        if bound:
            return "，".join(f"{slot}{value}人" for slot, value in bound)
    if any(k in question for k in ("占比", "比例")) and "高级" in question and staff.get("职称"):
        ranks = {name: num for name, num, _ in staff["职称"]}
        if {"正高", "副高"} <= set(ranks):
            # 分母用报告正文的专任教师数，不用三个职称相加时的舍入。
            base = re.search(r"专任教师\s*(\d+)\s*人", re.sub(r"\s+", "", text))
            if base:
                pct = (Decimal(ranks["正高"]) + Decimal(ranks["副高"])) / Decimal(base.group(1)) * 100
                return f"{_q(pct, 2)}%"

    disciplines = _disciplines()
    if disciplines and "学科门类" in question and "最多" in question:
        top = max(disciplines, key=lambda item: item[1])
        return f"{len(disciplines)}个门类，{top[0]}最多，共{top[1]}个"
    if disciplines and "合计" in question and "专业" in question:
        named = [item for item in disciplines if _mentioned(question, item[0])]
        if len(named) >= 2:
            total = sum(item[1] for item in named)
            denom = re.search(r"(\d+)个专业", question)
            if denom:
                pct = Decimal(total) / Decimal(denom.group(1)) * 100
                return f"{total}个，约{_q(pct, 2)}%"
            return f"{total}个"
    if disciplines and any(k in question for k in ("归入", "属于")) and "门类" in question:
        pieces = []
        for name, _count, _pct, majors in disciplines:
            for major in re.findall(r"[\u4e00-\u9fff]{2,12}", majors):
                if major in question and major not in _DISCIPLINES:
                    pieces.append((question.find(major), f"{major}属{name}"))
        pieces = list(dict.fromkeys(pieces))
        if len(pieces) >= 2:
            pieces.sort()
            return "，".join(text for _, text in pieces)

    if "相差" in question and "投档" in question:
        groups = re.findall(r"([\u4e00-\u9fff]{2,8}不限组)最低投档分\s*(\d+)\s*分", text)
        picked = [(name, int(score)) for name, score in groups if name[-4:] in question or name in question]
        # 「江苏省物理不限组」含「物理不限组」。
        if len(picked) < 2:
            picked = [(name, int(score)) for name, score in groups if any(part in name for part in re.findall(r"[\u4e00-\u9fff]{2,6}不限组", question))]
        uniq = list(dict.fromkeys(picked))
        if len(uniq) == 2:
            return f"{abs(uniq[0][1] - uniq[1][1])}分"

    if any(k in question for k in ("相加", "合计")) and "日常运行" in question and "专项" in question and "分别" not in question:
        daily = _money_after(text, "日常运行经费")
        special = _money_after(text, "专项经费")
        if daily and special:
            return f"{_q(daily + special, 2)}万元"
    if "反推" in question and "增加" in question:
        grown = re.search(r"增加\s*(\d+\.\d+)\s*万元", question)
        compact = re.sub(r"\s+", "", text)
        found = re.search(r"(\d+\.\d+)万元，比去年增长(\d+\.\d+)万元", compact)
        if grown and found and found.group(2) == grown.group(1):
            return f"{_q(Decimal(found.group(1)) - Decimal(grown.group(1)), 2)}万元"
    if "合计" in question and "实验经费" in question and "实习经费" in question:
        practice = _money_after(text, "实习经费")
        experiment = _money_after(text, "实验经费")
        if practice and experiment:
            return f"{_q(practice + experiment, 2)}元"
    if "分别" in question and all(k in question for k in ("日常运行", "专项", "实验经费")):
        daily = _money_after(text, "日常运行经费")
        special = _money_after(text, "专项经费")
        experiment = _money_after(text, "实验经费")
        if daily and special and experiment:
            return (
                f"2023年度，学校本科教学日常运行经费总额达{_q(daily, 2)}万元，"
                f"本科教学专项经费达{_q(special, 2)}万元，"
                f"生均本科实验经费{_q(experiment, 2)}元。"
            )
    if all(k in question for k in ("生均占地", "教学行政用房")):
        land = _money_after(text, "生均占地面积为")
        room = _money_after(text, "生均教学行政用房")
        if land and room:
            return f"{_q(land, 2)}平方米、{_q(room, 2)}平方米"
    if "新增" in question and "仪器" in question and any(k in question for k in ("占", "比例")):
        total = _money_after(text, "仪器设备总价值达")
        added = _money_after(text, "新增教学科研仪器设备总值")
        if total and added and total != 0:
            pct = added / total * 100
            return f"新增{_q(added, 2)}万元，占总值{_q(total, 2)}万元约{_q(pct, 2)}%"
    if "回收" in question and "有效" in question and not re.search(r"有效问卷\s*\d+", question):
        compact = re.sub(r"\s+", "", text)
        surveyed = re.search(r"调查人数为(\d+)人", compact)
        valid = re.search(r"有效问卷(\d+)份", compact)
        if surveyed and valid:
            pct = Decimal(valid.group(1)) / Decimal(surveyed.group(1)) * 100
            return f"{_q(pct, 2)}%"

    table = _dachuang()
    if table and any(k in question for k in ("大创", "创新创业")):
        years = [y for y in _years(question) if y in {"2024", "2023", "2022"}]
        year_idx = {"2024": 0, "2023": 1, "2022": 2}.get(years[0], None) if years else None
        if "累计" in question or ("2022" in question and "2024" in question and "分别" in question and year_idx is None):
            return f"国家级{table['国家级'][3]}项，省级{table['省级'][3]}项，校级{table['校级'][3]}项"
        if year_idx is not None and "分别" in question and "比例" not in question and "增长" not in question:
            cells = [table["国家级"][year_idx], table["省级"][year_idx], table["校级"][year_idx], table["合计"][year_idx]]
            if sum(str(n) in question for n in cells[:3]) < 2:
                return f"{cells[0]}项、{cells[1]}项、{cells[2]}项，合计{cells[3]}项"
        if year_idx is not None and "国家级" in question and "比例" in question:
            part = Decimal(table["国家级"][year_idx])
            whole = Decimal(table["合计"][year_idx])
            if whole:
                return f"{_q(part / whole * 100, 2)}%"
        if "省级" in question and "增长" in question and "2024" in question and "2023" in question:
            now, prev = table["省级"][0], table["省级"][1]
            delta = now - prev
            pct = Decimal(delta) / Decimal(prev) * 100 if prev else None
            if pct is not None:
                return f"增加{delta}项，约{_q(pct, 2)}%"

    labs = _labs()
    if labs and "实验" in question and "面积" in question and "最小" not in question:
        if "最大" in question:
            name, college, area = max(labs, key=lambda item: item[2])
            college_bit = f"，所在学院为{college}" if college else ""
            return f"{name}面积最大，为{area}平方米{college_bit}"
        if re.search(r"小于\s*1000|不足\s*1000|不到\s*1000", question):
            small = [row for row in labs if row[2] < 1000]
            if small:
                return "，".join(f"{name}{area}" for name, _college, area in small)
        if any(k in question for k in ("之和", "合计")):
            named = []
            for cue in ("数学", "物理", "计算机", "自动化", "物联网"):
                if cue in question:
                    hit = _find_lab(cue + "实验", labs)
                    if hit and hit not in named:
                        named.append(hit)
            if len(named) >= 2:
                return f"{sum(row[2] for row in named)}平方米"
        compared = re.search(r"(.+?)的?面积比(.+?)大多少", question)
        if compared:
            left = _find_lab(compared.group(1), labs)
            right = _find_lab(compared.group(2), labs)
            if left and right and left[0] != right[0]:
                return f"{abs(left[2] - right[2])}平方米"
    if "平均" in question and "开放实验" in question:
        matched = re.search(r"立项项目数\s*参加学生数\s*20\d\d-\d{4}\s*学年\s*(\d+)\s+(\d+)", re.sub(r"\s+", " ", text))
        if not matched:
            matched = re.search(r"20\d\d-\d{4}\s*学年\s*(\d+)\s+(\d+)", _slice(text, "表 3-4", "3.5", 400))
        if matched:
            projects, students = int(matched.group(1)), int(matched.group(2))
            if projects:
                avg = Decimal(students) / Decimal(projects)
                return f"{projects}项，{students}人，平均约{_q(avg, 2)}人"
    if re.search(r"20\d\d", question) and "立项" in question and "经费" in question and "哪" in question:
        years = _years(question)
        hits = [row for row in _platforms() if row[0] in years]
        if len(hits) == 1:
            _year, name, _level, fee = hits[0]
            return f"{name}，{fee}万元"
    if "出口" in question and "是否" not in question and any(k in question for k in ("各多大", "合计", "分别")):
        compact = re.sub(r"\s+", "", text)
        matched = re.search(r"出口总带宽为(\d+)M.*?电信.*?(\d+)M.*?移动出口(\d+)M.*?教育科研网.*?(\d+)M", compact)
        if matched:
            total, telecom, mobile, edu = matched.groups()
            return f"电信{telecom}M，移动{mobile}M，教育科研网{edu}M，合计{total}M"
    if "下行" in question and "上行" in question and any(k in question for k in ("倍率", "几倍", "多少倍")):
        down = re.search(r"(\d+)\s*M?\s*下行", question) or re.search(r"(\d+)M（下行）", text)
        up = re.search(r"(\d+)\s*M?\s*上行", question) or re.search(r"(\d+)M（上行）", text)
        if down and up and Decimal(up.group(1)) != 0:
            times = Decimal(down.group(1)) / Decimal(up.group(1))
            if times == times.to_integral():
                return f"下行是上行的{int(times)}倍"
    if "外文电子期刊" in question and "中文电子期刊" in question and "多" in question:
        compact = re.sub(r"\s+", "", text)
        foreign = re.search(r"外文电子期刊(\d+)", compact)
        chinese = re.search(r"中文电子期刊(\d+)", compact)
        if foreign and chinese:
            return f"多{int(foreign.group(1)) - int(chinese.group(1))}册"
    return None


def report_answer(question: str) -> str | None:
    if not _is_teaching_report(question):
        return None
    stated = _stated_numbers(question)
    if stated:
        return stated
    try:
        return _from_tables(question)
    except Exception:
        return None
