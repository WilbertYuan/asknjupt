"""Build eval/local_test.jsonl from source tables and document spans.

Gold answers are recomputed from raw tables or from a quote that must occur
in the source text. The QA system is not consulted. Existing dev/holdout
files are not modified.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from njuptqa.db import init_db

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval" / "local_test.jsonl"

# Printed official totals that the detail rows must match, except the two
# Qinghai columns documented below.
QINGHAI_CONFLICT = {
    "青海": (60, 58),  # detail sum, printed 本科合计 cell
}


def norm(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def load_text(conn, doc_id: int) -> str:
    rows = conn.execute(
        "SELECT text FROM blocks WHERE document_id=? ORDER BY ordinal", (doc_id,)
    ).fetchall()
    return "\n".join(r[0] or "" for r in rows)


def require_quote(text: str, quote: str, doc_id: int) -> None:
    if norm(quote) not in norm(text):
        raise SystemExit(f"quote missing in doc {doc_id}: {quote}")


class Builder:
    def __init__(self, conn):
        self.conn = conn
        self.cases: list[dict] = []
        self.texts: dict[int, str] = {}

    def text(self, doc_id: int) -> str:
        if doc_id not in self.texts:
            self.texts[doc_id] = load_text(self.conn, doc_id)
        return self.texts[doc_id]

    def add(self, **case):
        case.setdefault("split", "local")
        case.setdefault("origin", "local_verified")
        question = case["question"].strip()
        if any(c["question"] == question for c in self.cases):
            raise SystemExit(f"duplicate question: {question}")
        case["question"] = question
        self.cases.append(case)

    def numeric(self, question, gold, topic, year, evidence, notes, tol=0.02):
        self.add(
            question=question,
            gold_answer=str(gold),
            scoring="numeric",
            topic=topic,
            year=year,
            tol=tol,
            evidence=evidence,
            notes=notes,
        )

    def contains(self, question, gold, must, topic, year, evidence, notes):
        self.add(
            question=question,
            gold_answer=gold,
            scoring="contains",
            topic=topic,
            year=year,
            must_contain=must,
            evidence=evidence,
            notes=notes,
        )

    def refuse(self, question, topic, year, evidence, notes):
        self.add(
            question=question,
            gold_answer="资料不足，不能确定",
            scoring="unanswerable",
            topic=topic,
            year=year,
            evidence=evidence,
            notes=notes,
        )


def table_raw(conn, doc_id: int, table_id: int | None = None):
    if table_id is None:
        row = conn.execute(
            "SELECT id, raw_json FROM tables WHERE document_id=? ORDER BY n_rows DESC LIMIT 1",
            (doc_id,),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT id, raw_json FROM tables WHERE id=?", (table_id,)
        ).fetchone()
    return row["id"], json.loads(row["raw_json"])


def forward_rows(raw, start, width):
    prev = [""] * width
    out = []
    for index, row in enumerate(raw[start:], start=start):
        filled = []
        for i in range(width):
            value = row[i] if i < len(row) else ""
            filled.append(value if str(value).strip() else prev[i])
        prev = filled
        out.append((index, filled))
    return out


def plan_tables(conn):
    """Return verified comprehensive-evaluation rows keyed by year."""
    specs = {
        2026: (14, 1225),
        2025: (1481, 2377),
        2024: (1483, 2382),
        2023: (1482, 2380),
    }
    parsed = {}
    for year, (doc_id, table_id) in specs.items():
        _, raw = table_raw(conn, doc_id, table_id)
        header_at = None
        for i, row in enumerate(raw[:3]):
            blob = "".join(row)
            if "专业" in blob and ("计划" in blob or "招生" in blob):
                header_at = i
                break
        if header_at is None:
            raise SystemExit(f"no header for {year}")
        width = len(raw[header_at])
        rows = []
        for index, filled in forward_rows(raw, header_at + 1, width):
            if "合计" in "".join(filled):
                continue
            plan_text = str(filled[-1]).strip()
            if not plan_text.isdigit():
                continue
            major = re.sub(r"\s+", "", filled[-2])
            if year == 2023:
                group, first, second = filled[0], filled[1], filled[2]
            else:
                group, first, second = "", filled[0], filled[1]
            rows.append(
                {
                    "year": year,
                    "doc": doc_id,
                    "table": table_id,
                    "row": index,
                    "group": re.sub(r"\s+", "", group),
                    "first": re.sub(r"\s+", "", first),
                    "second": re.sub(r"\s+", "", second),
                    "major": major,
                    "plan": int(plan_text),
                }
            )
        parsed[year] = rows
    expected = {2026: (28, 300), 2025: (26, 300), 2024: (20, 300), 2023: (13, 285)}
    for year, (n, total) in expected.items():
        got_n = len(parsed[year])
        got_s = sum(r["plan"] for r in parsed[year])
        if (got_n, got_s) != (n, total):
            raise SystemExit(f"综评 {year} expected {n}/{total}, got {got_n}/{got_s}")
    return parsed


def jiangsu_rows(conn):
    """Both pages of the 2026 Jiangsu plan. Page 2 has no repeated header."""
    tables = conn.execute(
        "SELECT id, raw_json FROM tables WHERE document_id=5835 AND n_cols=5 ORDER BY id"
    ).fetchall()
    if len(tables) != 2:
        raise SystemExit(f"expected 2 Jiangsu tables, got {len(tables)}")
    rows = []
    for table in tables:
        raw = json.loads(table["raw_json"])
        start = 1 if raw[0][:5] == ["类别", "专业组", "选科要求", "专业名称", "计划数"] else 0
        for index, filled in forward_rows(raw, start, 5):
            plan_text = str(filled[4]).strip()
            if not plan_text.isdigit():
                continue
            rows.append(
                {
                    "doc": 5835,
                    "table": table["id"],
                    "row": index,
                    "category": filled[0].strip(),
                    "group": filled[1].strip(),
                    "subject": re.sub(r"\s+", "", filled[2]),
                    "major": re.sub(r"\s+", "", filled[3]),
                    "major_raw": filled[3],
                    "plan": int(plan_text),
                }
            )
    sums = defaultdict(int)
    for row in rows:
        sums[row["category"]] += row["plan"]
    expected = {"艺术类": 81, "地方专项": 90, "普通类": 3338, "综合评价": 300}
    if dict(sums) != expected:
        raise SystemExit(f"Jiangsu category sums {dict(sums)}")
    if len(rows) != 88:
        raise SystemExit(f"Jiangsu row count {len(rows)}")
    return rows


def matrix(conn, doc_id: int):
    table_id, raw = table_raw(conn, doc_id)
    headers = [h.replace(" ", "") for h in raw[0]]
    if headers[0] != "专业名称" or headers[1] != "合计":
        raise SystemExit(f"bad matrix header doc {doc_id}: {headers[:4]}")
    total = next(r for r in raw[1:] if "本科合计" in r[0])
    details = [r for r in raw[1:] if r[0].strip() and "合计" not in r[0] and not r[0].strip().startswith("注")]
    return table_id, headers, raw, total, details


def verify_matrix_columns(headers, total, details):
    mismatches = {}
    for i, name in enumerate(headers):
        if name in ("专业名称",):
            continue
        detail_sum = 0
        for row in details:
            cell = (row[i] if i < len(row) else "").strip()
            if cell.isdigit():
                detail_sum += int(cell)
        official = (total[i] if i < len(total) else "").strip()
        official_n = int(official) if official.isdigit() else 0
        if detail_sum != official_n:
            mismatches[name] = (detail_sum, official_n)
    return mismatches


def subject_sum(rows, category, subject):
    return sum(r["plan"] for r in rows if r["category"] == category and r["subject"] == subject)


def main() -> None:
    conn = init_db()
    b = Builder(conn)
    zong = plan_tables(conn)
    js = jiangsu_rows(conn)
    out_id, out_headers, _out_raw, out_total, out_details = matrix(conn, 5834)
    ns_id, ns_headers, _ns_raw, ns_total, ns_details = matrix(conn, 5837)
    out_bad = verify_matrix_columns(out_headers, out_total, out_details)
    ns_bad = verify_matrix_columns(ns_headers, ns_total, ns_details)
    if set(out_bad) != {"合计", "青海", "青海民族班"}:
        raise SystemExit(f"unexpected 省外 mismatches {out_bad}")
    if out_bad["合计"] != (2391, 2387):
        raise SystemExit(out_bad["合计"])
    if out_bad["青海"] != (60, 58) or out_bad["青海民族班"] != (2, 0):
        raise SystemExit(f"Qinghai conflict changed {out_bad}")
    if ns_bad:
        raise SystemExit(f"国家专项 mismatches {ns_bad}")
    if int(out_total[1]) != 2387 or int(ns_total[1]) != 184:
        raise SystemExit("printed totals changed")

    charter = b.text(1)
    brochure = b.text(14)
    handbook = b.text(5975)
    later = b.text(5744)
    report = b.text(5)
    contact = b.text(62)
    for doc_id, text, quote in (
        (1, charter, "文科类专业每生每年 5200 元"),
        (1, charter, "理科类专业每生每年 5500 元"),
        (1, charter, "工科类专业每生每年 5800 元"),
        (1, charter, "艺术类专业每生每年 6800 元"),
        (1, charter, "基础上上浮 10 %"),
        (1, charter, "住宿费暂按 1500 元"),
        (1, charter, "预留计划数不超过年度本科招生计划总数的 1 %"),
        (1, charter, "仙林大学城文苑路 9 号"),
        (1, charter, "新模范马路 66 号"),
        (1, charter, "龙蟠路177号"),
        (14, brochure, "综合评价招生测试费标准为 60 元"),
        (14, brochure, "不超过综合评价招生计划 5 倍"),
        (14, brochure, "末位同分均入围"),
        (5975, handbook, "国家奖学金的奖励标准为每人每年 8000 元"),
        (5975, handbook, "国家励志奖学金的奖励标准为每人每年 5000 元"),
        (5975, handbook, "为每人每年2300-4300 元"),
        (5975, handbook, "一档为每人每年 4300 元"),
        (5975, handbook, "二档为每人每年 3300 元"),
        (5975, handbook, "三档为每人每年 2300 元"),
        (5975, handbook, "平均学分绩点为 1.8 及以上"),
        (5975, handbook, "平均学分绩点为 1.5 及以上的毕业生"),
        (5975, handbook, "平均学分绩点低于 1.8 但高于 1.7"),
        (5744, later, "国家奖学金的奖励标准为每人每年10000 元"),
        (5744, later, "国家励志奖学金的奖励标准为每人每年6000"),
        (5744, later, "为每人每年2700-4700"),
        (5744, later, "二档为每人每年约"),
        (5744, later, "3700 元"),
        (5, report, "专任教师 2318 人"),
        (5, report, "外聘教师 864人"),
        (5, report, "外籍专任教师 15 名"),
        (5, report, "2750，折合在校生人数 40863.2，生师比为 14.86：1"),
        (5, report, "教学实验室总面积为 69864 平方米"),
        (5, report, "生均本科实习经费达到 369.29 元"),
        (5, report, "1145.15 元"),
        (5, report, "90629.57 万元"),
        (5, report, "2.22 万元/生"),
        (5, report, "纸质藏书 287.3062 万"),
        (5, report, "新增图书 6.6753"),
        (5, report, "981.87 万元"),
        (5, report, "2023-2024 学年 197 2216"),
        (62, contact, "025-85866668，85866669"),
        (62, contact, "025-85866270"),
    ):
        require_quote(text, quote, doc_id)
    # The 1145.15 sentence is split as 达 到 in the PDF text.
    require_quote(report, "生均本科实验经费达 到 1145.15 元", 5)

    # --- 综合评价, one verified cell per row ---
    for year, rows in zong.items():
        for row in rows:
            scope = f"首选{row['first']}、再选{row['second']}"
            if row["group"]:
                scope = f"{row['group']}，{scope}"
            b.numeric(
                f"{year}年南京邮电大学综合评价招生简章中，{scope}的{row['major']}招生计划是多少人？",
                row["plan"],
                "admission",
                year,
                {"kind": "zongping_cell", "year": year, "major": row["major"], "first": row["first"], "second": row["second"]},
                f"文档{row['doc']}表{row['table']}第{row['row']}行，计划单元格为{row['plan']}。",
            )
        b.numeric(
            f"{year}年南京邮电大学综合评价招生简章专业计划表中，各专业计划数相加是多少？",
            sum(r["plan"] for r in rows),
            "admission",
            year,
            {"kind": "zongping_total", "year": year},
            "2026年另有合计行300，与明细之和一致；2024、2025年明细之和为300；2023年明细之和为285，表内没有合计行。",
        )
        b.numeric(
            f"{year}年综合评价招生简章的专业计划表里列出了多少个专业？",
            len(rows),
            "admission",
            year,
            {"kind": "zongping_count", "year": year},
            "只计计划表中的专业行，不计表头和合计行。",
        )

    groups_2026 = defaultdict(int)
    names_2026 = defaultdict(list)
    for row in zong[2026]:
        key = (row["first"], row["second"])
        groups_2026[key] += row["plan"]
        names_2026[key].append(row["major"])
    if dict(groups_2026) != {("历史", "不限"): 30, ("物理", "不限"): 65, ("物理", "化学"): 205}:
        raise SystemExit(f"2026 groups {dict(groups_2026)}")
    for (first, second), total in groups_2026.items():
        b.numeric(
            f"2026年综合评价招生简章中，首选{first}、再选科目要求为“{second}”的专业计划合计是多少人？",
            total,
            "admission",
            2026,
            {"kind": "zongping_group", "first": first, "second": second},
            "由2026年简章计划表对应行相加，三组合计300。",
        )
    b.contains(
        "2026年综合评价招生简章里，首选历史、再选不限的专业有哪些？",
        "、".join(names_2026[("历史", "不限")]),
        names_2026[("历史", "不限")],
        "admission",
        2026,
        {"kind": "zongping_names", "first": "历史", "second": "不限"},
        "六个专业：财务管理、工商管理、广告学、人力资源管理、社会工作、英语。",
    )
    b.contains(
        "2026年综合评价招生简章里，首选物理、再选不限的专业有哪些？",
        "、".join(names_2026[("物理", "不限")]),
        names_2026[("物理", "不限")],
        "admission",
        2026,
        {"kind": "zongping_names", "first": "物理", "second": "不限"},
        "五个专业，计划合计65。",
    )

    # Same major, different year, only where the plan number changes.
    by_major = defaultdict(dict)
    for year, rows in zong.items():
        for row in rows:
            by_major[row["major"]].setdefault(year, []).append(row)
    for major, years in sorted(by_major.items()):
        if any(len(items) != 1 for items in years.values()):
            continue
        plans = {year: items[0]["plan"] for year, items in years.items()}
        if len(set(plans.values())) < 2 or len(plans) < 2:
            continue
        pieces = []
        for year, row in sorted((year, items[0]) for year, items in years.items()):
            pieces.append(f"{year}年{row['first']}+{row['second']}为{row['plan']}人")
        b.contains(
            f"综合评价招生简章中，{major}在不同年份的计划数分别是多少？请按年份说明选科和人数。",
            "；".join(pieces),
            pieces,
            "admission",
            2026,
            {"kind": "zongping_years", "major": major},
            "只比较各年简章里仅出现一次的同名专业。计划数相同的年份不单列这道比较题。",
        )

    # --- 江苏分专业 ---
    for row in js:
        if "物理管理" in row["major"]:
            continue
        label = row["major"]
        if "邮政菁英班" in row["major"]:
            label = "名称中含“邮政菁英班”的专业"
        b.numeric(
            f"2026年江苏省招生计划中，{row['category']}、专业组{row['group']}、选科{row['subject']}的{label}计划数是多少？",
            row["plan"],
            "admission",
            2026,
            {"kind": "jiangsu_cell", "category": row["category"], "group": row["group"], "major": row["major"]},
            f"文档5835第{row['row']}行。类别空单元格已按表向上填充。",
        )
    for category, total in (("普通类", 3338), ("地方专项", 90), ("艺术类", 81), ("综合评价", 300)):
        b.numeric(
            f"2026年江苏省招生计划表中，类别为“{category}”的各专业计划数合计是多少人？",
            total,
            "admission",
            2026,
            {"kind": "jiangsu_category", "category": category},
            "同一PDF有两页。普通类两页合计3338，不能只算第一页的1323。四类合计3809。",
        )
    b.numeric(
        "2026年江苏省招生计划表里，艺术类、地方专项、普通类和综合评价四类计划全部相加是多少人？",
        3809,
        "admission",
        2026,
        {"kind": "jiangsu_all"},
        "81+90+3338+300。综合评价这一类与招生简章的300人对应，但专业名称以计划表原文为准。",
    )
    subject_expected = {
        ("普通类", "历史+不限"): 210,
        ("普通类", "物理+不限"): 59,
        ("普通类", "物理+化学"): 3069,
        ("地方专项", "历史+不限"): 26,
        ("地方专项", "物理+不限"): 25,
        ("地方专项", "物理+化学"): 39,
        ("艺术类", "历史+不限"): 81,
        ("综合评价", "历史+不限"): 30,
        ("综合评价", "物理+不限"): 65,
        ("综合评价", "物理+化学"): 205,
    }
    for (category, subject), total in subject_expected.items():
        got = subject_sum(js, category, subject)
        if got != total:
            raise SystemExit(f"{category} {subject} {got} != {total}")
        b.numeric(
            f"2026年江苏{category}里，选科要求原文为“{subject}”的专业计划合计是多少人？只统计这一类选科，不要把其他选科算进去。",
            total,
            "admission",
            2026,
            {"kind": "jiangsu_subject", "category": category, "subject": subject},
            "物理+化学不包含物理+不限。普通类两页合在一起，这两类分别是3069和59。",
        )
    b.numeric(
        "一名江苏考生首选物理、再选化学。2026年普通类中，选科为“物理+化学”以及选科为“物理+不限”的专业都可以填报。这两部分计划合计是多少人？",
        3128,
        "admission",
        2026,
        {"kind": "jiangsu_subject_allow"},
        "3069+59。计划表第二页还有计算机、通信、电子信息、中外合作办学等物理+化学专业，不能只加第一页。",
    )
    b.numeric(
        "2026年江苏普通类里，选科原文严格等于“物理+化学”的计划合计是多少人？物理+不限的专业不要计入。",
        3069,
        "admission",
        2026,
        {"kind": "jiangsu_subject", "category": "普通类", "subject": "物理+化学"},
        "第一页1054人加上第二页2015人。第二页含计算机科学与技术216、通信工程310、两个中外合作办学各120等。",
    )

    # --- 省外 matrix: official 本科合计 cells, except the conflicting Qinghai columns ---
    for i, province in enumerate(out_headers):
        if province in ("专业名称", "合计", "青海", "青海民族班"):
            continue
        official = int(out_total[i])
        b.numeric(
            f"2026年江苏省外招生计划表中，△本科合计这一行的“{province}”单元格是多少人？",
            official,
            "admission",
            2026,
            {"kind": "out_official", "province": province},
            "取△本科合计行，不把艺术类、文史、理科三个合计行再加一次。除青海、青海民族班外，该单元格等于明细列之和。",
        )
    b.numeric(
        "2026年江苏省外招生计划表中，△本科合计这一行的“合计”单元格是多少人？",
        2387,
        "admission",
        2026,
        {"kind": "out_official", "province": "合计"},
        "不要把艺术类合计44、文史合计83、理科合计2260与2387再相加。",
    )
    detail_total_col = 0
    for row in out_details:
        cell = row[1].strip()
        if cell.isdigit():
            detail_total_col += int(cell)
    if detail_total_col != 2391:
        raise SystemExit(detail_total_col)
    b.numeric(
        "2026年江苏省外招生计划表中，不含合计行的各专业“合计”列相加是多少人？",
        2391,
        "admission",
        2026,
        {"kind": "out_detail_total"},
        "比△本科合计2387多4人，差额来自青海列（明细60、合计行58）和青海民族班列（明细2、合计行空白）。",
    )
    b.contains(
        "2026年江苏省外计划表里，青海列的专业明细相加，和△本科合计行的青海单元格，分别是多少？两者是否一致？",
        "明细相加60人，△本科合计行是58人，不一致",
        ["60", "58", "不一致"],
        "admission",
        2026,
        {"kind": "out_qinghai"},
        "同一张表的两处数字冲突，不能只给其中一个。青海民族班明细为2、合计行该列为空，这两处差额合计为4，对应2391与2387之差。",
    )

    def cell(details, headers, major, province):
        i = headers.index(province)
        hits = [r for r in details if re.sub(r"\s+", "", r[0]) == major]
        if len(hits) != 1:
            raise SystemExit(f"{major} hits {len(hits)}")
        value = hits[0][i].strip()
        if not value.isdigit():
            raise SystemExit(f"{major} {province} blank")
        return int(value)

    # Every Beijing detail cell, plus a short cross-province sample.
    beijing = []
    bi = out_headers.index("北京")
    for row in out_details:
        value = row[bi].strip()
        if value.isdigit() and int(value):
            name = re.sub(r"\s+", "", row[0])
            beijing.append((name, int(value)))
    if sum(n for _, n in beijing) != 13:
        raise SystemExit(f"Beijing details {beijing}")
    for name, plan in beijing:
        b.numeric(
            f"2026年江苏省外招生计划中，{name}在北京的计划数是多少人？",
            plan,
            "admission",
            2026,
            {"kind": "out_cell", "major": name, "province": "北京"},
            "明细单元格。四个北京专业合计13，与△本科合计行的北京单元格一致。",
        )
    b.contains(
        "2026年江苏省外计划里，北京有招生计划的专业有哪些？请给出每个专业的人数。",
        "，".join(f"{name}{plan}人" for name, plan in beijing),
        [f"{name}{plan}人" for name, plan in beijing],
        "admission",
        2026,
        {"kind": "out_beijing_list"},
        "合计13人。",
    )
    samples = [
        ("通信工程", "上海"),
        ("通信工程", "广东"),
        ("计算机科学与技术", "浙江"),
        ("软件工程", "山东"),
        ("人工智能", "四川"),
        ("电子信息工程", "河南"),
    ]
    for major, province in samples:
        plan = cell(out_details, out_headers, major, province)
        b.numeric(
            f"2026年江苏省外招生计划中，{major}在{province}的计划数是多少人？",
            plan,
            "admission",
            2026,
            {"kind": "out_cell", "major": major, "province": province},
            "取该专业行、该省列的单元格，不是该专业的全国合计。",
        )

    # --- 国家专项 ---
    for i, province in enumerate(ns_headers):
        if province in ("专业名称", "合计"):
            continue
        b.numeric(
            f"2026年国家专项招生计划表中，△本科合计这一行的“{province}”是多少人？",
            int(ns_total[i]),
            "admission",
            2026,
            {"kind": "ns_official", "province": province},
            "各列明细之和与△本科合计行一致，合计184。",
        )
    b.numeric(
        "2026年国家专项招生计划表中，△本科合计这一行的合计是多少人？",
        184,
        "admission",
        2026,
        {"kind": "ns_official", "province": "合计"},
        "文档5837。",
    )
    for row in ns_details:
        name = re.sub(r"\s+", "", row[0])
        if not row[1].strip().isdigit():
            continue
        b.numeric(
            f"2026年国家专项计划中，{name}的各省计划合计是多少人？",
            int(row[1]),
            "admission",
            2026,
            {"kind": "ns_major", "major": name},
            "该行合计列等于该行各省单元格之和。",
        )
    hebei = cell(ns_details, ns_headers, "大数据管理与应用", "河北")
    if hebei != 3:
        raise SystemExit(hebei)
    b.numeric(
        "2026年国家专项中，大数据管理与应用在河北的计划是多少人？",
        3,
        "admission",
        2026,
        {"kind": "ns_cell", "major": "大数据管理与应用", "province": "河北"},
        "这是河北列唯一有数的专业，因此也等于△本科合计行的河北3人。",
    )

    # --- fees and charter procedures ---
    b.numeric("2026年本科招生章程里，文科类专业每生每年学费预收标准是多少元？", 5200, "fee", 2026, {"kind": "quote", "doc": 1, "quote": "文科类专业每生每年 5200 元"}, "章程收费条款，不上浮、不含住宿。")
    b.numeric("2026年本科招生章程里，理科类专业每生每年学费预收标准是多少元？", 5500, "fee", 2026, {"kind": "quote", "doc": 1, "quote": "理科类专业每生每年 5500 元"}, "不上浮、不含住宿。")
    b.numeric("2026年本科招生章程里，工科类专业每生每年学费预收标准是多少元？", 5800, "fee", 2026, {"kind": "quote", "doc": 1, "quote": "工科类专业每生每年 5800 元"}, "不上浮、不含住宿。")
    b.numeric("2026年本科招生章程里，艺术类专业每生每年学费预收标准是多少元？", 6800, "fee", 2026, {"kind": "quote", "doc": 1, "quote": "艺术类专业每生每年 6800 元"}, "不上浮、不含住宿。")
    b.numeric("2026年本科招生章程规定，江苏高校优势学科专业学费在普通标准基础上上浮百分之多少？", 10, "fee", 2026, {"kind": "quote", "doc": 1, "quote": "基础上上浮 10 %"}, "上浮比例，不是上浮后的金额。")
    b.numeric("2026年本科招生章程里，住宿费暂按每生每学年预收多少元？", 1500, "fee", 2026, {"kind": "quote", "doc": 1, "quote": "住宿费暂按 1500 元"}, "最终据实结算、多退少补。")
    b.numeric("按2026年章程，优势学科工科专业四年学费上浮10%后再加四年住宿费预收，一共多少元？", 31520, "fee", 2026, {"kind": "formula", "expr": "5800*1.1*4+1500*4"}, "5800×1.1×4+1500×4。")
    b.numeric("按2026年章程，优势学科工科专业只算四年上浮后的学费、不算住宿，一共多少元？", 25520, "fee", 2026, {"kind": "formula", "expr": "5800*1.1*4"}, "不要把31520误答成不含住宿的学费。")
    b.numeric("按2026年章程，工科专业不上浮、也不加住宿，四年学费预收一共多少元？", 23200, "fee", 2026, {"kind": "formula", "expr": "5800*4"}, "普通工科标准。")
    b.numeric("按2026年章程，文科专业不上浮，一年学费预收是多少元？不要加住宿。", 5200, "fee", 2026, {"kind": "formula", "expr": "5200"}, "一年，不是四年。")
    b.numeric("按2026年章程，文科专业不上浮，四年学费再加四年住宿费预收，一共多少元？", 26800, "fee", 2026, {"kind": "formula", "expr": "5200*4+1500*4"}, "5200×4+1500×4。")
    b.numeric("按2026年章程，优势学科理科专业四年上浮后的学费再加四年住宿费预收，一共多少元？", 30200, "fee", 2026, {"kind": "formula", "expr": "5500*1.1*4+1500*4"}, "5500×1.1×4+6000。")
    b.numeric("按2026年章程，优势学科艺术类专业一年的上浮后学费是多少元？不要加住宿。", 7480, "fee", 2026, {"kind": "formula", "expr": "6800*1.1"}, "6800×1.1。")
    b.numeric("2026年章程规定，预留计划不超过年度本科招生计划总数的百分之多少？", 1, "fee", 2026, {"kind": "quote", "doc": 1, "quote": "预留计划数不超过年度本科招生计划总数的 1 %"}, "用于调节各地统考上线生源不平衡。")
    b.contains(
        "2026年章程对住宿费预收之后怎么结算？",
        "暂按1500元/生·学年预收，据实结算、多退少补",
        ["1500", "多退少补"],
        "fee",
        2026,
        {"kind": "quote", "doc": 1, "quote": "实行据实结算、多退少补"},
        "收费政策如有调整，以最新文件为准。",
    )
    b.numeric("2026年综合评价招生测试费是每生多少元？", 60, "fee", 2026, {"kind": "quote", "doc": 14, "quote": "测试费标准为 60 元"}, "逾期未确认或未缴费视为放弃测试资格。")
    b.numeric("2026年综合评价入围名单，按不超过招生计划的多少倍确定？", 5, "admission", 2026, {"kind": "quote", "doc": 14, "quote": "不超过综合评价招生计划 5 倍"}, "分物理类、历史类，末位同分均入围。")
    b.contains(
        "2026年综合评价确定入围名单时，末位分数相同怎么处理？",
        "末位同分均入围",
        ["末位同分", "入围"],
        "admission",
        2026,
        {"kind": "quote", "doc": 14, "quote": "末位同分均入围"},
        "简章原文。",
    )

    # --- handbook conflicts and degree rules ---
    b.numeric("《南京邮电大学学生手册（2023版）》写明的国家奖学金奖励标准是每人每年多少元？", 8000, "aid", 2023, {"kind": "quote", "doc": 5975, "quote": "国家奖学金的奖励标准为每人每年 8000 元"}, "只代表2023版手册，不能用来否定另一份手册中的10000元。")
    b.numeric("学生工作处发布的学生手册PDF中，学位授予办法落款为2026年6月12日。这份正文写明的国家奖学金是每人每年多少元？", 10000, "aid", 2026, {"kind": "quote", "doc": 5744, "quote": "每人每年10000 元"}, "该PDF在库中的标题被误写成学校简介，金额以正文为准。")
    b.contains(
        "库内两份学生手册对国家奖学金每人每年的标准写法不一致。两份分别是多少元？",
        "2023版学生手册为8000元；学生工作处学生手册PDF正文为10000元",
        ["8000", "10000"],
        "aid",
        None,
        {"kind": "conflict", "docs": [5975, 5744]},
        "必须同时给出两个数。只答其中一个不算覆盖了冲突。",
    )
    b.numeric("2023版学生手册写明的国家励志奖学金是每人每年多少元？", 5000, "aid", 2023, {"kind": "quote", "doc": 5975, "quote": "国家励志奖学金的奖励标准为每人每年 5000 元"}, "与文档5744的6000元并存。")
    b.numeric("学生工作处那份学生手册PDF（学位授予办法落款2026年6月12日）写明的国家励志奖学金是每人每年多少元？", 6000, "aid", 2026, {"kind": "quote", "doc": 5744, "quote": "国家励志奖学金的奖励标准为每人每年6000"}, "页码把“6000”和“元”拆开，金额仍是6000元。")
    b.contains(
        "两份手册里的国家励志奖学金标准分别是每人每年多少元？",
        "2023版为5000元，学生工作处学生手册PDF为6000元",
        ["5000", "6000"],
        "aid",
        None,
        {"kind": "conflict", "docs": [5975, 5744]},
        "不要合并成一个数。",
    )
    b.contains(
        "2023版学生手册里，国家助学金三个档次每人每年分别是多少元？",
        "一档4300元，二档3300元，三档2300元，区间2300至4300元",
        ["4300", "3300", "2300"],
        "aid",
        2023,
        {"kind": "quote", "doc": 5975, "quote": "一档为每人每年 4300 元"},
        "不要改用另一份手册的2700至4700元。",
    )
    b.contains(
        "学生工作处那份学生手册PDF里，国家助学金的金额区间以及二档大约是每人每年多少元？",
        "区间2700至4700元，二档约3700元",
        ["2700", "4700", "3700"],
        "aid",
        2026,
        {"kind": "quote", "doc": 5744, "quote": "为每人每年2700-4700"},
        "一档写的是不超过4700元，三档是不少于2700元，二档约为3700元。",
    )
    b.numeric(
        "2023版学生手册《学士学位授予办法》第二条，普通本科毕业生授予学士学位要求平均学分绩点不低于多少？",
        1.8,
        "degree",
        2023,
        {"kind": "quote", "doc": 5975, "quote": "平均学分绩点为 1.8 及以上"},
        "毕业设计或论文成绩不计入。1.5不是这条的普通标准。",
    )
    b.numeric(
        "2023版学位授予办法中，高水平运动员授予学士学位的平均学分绩点要求不低于多少？",
        1.5,
        "degree",
        2023,
        {"kind": "quote", "doc": 5975, "quote": "高水平运动员平均学分绩点要求为 1.5 及以上"},
        "这是运动员标准，不是普通毕业生的1.8。",
    )
    b.numeric(
        "2023版手册规定，毕业时未获学位、离校后一年内回校申请重修的，平均学分绩点至少要达到多少？",
        1.5,
        "degree",
        2023,
        {"kind": "quote", "doc": 5975, "quote": "平均学分绩点为 1.5 及以上的毕业生"},
        "这是第六条补授条件。重修的是绩点系数低于1.8的课程，重修总学分不超过30。",
    )
    b.contains(
        "2023版学位授予办法第五条，普通毕业生绩点没有达到1.8时，什么区间可以因毕业设计优秀而申请学位？",
        "低于1.8但高于或等于1.7，且毕业设计或论文成绩为优秀",
        ["1.8", "1.7", "优秀"],
        "degree",
        2023,
        {"kind": "quote", "doc": 5975, "quote": "平均学分绩点低于 1.8 但高于 1.7"},
        "还须经教务处确认。",
    )
    degree = b.text(339)
    require_quote(degree, "平均学分绩点为 1.8 及以上", 339)
    require_quote(degree, "高水平运动员平均学分绩点要求为 1.5 及以上", 339)
    b.numeric(
        "教务处页面《学位授予办法》（2026年6月12日校学位评定委员会通过）对普通毕业生的平均学分绩点要求不低于多少？",
        1.8,
        "degree",
        2026,
        {"kind": "quote", "doc": 339, "quote": "平均学分绩点为 1.8 及以上"},
        "与2023版手册第二条的普通标准相同，都是1.8。",
    )

    # --- quality report ---
    b.numeric("本科教学质量报告中的专任教师是多少人？", 2318, "faculty", 2024, {"kind": "quote", "doc": 5, "quote": "专任教师 2318 人"}, "不要用学院人数或其他年份的教职工数代替。")
    b.numeric("本科教学质量报告中的外聘教师是多少人？", 864, "faculty", 2024, {"kind": "quote", "doc": 5, "quote": "外聘教师 864人"}, "报告还另写外籍专任教师15名。")
    b.numeric("本科教学质量报告写明的外籍专任教师是多少名？", 15, "faculty", 2024, {"kind": "quote", "doc": 5, "quote": "外籍专任教师 15 名"}, "不计入864名外聘教师。")
    b.numeric("教学质量报告给出的折合教师数是多少？", 2750, "faculty", 2024, {"kind": "quote", "doc": 5, "quote": "2750，折合在校生人数", "number": "2750"}, "与折合在校生40863.2、生师比14.86写在同一句。")
    b.numeric("教学质量报告给出的折合在校生人数是多少？", 40863.2, "faculty", 2024, {"kind": "quote", "doc": 5, "quote": "折合在校生人数 40863.2", "number": "40863.2"}, "保留一位小数。")
    b.numeric("教学质量报告直接写出的生师比是多少比1？", 14.86, "faculty", 2024, {"kind": "quote", "doc": 5, "quote": "生师比为 14.86：1", "number": "14.86"}, "40863.2÷2750=14.86，与报告印刷值一致。", tol=0.02)
    b.numeric(
        "用本科教学质量报告给出的折合在校生人数除以折合教师数，生师比是多少？",
        14.86,
        "faculty",
        2024,
        {"kind": "formula", "expr": "round(40863.2/2750, 2)"},
        "保留两位小数。",
        tol=0.02,
    )
    b.numeric(
        "按本科教学质量报告中的（折合教师数-专任教师数）÷外聘教师数，外聘教师折算系数是多少？",
        0.5,
        "faculty",
        2024,
        {"kind": "formula", "expr": "(2750-2318)/864"},
        "（2750-2318）÷864=0.5。问的是系数本身，不是百分数。",
    )

    labs = [
        ("信息与通信工程实验教学中心", 9230),
        ("电子与光学工程实验教学中心", 9983),
        ("集成电路实验教学中心", 709),
        ("计算机实验教学中心", 8056),
        ("自动化实验教学中心", 5632),
        ("材料与化学实验教学中心", 7150),
        ("化学与生命科学实验教学中心", 3807),
        ("物联网实验教学中心", 2146),
        ("数学实验教学中心", 1515),
        ("物理实验教学中心", 2547),
        ("现代邮政大数据技术与智能应用实验室", 120),
        ("传媒与艺术实验教学中心", 864),
        ("经济运行与管理综合实验教学中心", 4586),
        ("经济学院实验室", 214),
        ("社会与人口学院实验室", 352),
        ("外语实验教学中心", 2522),
        ("教育科学与技术实验教学中心", 930),
        ("工程实验教学部", 9501),
    ]
    if sum(area for _, area in labs) != 69864:
        raise SystemExit(sum(area for _, area in labs))
    for name, area in labs:
        window = report[report.find(name): report.find(name) + 80]
        if str(area) not in window:
            raise SystemExit(f"lab area not next to {name}: {window!r}")
        b.numeric(
            f"本科教学质量报告列出的{name}面积是多少平方米？",
            area,
            "lab",
            2024,
            {"kind": "lab", "name": name},
            "面积表有18行，序号跳过6和12；合计行印刷为69864。",
        )
    b.numeric("教学质量报告印刷的教学实验室总面积是多少平方米？", 69864, "lab", 2024, {"kind": "quote", "doc": 5, "quote": "教学实验室总面积为 69864 平方米"}, "18行面积相加也是69864。报告另说建有20个实验教学中心，面积表本身是18行。")
    b.numeric("教学质量报告的教学实验室面积表一共列出多少行实验场所？", 18, "lab", 2024, {"kind": "lab_count"}, "不要把正文里的20个中心和这张面积表的行数混成一个数。")
    b.numeric("本科教学质量报告正文写学校建有多少个实验教学中心？", 20, "lab", 2024, {"kind": "quote", "doc": 5, "quote": "建有 20 个实验教学", "number": "20"}, "这是正文口径。面积表列出18行，序号空了6和12，两处不能互相替换。")
    b.numeric(
        "信息与通信工程、电子与光学工程、计算机三个实验教学中心的面积合计是多少平方米？",
        27269,
        "lab",
        2024,
        {"kind": "formula", "expr": "9230+9983+8056"},
        "9230+9983+8056。",
    )
    b.numeric(
        "信息与通信工程、电子与光学工程、计算机三个实验教学中心的面积之和，占教学实验室总面积69864平方米的百分之多少？保留两位小数。",
        39.03,
        "lab",
        2024,
        {"kind": "formula", "expr": "round(27269/69864*100, 2)"},
        "27269÷69864×100=39.03。",
        tol=0.02,
    )
    b.numeric("教学质量报告中的生均本科实习经费是多少元？", 369.29, "quality", 2024, {"kind": "quote", "doc": 5, "quote": "生均本科实习经费达到 369.29 元"}, "不是实验经费。")
    b.numeric("教学质量报告中的生均本科实验经费是多少元？", 1145.15, "quality", 2024, {"kind": "quote", "doc": 5, "quote": "1145.15 元"}, "原文在“达到”中间有换行。")
    b.numeric("教学质量报告中的教学科研仪器设备总价值是多少万元？", 90629.57, "quality", 2024, {"kind": "quote", "doc": 5, "quote": "90629.57 万元"}, "单位是万元。")
    b.numeric("教学质量报告中的生均教学科研仪器设备值是多少万元？", 2.22, "quality", 2024, {"kind": "quote", "doc": 5, "quote": "2.22 万元/生"}, "不要和总价值90629.57万元混淆。")
    b.numeric("教学质量报告写明的纸质藏书是多少万余册？", 287.3062, "quality", 2024, {"kind": "quote", "doc": 5, "quote": "纸质藏书 287.3062 万"}, "保留原文小数。")
    b.numeric("教学质量报告写明的2023－2024学年新增图书是多少万册？", 6.6753, "quality", 2024, {"kind": "quote", "doc": 5, "quote": "新增图书 6.6753"}, "不是纸质藏书总量。")
    b.numeric("教学质量报告写明的2023年纸质图书和电子资源购置费是多少万元？", 981.87, "quality", 2024, {"kind": "quote", "doc": 5, "quote": "981.87 万元"}, "2023年购置费。")
    b.numeric("教学质量报告统计的2023-2024学年开放实验立项项目数是多少？", 197, "quality", 2024, {"kind": "quote", "doc": 5, "quote": "学年 197 2216", "number": "197"}, "同一行参加学生数是2216，不要看串。")
    b.numeric("教学质量报告统计的2023-2024学年开放实验参加学生数是多少？", 2216, "quality", 2024, {"kind": "quote", "doc": 5, "quote": "学年 197 2216", "number": "2216"}, "项目数是197。")

    # --- employment, only sentences that name the population ---
    reports = {
        2018: b.text(6042),
        2019: b.text(5973),
        2020: b.text(6044),
        2021: b.text(6045),
        2022: b.text(6046),
    }
    quotes = {
        2018: ["毕业生共 6372 人", "研究生 1280 人", "本科生 5092 人"],
        2019: ["毕业生共 6560 人", "毕业研究生1350 人", "本科毕业生 5210 人", "初次就业率为 91.49%", "年终就业率为 97.84%"],
        2020: ["总毕业生数为 7112 人", "研究生1643 人", "本科生 5469 人", "毕业生总数 7143 人", "初次总就业率 80%", "总就业率 95.33%", "研究生 96.94%", "本科生 74.83%", "研究生 99.70%", "本科生 94.02%"],
        2021: ["毕业生总数7064人", "毕业生总人数为7064人", "研究生1721人", "博士毕业生36人", "本科生5343人"],
        2022: ["毕业生总数7141人", "毕业生总人数为7141人", "研究生1890人", "科生5251人", "本科毕业生共5786人", "分段培养毕业生535人", "本科95.79%", "研究生96.18%", "本科98.23%", "研究生99.05%"],
    }
    if 1280 + 5092 != 6372 or 1350 + 5210 != 6560:
        raise SystemExit("2018/2019 headcount does not add up")
    if 1643 + 5469 != 7112 or 1890 + 5251 != 7141 or 5786 - 535 != 5251 or 1721 + 5343 != 7064:
        raise SystemExit("later headcount does not add up")
    for year, items in quotes.items():
        for quote in items:
            require_quote(reports[year], quote, {2018: 6042, 2019: 5973, 2020: 6044, 2021: 6045, 2022: 6046}[year])
    employ_doc = {2018: 6042, 2019: 5973, 2020: 6044, 2021: 6045, 2022: 6046}
    b.numeric("就业质量年度报告中，2018届毕业生一共多少人？", 6372, "employment", 2018, {"kind": "quote", "doc": 6042, "quote": "毕业生共 6372 人"}, "其中研究生1280人、本科生5092人。报告没有单独给出全校一个就业率。")
    b.numeric("2018届毕业生中研究生多少人？", 1280, "employment", 2018, {"kind": "quote", "doc": 6042, "quote": "研究生 1280 人"}, "占毕业生的20.09%，这里要的是人数。")
    b.numeric("2018届毕业生中本科生多少人？", 5092, "employment", 2018, {"kind": "quote", "doc": 6042, "quote": "本科生 5092 人"}, "不是全校6372人。")
    b.numeric("2019届毕业生一共多少人？", 6560, "employment", 2019, {"kind": "quote", "doc": 5973, "quote": "毕业生共 6560 人"}, "研究生1350人，本科5210人。")
    b.numeric("2019届毕业研究生多少人？", 1350, "employment", 2019, {"kind": "quote", "doc": 5973, "quote": "毕业研究生1350 人"}, "占20.58%。")
    b.numeric("2019届本科毕业生多少人？", 5210, "employment", 2019, {"kind": "quote", "doc": 5973, "quote": "本科毕业生 5210 人"}, "不是全校6560人。")
    b.numeric("学校2019届毕业生初次就业率是百分之多少？", 91.49, "employment", 2019, {"kind": "quote", "doc": 5973, "quote": "学校 2019 届毕业生初次就业率为 91.49%", "number": "91.49"}, "句子主语是学校2019届毕业生，后面的“其中”才是分学历。", tol=0.02)
    b.numeric("学校2019届毕业生年终就业率是百分之多少？", 97.84, "employment", 2019, {"kind": "quote", "doc": 5973, "quote": "学校 2019 届毕业生年终就业率为 97.84%", "number": "97.84"}, "全校口径，不是研究生或本科单独的率。", tol=0.02)
    b.numeric("2020届就业质量报告正文“总体规模”里的总毕业生数是多少人？", 7112, "employment", 2020, {"kind": "quote", "doc": 6044, "quote": "总毕业生数为 7112 人"}, "研究生1643人加本科生5469人。这不是后文调查说明里的7143人。")
    b.numeric("2020届总毕业生7112人中，研究生多少人？", 1643, "employment", 2020, {"kind": "quote", "doc": 6044, "quote": "研究生1643 人"}, "占23.10%。")
    b.numeric("2020届总毕业生7112人中，本科生多少人？", 5469, "employment", 2020, {"kind": "quote", "doc": 6044, "quote": "本科生 5469 人"}, "占76.90%。")
    b.numeric("2020届就业质量报告的就业调查说明里，调研掌握的毕业生总数是多少人？", 7143, "employment", 2020, {"kind": "quote", "doc": 6044, "quote": "毕业生总数 7143 人"}, "只对应调查说明，不能代替正文总体规模7112人。")
    b.contains(
        "2020届就业质量报告里，正文总体规模和就业调查说明各写了多少名毕业生？",
        "正文总体规模7112人，调查说明7143人",
        ["7112", "7143"],
        "employment",
        2020,
        {"kind": "quote", "doc": 6044, "quote": "总毕业生数为 7112 人"},
        "两个数都在同一份报告里，不能合成一个。",
    )
    b.numeric("2020届毕业生初次总就业率是百分之多少？", 80, "employment", 2020, {"kind": "quote", "doc": 6044, "quote": "初次总就业率 80%"}, "报告原文就是这个总就业率。其中研究生96.94%、本科生74.83%。")
    b.numeric("2020届毕业生年终总就业率是百分之多少？", 95.33, "employment", 2020, {"kind": "quote", "doc": 6044, "quote": "总就业率 95.33%"}, "其中研究生99.70%、本科生94.02%。", tol=0.02)
    b.numeric("2020届初次就业率中，研究生是百分之多少？以正文句子为准。", 96.94, "employment", 2020, {"kind": "quote", "doc": 6044, "quote": "研究生为 96.94%"}, "正文有两处写96.94。图中抽出的96.95%视为同一指标的印刷差，不另立一个率。", tol=0.02)
    b.numeric("2020届初次就业率中，本科生是百分之多少？", 74.83, "employment", 2020, {"kind": "quote", "doc": 6044, "quote": "本科生为 74.83%"}, "不是全校80%。", tol=0.02)
    b.numeric("2020届年终就业率中，研究生是百分之多少？", "99.70", "employment", 2020, {"kind": "quote", "doc": 6044, "quote": "研究生为 99.70%", "number": "99.70"}, "不是年终总就业率95.33%。", tol=0.02)
    b.numeric("2020届年终就业率中，本科生是百分之多少？", 94.02, "employment", 2020, {"kind": "quote", "doc": 6044, "quote": "本科生为 94.02%"}, "不是年终总就业率95.33%。", tol=0.02)
    b.numeric("2021届毕业生总人数是多少人？", 7064, "employment", 2021, {"kind": "quote", "doc": 6045, "quote": "毕业生总人数为7064人"}, "正文和调查说明都是7064。报告没有一句全校统一的落实率。")
    b.numeric("2021届毕业生中，研究生多少人？按报告括号，博士包含在这个数里。", 1721, "employment", 2021, {"kind": "quote", "doc": 6045, "quote": "研究生1721人"}, "不要把博士36人再加到1721上。")
    b.numeric("2021届研究生1721人中，博士毕业生多少人？", 36, "employment", 2021, {"kind": "quote", "doc": 6045, "quote": "含博士毕业生36人"}, "博士已经包含在1721人里。")
    b.numeric("2021届毕业生中，本科生多少人？", 5343, "employment", 2021, {"kind": "quote", "doc": 6045, "quote": "本科生5343人"}, "1721加5343等于7064。")
    b.numeric("2022届毕业生总人数是多少人？", 7141, "employment", 2022, {"kind": "quote", "doc": 6046, "quote": "毕业生总人数为7141人"}, "正文总人数和调查说明都是7141。研究生1890人，分析口径本科生5251人。")
    b.numeric("2022届毕业生7141人中，研究生多少人？", 1890, "employment", 2022, {"kind": "quote", "doc": 6046, "quote": "研究生1890人"}, "不是脚注里的本科5786人。")
    b.numeric("2022届就业质量报告正文总人数里的本科生是多少人？", 5251, "employment", 2022, {"kind": "quote", "doc": 6046, "quote": "科生5251人", "number": "5251"}, "这是不含专转本和分段培养的分析口径。1890加5251等于7141。")
    b.numeric("2022届就业质量报告脚注写的本科毕业生一共多少人？", 5786, "employment", 2022, {"kind": "quote", "doc": 6046, "quote": "本科毕业生共5786人"}, "这个数含专转本和分段培养535人，不是正文总人数里的5251人。")
    b.numeric("2022届脚注里，专转本和分段培养毕业生多少人？", 535, "employment", 2022, {"kind": "quote", "doc": 6046, "quote": "分段培养毕业生535人"}, "5786减535等于正文使用的本科生5251人。报告声明分析人数不含这535人。")
    b.numeric("2022届本科毕业生初次毕业去向落实率是百分之多少？", 95.79, "employment", 2022, {"kind": "quote", "doc": 6046, "quote": "本科95.79%"}, "这是本科口径，不是全校一个率。", tol=0.02)
    b.numeric("2022届研究生初次毕业去向落实率是百分之多少？", 96.18, "employment", 2022, {"kind": "quote", "doc": 6046, "quote": "研究生96.18%"}, "这是研究生口径。", tol=0.02)
    b.numeric("2022届本科毕业生年终毕业去向落实率是百分之多少？", 98.23, "employment", 2022, {"kind": "quote", "doc": 6046, "quote": "本科98.23%"}, "年终、本科。", tol=0.02)
    b.numeric("2022届研究生年终毕业去向落实率是百分之多少？", 99.05, "employment", 2022, {"kind": "quote", "doc": 6046, "quote": "研究生99.05%"}, "年终、研究生。不能当成全校落实率。", tol=0.02)

    # --- contact and addresses ---
    b.contains(
        "2026年招生章程第六条写出的仙林校区和三牌楼校区地址分别是什么？",
        "仙林校区为南京市仙林大学城文苑路9号，三牌楼校区为南京市新模范马路66号",
        ["文苑路9号", "新模范马路66号"],
        "contact",
        2026,
        {"kind": "quote", "doc": 1, "quote": "仙林大学城文苑路 9 号"},
        "具体专业学习地点以各省级招生部门公布的为准。",
    )
    b.contains(
        "招生章程所在页面给出的锁金村校区地址是什么？",
        "南京市龙蟠路177号",
        ["龙蟠路177号"],
        "contact",
        2026,
        {"kind": "quote", "doc": 1, "quote": "锁金村校区地址：南京市龙蟠路177号"},
        "与第六条的仙林、三牌楼地址同页出现。",
    )
    b.contains(
        "本科招生网“联系方式”页面上的招生办公室电话是多少？",
        "025-85866668，85866669",
        ["025-85866668", "85866669"],
        "contact",
        None,
        {"kind": "quote", "doc": 62, "quote": "025-85866668，85866669"},
        "办公地点写的是仙林校区教学4号楼220室、221室。",
    )
    b.contains(
        "本科招生办公室的传真号码是多少？",
        "025-85866270",
        ["025-85866270"],
        "contact",
        None,
        {"kind": "quote", "doc": 62, "quote": "025-85866270"},
        "不要和招生电话混用。",
    )

    # --- genuinely unsupported ---
    ordinary_2025 = conn.execute(
        "SELECT COUNT(*) FROM admission_plans WHERE year=2025 AND admission_type='普通'"
    ).fetchone()[0]
    if ordinary_2025 != 0:
        raise SystemExit("2025 ordinary plans exist; do not mark them unanswerable")
    missing_reports = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE title LIKE '%2023届毕业生就业质量%' OR title LIKE '%2017届毕业生就业质量%' OR title LIKE '%2016届毕业生就业质量%'"
    ).fetchone()[0]
    if missing_reports != 0:
        raise SystemExit("employment report unexpectedly present")
    undergrad_plans = conn.execute(
        """SELECT id, title FROM documents
           WHERE parse_status='parsed' AND title LIKE '%培养方案%'
             AND (title LIKE '%通信与信息工程学院%' OR title LIKE '%电子与光学工程学院%' OR title LIKE '%外国语学院%')
             AND title LIKE '%本科%'"""
    ).fetchall()
    if undergrad_plans:
        raise SystemExit(f"undergrad plans exist { [dict(r) for r in undergrad_plans] }")

    b.refuse("2025年江苏省普通类招生计划一共多少人？", "admission", 2025, {"kind": "absent", "check": "admission_plans year=2025 type=普通 is 0"}, "2025年分省计划扫描件没有形成可核对的数字表。2025年综合评价300人不能用来回答普通类。")
    b.refuse("2025年学校在北京市的普通本科招生计划是多少人？", "admission", 2025, {"kind": "absent", "check": "no 2025 province plan rows"}, "省外分省计划只核到2026年。")
    b.refuse("2018届全校毕业生的就业率是多少？请给一个全校数字。", "employment", 2018, {"kind": "absent", "check": "2018 text has subgroup rates only"}, "6372人可以回答，但正文没有一句全校就业率。研究生94.77%不能当成全校。")
    b.refuse("2021届全校毕业生的毕业去向落实率是多少？", "employment", 2021, {"kind": "absent", "check": "2021 body has 7064 people and no school-wide rate sentence"}, "人数是7064。图表里的百分数没有对应的全校句子，不能当成落实率。")
    b.refuse("2022届全校毕业生统一的毕业去向落实率是多少？不要分本科和研究生。", "employment", 2022, {"kind": "absent", "check": "2022 rates are split"}, "本科初次95.79%、研究生初次96.18%，报告没有再给一个全校合计率。")
    b.refuse("2023届毕业生就业质量年度报告里的全校就业率是多少？", "employment", 2023, {"kind": "absent", "check": "no 2023 report document"}, "公开栏目目前入库的是2018至2022届。")
    b.refuse("2016届毕业生有多少人？", "employment", 2016, {"kind": "absent", "check": "no 2016 report"}, "就业质量报告只核到2018届及以后。")
    b.refuse("通信与信息工程学院本科培养方案规定的通信工程专业毕业学分是多少？", "program", None, {"kind": "absent", "check": "no parsed undergrad plan for that college"}, "库内没有该学院本科培养方案。研究生培养方案的学分不能拿来回答。")
    b.refuse("电子与光学工程学院本科培养方案里，电子科学与技术的总学分是多少？", "program", None, {"kind": "absent", "check": "no parsed undergrad plan"}, "不要用研究生培养方案或招生计划人数代替学分。")
    b.refuse("外国语学院本科培养方案里英语专业要修多少学分？", "program", None, {"kind": "absent", "check": "no parsed undergrad plan"}, "招生计划里的英语专业人数不是培养方案学分。")
    b.refuse("请查询学号B22040101这名在校生目前的平均学分绩点。", "personal", None, {"kind": "absent", "check": "no personal grade data"}, "公开资料没有个人成绩。")
    b.refuse("我考了620分，选考物理和化学，能不能被南邮通信工程录取？", "personal", 2026, {"kind": "absent", "check": "no individual admission decision"}, "计划数不能换成录取结论。")
    b.refuse("登录教务系统后，我本学期有几门不及格？", "personal", None, {"kind": "absent", "check": "login systems are not crawled"}, "需要登录的个人页面不在库内。")
    b.refuse("我的校园卡余额是多少？", "personal", None, {"kind": "absent", "check": "no card system"}, "个人账户数据不在公开资料里。")

    verify(b, zong, js, out_headers, out_total, out_details, ns_headers, ns_total, ns_details)
    for index, case in enumerate(b.cases, start=1):
        case["id"] = f"L{index:03d}"
    OUT.write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in b.cases), encoding="utf-8")
    by_topic = defaultdict(int)
    by_scoring = defaultdict(int)
    for case in b.cases:
        by_topic[case["topic"]] += 1
        by_scoring[case["scoring"]] += 1
    print(f"wrote {len(b.cases)} cases to {OUT}")
    print("scoring", dict(by_scoring))
    print("topic", dict(by_topic))
    conn.close()


def verify(b, zong, js, out_headers, out_total, out_details, ns_headers, ns_total, ns_details):
    """Recompute every numeric gold from the source structures already loaded."""
    for case in b.cases:
        evidence = case["evidence"]
        kind = evidence["kind"]
        gold = case["gold_answer"]
        if case["scoring"] == "numeric":
            value = float(gold)
            got = recompute(b, evidence, zong, js, out_headers, out_total, out_details, ns_headers, ns_total, ns_details)
            if abs(got - value) > 1e-9:
                raise SystemExit(f"gold mismatch {case['question']} gold={value} recomputed={got}")
        elif case["scoring"] == "contains":
            if kind == "quote":
                require_quote(b.text(evidence["doc"]), evidence["quote"], evidence["doc"])
            if kind == "zongping_years":
                rows = []
                for year, items in zong.items():
                    hits = [r for r in items if r["major"] == evidence["major"]]
                    if len(hits) == 1:
                        rows.append(hits[0])
                for row in rows:
                    token = f"{row['year']}年{row['first']}+{row['second']}为{row['plan']}人"
                    if token not in case["gold_answer"]:
                        raise SystemExit(f"year token missing {token}")
            if kind in ("out_beijing_list", "out_qinghai", "conflict"):
                pass
            for key in case["must_contain"]:
                if key not in case["gold_answer"] and norm(key) not in norm(case["gold_answer"]):
                    raise SystemExit(f"must_contain {key} not in gold: {case['question']}")
        elif case["scoring"] != "unanswerable":
            raise SystemExit(case["scoring"])
    if len(b.cases) < 200:
        raise SystemExit(f"only {len(b.cases)} cases")


def recompute(b, evidence, zong, js, out_headers, out_total, out_details, ns_headers, ns_total, ns_details):
    kind = evidence["kind"]
    if kind == "zongping_cell":
        hits = [
            r for r in zong[evidence["year"]]
            if r["major"] == evidence["major"] and r["first"] == evidence["first"] and r["second"] == evidence["second"]
        ]
        if len(hits) != 1:
            raise SystemExit(hits)
        return hits[0]["plan"]
    if kind == "zongping_total":
        return sum(r["plan"] for r in zong[evidence["year"]])
    if kind == "zongping_count":
        return len(zong[evidence["year"]])
    if kind == "zongping_group":
        return sum(r["plan"] for r in zong[2026] if r["first"] == evidence["first"] and r["second"] == evidence["second"])
    if kind == "jiangsu_cell":
        hits = [r for r in js if r["category"] == evidence["category"] and r["group"] == evidence["group"] and r["major"] == evidence["major"]]
        if len(hits) != 1:
            raise SystemExit(evidence)
        return hits[0]["plan"]
    if kind == "jiangsu_category":
        return sum(r["plan"] for r in js if r["category"] == evidence["category"])
    if kind == "jiangsu_all":
        return sum(r["plan"] for r in js)
    if kind == "jiangsu_subject":
        return subject_sum(js, evidence["category"], evidence["subject"])
    if kind == "jiangsu_subject_allow":
        return subject_sum(js, "普通类", "物理+化学") + subject_sum(js, "普通类", "物理+不限")
    if kind == "out_official":
        return int(out_total[out_headers.index(evidence["province"])])
    if kind == "out_detail_total":
        return sum(int(r[1]) for r in out_details if r[1].strip().isdigit())
    if kind == "out_cell":
        i = out_headers.index(evidence["province"])
        hits = [r for r in out_details if re.sub(r"\s+", "", r[0]) == evidence["major"]]
        return int(hits[0][i])
    if kind == "ns_official":
        return int(ns_total[ns_headers.index(evidence["province"])])
    if kind == "ns_major":
        hits = [r for r in ns_details if re.sub(r"\s+", "", r[0]) == evidence["major"]]
        return int(hits[0][1])
    if kind == "ns_cell":
        i = ns_headers.index(evidence["province"])
        hits = [r for r in ns_details if re.sub(r"\s+", "", r[0]) == evidence["major"]]
        return int(hits[0][i])
    if kind == "formula":
        return float(eval(evidence["expr"], {"__builtins__": {}}, {"round": round}))
    if kind == "quote":
        require_quote(b.text(evidence["doc"]), evidence["quote"], evidence["doc"])
        nums = re.findall(r"\d+(?:\.\d+)?", evidence["quote"])
        wanted = evidence.get("number")
        if wanted is None:
            if len(nums) != 1:
                raise SystemExit(f"ambiguous quote {evidence['quote']} -> {nums}")
            return float(nums[0])
        if wanted not in nums:
            raise SystemExit(f"{wanted} not in {evidence['quote']}")
        return float(wanted)
    if kind == "lab":
        areas = {
            "信息与通信工程实验教学中心": 9230,
            "电子与光学工程实验教学中心": 9983,
            "集成电路实验教学中心": 709,
            "计算机实验教学中心": 8056,
            "自动化实验教学中心": 5632,
            "材料与化学实验教学中心": 7150,
            "化学与生命科学实验教学中心": 3807,
            "物联网实验教学中心": 2146,
            "数学实验教学中心": 1515,
            "物理实验教学中心": 2547,
            "现代邮政大数据技术与智能应用实验室": 120,
            "传媒与艺术实验教学中心": 864,
            "经济运行与管理综合实验教学中心": 4586,
            "经济学院实验室": 214,
            "社会与人口学院实验室": 352,
            "外语实验教学中心": 2522,
            "教育科学与技术实验教学中心": 930,
            "工程实验教学部": 9501,
        }
        require_quote(b.text(5), evidence["name"], 5)
        require_quote(b.text(5), str(areas[evidence["name"]]), 5)
        return areas[evidence["name"]]
    if kind == "lab_count":
        return 18
    raise SystemExit(f"cannot recompute {kind}")


if __name__ == "__main__":
    main()
