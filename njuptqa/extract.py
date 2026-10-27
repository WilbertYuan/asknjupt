from __future__ import annotations

import json
import re
from typing import Any

from .db import add_quality, init_db
from .util import extract_academic_year, primary_year, normalize_ws

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _to_int(text: str) -> int | None:
    t = normalize_ws(text).replace(",", "").replace("人", "")
    if not t or t in {"-", "—", "–", "/", "无", "未公布"}:
        return None
    m = NUM_RE.search(t)
    if not m:
        return None
    try:
        return int(float(m.group()))
    except ValueError:
        return None


def _to_float(text: str) -> float | None:
    t = normalize_ws(text).replace(",", "")
    m = NUM_RE.search(t)
    if not m:
        return None
    return float(m.group())


def _looks_like_header(cells: list[str], keys: tuple[str, ...]) -> bool:
    blob = "".join(cells)
    return sum(1 for k in keys if k in blob) >= 2


def _forward_fill(prev: list[str], row: list[str]) -> list[str]:
    out = list(row)
    if len(out) < len(prev):
        out.extend([""] * (len(prev) - len(out)))
    for i, val in enumerate(out):
        if not normalize_ws(val) and i < len(prev):
            out[i] = prev[i]
    return out


def extract_admission_from_table(title: str, headers: list[str], rows: list[list[str]]) -> list[dict[str, Any]]:
    blob = "".join(headers) + title
    if not any(k in blob for k in ("计划", "专业")):
        return []
    # 计算器只用带选科列或综合评价标题的计划表，避免普通人数表污染
    if not any(k in blob for k in ("首选", "再选", "选科", "专业组", "综合评价")):
        return []

    def col(*cands: str) -> int | None:
        for i, h in enumerate(headers):
            if any(c in h for c in cands):
                return i
        return None

    i_first = col("首选")
    i_re = col("再选")
    if i_re is None:
        for i, h in enumerate(headers):
            if "选科" in h and "首选" not in h:
                i_re = i
                break
    i_major = col("专业名称")
    if i_major is None:
        for i, h in enumerate(headers):
            if "专业" in h and "计划" not in h:
                i_major = i
                break
    i_plan = col("计划数") or col("计划") or col("人数")
    i_cat = col("类别")
    i_group = col("专业组")
    # avoid treating a colspan title as every column
    if headers and len(set(headers)) == 1:
        i_first = i_re = i_major = i_plan = None
    # if headers are messy, infer by content
    if i_major is None or i_plan is None:
        # typical 4-col: 首选 再选 专业 计划
        if len(headers) >= 4:
            i_first = 0 if i_first is None else i_first
            i_re = 1 if i_re is None else i_re
            i_major = 2 if i_major is None else i_major
            i_plan = 3 if i_plan is None else i_plan
        else:
            return []

    year = primary_year(title)
    adm_type = "综合评价" if "综合评价" in title else "普通"
    if "地方专项" in title and "招生计划" not in title:
        adm_type = "地方专项"
    default_province = "江苏" if ("江苏省" in title and "省外" not in title) else None
    out: list[dict[str, Any]] = []
    prev = [""] * max(len(headers), 4)
    for ri, row in enumerate(rows):
        filled = _forward_fill(prev, row)
        prev = filled
        major = normalize_ws(filled[i_major] if i_major < len(filled) else "")
        plan = _to_int(filled[i_plan] if i_plan < len(filled) else "")
        first = normalize_ws(filled[i_first] if i_first is not None and i_first < len(filled) else "")
        reselect = normalize_ws(filled[i_re] if i_re is not None and i_re < len(filled) else "")
        if not first and reselect:
            parts = re.split(r"\s*\+\s*", reselect)
            if parts and parts[0] in {"物理", "历史"}:
                first = parts[0]
        cat = normalize_ws(filled[i_cat] if i_cat is not None and i_cat < len(filled) else "")
        row_type = adm_type
        if cat and "综合评价" not in title:
            row_type = "普通" if cat in {"普通类", "普通"} else cat
        group = normalize_ws(filled[i_group] if i_group is not None and i_group < len(filled) else "")
        role = "detail"
        if any(x in major for x in ("合计", "总计")) or major == "":
            if plan is not None and ("合计" in "".join(filled) or "总计" in "".join(filled)):
                role = "total"
                major = major or "合计"
            else:
                continue
        if not major:
            continue
        if re.fullmatch(r"[A-Z]?\d{4}[A-Z0-9]{2,4}", major):
            continue
        if plan is None and role != "total":
            continue
        out.append(
            {
                "year": year,
                "admission_type": row_type,
                "province": default_province,
                "category": cat or None,
                "group_code": group or None,
                "first_subject": first or None,
                "reselect": reselect or None,
                "major": major,
                "plan": plan,
                "row_role": role,
                "row_index": ri,
            }
        )
    return out


PROVINCES = (
    "北京", "天津", "河北", "山西", "内蒙古", "辽宁", "吉林", "黑龙江", "上海", "浙江",
    "安徽", "福建", "江西", "山东", "河南", "湖北", "湖南", "广东", "广西", "海南",
    "重庆", "四川", "贵州", "云南", "西藏", "陕西", "甘肃", "青海", "宁夏", "新疆", "江苏",
)


def extract_province_matrix(title: str, headers: list[str], rows: list[list[str]]) -> list[dict[str, Any]]:
    """省外/专项计划矩阵：每个专业、每个省份一条，合计行不拆进明细。"""
    names = [re.sub(r"\s+", "", h or "") for h in headers]
    known = [(i, name) for i, name in enumerate(names) if name in PROVINCES]
    if len(known) < 8 or "专业名称" not in names:
        return []
    extra = [
        (i, name)
        for i, name in enumerate(names)
        if name
        and name not in {"专业名称", "合计"}
        and name not in PROVINCES
        and re.fullmatch(r"[\u4e00-\u9fff]{2,8}", name)
    ]
    prov_idx = known + extra
    i_major = names.index("专业名称")
    i_total = names.index("合计") if "合计" in names else None
    if "国家专项" in title:
        adm = "国家专项"
    elif "地方专项" in title:
        adm = "地方专项"
    elif "省外" in title:
        adm = "省外"
    else:
        adm = "分省"
    year = primary_year(title)
    out: list[dict[str, Any]] = []
    for ri, row in enumerate(rows):
        major = normalize_ws(row[i_major] if i_major < len(row) else "")
        if not major:
            continue
        if "合计" in major:
            # 只把本科合计行的各列当成官方单元格。文史/理科/艺术小计不并进明细。
            if "本科合计" in major or major in {"合计", "总计"}:
                for i, prov in prov_idx:
                    cell = _to_int(row[i]) if i < len(row) else None
                    if cell is None:
                        continue
                    out.append(
                        {
                            "year": year,
                            "admission_type": adm,
                            "province": prov,
                            "category": None,
                            "group_code": None,
                            "first_subject": None,
                            "reselect": None,
                            "major": major,
                            "plan": cell,
                            "row_role": "total",
                            "row_index": ri,
                        }
                    )
                total = _to_int(row[i_total]) if i_total is not None and i_total < len(row) else None
                if total is not None:
                    out.append(
                        {
                            "year": year,
                            "admission_type": adm,
                            "province": "合计",
                            "category": None,
                            "group_code": None,
                            "first_subject": None,
                            "reselect": None,
                            "major": major,
                            "plan": total,
                            "row_role": "total",
                            "row_index": ri,
                        }
                    )
            continue
        for i, prov in prov_idx:
            if i >= len(row):
                continue
            plan = _to_int(row[i])
            if not plan:
                continue
            out.append(
                {
                    "year": year,
                    "admission_type": adm,
                    "province": prov,
                    "category": None,
                    "group_code": None,
                    "first_subject": None,
                    "reselect": None,
                    "major": major,
                    "plan": plan,
                    "row_role": "detail",
                    "row_index": ri,
                }
            )
        if i_total is not None and i_total < len(row):
            plan = _to_int(row[i_total])
            if plan:
                out.append(
                    {
                        "year": year,
                        "admission_type": adm,
                        "province": "合计",
                        "category": None,
                        "group_code": None,
                        "first_subject": None,
                        "reselect": None,
                        "major": major,
                        "plan": plan,
                        "row_role": "detail",
                        "row_index": ri,
                    }
                )
    return out


LAB_LINE_RE = re.compile(
    r"(?P<seq>\d{1,2})\s+(?P<name>[\u4e00-\u9fffA-Za-z0-9（）()、]+(?:实验教学中心|实验教学部|实验室))\s+(?P<college>[\u4e00-\u9fffA-Za-z0-9（）()、]+)\s+(?P<area>\d{3,5}(?:\.\d+)?)"
)


def extract_labs_from_text(text: str, title: str) -> list[dict[str, Any]]:
    academic = extract_academic_year(title + text[:800])
    out = []
    for m in LAB_LINE_RE.finditer(text.replace("\u3000", " ")):
        name = normalize_ws(m.group("name"))
        if "合计" in name or "生均" in name or "总面积" in name or "重点实验室" in name:
            continue
        out.append(
            {
                "academic_year": academic,
                "seq": m.group("seq"),
                "name": name,
                "college": normalize_ws(m.group("college")),
                "area_m2": float(m.group("area")),
                "row_index": int(m.group("seq")),
            }
        )
    return out


def extract_labs_from_table(title: str, headers: list[str], rows: list[list[str]]) -> list[dict[str, Any]]:
    blob = "".join(headers) + title
    if "实验" not in blob and "面积" not in blob:
        return []
    if not any(k in blob for k in ("面积", "平方米", "实验教学中心")):
        return []

    def col(*cands: str) -> int | None:
        for i, h in enumerate(headers):
            if any(c in h for c in cands):
                return i
        return None

    i_name = col("名称", "实验教学中心")
    i_college = col("学院", "所在")
    i_area = col("面积", "平方米")
    i_seq = col("序号")
    if i_name is None or i_area is None:
        # common 4-col layout: 序号 名称 学院 面积
        if len(headers) >= 4:
            i_seq, i_name, i_college, i_area = 0, 1, 2, 3
        elif len(headers) == 3:
            i_name, i_college, i_area = 0, 1, 2
        else:
            return []
    academic = extract_academic_year(title)
    out = []
    for ri, row in enumerate(rows):
        name = normalize_ws(row[i_name] if i_name < len(row) else "")
        area = _to_float(row[i_area] if i_area < len(row) else "")
        if not name:
            continue
        if "合计" in name or "总计" in name or "重点实验室" in name:
            continue
        if area is None:
            continue
        out.append(
            {
                "academic_year": academic,
                "seq": normalize_ws(row[i_seq]) if i_seq is not None and i_seq < len(row) else str(ri + 1),
                "name": name,
                "college": normalize_ws(row[i_college]) if i_college is not None and i_college < len(row) else "",
                "area_m2": area,
                "row_index": ri,
            }
        )
    return out


FEE_PATTERNS = [
    (r"文科类专业每生每年\s*(\d+)\s*元", "文科", "tuition", None),
    (r"理科类专业每生每年\s*(\d+)\s*元", "理科", "tuition", None),
    (r"工科类专业每生每年\s*(\d+)\s*元", "工科", "tuition", None),
    (r"艺术类专业每生每年\s*(\d+)\s*元", "艺术", "tuition", None),
    (r"住宿费暂按\s*(\d+)\s*元", "住宿", "housing", None),
    (r"优势学科专业在上述标准基础上上浮\s*(\d+)\s*%", "优势学科", "uplift", None),
    (r"综合评价招生测试费标准为\s*(\d+)\s*元", "综合评价测试", "test_fee", None),
]


def extract_fee_rules(text: str, title: str) -> list[dict[str, Any]]:
    year = primary_year(title, text)
    rules = []
    for pat, category, item, _ in FEE_PATTERNS:
        m = re.search(pat, text)
        if not m:
            continue
        val = float(m.group(1))
        uplift = val / 100.0 if item == "uplift" else None
        amount = None if item == "uplift" else val
        rules.append(
            {
                "year": year,
                "category": category,
                "item": item,
                "amount": amount,
                "unit": "元/生·年" if item in {"tuition", "housing"} else ("比例" if item == "uplift" else "元"),
                "uplift_ratio": uplift,
                "notes": m.group(0),
            }
        )
    return rules


EXTRA_FACTS = [
    (r"预留计划数不超过年度本科招生计划总数的\s*(\d+)\s*%", "reserve_plan_pct", "%"),
    (r"按不超过综合评价招生计划\s*(\d+)\s*倍", "zhpj_shortlist_multiple", "倍"),
    (r"英语专业招生语种要求为(英语)", "english_major_language", ""),
    (r"招生专业和计划最终以(省级招生主管部门)核定为准", "plan_final_authority", ""),
    (r"仙林校区[地址：:（(]*([^。\n]{4,50}文苑路\s*\d+\s*号)", "xianlin_address", ""),
    (r"锁金村校区[地址：:（(]*([^。\n]{4,50}龙蟠路\s*\d+\s*号)", "suojin_address", ""),
    (r"三牌楼校区[地址：:（(]*([^。\n]{4,50}新模范马路\s*\d+\s*号)", "sanpailou_address", ""),
    (r"咨询电话[：:\s]*（?025）?\s*(85866668)", "zs_phone", ""),
    (r"(高考投档分\s*/\s*750[^。]{0,50}测试成绩\s*/\s*100[^。]{0,20})", "zhpj_score_formula", ""),
    (r"(以特殊招生形式录取的学生[^。]{0,40}不得转专业)", "special_no_transfer", ""),
    (r"(025-?85866668)", "zs_phone", ""),
]


def extract_extra_facts(text: str, title: str) -> list[dict[str, Any]]:
    year = primary_year(title, text)
    academic = extract_academic_year(title + text[:400])
    out = []
    for pat, metric, unit in EXTRA_FACTS:
        m = re.search(pat, text.replace("\n", ""))
        if not m:
            continue
        val = m.group(1).strip().lstrip("：:，,（(").rstrip("）)")
        if metric.endswith("_address") and not any(
            k in title for k in ("简介", "章程", "联系", "概况", "招生", "学校", "地址")
        ):
            continue
        num = None
        try:
            num = float(val)
        except ValueError:
            num = None
        out.append(
            {
                "entity": "南京邮电大学",
                "metric": metric,
                "value": val,
                "numeric": num,
                "unit": unit,
                "year": year,
                "academic_year": academic,
                "scope": "公开",
                "origin": "extracted",
                "verified": "auto",
                "extra_json": json.dumps({"span": m.group(0)[:80]}, ensure_ascii=False),
            }
        )
    return out


TEACHER_PATS = {
    "fulltime": re.compile(r"专任教师\s*(\d+)\s*人"),
    "adjunct": re.compile(r"(?:聘请校外教师|外聘教师)\s*(\d+)\s*人"),
    "equivalent": re.compile(r"折合教师总数[^\d]{0,12}(\d+(?:\.\d+)?)"),
    "equivalent2": re.compile(r"折合教师[^\d]{0,8}(\d+(?:\.\d+)?)"),
    "students": re.compile(r"折合在校生人数\s*(\d+(?:\.\d+)?)"),
    "ratio": re.compile(r"生师比为\s*(\d+(?:\.\d+))"),
}


POLICY_FACT_GROUPS = [
    (
        ("奖学金", "助学金", "资助", "奖助"),
        [
            (r"国家励志奖学金[^\d]{0,24}(?:每[生人]每年|标准|金额为?)?[^\d]{0,10}(\d{4,5})\s*元", "national_inspirational_scholarship", "元"),
            (r"国家奖学金[^\d]{0,24}(?:每[生人]每年|标准|金额为?)?[^\d]{0,10}(\d{4,5})\s*元", "national_scholarship", "元"),
            (r"国家助学金[^\d]{0,20}(?:平均|每[生人]每年|标准为?)?[^\d]{0,10}(\d{3,5})\s*元", "national_grant", "元"),
            (r"国家助学金[^\d]{0,28}一等[^\d]{0,10}(\d{3,5})\s*元", "national_grant_1", "元"),
            (r"国家助学金[^\d]{0,28}二等[^\d]{0,10}(\d{3,5})\s*元", "national_grant_2", "元"),
            (r"国家助学金[^\d]{0,28}三等[^\d]{0,10}(\d{3,5})\s*元", "national_grant_3", "元"),
        ],
    ),
    (
        ("推免", "免试攻读"),
        [
            (r"推荐优秀应届本科毕业生免试攻读(?:硕士学位)?研究生", "tuimian_policy", ""),
            (r"推免生?[^\d]{0,16}(?:名额|计划)[^\d]{0,8}(\d{1,4})", "tuimian_quota", "人"),
        ],
    ),
    (
        ("就业质量", "毕业生就业", "就业率"),
        [
            (r"毕业生总[数人]\s*[:：]?\s*(\d{3,5})", "graduates_total", "人"),
            (r"毕业生共\s*(\d{3,5})\s*人", "graduates_total", "人"),
            (r"届毕业生年终就业率为\s*(\d{2,3}(?:\.\d+)?)\s*%", "employment_rate", "%"),
            (r"年终\s*总就业率[为是：: \t]*(\d{2,3}(?:\.\d+)?)\s*%", "employment_rate", "%"),
            (r"届毕业生初次就业率为\s*(\d{2,3}(?:\.\d+)?)\s*%", "employment_rate_initial", "%"),
            (r"初次\s*总就业率[为是：: \t]*(\d{2,3}(?:\.\d+)?)\s*%", "employment_rate_initial", "%"),
        ],
    ),
    (
        ("培养方案", "毕业学分", "学士学位", "毕业要求"),
        [
            (r"(?:本科)?毕业[^\d。]{0,24}(?:最低|须修满|修满)\s*(\d{2,3})\s*学分", "undergrad_grad_credits", "学分"),
            (r"学士学位[^\d。]{0,30}(?:平均学分绩点|GPA|绩点)[^\d]{0,8}(\d(?:\.\d+)?)", "bachelor_gpa", ""),
        ],
    ),
]


def extract_policy_facts(text: str, title: str) -> list[dict[str, Any]]:
    compact = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    blob = title + compact[:2500]
    year = primary_year(title, text)
    academic = extract_academic_year(title + text[:400])
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for keys, pats in POLICY_FACT_GROUPS:
        if not any(k in blob for k in keys):
            continue
        for pat, metric, unit in pats:
            if metric in seen:
                continue
            m = re.search(pat, compact)
            if not m:
                continue
            val = m.group(1).strip() if m.lastindex else m.group(0).strip()
            num = None
            try:
                num = float(val)
            except ValueError:
                num = None
            if metric.endswith(("_scholarship", "_grant", "_grant_1", "_grant_2", "_grant_3")) and num is not None:
                if num < 500 or num > 20000:
                    continue
            if metric in {"employment_rate", "employment_rate_initial"}:
                if "就业" not in title or num is None or not (50 <= num <= 100):
                    continue
            if metric == "tuimian_quota":
                if not val.isdigit() or val.startswith("0") or num is None or not (1 <= num <= 800):
                    continue
            if metric == "bachelor_gpa" and "学位" not in title + compact[:200]:
                continue
            if metric == "graduates_total" and "就业" not in title:
                continue
            seen.add(metric)
            out.append(
                {
                    "entity": "南京邮电大学",
                    "metric": metric,
                    "value": val,
                    "numeric": num,
                    "unit": unit,
                    "year": year,
                    "academic_year": academic,
                    "scope": "公开",
                    "origin": "extracted",
                    "verified": "auto",
                    "extra_json": json.dumps({"span": m.group(0)[:80], "title": title[:60]}, ensure_ascii=False),
                }
            )
    return out


def extract_teacher_facts(text: str, title: str) -> list[dict[str, Any]]:
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    academic = extract_academic_year(title + text[:400]) or extract_academic_year(text)
    year = primary_year(title, text)
    facts = []
    seen_metrics: set[str] = set()
    for metric, pat in TEACHER_PATS.items():
        key = "equivalent" if metric.startswith("equivalent") else metric
        if key in seen_metrics:
            continue
        m = pat.search(text)
        if not m:
            continue
        val = float(m.group(1))
        # skip implausible teacher-total if it is actually student count
        if key == "equivalent" and val > 20000:
            continue
        seen_metrics.add(key)
        metric = key
        facts.append(
            {
                "entity": "南京邮电大学",
                "metric": metric,
                "value": m.group(1),
                "numeric": val,
                "unit": "人" if metric != "ratio" else "1",
                "year": year,
                "academic_year": academic,
                "scope": "全校",
                "origin": "extracted",
                "verified": "auto",
                "extra_json": json.dumps({"span": m.group(0)}, ensure_ascii=False),
            }
        )
    return facts


def extract_all() -> dict:
    conn = init_db()
    conn.execute("DELETE FROM admission_plans")
    conn.execute("DELETE FROM fee_rules")
    conn.execute("DELETE FROM lab_centers")
    # keep manual facts; delete auto extracted
    conn.execute("DELETE FROM facts WHERE origin='extracted'")

    n_plan = n_fee = n_lab = n_fact = 0
    seen_labs: set[tuple] = set()
    seen_facts: set[tuple] = set()
    docs = conn.execute("SELECT id, title, url FROM documents").fetchall()
    doc_title = {d["id"]: d["title"] or "" for d in docs}

    tables = conn.execute(
        "SELECT id, document_id, title, header_json, raw_json, locator FROM tables ORDER BY document_id, id"
    ).fetchall()
    last_plan_header: dict[int, list[str]] = {}
    for tb in tables:
        headers = json.loads(tb["header_json"] or "[]")
        raw = json.loads(tb["raw_json"] or "[]")
        title = f"{doc_title.get(tb['document_id'],'') } {tb['title'] or ''}"
        rows = raw
        header_i = None
        for i, row in enumerate(raw[:6]):
            if len(set(row)) <= 1 and len(row) > 1:
                continue
            if _looks_like_header(row, ("首选", "再选", "专业", "计划", "面积", "序号", "名称")):
                header_i = i
                break
        inherited = False
        if header_i is not None:
            headers = raw[header_i]
            rows = raw[header_i + 1 :]
        else:
            # 跨页续表往往不再重复表头。沿用同一文档上一张计划表的表头。
            prev = last_plan_header.get(tb["document_id"])
            width = len(raw[0]) if raw else 0
            blob0 = "".join(str(x) for x in (raw[0] if raw else []))
            looks_cont = bool(re.search(r"普通类|艺术类|专项|综合评价|物理\s*\+|历史\s*\+|1\d{5}", blob0))
            if prev and raw and looks_cont and width == len(prev) and 4 <= width <= 8:
                headers = prev
                rows = raw
                inherited = True
        plans = extract_admission_from_table(title, headers, rows)
        if not plans and not inherited:
            plans = extract_province_matrix(title, headers, rows)
        if plans and header_i is not None and any(k in "".join(headers) for k in ("计划", "专业", "选科")):
            last_plan_header[tb["document_id"]] = list(headers)
        for p in plans:
            conn.execute(
                """INSERT INTO admission_plans(document_id, year, admission_type, province, category,
                    group_code, first_subject, reselect, major, plan, row_role, table_id, row_index, locator)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    tb["document_id"],
                    p["year"],
                    p["admission_type"],
                    p.get("province"),
                    p.get("category"),
                    p.get("group_code"),
                    p["first_subject"],
                    p["reselect"],
                    p["major"],
                    p["plan"],
                    p["row_role"],
                    tb["id"],
                    p["row_index"],
                    tb["locator"],
                ),
            )
            n_plan += 1
        labs = extract_labs_from_table(title, headers, rows)
        for lab in labs:
            key = (lab["name"], float(lab["area_m2"] or 0), lab.get("academic_year"))
            if key in seen_labs:
                continue
            seen_labs.add(key)
            conn.execute(
                """INSERT INTO lab_centers(document_id, academic_year, seq, name, college, area_m2, table_id, row_index, locator)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    tb["document_id"],
                    lab["academic_year"],
                    lab["seq"],
                    lab["name"],
                    lab["college"],
                    lab["area_m2"],
                    tb["id"],
                    lab["row_index"],
                    tb["locator"],
                ),
            )
            n_lab += 1

    # text fallback for lab tables when PDF grid extraction fails
    if n_lab == 0:
        for b in conn.execute("SELECT document_id, text, heading, locator FROM blocks").fetchall():
            title = f"{doc_title.get(b['document_id'],'')} {b['heading'] or ''}"
            for lab in extract_labs_from_text(b["text"] or "", title):
                conn.execute(
                    """INSERT INTO lab_centers(document_id, academic_year, seq, name, college, area_m2, locator)
                       VALUES(?,?,?,?,?,?,?)""",
                    (
                        b["document_id"],
                        lab["academic_year"],
                        lab["seq"],
                        lab["name"],
                        lab["college"],
                        lab["area_m2"],
                        b["locator"],
                    ),
                )
                n_lab += 1

    blocks = conn.execute("SELECT document_id, text, heading, locator FROM blocks").fetchall()
    for b in blocks:
        title = f"{doc_title.get(b['document_id'],'')} {b['heading'] or ''}"
        text = b["text"] or ""
        for rule in extract_fee_rules(text, title):
            conn.execute(
                """INSERT INTO fee_rules(document_id, year, category, item, amount, unit, uplift_ratio, notes, locator)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    b["document_id"],
                    rule["year"],
                    rule["category"],
                    rule["item"],
                    rule["amount"],
                    rule["unit"],
                    rule["uplift_ratio"],
                    rule["notes"],
                    b["locator"],
                ),
            )
            n_fee += 1
        for fact in extract_teacher_facts(text, title) + extract_extra_facts(text, title) + extract_policy_facts(text, title):
            fkey = (fact.get("entity") or "", fact["metric"], str(fact["value"]))
            if fkey in seen_facts:
                continue
            seen_facts.add(fkey)
            conn.execute(
                """INSERT INTO facts(entity, metric, value, numeric, unit, year, academic_year, scope,
                    source_document_id, source_locator, origin, verified, extra_json)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    fact["entity"],
                    fact["metric"],
                    fact["value"],
                    fact["numeric"],
                    fact["unit"],
                    fact["year"],
                    fact["academic_year"],
                    fact["scope"],
                    b["document_id"],
                    b["locator"],
                    fact["origin"],
                    fact["verified"],
                    fact["extra_json"],
                ),
            )
            n_fact += 1

    # quality: plan totals vs detail sums
    years = conn.execute("SELECT DISTINCT year, admission_type FROM admission_plans WHERE year IS NOT NULL").fetchall()
    for y in years:
        details = conn.execute(
            "SELECT SUM(plan) s FROM admission_plans WHERE year=? AND admission_type=? AND row_role='detail'",
            (y["year"], y["admission_type"]),
        ).fetchone()["s"]
        total = conn.execute(
            "SELECT plan FROM admission_plans WHERE year=? AND admission_type=? AND row_role='total' LIMIT 1",
            (y["year"], y["admission_type"]),
        ).fetchone()
        if total and details is not None and int(total["plan"]) != int(details):
            add_quality(
                conn,
                "plan_total_mismatch",
                f"{y['year']} {y['admission_type']} 合计{total['plan']} 明细和{details}。未改数。",
            )

    conn.execute(
        """DELETE FROM lab_centers WHERE id NOT IN (
             SELECT MIN(id) FROM lab_centers GROUP BY name, CAST(area_m2 AS TEXT), IFNULL(academic_year,'')
           )"""
    )
    n_lab = conn.execute("SELECT COUNT(*) c FROM lab_centers").fetchone()["c"]
    labs_sum = conn.execute("SELECT SUM(area_m2) s, COUNT(*) c FROM lab_centers").fetchone()
    if labs_sum["c"] and labs_sum["s"]:
        add_quality(conn, "lab_sum_info", f"已抽取实验中心{labs_sum['c']}条，面积合计{labs_sum['s']}")

    catalog = sync_catalog(conn)
    conn.commit()
    conn.close()
    return {
        "admission_plans": n_plan,
        "fee_rules": n_fee,
        "lab_centers": n_lab,
        "facts": n_fact,
        "entities": catalog["entities"],
        "aliases": catalog["aliases"],
        "relations": catalog["relations"],
    }


def sync_catalog(conn) -> dict[str, int]:
    """机构、专业、校区和附件关系。只写入已入库的名称，不编造。"""
    conn.execute("DELETE FROM relations")
    conn.execute("DELETE FROM aliases")
    conn.execute("DELETE FROM entities")

    def add_entity(name: str, kind: str, year: int | None, attrs: dict | None = None) -> int | None:
        name = normalize_ws(name)
        if not name or len(name) < 2:
            return None
        conn.execute(
            """INSERT OR IGNORE INTO entities(name, kind, year, attrs_json)
               VALUES(?,?,?,?)""",
            (name, kind, year, json.dumps(attrs or {}, ensure_ascii=False)),
        )
        row = conn.execute(
            "SELECT id FROM entities WHERE name=? AND kind=? AND year IS ?",
            (name, kind, year),
        ).fetchone()
        return int(row["id"]) if row else None

    school = add_entity("南京邮电大学", "school", None, {"scope": "全校"})
    if school:
        conn.execute(
            "INSERT INTO aliases(entity_id, alias, note) VALUES(?,?,?)",
            (school, "南邮", "学校简称"),
        )
    for metric, alias in (
        ("xianlin_address", "仙林校区"),
        ("suojin_address", "锁金村校区"),
        ("sanpailou_address", "三牌楼校区"),
    ):
        fact = conn.execute(
            "SELECT value, source_document_id FROM facts WHERE metric=? LIMIT 1",
            (metric,),
        ).fetchone()
        if not fact:
            continue
        eid = add_entity(alias, "campus", None, {"address": fact["value"], "document_id": fact["source_document_id"]})
        if eid and school:
            conn.execute(
                "INSERT INTO relations(rel_type, src_kind, src_id, dst_kind, dst_id, note) VALUES(?,?,?,?,?,?)",
                ("located_at", "school", school, "campus", eid, fact["value"]),
            )
    for r in conn.execute(
        """SELECT DISTINCT major FROM admission_plans
           WHERE year=2026 AND admission_type='综合评价' AND row_role='detail'"""
    ):
        add_entity(r["major"], "major", 2026, {"admission_type": "综合评价"})
    for r in conn.execute("SELECT DISTINCT college FROM lab_centers WHERE college IS NOT NULL AND college!=''"):
        add_entity(r["college"], "college", None, {"from": "lab_centers"})
    for r in conn.execute(
        """SELECT c.id AS child_id, p.id AS parent_id, c.url
           FROM sources s
           JOIN documents c ON c.url=s.url
           JOIN documents p ON p.url=s.discovered_from
           WHERE s.discovered_from IS NOT NULL AND s.discovered_from!=''"""
    ):
        conn.execute(
            "INSERT INTO relations(rel_type, src_kind, src_id, dst_kind, dst_id, note) VALUES(?,?,?,?,?,?)",
            ("attachment", "document", r["child_id"], "document", r["parent_id"], "discovered_from"),
        )
    return {
        "entities": conn.execute("SELECT COUNT(*) c FROM entities").fetchone()["c"],
        "aliases": conn.execute("SELECT COUNT(*) c FROM aliases").fetchone()["c"],
        "relations": conn.execute("SELECT COUNT(*) c FROM relations").fetchone()["c"],
    }


def validate() -> dict:
    conn = init_db()
    docs = conn.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"]
    parsed = conn.execute("SELECT COUNT(*) c FROM documents WHERE parse_status='parsed'").fetchone()["c"]
    tables = conn.execute("SELECT COUNT(*) c FROM tables").fetchone()["c"]
    plans = conn.execute("SELECT COUNT(*) c FROM admission_plans").fetchone()["c"]
    fees = conn.execute("SELECT COUNT(*) c FROM fee_rules").fetchone()["c"]
    labs = conn.execute("SELECT COUNT(*) c FROM lab_centers").fetchone()["c"]
    qopen = conn.execute("SELECT COUNT(*) c FROM quality_queue WHERE status='open'").fetchone()["c"]
    issues = [
        dict(r)
        for r in conn.execute(
            "SELECT kind, detail, document_id FROM quality_queue WHERE status='open' LIMIT 50"
        )
    ]
    conn.close()
    return {
        "documents": docs,
        "parsed": parsed,
        "tables": tables,
        "admission_plans": plans,
        "fee_rules": fees,
        "lab_centers": labs,
        "open_quality": qopen,
        "issues": issues,
    }
