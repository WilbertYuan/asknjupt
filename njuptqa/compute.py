from __future__ import annotations

import copy
import json
import re
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from .db import init_db
from .retrieve import QueryPlan
from .util import focus_text, utc_now


def _dec(x: float | int | str | Decimal) -> Decimal:
    return Decimal(str(x))


def _show(x: Any) -> str:
    if x is None or x == "":
        return ""
    d = _dec(x)
    if d == d.to_integral():
        return str(int(d))
    text = format(d, "f")
    return text.rstrip("0").rstrip(".")


def _q(x: Decimal, places: int = 2) -> Decimal:
    return x.quantize(Decimal("0." + "0" * places) if places else Decimal("1"), rounding=ROUND_HALF_UP)


def subject_allows(required: str | None, first: str | None, seconds: list[str]) -> bool:
    """是否满足专业的首选/再选要求。"""
    req = (required or "").replace("＋", "+").replace(" ", "")
    if not req or req in {"不限", "无", "-", "——"}:
        return True
    if "不限" in req and not any(s in req for s in ("化学", "生物", "政治", "地理", "技术")):
        return True
    # first subject checked separately
    if req in {"物理", "历史"}:
        return first == req
    if "不限" in req and "+" in req:
        # 物理+不限
        left = req.split("+")[0]
        return first == left if left in {"物理", "历史"} else True
    parts = re.split(r"[+/及和、或]", req)
    parts = [p for p in parts if p and p not in {"首选", "再选"}]
    need_and = any(k in (required or "") for k in ("且", "和", "+")) and "或" not in (required or "")
    # 再选化学 => must include 化学
    subjects = [p for p in parts if p in {"化学", "生物", "政治", "地理", "技术"}]
    if not subjects:
        return True
    if "或" in (required or ""):
        return any(s in seconds for s in subjects)
    return all(s in seconds for s in subjects)


def first_allows(cell: str | None, first: str | None) -> bool:
    if not first:
        return True
    c = cell or ""
    if not c or c == "不限":
        return True
    return first in c


def _compact(text: str | None) -> str:
    return re.sub(r"\s+", "", text or "")


def subject_key(row: dict[str, Any]) -> str:
    """把一行收成可比较的选科原文，例如 物理+化学、历史+不限。"""
    first = _compact(row.get("first_subject"))
    reselect = _compact(row.get("reselect")).replace("＋", "+")
    if "+" in reselect:
        return reselect
    if first and reselect:
        return f"{first}+{reselect}"
    return reselect or first


def _major_ok(plan: QueryPlan, major: str | None) -> bool:
    row_m = _compact(major)
    if plan.major_exact:
        return row_m == _compact(plan.major_exact)
    if plan.major_contains:
        return _compact(plan.major_contains) in row_m
    asked = [_compact(m) for m in plan.majors if m]
    if not asked:
        return True
    q_norm = _compact(plan.question)
    longer = any(row_m != other and row_m in other and other in q_norm for other in asked)
    if row_m in asked or (row_m and row_m in q_norm):
        return not longer
    return False


def _select_plans(conn, plan: QueryPlan, role: str) -> list[dict[str, Any]]:
    q = plan.question
    sql = """SELECT p.* FROM admission_plans p
             JOIN documents d ON d.id = p.document_id
             WHERE p.row_role=?"""
    params: list[Any] = [role]
    if plan.years and plan.answer_form != "by_year":
        sql += " AND p.year IN ({})".format(",".join("?" * len(plan.years)))
        params.extend(plan.years)
    if plan.admission_type:
        sql += " AND p.admission_type=?"
        params.append(plan.admission_type)
    scope = plan.doc_scope
    if not scope and plan.admission_type == "综合评价" and "招生计划" not in q and "计划表" not in q:
        scope = "简章"
    if scope == "简章":
        sql += " AND d.title LIKE '%简章%'"
    elif scope == "江苏省招生计划":
        sql += " AND d.title LIKE '%江苏省招生计划%'"
    if plan.province:
        sql += " AND p.province=?"
        params.append(plan.province)
    elif role == "detail":
        sql += " AND (p.province IS NULL OR p.province!='合计')"
    if plan.group_code:
        sql += " AND upper(replace(ifnull(p.group_code,''), ' ', ''))=?"
        params.append(plan.group_code.upper())
    if role == "total" and ("本科合计" in q or "△" in q):
        sql += " AND p.major LIKE '%本科合计%'"
    rows = [dict(r) for r in conn.execute(sql, params)]
    matched = []
    for r in rows:
        if plan.literal_subject and plan.literal_target:
            if subject_key(r) != plan.literal_target:
                continue
        else:
            if plan.first_subject and not first_allows(r.get("first_subject"), plan.first_subject):
                continue
            if plan.second_subjects and not subject_allows(r.get("reselect"), plan.first_subject, plan.second_subjects):
                continue
        if not _major_ok(plan, r.get("major")):
            continue
        matched.append(r)
    return matched


def _plan_docs(conn, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    docs = []
    for i in sorted({r["document_id"] for r in rows if r.get("document_id")}):
        d = conn.execute("SELECT id, url, title FROM documents WHERE id=?", (i,)).fetchone()
        if d:
            docs.append(dict(d))
    return docs


def filter_plans(plan: QueryPlan) -> dict[str, Any]:
    conn = init_db()
    filters = {
        "year": plan.years,
        "admission_type": plan.admission_type,
        "first_subject": plan.first_subject,
        "second_subjects": plan.second_subjects,
        "majors": plan.majors,
        "province": plan.province,
        "literal": plan.literal_target if plan.literal_subject else None,
        "form": plan.answer_form,
    }
    if plan.answer_form == "conflict":
        details = _select_plans(conn, plan, "detail")
        officials = _select_plans(conn, plan, "total")
        detail_sum = sum(int(r["plan"] or 0) for r in details)
        official = sum(int(r["plan"] or 0) for r in officials)
        conn.close()
        if not details and not officials:
            return {"op": "filter_sum_plans", "error": "没有匹配的计划明细或合计单元格", "filters": filters}
        same = detail_sum == official
        return {
            "op": "filter_sum_plans",
            "matched": details,
            "count": len(details),
            "sum": detail_sum,
            "official": official,
            "headline": f"明细相加{detail_sum}人，合计行{official}人，{'一致' if same else '不一致'}",
            "formula": f"detail={detail_sum}; official_cell={official}",
            "documents": [],
            "filters": filters,
        }
    role = "total" if plan.answer_form == "official" else "detail"
    matched = _select_plans(conn, plan, role)
    docs = _plan_docs(conn, matched)
    conn.close()
    if not matched:
        return {"op": "filter_sum_plans", "error": "没有匹配到计划明细", "filters": filters, "matched": []}
    total = sum(int(r["plan"] or 0) for r in matched)
    headline = f"{total}人"
    asks_both = plan.answer_form == "count" and any(
        k in focus_text(plan.question) for k in ("总计划", "计划数", "多少人")
    )
    if asks_both:
        headline = f"{len(matched)}个，计划合计{total}人"
    elif plan.answer_form == "count":
        headline = f"{len(matched)}个"
    elif plan.answer_form == "official":
        headline = f"{int(matched[0]['plan'])}人"
        total = int(matched[0]["plan"])
    elif plan.answer_form == "list":
        headline = "、".join(f"{r['major']}{int(r['plan'])}人" for r in matched)
    elif plan.answer_form == "by_year":
        lines = []
        for r in sorted(matched, key=lambda item: (item.get("year") or 0, item.get("major") or "")):
            lines.append(f"{r.get('year')}年{subject_key(r)}为{int(r['plan'])}人")
        headline = "；".join(lines)
    return {
        "op": "filter_sum_plans",
        "matched": matched,
        "count": len(matched),
        "sum": total,
        "headline": headline,
        "formula": "sum(plan where the row's full major and subject operator match; total rows stay out of detail sums)",
        "documents": docs,
        "filters": filters,
    }


def _province_headline(plan: QueryPlan, total: int) -> str | None:
    """问开展省份和计划合计时，省份和核定口径用简章原句，人数用计划表合计。"""
    q = plan.question
    if "计划合计" not in q or "哪个省" not in q or len(plan.years) != 1:
        return None
    year = plan.years[0]
    conn = init_db()
    docs = conn.execute(
        "SELECT id FROM documents WHERE title LIKE ?",
        (f"%{year}年综合评价招生简章%",),
    ).fetchall()
    provinces: set[str] = set()
    has_final = False
    for doc in docs:
        parts = conn.execute(
            "SELECT text FROM retrieval_units WHERE document_id=? ORDER BY id",
            (doc["id"],),
        ).fetchall()
        blob = re.sub(r"\s+", "", "".join(part["text"] or "" for part in parts))
        provinces.update(re.findall(r"在([\u4e00-\u9fff]{2,3})省开展", blob))
        if "最终以省级招生主管部门核定为准" in blob:
            has_final = True
    conn.close()
    if provinces != {"江苏"} or not has_final:
        return None
    return f"在江苏省开展，计划合计{total}人，最终以省级招生主管部门核定为准"


def plan_exact_size(plan: QueryPlan, size: int) -> dict[str, Any]:
    """问「哪些专业的计划为 N 人」时，只列出计划数恰好是 N 的专业。"""
    result = filter_plans(plan)
    if result.get("error"):
        return result
    names: list[str] = []
    for row in result.get("matched") or []:
        if int(row.get("plan") or 0) != size:
            continue
        name = re.sub(r"\s+", "", str(row.get("major") or ""))
        if name and name not in names:
            names.append(name)
    if not names:
        return {"op": "filter_sum_plans", "error": "没有这个计划数", "filters": result.get("filters")}
    picked = dict(result)
    picked["headline"] = "、".join(names)
    picked["matched"] = [row for row in result.get("matched") or [] if int(row.get("plan") or 0) == size]
    picked["count"] = len(picked["matched"])
    picked["sum"] = size * len(names)
    return picked


def plan_share(plan: QueryPlan) -> dict[str, Any]:
    """某一首选科目的计划占该类型全部计划的比例。分子分母都来自同一张计划表。"""
    part = filter_plans(plan)
    whole_plan = copy.copy(plan)
    whole_plan.first_subject = None
    whole_plan.second_subjects = []
    whole_plan.literal_subject = False
    whole_plan.literal_target = None
    whole = filter_plans(whole_plan)
    if part.get("error") or whole.get("error") or not whole.get("sum"):
        return {"op": "filter_sum_plans", "error": "计划比例缺少分子或分母", "filters": part.get("filters")}
    pct = _q(_dec(part["sum"]) / _dec(whole["sum"]) * 100, 2)
    shown = _show(pct)
    shared = dict(part)
    shared["headline"] = f"{shown}%"
    shared["formula"] = f"{part['sum']}/{whole['sum']}={shown}%"
    return shared


def _study_years(question: str) -> int:
    """学年数看问句本身。“不要按四年”和“有人按四年”里的年数不算。"""
    focus = focus_text(question)
    for word, years in (
        ("六年", 6), ("五年", 5), ("四年", 4), ("4年", 4),
        ("三年", 3), ("3年", 3), ("两年", 2), ("二年", 2), ("2年", 2),
        ("一年", 1), ("1年", 1),
    ):
        if word in focus:
            return years
    return 1


def fee_prepaid(plan: QueryPlan) -> dict[str, Any]:
    conn = init_db()
    year = plan.years[0] if plan.years else None
    q = "SELECT * FROM fee_rules"
    params: list[Any] = []
    if year:
        q += " WHERE year=? OR year IS NULL"
        params.append(year)
    rules = [dict(r) for r in conn.execute(q, params)]
    # prefer matching year
    if year:
        year_rules = [r for r in rules if r.get("year") == year]
        if year_rules:
            rules = year_rules

    def pick(item: str, category: str | None = None) -> dict | None:
        cands = [r for r in rules if r["item"] == item]
        if category:
            cands = [r for r in cands if r.get("category") == category] or cands
        return cands[0] if cands else None

    focus = focus_text(plan.question)
    # 「文、理、工、艺术各是多少」要报每一类基准，不能只留下最后一类。
    if "各" in plan.question and all(k in plan.question for k in ("文", "理", "工", "艺术")):
        listed = []
        for name in ("文科", "理科", "工科", "艺术"):
            rule = pick("tuition", name)
            if not rule:
                continue
            amount = int(_dec(rule["amount"]))
            listed.append(f"{name}{amount}元")
        if len(listed) >= 3:
            docs = []
            for name in ("文科", "理科", "工科", "艺术"):
                rule = pick("tuition", name)
                if not rule:
                    continue
                d = conn.execute("SELECT id,url,title FROM documents WHERE id=?", (rule["document_id"],)).fetchone()
                if d and dict(d) not in docs:
                    docs.append(dict(d))
            conn.close()
            return {
                "op": "fee_prepaid",
                "headline": "、".join(listed),
                "formula": "各类学费基准",
                "documents": docs,
                "sum": None,
            }
    category = None
    for name in ("工科", "理科", "文科", "艺术"):
        if name in focus:
            category = name
            break
    tuition_rule = pick("tuition", category)
    housing_rule = pick("housing")
    uplift_rule = pick("uplift")
    if not tuition_rule:
        conn.close()
        return {"op": "fee", "error": "未找到对应类别学费规则", "rules": rules}

    tuition = _dec(tuition_rule["amount"])
    used = [tuition_rule]
    formula_parts = [f"{category or tuition_rule['category']}学费{tuition}"]
    want_uplift = ("不上浮" not in plan.question) and ("优势" in plan.question or "上浮" in plan.question)
    uplift_ratio = None
    if want_uplift:
        uplift_ratio = _dec(uplift_rule["uplift_ratio"]) if uplift_rule and uplift_rule.get("uplift_ratio") else _dec("0.10")
        tuition = tuition * (1 + uplift_ratio)
        formula_parts.append(f"上浮{uplift_ratio}*100%后={tuition}")
        if uplift_rule:
            used.append(uplift_rule)
    housing = _dec(0)
    exclude_housing = any(k in plan.question for k in ("不含宿", "不加住宿", "不要加住宿")) or bool(
        re.search(r"不(算|含|加)住宿", plan.question)
    )
    include_housing = "住宿" in plan.question and not exclude_housing
    if include_housing:
        if not housing_rule:
            conn.close()
            return {"op": "fee", "error": "题目需要住宿费但未找到住宿费规则", "rules": rules}
        housing = _dec(housing_rule["amount"])
        used.append(housing_rule)
        formula_parts.append(f"住宿费{housing}")
    only_housing = bool(re.search(r"只(统计|问|算|收)住宿", plan.question)) or bool(
        re.search(r"不要算学费|不算学费|不含学费|不包括学费", plan.question)
    )
    if only_housing:
        tuition = _dec(0)
        formula_parts.append("题目要求不算学费")
    annual = tuition + housing
    years_n = _study_years(plan.question)
    total = annual * years_n
    formula_parts.append(f"学年数{years_n}")
    formula_parts.append(f"合计=({tuition}+{housing})*{years_n}={total}")
    housing_only = include_housing and "学费" not in plan.question and "一共" not in plan.question and not only_housing
    asks_ratio = "百分之多少" in plan.question and "多少元" not in plan.question
    asks_delta = "较" in plan.question and "多多少" in plan.question and uplift_ratio is not None and "一共" not in plan.question
    if asks_delta:
        base = _dec(tuition_rule["amount"])
        diff = tuition - base
        shown = int(diff) if diff == int(diff) else diff
        headline = f"{shown}元"
    elif asks_ratio and uplift_ratio is not None:
        headline = f"{int(uplift_ratio * 100)}"
    elif only_housing:
        shown = int(total) if total == int(total) else total
        headline = f"{shown}元"
    elif housing_only:
        headline = f"{int(housing)}元"
    else:
        shown = int(total) if total == int(total) else total
        headline = f"{shown}元"
    settlement = ""
    if housing_rule and any(k in plan.question for k in ("结算", "多退少补")):
        settlement = housing_rule.get("notes") or "据实结算、多退少补"

    docs = []
    for r in used:
        d = conn.execute("SELECT id,url,title FROM documents WHERE id=?", (r["document_id"],)).fetchone()
        if d:
            docs.append(dict(d))
    conn.close()
    return {
        "op": "fee_prepaid",
        "tuition": float(tuition),
        "housing": float(housing),
        "years": years_n,
        "annual": float(annual),
        "sum": float(total),
        "headline": headline,
        "settlement": settlement,
        "formula": "；".join(str(x) for x in formula_parts),
        "documents": docs,
        "used_rules": used,
    }


FACT_HINTS = (
    ("预留计划", "reserve_plan_pct"),
    ("入围", "zhpj_shortlist_multiple"),
    ("多少倍", "zhpj_shortlist_multiple"),
    ("几倍", "zhpj_shortlist_multiple"),
    ("语种", "english_major_language"),
    ("核定", "plan_final_authority"),
    ("仙林", "xianlin_address"),
    ("锁金", "suojin_address"),
    ("咨询电话", "zs_phone"),
    ("电话", "zs_phone"),
    ("转专业", "special_no_transfer"),
    ("综合成绩", "zhpj_score_formula"),
    ("推免名额", "tuimian_quota"),
    ("推免计划", "tuimian_quota"),
    ("推免", "tuimian_policy"),
)


def lookup_fact(plan: QueryPlan) -> dict[str, Any]:
    metric = None
    for hint, name in FACT_HINTS:
        if hint not in plan.question:
            continue
        # 「转专业」三个字也会出现在办法的修订日期、流程里，不能一律答成特殊招生限制。
        if name == "special_no_transfer" and not any(
            k in plan.question for k in ("能否", "能不能", "可不可以", "可以转", "不得", "特殊招生")
        ):
            continue
        # 这条事实只有办法名称。问对象、排名、条件时要回到原文条款。
        if name == "reserve_plan_pct" and any(k in plan.question for k in ("用途", "用于")):
            continue
        # 「入围后达到什么控制线」不是在问入围倍数。倍数题里自带「倍」。
        if name == "zhpj_shortlist_multiple" and "倍" not in plan.question:
            continue
        if name in {"xianlin_address", "suojin_address"} and not any(
            k in plan.question for k in ("地址", "在哪", "哪里", "哪儿", "门牌", "邮编")
        ):
            continue
        if name == "tuimian_policy" and any(
            k in plan.question
            for k in (
                "对象", "排名", "条件", "哪些", "不含", "以内", "英语", "外语", "取消", "一次",
                "微专业", "学分", "年级", "竞赛", "论文", "实验班", "专升本",
                "月份", "每年", "何时", "时间", "几月",
                # 事实库只有政策名称。次数、区别、后果都在原文条款里。
                "几次", "机会", "区别", "怎样", "影响", "吗",
            )
        ):
            continue
        metric = name
        break
    if not metric:
        return {"op": "lookup", "error": "未识别可查询的结构化事实"}
    conn = init_db()
    row = conn.execute(
        "SELECT * FROM facts WHERE metric=? ORDER BY year DESC LIMIT 1", (metric,)
    ).fetchone()
    docs = []
    if row:
        d = conn.execute(
            "SELECT id,url,title FROM documents WHERE id=?", (row["source_document_id"],)
        ).fetchone()
        if d:
            docs.append(dict(d))
    conn.close()
    if not row:
        return {"op": "lookup", "error": f"未找到事实 {metric}"}
    return {
        "op": "fact_lookup",
        "metric": metric,
        "value": row["value"],
        "unit": row["unit"] or "",
        "sum": row["numeric"],
        "formula": f"{metric}={row['value']}{row['unit'] or ''}",
        "documents": docs,
    }


def lookup_test_fee(plan: QueryPlan) -> dict[str, Any]:
    conn = init_db()
    year = plan.years[0] if plan.years else None
    q = "SELECT * FROM fee_rules WHERE item='test_fee'"
    params: list[Any] = []
    if year:
        q += " AND (year=? OR year IS NULL)"
        params.append(year)
    row = conn.execute(q, params).fetchone()
    docs = []
    if row:
        d = conn.execute("SELECT id,url,title FROM documents WHERE id=?", (row["document_id"],)).fetchone()
        if d:
            docs.append(dict(d))
    conn.close()
    if not row:
        return {"op": "lookup", "error": "未找到综合评价测试费"}
    return {
        "op": "lookup",
        "item": "test_fee",
        "sum": float(row["amount"]),
        "formula": row["notes"],
        "documents": docs,
    }


def teacher_ratio(plan: QueryPlan) -> dict[str, Any]:
    conn = init_db()
    ay = plan.academic_year
    facts = [dict(r) for r in conn.execute("SELECT * FROM facts WHERE metric IN ('fulltime','adjunct','equivalent','students','ratio')")]
    if ay:
        ay_docs = {f.get("source_document_id") for f in facts if f.get("academic_year") == ay}
        scoped = [
            f
            for f in facts
            if f.get("academic_year") == ay or (ay_docs and f.get("source_document_id") in ay_docs)
        ]
        if scoped:
            facts = scoped
    elif plan.years:
        y = str(plan.years[0])
        scoped = [f for f in facts if (f.get("academic_year") or "").startswith(y) or str(f.get("year")) == y]
        if scoped:
            facts = scoped

    # prefer a consistent triple from the same document
    by_doc: dict[int, dict[str, dict]] = {}
    for f in facts:
        did = f.get("source_document_id") or 0
        by_doc.setdefault(did, {})[f["metric"]] = f
    complete = [
        bundle
        for bundle in by_doc.values()
        if {"fulltime", "adjunct", "equivalent"} <= set(bundle)
    ]
    by: dict[str, list[dict]] = {}
    for f in facts:
        by.setdefault(f["metric"], []).append(f)

    def one(metric: str) -> dict | None:
        if complete:
            return complete[0].get(metric)
        vals = by.get(metric) or []
        return vals[0] if vals else None

    ft, adj, eq, stu, ratio = one("fulltime"), one("adjunct"), one("equivalent"), one("students"), one("ratio")
    if not ft or not adj or not eq:
        conn.close()
        return {"op": "teacher", "error": "专任/外聘/折合教师人数不完整", "facts": facts}

    fulltime = _dec(ft["numeric"])
    adjunct = _dec(adj["numeric"])
    equivalent = _dec(eq["numeric"])
    if adjunct == 0:
        conn.close()
        return {"op": "teacher", "error": "外聘教师为0，无法反推折算系数", "facts": facts}
    coef = (equivalent - fulltime) / adjunct
    out: dict[str, Any] = {
        "op": "teacher_ratio",
        "fulltime": float(fulltime),
        "adjunct": float(adjunct),
        "equivalent": float(equivalent),
        "coef": float(_q(coef, 4)),
        "formula": f"折合={fulltime}+外聘×k => k=({equivalent}-{fulltime})/{adjunct}={_q(coef,4)}",
        "facts_used": [ft, adj, eq],
    }
    if stu:
        students = _dec(stu["numeric"])
        recomputed = students / equivalent
        out["students"] = float(students)
        out["ratio"] = float(_q(recomputed, 2))
        out["ratio_exact"] = float(recomputed)
        out["formula"] += f"；生师比={students}/{equivalent}={_q(recomputed,2)}"
        out["facts_used"].append(stu)
        if ratio:
            out["reported_ratio"] = float(ratio["numeric"])
            out["facts_used"].append(ratio)
    foreign = None
    blob = " ".join(
        r["text"] or ""
        for r in conn.execute(
            """SELECT b.text FROM blocks b JOIN documents d ON d.id=b.document_id
               WHERE d.title LIKE '%教学质量%'"""
        )
    )
    found = re.search(r"外籍专任教师\s*(\d+(?:\.\d+)?)", re.sub(r"[ \t]+", "", blob))
    if found:
        foreign = found.group(1)
    out["foreign"] = foreign
    q = focus_text(plan.question)
    asked = q[: q.rfind("多少")] if "多少" in q else q
    tail = asked[-12:]
    if any(k in tail for k in ("折算系数", "反推")) or any(k in q for k in ("折算系数", "反推")) and "生师比" not in tail:
        headline = _show(out["coef"])
    elif "生师比" in tail:
        headline = _show(out.get("ratio"))
    elif "折合在校生" in tail or ("折合在校生" in q and "生师比" not in q):
        headline = _show(out.get("students"))
    elif "生师比" in q:
        headline = _show(out.get("ratio"))
    elif "折合教师" in q:
        headline = _show(out["equivalent"])
    elif "外籍" in q:
        headline = foreign or ""
    elif "外聘" in q:
        headline = _show(out["adjunct"])
    elif "专任" in q:
        headline = _show(out["fulltime"])
    else:
        headline = _show(out["coef"])
    out["headline"] = headline
    docs = []
    for f in out["facts_used"]:
        d = conn.execute("SELECT id,url,title FROM documents WHERE id=?", (f["source_document_id"],)).fetchone()
        if d:
            docs.append(dict(d))
    out["documents"] = docs
    # conflict check
    if len({(f["metric"], f["numeric"]) for f in facts}) < len(facts):
        out["conflict"] = "同一指标存在多个数值，已优先使用匹配学年的第一条，请核对立档附件。"
    conn.close()
    return out


def lab_share(plan: QueryPlan) -> dict[str, Any]:
    conn = init_db()
    labs = [dict(r) for r in conn.execute("SELECT * FROM lab_centers")]
    if plan.academic_year:
        scoped = [r for r in labs if r.get("academic_year") == plan.academic_year]
        if scoped:
            labs = scoped
    uniq, seen = [], set()
    for r in labs:
        key = (r.get("name"), r.get("area_m2"))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    labs = uniq
    def core(name: str) -> str:
        return name.replace("实验教学中心", "").replace("实验室", "").replace("实验教学部", "")

    hits = []
    for r in labs:
        name = r.get("name") or ""
        c = core(name)
        abbrev = False
        if c and len(c) >= 4:
            for size in range(min(len(c), 8), 3, -1):
                if c[:size] in plan.question:
                    abbrev = True
                    break
        short_ok = bool(c) and len(c) >= 3 and c in plan.question
        if name and (name in plan.question or short_ok or abbrev):
            hits.append(r)
    hits.sort(key=lambda r: len(r.get("name") or ""), reverse=True)
    matched = []
    for r in hits:
        name = r.get("name") or ""
        c = core(name)
        if any(c and c != core(k.get("name") or "") and c in core(k.get("name") or "") for k in matched):
            continue
        if any(name and name != (k.get("name") or "") and name in (k.get("name") or "") for k in matched):
            continue
        matched.append(r)
    asks_table = any(k in plan.question for k in ("多少行", "行数", "列出多少", "多少个", "建有", "总面积", "最大", "最小"))
    if not matched and not asks_table:
        conn.close()
        return {"op": "lab", "error": "未能从问题中定位实验中心名称", "labs": labs}

    part = sum(_dec(r["area_m2"]) for r in matched)
    total_rows = labs
    total = sum(_dec(r["area_m2"]) for r in total_rows)
    # prefer document-stated 合计 if present in facts? use sum of table
    if total == 0:
        conn.close()
        return {"op": "lab", "error": "表内总面积为0", "matched": matched}
    share = part / total if total else Decimal(0)
    q = focus_text(plan.question)
    asked_at = q.find("多少")
    tail = q[max(0, asked_at - 16) : asked_at] if asked_at >= 0 else q[-16:]
    built = None
    prose = " ".join(
        r["text"] or ""
        for r in conn.execute(
            """SELECT b.text FROM blocks b JOIN documents d ON d.id=b.document_id
               WHERE d.title LIKE '%教学质量%'"""
        )
    )
    built_m = re.search(r"建有\s*(\d+)\s*个实验教学", re.sub(r"[ \t]+", "", prose))
    if built_m:
        built = built_m.group(1)
    if "百分之" in q or ("占" in q and "面积" in q):
        headline = _show(_q(share * 100, 2))
    elif "最小" in tail and labs:
        headline = _show(min(_dec(r["area_m2"]) for r in labs))
    elif "最大" in tail and labs:
        headline = _show(max(_dec(r["area_m2"]) for r in labs))
    elif "多少行" in q or "行数" in tail or "列出多少" in q:
        headline = str(len(labs))
    elif "建有" in tail or ("多少个" in tail and "实验" in q):
        headline = built or str(len(labs))
    elif "总面积" in q and len(matched) != 1:
        headline = _show(total)
    elif len(matched) > 1:
        headline = _show(part)
    elif matched:
        headline = _show(matched[0]["area_m2"])
    else:
        headline = _show(total)
    docs = []
    for r in matched:
        d = conn.execute("SELECT id,url,title FROM documents WHERE id=?", (r["document_id"],)).fetchone()
        if d:
            docs.append(dict(d))
    conn.close()
    return {
        "op": "lab_share",
        "matched": matched,
        "part": float(part),
        "total": float(total),
        "ratio": float(share),
        "percent": float(_q(share * 100, 2)),
        "headline": headline,
        "built": built,
        "formula": f"{'+'.join(str(r['area_m2']) for r in matched)}={part}；{part}/{total}={_q(share*100,2)}%",
        "documents": docs,
        "n_centers_in_table": len(labs),
    }


def execute(plan: QueryPlan) -> dict[str, Any] | None:
    q = plan.question
    result: dict[str, Any] | None = None
    wants_lab = (
        "实验教学" in q
        or "面积表" in q
        or ("实验室" in q and ("面积" in q or "多少" in q))
        or ("面积" in q and any(k in q for k in ("最大", "最小", "平方米", "实验中心", "行数", "相加")))
    )
    asks_plan = any(
        k in q
        for k in ("多少人", "计划数", "可报", "能报", "合计多少", "有哪些", "多少个", "列出了多少", "相加", "单元格")
    )
    asks_share = (
        bool(plan.first_subject)
        and bool(plan.admission_type)
        and "计划" in q
        and any(k in q for k in ("比例", "占比"))
        and "倍" not in q
    )
    has_plan = bool(
        plan.admission_type
        or plan.province
        or plan.group_code
        or plan.majors
        or plan.doc_scope
        or plan.major_contains
        or plan.major_exact
    )
    blocked = any(k in q for k in ("测试费", "入围", "几倍", "语种", "预留", "学费", "住宿费"))
    exact_size = re.search(r"计划(?:仅为|为)\s*(\d+)\s*人", q)
    if wants_lab and "招生计划" not in q and "多少人" not in q:
        result = lab_share(plan)
    elif exact_size and "哪些" in q and has_plan and not blocked:
        result = plan_exact_size(plan, int(exact_size.group(1)))
    elif asks_share and has_plan and not blocked:
        result = plan_share(plan)
    elif asks_plan and has_plan and not blocked:
        result = filter_plans(plan)
        if result and not result.get("error") and result.get("sum") is not None:
            head = _province_headline(plan, int(result["sum"]))
            if head:
                result["headline"] = head
    elif "测试费" in q:
        result = lookup_test_fee(plan)
    elif any(
        k in q
        for k in (
            "预留计划",
            "入围",
            "多少倍",
            "几倍",
            "语种",
            "核定为准",
            "仙林校区",
            "锁金村",
            "咨询电话",
            "转专业",
            "综合成绩",
            "推免",
        )
    ):
        result = lookup_fact(plan)
    elif (
        ("学费" in q or "住宿" in q or "上浮" in q or plan.topic == "fee")
        and "怎么" not in q
        and "如何" not in q
        and "怎样" not in q
        and "能否" not in q
        and "是否" not in q
        and ("比例" not in q or "上浮" in q)
        and "奖学金" not in q
        and "助学金" not in q
    ):
        result = fee_prepaid(plan)
    elif not (
        "截至" in q and any(k in q for k in ("教职工", "专任教师", "导师", "留学生"))
    ) and "正高" not in q and "副高" not in q and "中级" not in q and (
        plan.topic == "teacher" or any(k in q for k in ("生师比", "折合教师", "外聘", "专任教师", "外籍"))
    ):
        result = teacher_ratio(plan)
    elif (
        "实验教学" in q
        or "面积表" in q
        or ("实验室" in q and ("面积" in q or "多少" in q))
        or ("面积" in q and any(k in q for k in ("最大", "最小", "平方米", "实验中心", "行数")))
    ):
        result = lab_share(plan)

    if result and result.get("error"):
        # 算不出时把原文交回去。实验室面积题算成功时不会走到这里。
        if result.get("op") in {"lookup", "fact_lookup", "lab"}:
            return None
        # 教学质量报告里的招生人数不在计划表里。计划表本身对不上时仍要拒答，避免拿别的年份充数。
        if result.get("op") == "filter_sum_plans" and "教学质量报告" in plan.question:
            return None
    if result:
        conn = init_db()
        conn.execute(
            """INSERT INTO calculation_traces(question, op, inputs_json, formula, result, numeric, unit, created_at)
               VALUES(?,?,?,?,?,?,?,?)""",
            (
                plan.question,
                result.get("op"),
                json.dumps({k: result.get(k) for k in ("filters", "used_rules") if k in result}, ensure_ascii=False, default=str),
                result.get("formula"),
                json.dumps(result, ensure_ascii=False, default=str)[:8000],
                result.get("sum") or result.get("percent") or result.get("ratio") or result.get("coef"),
                None,
                utc_now(),
            ),
        )
        conn.commit()
        conn.close()
    return result


def complete_postal(question: str, value: str) -> str | None:
    """问句要邮编、事实只有门牌时，从原文里取紧挨着这扇门牌的邮编。

    同一门牌对应多个邮编就放弃。返回的是原文切片，不是拼出来的句子。
    """
    if "邮编" not in (question or "") or not value:
        return None
    campuses = [c for c in ("锁金村", "三牌楼", "仙林") if c in question]
    if len(campuses) != 1:
        return None
    campus = campuses[0]
    # 从长到短试门牌。「玄武区龙蟠路177号」对不上原文，退到「龙蟠路177号」。
    # 「仙林大学城文苑路9号」能唯一对上邮编时，不要再缩短成「文苑路9号」。
    compact_value = re.sub(r"\s+", "", value)
    tails = [compact_value]
    cut = re.search(r".*[省市区县](.+?\d+号)$", compact_value)
    if cut and cut.group(1) not in tails:
        tails.append(cut.group(1))
    short = re.search(r"([\u4e00-\u9fff]{2,3}?[路街道巷]\d+号)$", compact_value)
    if short and short.group(1) not in tails:
        tails.append(short.group(1))
    conn = init_db()
    try:
        for street in tails:
            if not re.search(r"\d+号$", street):
                continue
            rows = conn.execute(
                """SELECT text FROM retrieval_units
                   WHERE replace(replace(text, ' ', ''), '　', '') LIKE ?
                     AND text LIKE '%邮编%'""",
                (f"%{street}%",),
            ).fetchall()
            counts: dict[str, int] = {}
            quotes: dict[str, str] = {}
            for (text,) in rows:
                compact = re.sub(r"\s+", "", text or "")
                for m in re.finditer(re.escape(street) + r"邮编[:：]?(\d{6})", compact):
                    zip_code = m.group(1)
                    counts[zip_code] = counts.get(zip_code, 0) + 1
                    label = campus + "校区地址"
                    start = compact.rfind(label, max(0, m.start() - 24), m.start() + 1)
                    quote = compact[start if start >= 0 else m.start() : m.end()]
                    prev = quotes.get(zip_code)
                    if prev is None or (label in quote and label not in prev) or len(quote) < len(prev):
                        if label in quote or prev is None:
                            quotes[zip_code] = quote
            if not counts:
                continue
            ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            zip_code, n = ranked[0]
            second = ranked[1][1] if len(ranked) > 1 else 0
            if n < 5 or n < second * 10:
                continue
            return quotes.get(zip_code)
    finally:
        conn.close()
    return None


def format_computed(result: dict[str, Any]) -> str | None:
    if not result or result.get("error"):
        return None
    op = result.get("op")
    if op == "filter_sum_plans":
        names = "、".join(f"{r['major']}{r['plan']}" for r in result.get("matched") or [])
        filt = result.get("filters") or {}
        head = result.get("headline") or f"{result.get('sum')}人"
        return (
            f"{head}。"
            f"筛选条件：年份={filt.get('year')}，类型={filt.get('admission_type')}，"
            f"省份={filt.get('province')}，首选={filt.get('first_subject')}，再选={filt.get('second_subjects')}，"
            f"选科原文={filt.get('literal')}。"
            f"命中{result.get('count')}行：{names}。"
            f"计算：{result.get('formula')}。"
        )
    if op == "fact_lookup":
        value = str(result.get("value") or "")
        unit = result.get("unit") or ""
        metric = str(result.get("metric") or "")
        if (metric.endswith("pct") or unit == "%") and "%" not in value:
            value = f"{value}%"
        return f"{value}。{result.get('formula')}。"
    if op in {"fee_prepaid", "lookup"}:
        head = result.get("headline")
        if not head:
            val = result.get("sum")
            head = f"{int(val) if val is not None and float(val) == int(val) else val}元"
        settlement = result.get("settlement") or ""
        return f"{head}。{settlement}{result.get('formula')}。"
    if op == "teacher_ratio":
        head = result.get("headline")
        if not head:
            return None
        return (
            f"{head}。"
            f"专任教师{_show(result.get('fulltime'))}人，外聘教师{_show(result.get('adjunct'))}人，"
            f"折合教师{_show(result.get('equivalent'))}，折合在校生{_show(result.get('students'))}，"
            f"生师比{_show(result.get('ratio'))}，外聘折算系数{_show(result.get('coef'))}。"
            f"外籍专任教师{result.get('foreign') or '未写明'}。"
            f"{result.get('formula')}。"
        )
    if op == "lab_share":
        detail = "、".join(f"{r['name']}{r['area_m2']}平方米" for r in result.get("matched") or [])
        head = result.get("headline")
        return (
            f"{head}。"
            f"{detail}。表内{result.get('n_centers_in_table')}行，总面积{result.get('total')}平方米。"
            f"正文建有{result.get('built') or '未写明'}个实验教学中心。"
            f"{result.get('formula')}。"
        )
    return None
