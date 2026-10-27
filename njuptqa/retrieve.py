from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .db import init_db
from .util import extract_academic_year, extract_years, focus_text, in_focus, normalize_ws

SUBJECTS = ("物理", "历史", "化学", "生物", "政治", "地理", "技术")
PROVINCES = (
    "北京", "天津", "河北", "山西", "内蒙古", "辽宁", "吉林", "黑龙江", "上海", "浙江",
    "安徽", "福建", "江西", "山东", "河南", "湖北", "湖南", "广东", "广西", "海南",
    "重庆", "四川", "贵州", "云南", "西藏", "陕西", "甘肃", "青海", "宁夏", "新疆", "江苏",
    "新疆定向", "青海民族班",
)


@dataclass
class QueryPlan:
    question: str
    years: list[int] = field(default_factory=list)
    academic_year: str | None = None
    first_subject: str | None = None
    second_subjects: list[str] = field(default_factory=list)
    admission_type: str | None = None
    province: str | None = None
    majors: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    ops: list[str] = field(default_factory=list)
    topic: str = "general"
    need_complete_set: bool = False
    keywords: list[str] = field(default_factory=list)
    group_code: str | None = None
    literal_subject: bool = False
    literal_target: str | None = None
    answer_form: str = "cell"
    doc_scope: str | None = None
    major_contains: str | None = None
    major_exact: str | None = None


def plan_query(question: str) -> QueryPlan:
    q = normalize_ws(question)
    p = QueryPlan(question=q, years=extract_years(q), academic_year=extract_academic_year(q))
    focus = focus_text(q)
    if "国家专项" in focus:
        p.admission_type = "国家专项"
        p.topic = "admission"
    elif "地方专项" in focus:
        p.admission_type = "地方专项"
        p.topic = "admission"
    elif "艺术类" in focus:
        p.admission_type = "艺术类"
        p.topic = "admission"
    elif "综合评价" in focus:
        p.admission_type = "综合评价"
        p.topic = "admission"
    elif "省外" in focus and "计划" in q:
        p.admission_type = "省外"
        p.topic = "admission"
    elif "普通类" in focus or ("普通批次" in focus and "计划" in q):
        p.admission_type = "普通"
        p.topic = "admission"
    # “明细全部相加”是把已经选中的类型加总，不是把四类合并。
    if ("四类" in focus) or re.search(r"(?<!明细)全部相加", q):
        p.admission_type = None
        p.topic = "admission"
    focused_years = [y for y in p.years if str(y) in focus]
    if focused_years and len(focused_years) < len(p.years):
        p.years = focused_years
    if "四类" in focus and "招生计划" in q:
        p.doc_scope = "江苏省招生计划"
    elif "简章" in focus:
        p.doc_scope = "简章"
    elif "江苏省招生计划" in q:
        p.doc_scope = "江苏省招生计划"
    province_text = q.replace("江苏省外", "").replace("江苏省", "")
    quoted = re.findall(r"[“\"]([^”\"]+)[”\"]", q)
    for name in sorted(PROVINCES, key=len, reverse=True):
        # 只在干扰句里出现的省名不算。长名先匹配，避免“新疆定向”误伤“新疆”。
        if not in_focus(q, name):
            continue
        if name in quoted or (name in province_text and "计划" in q and name != "江苏"):
            p.province = name
            p.topic = "admission"
            break
    # “青海合计行”问的是青海；只有没点明省份、又在问合计行本身时，才取合计列。
    named_cell = None
    for name in sorted(PROVINCES, key=len, reverse=True):
        if name == "江苏":
            continue
        if re.search(rf"{re.escape(name)}的?合计", focus) or re.search(
            rf"这一行.{{0,3}}{re.escape(name)}", q
        ):
            named_cell = name
            break
    if named_cell:
        p.province = named_cell
    elif re.search(r"合计行.{0,8}多少", focus) and "这一行" not in q:
        p.province = "合计"
    if "合计" in quoted and "单元格" in q:
        p.province = "合计"
    gm = re.search(r"专业组\s*([0-9A-Za-z]{5,8})", q)
    if gm:
        p.group_code = gm.group(1).upper()
    p.literal_subject = any(k in q for k in ("原文", "严格", "不要把", "只统计"))
    for token in quoted:
        compact = token.replace("＋", "+").replace(" ", "")
        if re.fullmatch(r"(?:物理|历史)(?:\+(?:不限|化学|生物|政治|地理|技术))+", compact):
            p.literal_target = compact
            # “历史＋不限”组是这一格本身。只有问考生还能不能报时，才把不限和指定再选加在一起。
            if not any(k in q for k in ("考生", "可报", "可以报", "符合再选")):
                p.literal_subject = True
            break
    fm = re.search(r"名称中含[“\"]([^”\"]+)[”\"]", q)
    if fm:
        p.major_contains = fm.group(1).strip()
    if any(k in q for k in ("多少个专业", "列出了多少个")):
        p.answer_form = "count"
    elif "不同年份" in q or "各年" in q:
        p.answer_form = "by_year"
    elif "有哪些" in q and (p.admission_type or p.province or p.first_subject or "计划" in q or "专业" in q):
        p.answer_form = "list"
    elif any(k in q for k in ("不一致", "是否一致")):
        p.answer_form = "conflict"
    elif any(k in q for k in ("单元格", "这一行的")):
        p.answer_form = "official"
    elif any(k in q for k in ("明细相加", "不含合计行")):
        p.answer_form = "detail_sum"
    elif any(k in q for k in ("相加", "合计", "一共")):
        p.answer_form = "sum"
    if any(k in q for k in ("学费", "住宿费", "测试费")):
        p.topic = "fee"
    elif "招生章程" in q or "章程" in q:
        p.topic = "charter"
    elif any(k in q for k in ("生师比", "专任", "外聘", "折合教师", "折合在校生", "外籍")):
        p.topic = "teacher"
    elif any(k in q for k in ("实验教学中心", "实验室", "面积")):
        p.topic = "lab"
    elif any(k in q for k in ("转专业", "学籍", "毕业", "推免", "培养方案")) or (
        "学位" in q and "学位授权" not in q
    ):
        p.topic = "academic"
    elif any(k in q for k in ("奖学金", "助学金", "资助", "勤工", "奖助")):
        p.topic = "aid"
    elif any(k in q for k in ("就业质量", "就业率", "落实率", "毕业生")):
        p.topic = "employment"
    elif any(k in q for k in ("图书馆", "借阅", "续借", "开馆", "借书", "书库", "阅览")):
        p.topic = "library"
    elif any(k in q for k in ("校训", "校风", "校歌", "校庆", "校标", "校旗", "南邮精神", "前身", "双一流")):
        p.topic = "profile"

    try:
        conn = init_db()
        known = [
            r["major"]
            for r in conn.execute(
                """SELECT DISTINCT major FROM admission_plans
                   WHERE row_role='detail' AND major IS NOT NULL AND major!=''
                     AND major NOT IN ('合计','总计','小计','职称','所学专业')
                     AND length(major) BETWEEN 2 AND 80"""
            )
        ]
        conn.close()
        focus_compact = re.sub(r"\s+", "", focus)
        hits = [n for n in known if n and re.sub(r"\s+", "", n) in focus_compact and n not in SUBJECTS]
        hits.sort(key=len, reverse=True)
        kept: list[str] = []
        for n in hits:
            if not any(n in k for k in kept):
                kept.append(n)
        p.majors = kept
    except Exception:
        p.majors = []
    exact = re.search(r"(?:恰好是|就是)[「“\"]([^」”\"]+)[」”\"]", focus)
    written = re.search(r"原文写成[「“\"]([^」”\"]+)[」”\"]", focus)
    if exact:
        p.major_exact = re.sub(r"\s+", "", exact.group(1))
        p.majors = [p.major_exact]
    elif written:
        p.major_contains = re.sub(r"\s+", "", written.group(1))
        p.majors = []

    subject_q = re.sub(r"[「“\"][^」”\"]+[」”\"]", " ", focus)
    for name in sorted(p.majors, key=len, reverse=True):
        subject_q = subject_q.replace(name, " ")
    if p.major_contains:
        subject_q = subject_q.replace(p.major_contains, " ")
    for s in SUBJECTS:
        if s not in subject_q:
            continue
        if s in {"物理", "历史"} and re.search(rf"首选\s*{s}|{s}\s*类", subject_q):
            p.first_subject = s
        elif s in {"物理", "历史"} and not p.first_subject:
            p.first_subject = s
        elif s not in {"物理", "历史"}:
            p.second_subjects.append(s)
    # 停在“的”之前，避免专业名里的“技术/地理”被当成再选科目
    m = re.search(r"再选([^，。；的]+)", q)
    if m:
        p.second_subjects = [s for s in SUBJECTS if s in m.group(1) and s not in {"物理", "历史"}]
    if p.answer_form == "detail_sum" and ("合计" in quoted or "合计列" in q):
        p.province = "合计"
    if p.first_subject:
        for token in quoted:
            piece = token.replace("＋", "+").replace(" ", "")
            if piece in {"不限", "化学", "生物", "政治", "地理", "技术"} and not p.literal_target:
                p.literal_target = f"{p.first_subject}+{piece}"
                p.literal_subject = True
                break
    # 没有引号时，“物理加化学的计划”指这一格原文，不是“考生可以兼报不限”。
    if not p.literal_target and not any(k in q for k in ("考生", "可报", "可以报", "符合再选")):
        for pat, target in (
            (r"物理\s*\+\s*化学", "物理+化学"),
            (r"物理加化学", "物理+化学"),
            (r"物理且再选化学", "物理+化学"),
            (r"物理且再选不限", "物理+不限"),
            (r"物理\s*\+\s*不限", "物理+不限"),
            (r"历史且再选不限", "历史+不限"),
            (r"历史\s*\+\s*不限", "历史+不限"),
        ):
            found = re.search(pat, q)
            if not found:
                continue
            pre = q[max(0, found.start() - 8) : found.start()]
            if re.search(r"不要|不含|除了|不算|不把", pre):
                continue
            p.literal_target = target
            p.literal_subject = True
            break
    if p.answer_form == "official" and p.province is None and "合计" in q:
        p.province = "合计"

    if any(k in q for k in ("合计", "多少人", "之和", "占比", "反推", "复算", "预收额")):
        p.need_complete_set = True
    if any(k in q for k in ("合计", "之和", "加总")):
        p.ops.append("sum")
    if "占比" in q or "占" in q and "面积" in q:
        p.ops.append("ratio")
    if "反推" in q or "折算系数" in q:
        p.ops.append("solve_linear")
    if "复算" in q or "生师比" in q:
        p.ops.append("ratio")
    if "上浮" in q:
        p.ops.append("uplift")
    if re.search(r"四年|4年", q):
        p.ops.append("multi_year")

    p.keywords = content_keywords(q)
    return p


# 问句里的校名几乎每页都有，不能拿来检索。
_SCHOOL_NAMES = ("南京邮电大学", "南京邮电学院")
_ANCHORS = (
    "综合评价", "招生章程", "转专业", "学籍管理办法", "仙林校区", "锁金村", "三牌楼",
    "咨询电话", "综合成绩", "测试费", "文苑路", "龙蟠路", "新模范马路",
    "国家专项", "地方专项", "普通类", "博士后流动站", "南邮精神",
    "校训", "校风", "校歌", "校庆", "校标", "校旗",
)
# 整词等于这些才丢掉。不从专业名内部删。
_DROP_WORDS = {
    "学生", "学校", "情况", "问题", "方面", "内容", "工作", "相关", "规定", "要求",
    "我们", "可以", "能够", "需要", "目前", "现在", "分别", "一般", "通常", "具体",
    "主要", "其他", "已经", "如果", "但是", "因为", "所以", "其中", "以及", "进行",
    "什么", "哪些", "如何", "多少", "怎么", "怎样", "是否", "这个", "那个", "一个",
}
_SPLIT = (
    "有没有", "是什么", "有什么", "叫什么", "哪一类", "哪一个", "哪一年", "哪一天",
    "哪些", "何时", "哪里", "哪儿", "哪个", "什么", "如何", "怎么", "怎样", "多少",
    "几个", "是否", "属于", "具有", "截至", "目前", "现在", "分别", "通常", "一般",
    "请问", "根据", "按照", "对于", "关于", "通过", "进行", "可以", "能够", "以及",
    "或者", "还是", "并且", "如果", "但是", "其中",
)
_GENERIC_BIGRAM = {
    "南京", "京邮", "邮电", "电大", "大学", "学院", "学校", "南邮", "什么", "如何",
    "怎么", "多少", "目前", "属于", "一个", "这个", "那个", "哪些", "是否", "以及",
    "其中", "根据", "按照", "关于", "现在", "分别", "通常", "一般", "可以", "进行",
    "学生", "我们", "问题", "情况", "工作", "相关",
}


def content_keywords(question: str) -> list[str]:
    """取问句里真正要查的说法。

    不再按 8 个字硬切。硬切会把“校训”拆进两个词，再拆成“南京、大学”，
    录取分数表的标题就能占满检索结果。
    """
    raw = focus_text(question) or question
    text = raw
    for name in _SCHOOL_NAMES:
        text = text.replace(name, " ")
    text = re.sub(r"20\d{2}\s*年(?:\s*\d{1,2}\s*月)?(?:\s*\d{1,2}\s*日)?", " ", text)
    splitter = re.compile("|".join(re.escape(s) for s in _SPLIT) + r"|[^\u4e00-\u9fffA-Za-z0-9.+]+|[的了吗呢吧啊]")
    parts: list[str] = []
    for piece in splitter.split(text):
        piece = piece.strip()
        piece = re.sub(r"^(有|是|为|将|把|其|该|本|这|那|请|问)+", "", piece)
        piece = re.sub(r"^哪", "", piece)
        piece = re.sub(r"^个", "", piece)
        piece = re.sub(r"(会有|是|为|有)$", "", piece)
        if len(piece) >= 2 and piece not in _DROP_WORDS:
            parts.append(piece)
    anchors = [a for a in _ANCHORS if a in raw]
    full: list[str] = []
    for word in anchors + sorted(parts, key=len, reverse=True):
        if word in full or word in _DROP_WORDS:
            continue
        full.append(word)
    slices: list[str] = []
    for word in full:
        if len(word) <= 6:
            continue
        for i in range(0, len(word) - 3):
            piece = word[i : i + 4]
            if piece in full or piece in slices or piece in _DROP_WORDS:
                continue
            slices.append(piece)
            if len(slices) >= 8:
                break
        if len(slices) >= 8:
            break
    return (full + slices)[:12]


def _phrase_clause(keyword: str) -> str | None:
    han = "".join(ch for ch in keyword if "\u4e00" <= ch <= "\u9fff")
    grams: list[str] = []
    if len(han) >= 2:
        for i in range(len(han) - 1):
            gram = han[i : i + 2]
            if gram in _GENERIC_BIGRAM:
                continue
            grams.append(gram)
    grams.extend(tok.lower() for tok in re.findall(r"[A-Za-z0-9]{2,}", keyword))
    grams = grams[:6]
    if not grams:
        return None
    if len(grams) == 1:
        return grams[0]
    return "(" + " AND ".join(grams) + ")"


def _fts_query(keywords: list[str]) -> str:
    # 每个关键词内部必须连续出现，关键词之间用 OR。
    # 这样“博士后流动站”不会退化成“大学 OR 南京”。
    clauses = []
    for keyword in keywords:
        clause = _phrase_clause(keyword)
        if clause and clause not in clauses:
            clauses.append(clause)
        if len(clauses) >= 8:
            break
    return " OR ".join(clauses)


def _year_locked(plan: QueryPlan) -> bool:
    """“截至2026年9月”是口径日期，不是只准查 2026 年文件。招生、章程、届、版才锁定年份。"""
    if not plan.years:
        return False
    if plan.topic in {"admission", "fee", "employment", "charter"}:
        return True
    return any(k in plan.question for k in ("学年", "届", "版", "简章", "章程", "招生计划"))


def _asks_score(question: str) -> bool:
    return any(k in question for k in ("分数", "位次", "投档分", "最低分", "最高分"))


def _title_bonus(question: str, doc_title: str, unit_title: str) -> float:
    blob = f"{doc_title or ''} {unit_title or ''}"
    bonus = 0.0
    rules = (
        (("校训", "校风", "校庆", "校标", "校旗", "标识色", "域名"), ("学校章程", "章程")),
        (("校歌", "作词", "作曲"), ("校歌",)),
        (("南邮精神",), ("南邮精神", "学校简介")),
        (("被誉", "前身", "建设高校", "办学", "教学机构", "在籍", "博士后", "校区", "双一流", "摇篮", "并入", "更名", "流动站"), ("学校简介", "单位概况")),
        (("借阅", "续借", "开馆", "借书", "光盘", "书库", "阅览", "图书馆"), ("图书馆",)),
        (("重修", "缓考", "旷课", "免修", "学业预警", "选课", "补考"), ("选课", "手册", "学籍", "考试")),
        (("转专业", "推免"), ("转专业", "推免")),
        (("学士学位", "学位授予", "绩点"), ("学位",)),
    )
    for cues, titles in rules:
        if any(c in question for c in cues) and any(t in blob for t in titles):
            bonus += 6
    return bonus


_SELECT = """
    SELECT ru.id, ru.document_id, ru.unit_type, ru.title, ru.text, ru.locator,
           ru.year, ru.academic_year, d.url, d.title AS doc_title, d.org
    FROM retrieval_fts f
    JOIN retrieval_units ru ON ru.id = f.rowid
    JOIN documents d ON d.id = ru.document_id
    WHERE retrieval_fts MATCH ?
"""


def _score_sql(question: str) -> str:
    if _asks_score(question):
        return ""
    return (
        " AND instr(ifnull(d.title,''), '录取分数') = 0"
        " AND instr(ifnull(ru.title,''), '录取分数') = 0"
        " AND instr(substr(ifnull(ru.text,''), 1, 80), '录取分数统计') = 0"
    )


_INTRO_CUES = (
    "教职工", "专任教师", "导师", "一流课程", "规划教材", "重点实验室", "科技园",
    "校地", "校企", "留学生", "国际交流", "国际合作", "工程教育", "本科招生专业",
    "一流专业", "院士",
)


def _profile_rows(conn, question: str, keywords: list[str], topic: str) -> list[Any]:
    """概况类问题把《学校简介》《学校章程》放进候选，避免短新闻把它们挤出 bm25 前几名。"""
    introish = any(k in question for k in _INTRO_CUES) or "ESI" in question
    if topic in {"library", "admission", "fee", "employment", "lab", "aid"}:
        return []
    if topic in {"teacher", "academic"} and not introish:
        return []
    if any(k in question for k in ("实验室",)) and "重点实验室" not in question:
        return []
    want: list[str] = []
    if any(k in question for k in ("校训", "校风", "校庆", "校标", "校旗", "标识色", "域名", "邮编")):
        want.append("学校章程")
    if any(k in question for k in ("校歌", "作词", "作曲")):
        want.append("校歌")
    if any(k in question for k in ("南邮精神",)):
        want.extend(["南邮精神", "学校简介"])
    if introish or any(k in question for k in ("被誉", "前身", "建设高校", "办学", "教学机构", "在籍", "博士后", "校区", "双一流", "摇篮", "并入", "更名", "流动站", "学科", "地位", "定名", "第一所", "改建", "邮电学校")):
        want.extend(["学校简介", "单位概况"])
    if not want:
        return []
    keys = sorted((kw for kw in keywords if len(kw) >= 2), key=len, reverse=True)[:4]
    clauses = " OR ".join("d.title LIKE ?" for _ in want)
    params: list[Any] = [f"%{w}%" for w in want]
    sql = f"""
        SELECT ru.id, ru.document_id, ru.unit_type, ru.title, ru.text, ru.locator,
               ru.year, ru.academic_year, d.url, d.title AS doc_title, d.org
        FROM retrieval_units ru JOIN documents d ON d.id = ru.document_id
        WHERE ({clauses})
    """
    filtered = sql
    params_filtered = list(params)
    if keys:
        filtered += " AND (" + " OR ".join("ru.text LIKE ?" for _ in keys) + ")"
        params_filtered.extend(f"%{kw}%" for kw in keys)
    filtered += " LIMIT 8"
    rows = conn.execute(filtered, params_filtered).fetchall()
    if rows:
        return rows
    if introish or any(k in question for k in ("地位", "被誉", "前身", "定名", "第一所", "改建", "摇篮", "更名")):
        return conn.execute(
            sql + " AND d.title LIKE '%学校简介%' ORDER BY length(ru.text) DESC LIMIT 2",
            params,
        ).fetchall()
    return []


# 问句点名要读哪一份文件时，按标题把那一份找出来。长的说法放前面。
_NAMED_DOCS: tuple[tuple[str, str, str], ...] = (
    ("本科教学质量报告", "%本科教学质量报告%", ""),
    ("教学质量报告", "%本科教学质量报告%", ""),
    ("普通本科招生章程", "%普通本科招生章程%", ""),
    ("综合评价招生简章", "%综合评价招生简章%", ""),
    ("综合评价招生报考指南", "%综合评价招生报考指南%", ""),
    ("综评报考指南", "%综合评价招生报考指南%", ""),
    ("综评测试公告", "%综合评价招生测试公告%", ""),
    ("本科生学籍管理办法", "%本科生学籍管理办法%", ""),
    ("学籍管理办法", "%本科生学籍管理办法%", ""),
    ("缓考", "%缓考、补考和重修%", ""),
    ("推免管理办法", "%免试攻读研究生管理办法%", ""),
    ("小米奖助学金", "%小米奖助学金%评选办法%", ""),
    ("勤工助学", "%勤工助学管理办法%", ""),
    ("甘霖励志奖学金", "%甘霖励志奖学金%评选办法%", ""),
    ("共进电子奖学金", "%共进电子奖学金%", ""),
    ("华为奖学金", "%华为奖学金%评选办法%", ""),
    ("临时困难补助", "%临时困难补助%", ""),
    ("学杂费减免", "%学杂费减免%", ""),
    ("学士学位授予", "%学士学位授予办法%", ""),
    ("借阅规则", "%借阅规则%", "lib.njupt.edu.cn"),
    ("入馆须知", "%入馆须知%", "lib.njupt.edu.cn"),
    ("开放时间", "%开放时间%", "lib.njupt.edu.cn"),
    ("研读室", "%研读室%", "lib.njupt.edu.cn"),
    ("图书馆常见问题", "%常见问题%", "lib.njupt.edu.cn"),
    ("本馆简介", "%本馆简介%", ""),
    ("图书馆2026年3月简介", "%本馆简介%", ""),
    ("学校章程", "%学校章程%", "www.njupt.edu.cn"),
    ("学校简介", "%学校简介%", "www.njupt.edu.cn"),
    ("校标", "%校标校训%", ""),
    ("校庆标识", "%校庆标识%", ""),
    ("南邮精神", "%南邮精神%", ""),
    ("海外访学奖学金", "%海外访学%", ""),
)


def _named_doc_spec(question: str) -> tuple[str, str] | None:
    """问句写了「按某文件」，就返回该文件的标题条件和站点。"""
    if "按" not in question:
        return None
    for cue, pattern, host in _NAMED_DOCS:
        if cue in question:
            return pattern, host
    return None


def _named_doc_rows(conn, question: str, keywords: list[str]) -> list[Any]:
    spec = _named_doc_spec(question)
    if not spec:
        return []
    pattern, host = spec
    sql = """
        SELECT ru.id, ru.document_id, ru.unit_type, ru.title, ru.text, ru.locator,
               ru.year, ru.academic_year, d.url, d.title AS doc_title, d.org
        FROM retrieval_units ru JOIN documents d ON d.id = ru.document_id
        WHERE d.title LIKE ?
          AND d.title NOT LIKE '%公示%'
          AND d.title NOT LIKE '%名单%'
          AND d.title NOT LIKE '%临时%'
    """
    params: list[Any] = [pattern]
    if host:
        sql += " AND d.url LIKE ?"
        params.append(f"%{host}%")
    years = re.findall(r"20\d{2}", question)
    yearly = any(k in pattern for k in ("章程", "简章", "公告", "指南", "教学质量报告"))
    if len(years) == 1 and yearly:
        sql += " AND d.title LIKE ?"
        params.append(f"%{years[0]}%")
    base_sql, base_params = sql, list(params)
    keys = [kw for kw in _match_terms(keywords) if 2 <= len(kw) <= 8][:4]
    if keys:
        sql += " AND (" + " OR ".join("ru.text LIKE ?" for _ in keys) + ")"
        params.extend(f"%{kw}%" for kw in keys)
    rows = conn.execute(sql + " LIMIT 6", params).fetchall()
    if rows:
        return rows
    return conn.execute(
        base_sql + " ORDER BY length(ru.text) DESC LIMIT 2",
        base_params,
    ).fetchall()


def _anchor_rows(conn, question: str, keywords: list[str]) -> list[Any]:
    """图书馆简介和学生手册往往排不进 bm25 前列，按文档标题直接补进候选。"""
    titles: list[str] = []
    if any(k in question for k in ("图书馆", "借阅", "续借", "开馆", "借书", "书库", "阅览", "馆藏", "藏书")):
        titles.append("%本馆简介%")
    graduate = any(k in question for k in ("研究生", "硕士研究生", "博士研究生"))
    student = any(k in question for k in (
        "选课", "考勤", "请假", "旷课", "综合素质", "学籍", "注册", "推免", "微专业",
        "学业预警", "重修", "补考", "缓考", "转专业", "入学", "报到",
    ))
    if student and not graduate:
        titles.append("%学生手册（2026版）%")
    years = re.findall(r"20\d{2}", question)
    if len(years) == 1 and "章程" in question:
        titles.append(f"%{years[0]}年%章程%")
    if len(years) == 1 and "简章" in question:
        titles.append(f"%{years[0]}年%简章%")
    latest_charter = any(k in question for k in ("日常工作", "院校标识码", "预留计划", "投档成绩")) and "研究生" not in question
    if not titles and not latest_charter:
        return []
    keys = [kw for kw in keywords if 2 <= len(kw) <= 8][:4]
    clause = " OR ".join("d.title LIKE ?" for _ in titles)
    base = """
        SELECT ru.id, ru.document_id, ru.unit_type, ru.title, ru.text, ru.locator,
               ru.year, ru.academic_year, d.url, d.title AS doc_title, d.org
        FROM retrieval_units ru JOIN documents d ON d.id = ru.document_id
        WHERE (""" + clause + ")"
    params: list[Any] = list(titles)
    sql = base
    if keys:
        sql += " AND (" + " OR ".join("ru.text LIKE ?" for _ in keys) + ")"
        params.extend(f"%{kw}%" for kw in keys)
    rows = []
    if titles:
        rows = conn.execute(sql + " LIMIT 8", params).fetchall()
        if not rows and keys:
            rows = conn.execute(
                base + " ORDER BY length(ru.text) DESC LIMIT 2",
                titles,
            ).fetchall()
    if latest_charter:
        rows = list(rows) + list(conn.execute(
            """
            SELECT ru.id, ru.document_id, ru.unit_type, ru.title, ru.text, ru.locator,
                   ru.year, ru.academic_year, d.url, d.title AS doc_title, d.org
            FROM retrieval_units ru JOIN documents d ON d.id = ru.document_id
            WHERE d.title LIKE '%普通本科招生章程%'
            ORDER BY d.title DESC LIMIT 3
            """
        ).fetchall())
    return rows


def lexical_search(question: str, plan: QueryPlan, k: int = 12) -> list[dict[str, Any]]:
    conn = init_db()
    fts_q = _fts_query(plan.keywords)
    rows: list[Any] = []
    if fts_q:
        sql = _SELECT + _score_sql(question)
        params: list[Any] = [fts_q]
        if _year_locked(plan):
            sql += " AND (ru.year IS NULL OR ru.year IN ({}) OR ru.academic_year LIKE ?)".format(
                ",".join("?" * len(plan.years))
            )
            params.extend(plan.years)
            params.append(f"%{plan.years[0]}%")
        sql += " ORDER BY bm25(retrieval_fts) LIMIT ?"
        params.append(k * 6)
        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception:
            rows = []
    if not rows and plan.keywords:
        key = max(plan.keywords, key=len)
        score_sql = "" if _asks_score(question) else " AND ifnull(d.title,'') NOT LIKE '%录取分数%'"
        rows = conn.execute(
            f"""
            SELECT ru.id, ru.document_id, ru.unit_type, ru.title, ru.text, ru.locator,
                   ru.year, ru.academic_year, d.url, d.title AS doc_title, d.org
            FROM retrieval_units ru JOIN documents d ON d.id = ru.document_id
            WHERE (ru.text LIKE ? OR ru.title LIKE ?){score_sql}
            LIMIT ?
            """,
            (f"%{key}%", f"%{key}%", k * 3),
        ).fetchall()
    seen_ids = {r["id"] if isinstance(r, dict) else r[0] for r in rows}
    for extra in (
        list(_profile_rows(conn, question, plan.keywords, plan.topic))
        + list(_anchor_rows(conn, question, plan.keywords))
        + list(_named_doc_rows(conn, question, plan.keywords))
    ):
        eid = extra["id"] if isinstance(extra, dict) else extra[0]
        if eid not in seen_ids:
            rows.append(extra)
            seen_ids.add(eid)
    graduate = any(k in question for k in ("研究生", "硕士研究生", "博士研究生"))
    student_rule = any(k in question for k in (
        "选课", "考勤", "请假", "旷课", "综合素质", "学籍", "注册", "推免", "微专业",
        "学业预警", "重修", "补考", "缓考", "转专业", "入学", "报到",
    ))
    libraryish = any(k in question for k in ("图书馆", "借阅", "开放", "书库", "委托", "借还"))
    scored: list[dict[str, Any]] = []
    for r in rows:
        r = dict(r)
        text = r["text"] or ""
        score = 0.0
        for kw in plan.keywords:
            if kw and (kw in text or kw in (r["title"] or "")):
                score += 2 + min(len(kw), 8) / 4
        if _year_locked(plan) and r["year"] in plan.years:
            score += 3
        if plan.admission_type and plan.admission_type in (r["title"] or "") + text:
            score += 3
        if plan.topic == "charter" and "章程" in (r["doc_title"] or "") + (r["title"] or ""):
            score += 4
        if _year_locked(plan) and len(plan.years) == 1 and any(k in question for k in ("章程", "简章")):
            titled = [y for y in extract_years(r["doc_title"] or "") if 2000 <= y <= 2035]
            if titled and plan.years[0] not in titled:
                score -= 8
            elif str(plan.years[0]) in (r["doc_title"] or ""):
                score += 8
        if plan.topic == "profile" and any(k in (r["doc_title"] or "") for k in ("学校简介", "学校章程", "概况", "校史", "校歌", "南邮精神")):
            score += 8
        if any(k in (r["doc_title"] or "") for k in ("学校简介", "单位概况")) and (
            "ESI" in question
            or any(k in question for k in (
                "被誉", "前身", "建设高校", "办学", "教学机构", "在籍", "博士后", "几个校区", "三个校区",
                "双一流", "摇篮", "并入", "更名", "流动站", "地位", "定名", "第一所", "改建", *_INTRO_CUES,
            ))
        ):
            score += 12
        if "学校章程" in (r["doc_title"] or "") and any(k in question for k in ("校训", "校风", "校庆", "校标", "校旗", "标识色", "域名")):
            score += 12
        if any(k in plan.question for k in ("地址", "电话")) and any(
            k in text for k in ("文苑路", "85866668", "龙蟠路", "新模范")
        ):
            score += 4
        score += _title_bonus(question, r["doc_title"] or "", r["title"] or "")
        title_blob = (r["doc_title"] or "") + (r["title"] or "") + (r.get("url") or "")
        blob0 = (text or "")[:120]
        if ("录取分数" in title_blob or "录取分数统计" in blob0) and not _asks_score(question):
            continue
        if not libraryish and "锁金" in plan.question and "龙蟠路" not in text and "锁金村" not in text:
            continue
        if not libraryish and "仙林" in plan.question and "文苑路" not in text and "仙林大学城" not in text:
            continue
        if plan.topic == "academic" and any(k in (r["doc_title"] or "") for k in ("学籍", "转专业", "学位", "推免", "培养方案")):
            score += 5
        if plan.topic == "aid" and any(k in (r["doc_title"] or "") + text[:80] for k in ("奖学金", "助学金", "资助", "奖助")):
            score += 5
        if plan.topic == "employment" and any(k in (r["doc_title"] or "") for k in ("就业质量", "就业率", "就业")):
            score += 5
        named = _named_doc_spec(question)
        if named:
            pattern, host = named
            title = r["doc_title"] or ""
            if pattern.strip("%") in title and (not host or host in (r.get("url") or "")):
                if not any(k in title for k in ("公示", "名单", "临时")):
                    score += 22
        if plan.topic == "library" and "图书馆" in (r["doc_title"] or "") + (r["title"] or ""):
            score += 5
        if "本馆简介" in (r["doc_title"] or "") and plan.topic == "library":
            score += 14
        if student_rule and not graduate and "学生手册（2026版）" in (r["doc_title"] or ""):
            score += 12
        if student_rule and "工作方案" in (r["doc_title"] or "") and "方案" not in question:
            score -= 8
        if student_rule and not graduate and (
            "pg.njupt.edu.cn" in (r.get("url") or "") or "研究生" in (r["doc_title"] or "")
        ):
            score -= 8
        if (text or "").count("招生章程") >= 3 and "第一条" not in (text or "")[:500]:
            score -= 10
        charter_years = [y for y in extract_years(r["doc_title"] or "") if 2000 <= y <= 2035]
        if (
            charter_years
            and "招生章程" in (r["doc_title"] or "")
            and not plan.years
            and any(k in question for k in ("日常工作", "标识码", "预留", "投档", "章程"))
        ):
            score += (max(charter_years) - 2018) * 3
        if any(k in question for k in ("为主体", "为特色", "学科门类", "相互交融")) and "教学质量报告" not in question:
            if "招生章程" in (r["doc_title"] or ""):
                score += 9
            if "教学质量报告" in (r["doc_title"] or ""):
                score -= 4
        if "综合成绩" in plan.question and "综合成绩" in text:
            score += 5
        if r["unit_type"] in {"table", "table_row"} and plan.need_complete_set:
            score += 1.5
        scored.append({**dict(r), "score": score})
    # 问句已经点名文件时，只在那一份里面挑句子，避免同名新闻把正文挤掉。
    if _named_doc_spec(question):
        named_ids = {extra["document_id"] for extra in _named_doc_rows(conn, question, plan.keywords)}
        narrowed = [item for item in scored if item["document_id"] in named_ids]
        if narrowed:
            scored = narrowed
    scored.sort(key=lambda x: x["score"], reverse=True)
    # dedup near-identical
    out: list[dict[str, Any]] = []
    seen = set()
    for item in scored:
        key = (item["document_id"], (item["text"] or "")[:80])
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= k:
            break
    out = _promote_best_sentence(conn, out, plan.keywords)
    out = _focus_cited_passage(conn, out, question)
    conn.close()
    return out


_TITLE_ROWS: list[Any] | None = None
_DOC_CHARS: dict[int, int] | None = None
_BI_STOP = {
    "南京", "京邮", "邮电", "电大", "大学", "学院", "学校", "南邮", "什么", "如何",
    "怎么", "多少", "一个", "这个", "那个", "哪些", "是否", "以及", "其中", "根据",
    "按照", "关于", "现在", "分别", "可以", "进行", "我们", "工作", "相关", "规定",
    "要求", "情况", "问题", "内容", "方面", "主要", "具体", "已经", "如果", "但是",
}


def _flat(text: str) -> str:
    return re.sub(r"[\s“”\"「」『』]", "", text or "")


def _bigrams(text: str) -> list[str]:
    """只取真正相邻的两个汉字。跨过数字拼成的「年月」会把任何日期都配上简介。"""
    flat = _flat(text)
    grams = []
    for index in range(len(flat) - 1):
        left, right = flat[index], flat[index + 1]
        if not ("\u4e00" <= left <= "\u9fff" and "\u4e00" <= right <= "\u9fff"):
            continue
        gram = left + right
        if gram not in _BI_STOP:
            grams.append(gram)
    grams.extend(num for num in re.findall(r"\d+(?:\.\d+)?", flat) if len(num) >= 2)
    return grams


def _cite_phrase(question: str) -> str:
    """「按某份文件，……」里的文件名。不查表，后面对标题做重合。"""
    match = re.search(r"按([^，。？?]{2,48})", question)
    if not match:
        return ""
    phrase = match.group(1)
    for word in ("南京邮电大学", "南邮", "官方", "专题说明", "页面", "网页", "附件", "历史办法", "历史文本", "公开"):
        phrase = phrase.replace(word, "")
    return phrase.strip()


def _title_rows(conn) -> list[Any]:
    global _TITLE_ROWS
    if _TITLE_ROWS is None:
        _TITLE_ROWS = conn.execute("SELECT id, title, url FROM documents").fetchall()
    return _TITLE_ROWS


def _doc_chars(conn) -> dict[int, int]:
    global _DOC_CHARS
    if _DOC_CHARS is None:
        _DOC_CHARS = {
            int(row["document_id"]): int(row["n"] or 0)
            for row in conn.execute(
                "SELECT document_id, coalesce(sum(length(text)), 0) AS n FROM retrieval_units GROUP BY document_id"
            )
        }
    return _DOC_CHARS


_USE_RARE_ANCHOR = True


def _align_doc_ids(conn, question: str) -> list[int]:
    phrase = _cite_phrase(question)
    grams = list(dict.fromkeys(_bigrams(phrase)))
    if len(grams) < 2:
        return []
    years = re.findall(r"20\d{2}", question)
    want_lib = "图书馆" in phrase or "图书馆" in question[:12]
    ranked: list[tuple[float, int]] = []
    for row in _title_rows(conn):
        title = row["title"] or ""
        if any(k in title for k in ("公示", "名单", "临时", "致谢")):
            continue
        have = set(_bigrams(title))
        overlap = sum(1 for gram in grams if gram in have)
        if overlap < 2:
            continue
        score = overlap / len(grams)
        url = row["url"] or ""
        if years and any(year in title or year in url for year in years):
            score += 0.3
        if want_lib and "lib.njupt.edu.cn" in url:
            score += 0.2
        if want_lib and "lib.njupt.edu.cn" not in url and "图书馆" not in title:
            score -= 0.45
        ranked.append((score, int(row["id"])))
    if not ranked:
        return []
    ranked.sort(reverse=True)
    best = ranked[0][0]
    if best < 0.45:
        return []
    ids: list[int] = []
    for score, doc_id in ranked:
        if score < best * 0.8:
            break
        if doc_id not in ids:
            ids.append(doc_id)
        if len(ids) >= 2:
            break
    # 标题更像的那一页如果只是几百字的导航，正文在同名的长文件里。
    chars = _doc_chars(conn)
    phrase_flat = _flat(phrase)
    if ids and chars.get(ids[0], 0) < 500:
        long_ids = []
        for score, doc_id in ranked:
            if score < 0.45 or chars.get(doc_id, 0) < 1500:
                continue
            title = ""
            for row in _title_rows(conn):
                if int(row["id"]) == doc_id:
                    title = row["title"] or ""
                    break
            han = "".join(ch for ch in title if "\u4e00" <= ch <= "\u9fff")
            if len(han) >= 4 and han in phrase_flat and doc_id not in long_ids:
                long_ids.append(doc_id)
        if long_ids:
            long_ids.sort(key=lambda doc_id: chars.get(doc_id, 0), reverse=True)
            return long_ids[:2]
    return ids


def _focus_cited_passage(conn, hits: list[dict[str, Any]], question: str) -> list[dict[str, Any]]:
    """点名的文件里，用问句自己的字（去掉文件名）给每一句打分，取最高的一句。"""
    doc_ids = _align_doc_ids(conn, question)
    if not doc_ids or not hits:
        return hits
    marks = ",".join("?" * len(doc_ids))
    units = conn.execute(
        f"""
        SELECT ru.id, ru.document_id, ru.unit_type, ru.title, ru.text, ru.locator,
               ru.year, ru.academic_year, d.url, d.title AS doc_title, d.org
        FROM retrieval_units ru JOIN documents d ON d.id = ru.document_id
        WHERE ru.document_id IN ({marks})
        ORDER BY length(ru.text) DESC
        """,
        doc_ids,
    ).fetchall()
    if not units:
        return hits
    body = _flat(re.sub(r"按[^，。？?]{2,48}[，,]?", "", question, count=1))
    cite = _flat(_cite_phrase(question))
    for phrase in ("完整内容", "什么", "多少", "哪些", "如何", "怎么", "是否", "哪一", "哪年", "何时", "几个", "分别", "具体", "可以"):
        body = body.replace(phrase, "")
    cited = re.search(r"按([^，。？?]{2,48})", question)
    # 校名写在「按……」里只是点文件。问的是举办、主管，不该被校友会那句的校名压过去。
    if cited and "南京邮电大学" in cited.group(1):
        body = body.replace("南京邮电大学", "")
    body = re.sub(r"20\d{2}年\d{0,2}月?", "", body)
    body = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", body)
    words = []
    for word in re.split(r"[的了吗呢吧啊]", body):
        if len(word) >= 2 and word not in cite and word not in words:
            words.append(word)
    # 文件名里的年份是版本，问句后半段的年份才是事实本身，例如「2001年并入」。
    cite_raw = _cite_phrase(question)
    for year in dict.fromkeys(re.findall(r"20\d{2}", question)):
        if year not in cite_raw and year not in words:
            words.append(year)
    # 长报告里按字数取前几段会漏掉「报到率」那一页。先留问句里的字出现得更多的几页。
    if len(units) > 12 and words:
        probes: list[str] = []
        for word in words:
            if 2 <= len(word) <= 8 and word not in probes:
                probes.append(word)
            for size in (4, 3):
                if len(word) < size + 1:
                    continue
                for offset in range(0, len(word) - size + 1, 2):
                    piece = word[offset : offset + size]
                    if piece in cite or piece in probes or piece == "南邮":
                        continue
                    probes.append(piece)
                    if len(probes) >= 16:
                        break
                if len(probes) >= 16:
                    break
            if len(probes) >= 16:
                break
        if probes:
            ranked_units = []
            for unit in units:
                flat = _flat(unit["text"] or "")
                hit = sum(1 for probe in probes if probe in flat)
                ranked_units.append((hit, len(flat), unit))
            ranked_units.sort(key=lambda item: (item[0], item[1]), reverse=True)
            picked = [unit for hit, _, unit in ranked_units if hit > 0][:12]
            if picked:
                units = picked
    clauses: list[tuple[str, Any]] = []
    for unit in units:
        for sent in _clauses(unit["text"] or ""):
            if len(sent) >= 6:
                clauses.append((sent, unit))
    if not words or not clauses:
        return hits
    stamps = list(dict.fromkeys(re.findall(r"20\d{2}年\d{1,2}月", question)))
    if stamps:
        dated = [item for item in clauses if any(stamp in _flat(item[1]["text"] or "") for stamp in stamps)]
        if dated:
            clauses = dated
    flats = [_flat(sent) for sent, _ in clauses]
    chars = [ch for ch in dict.fromkeys("".join(words)) if "\u4e00" <= ch <= "\u9fff" and ch not in cite]
    df_char = {ch: sum(ch in flat for flat in flats) for ch in chars}

    def spans(word: str, flat: str) -> list[str]:
        """问句里能在这句原文连着对上的几段。文件名和校名简称跳过。"""
        found = _span_pieces(word, flat, cite)
        # 「没有及格」和原文「不及格」是同一件事。
        if "没有" in word:
            for piece in _span_pieces(word.replace("没有", "不"), flat, cite):
                if piece not in found:
                    found.append(piece)
        return found

    def _span_pieces(word: str, flat: str, cite_text: str) -> list[str]:
        found: list[str] = []
        index = 0
        while index < len(word):
            size = 0
            while index + size < len(word) and word[index : index + size + 1] in flat:
                size += 1
            piece = word[index : index + size]
            if size >= 2 and (piece in cite_text or piece in found or piece == "南邮"):
                index += size
                continue
            if size >= 2:
                at = flat.find(piece)
                nxt = flat[at + size : at + size + 1] if at >= 0 else ""
                # 「主管」切进「自主管理」、「举办」切进「举办者」时，原文还连着字，问句并没有。
                if size == 2 and nxt and "\u4e00" <= nxt <= "\u9fff" and word[index + size : index + size + 1] != nxt:
                    index += 1
                    continue
                found.append(piece)
                index += size
                continue
            index += 1
        return found

    # 同一段话在很多句子里都有，说明它是话题而不是答案。少见的那段更该被选中。
    span_df: dict[str, int] = {}
    found_by: list[list[str]] = []
    for flat in flats:
        got: list[str] = []
        for word in words:
            for piece in spans(word, flat):
                if piece not in got:
                    got.append(piece)
        found_by.append(got)
        for piece in got:
            span_df[piece] = span_df.get(piece, 0) + 1

    def cover(flat: str, got: list[str]) -> float:
        def weight(piece: str) -> float:
            base = (len(piece) ** 2) / (span_df[piece] ** 0.5)
            # 「邮电学校」只是「华东邮电学校」的尾巴，不能压过写全了的那一句。
            if any(len(other) > len(piece) and piece in other for other in span_df):
                base *= 0.2
            return base

        total = sum(weight(piece) for piece in got)
        covered = "".join(got)
        # 「迁到」对不上「迁址」时，少见的「迁」仍能把那一句指出来。
        rare = 0.0
        for ch in chars:
            if ch in covered or ch not in flat or not (0 < df_char.get(ch, 0) <= 3):
                continue
            rare += 1.5 / df_char[ch]
        return (total + min(rare, 1.5)) / (len(flat) ** 0.5)

    best: tuple[float, str, Any, int] | None = None
    # 「哪一天」要的是月日，不是年份。年份加分会把章标题旁的旧年份抬上来。
    asks_day = any(k in question for k in ("哪一天", "哪天", "几月几日"))
    asks_when = (not asks_day) and any(k in question for k in ("哪年", "哪一", "何时", "从哪"))
    asks_count = any(k in question for k in ("多少", "几人", "几名", "规模"))
    anchor = ""
    anchor_df: dict[str, int] = {}
    for word in words:
        for size in (4, 3):
            if len(word) < size:
                continue
            for offset in range(0, len(word) - size + 1):
                gram = word[offset : offset + size]
                if gram in cite or gram in anchor_df or gram == "南邮":
                    continue
                if not all("\u4e00" <= ch <= "\u9fff" for ch in gram):
                    continue
                if "年" in gram or "月" in gram:
                    continue
                anchor_df[gram] = sum(gram in flat for flat in flats)
    rare_grams = [gram for gram, count in anchor_df.items() if 0 < count <= 3]
    if rare_grams:
        lowest = min(anchor_df[gram] for gram in rare_grams)
        anchor = max((gram for gram in rare_grams if anchor_df[gram] == lowest), key=len)
    for index, ((sent, unit), flat) in enumerate(zip(clauses, flats)):
        score = cover(flat, found_by[index])
        if score <= 0:
            continue
        for stamp in stamps:
            if stamp in _flat(unit["text"] or ""):
                score += 0.2
        chunk = sent
        if index + 1 < len(clauses) and len(sent) < 100 and clauses[index + 1][1]["document_id"] == unit["document_id"]:
            nxt = clauses[index + 1][0]
            # 短句补下一句时，数字不能和下一节的编号粘在一起。
            sep = "\n" if sent[-1:].isdigit() and nxt[:1].isdigit() else ""
            chunk = sent + sep + nxt
        if asks_when and re.search(r"(?:19|20)\d{2}", flat):
            score += 1.2
        # 问具体日子时，写了「月」「日」的条款高于只带话题的章标题。
        if asks_day and "月" in flat and "日" in flat and re.search(r"\d", flat):
            score += 1.6
        # 「第九章 标识与校庆日」这种短章名不是答案。
        if re.match(r"^第[一二三四五六七八九十百0-9]{1,3}章", sent) and len(sent) < 40:
            score *= 0.3
        # 问人数时，没有数字的口号不能靠话题词压过写了数量的那一句。
        if asks_count and not re.search(r"\d", flat):
            score *= 0.05
        if best is None or score > best[0]:
            best = (score, chunk[:480], unit, index)
    # 问句里最稀有的三四个字如果没出现在当前第一名里，改选含这几个字、而且仍然说得通的那一句。
    # 问哪一年时不改，避免「与地方共建」把写了年份的那句换掉。
    if best and anchor and not asks_when and anchor not in _flat(best[1]) and _USE_RARE_ANCHOR:
        longest = max((len(piece) for piece in found_by[best[3]]), default=0)
        # 当前这句已经对上了更长的一段问句，就不要被一个更短的稀有片段拽走。
        if longest <= len(anchor):
            anchored: tuple[float, str, Any, int] | None = None
            for index, ((sent, unit), flat) in enumerate(zip(clauses, flats)):
                if anchor not in flat:
                    continue
                # 问人数时，不能被没有数字的口号句拽回去。
                if asks_count and not re.search(r"\d", flat):
                    continue
                if re.match(r"^第[一二三四五六七八九十百0-9]{1,3}章", sent) and len(sent) < 40:
                    continue
                score = cover(flat, found_by[index])
                if score <= 0:
                    score = 0.4
                chunk = sent
                if index + 1 < len(clauses) and len(sent) < 100 and clauses[index + 1][1]["document_id"] == unit["document_id"]:
                    nxt = clauses[index + 1][0]
                    sep = "\n" if sent[-1:].isdigit() and nxt[:1].isdigit() else ""
                    chunk = sent + sep + nxt
                if anchored is None or score > anchored[0]:
                    anchored = (score, chunk[:480], unit, index)
            if anchored and anchored[0] >= 0.2:
                best = anchored
    if not best or best[0] < 0.35:
        return hits
    headline = best[1]
    chunk = headline
    covered = set(found_by[best[3]])
    if best[3] > 0:
        prev_sent, prev_unit = clauses[best[3] - 1]
        if prev_unit is clauses[best[3]][1] and len(prev_sent) <= 90 and prev_sent[:24] not in chunk:
            brought = [
                piece for piece in found_by[best[3] - 1]
                if piece not in covered and span_df.get(piece, 9) <= 2 and len(piece) >= 2
            ]
            # 条号和主语经常在上一句，例如「第六条 学校是非营利性事业单位」。
            if brought or re.match(r"^第", prev_sent):
                chunk += "\n" + prev_sent
                covered.update(brought)
    # 分号拆开的职责条目，主语在前面带冒号的那一句，例如「某某委员会的主要职责是：」。
    if "：" not in headline[:30]:
        for back in range(best[3] - 1, max(-1, best[3] - 6), -1):
            prev_sent, prev_unit = clauses[back]
            if prev_unit is not clauses[best[3]][1]:
                break
            if prev_sent.endswith("。"):
                break
            if "：" in prev_sent and len(prev_sent) <= 80 and prev_sent[:24] not in chunk:
                chunk += "\n" + prev_sent
                break
    extras: list[tuple[float, str]] = []
    for index, (sent, _unit) in enumerate(clauses):
        if index == best[3]:
            continue
        new = [
            piece for piece in found_by[index]
            if piece not in covered and span_df.get(piece, 9) <= 2 and len(piece) >= 2
        ]
        if not new:
            continue
        rare = sum((len(piece) ** 2) / (span_df[piece] ** 0.5) for piece in new)
        extras.append((rare, sent))
    extras.sort(key=lambda item: item[0], reverse=True)
    added = 0
    for _rare, sent in extras:
        if added >= 2 or len(chunk) >= 700:
            break
        if sent[:40] in chunk:
            continue
        # 换行再拼，避免「留学生701」和下一节「2.4」粘成 7012.4。
        chunk += "\n" + sent
        added += 1
    promoted = dict(best[2])
    promoted["text"] = chunk[:720]
    promoted["headline"] = headline[:480]
    promoted["focused"] = True
    promoted["score"] = (hits[0].get("score") or 0) + best[0]
    rest = [hit for hit in hits if hit.get("document_id") != promoted.get("document_id")]
    return [promoted] + rest[:4]


def _match_terms(keywords: list[str]) -> list[str]:
    """长词对不上原文时，再用它的头三个字和尾三个字去对句子。

    “教职工总规模”在简介里写成“教职工”，“学科为主体”的答案写成“以工学为主体”。
    这些片段只用于句内挑选，不放进全文检索，避免单字词把无关页面召回来。
    """
    terms: list[str] = []
    for kw in keywords:
        if kw and kw not in terms:
            terms.append(kw)
        if len(kw) < 5:
            continue
        for piece in (kw[:3], kw[:4], kw[-3:], kw[-4:]):
            if len(piece) >= 3 and piece not in terms and piece not in _DROP_WORDS:
                terms.append(piece)
    return terms


def _independent_keywords(sentence: str, keywords: list[str]) -> list[str]:
    """长词已经命中时，不再把它切出来的 4 字片段重复计分。

    否则章节标题里的“综合素质测评”会压过正文里的“德育、智育、能力测评”。
    """
    present = [kw for kw in keywords if len(kw) >= 2 and kw in sentence]
    kept: list[str] = []
    for kw in sorted(present, key=len, reverse=True):
        if any(kw != other and kw in other for other in kept):
            continue
        kept.append(kw)
    return kept


def _clauses(text: str) -> list[str]:
    """PDF 换行不是句号。先把没写完的行拼回去，再按句号切开。"""
    # 「留学生701」和下一节「2.4 招生」粘在一起时，701 不能变成 7012.4。
    text = re.sub(r"(\d{2,})(\d\.\d)(?=\s*[\u4e00-\u9fff])", r"\1 \2", text or "")
    lines = [ln.strip() for ln in re.split(r"\n+", text or "") if ln.strip()]
    glued: list[str] = []
    buf = ""
    for ln in lines:
        if not buf:
            buf = ln
            continue
        if buf[-1] in "。！？；" or re.match(r"^(?:[（(]|第|\d{1,2}[、]|\d{1,2}\.\d{1,2}\s|\d{1,2}\.\s)", ln):
            glued.append(buf)
            buf = ln
        elif buf[-1].isdigit() or ln[:1].isdigit():
            # 表格一行一个数。拼回去时留空格，避免 397 和 17.13% 粘成 39717.13。
            buf += " " + ln
        else:
            buf += ln
    if buf:
        glued.append(buf)
    parts: list[str] = []
    for piece in glued:
        parts.extend(p.strip() for p in re.split(r"(?<=[。！？；])", piece) if p and len(p.strip()) >= 8)
    return parts


def _sentence_score(sentence: str, keywords: list[str]) -> float:
    kept = _independent_keywords(sentence, _match_terms(keywords))
    if not kept:
        return 0.0
    return len(kept) * 10 + sum(len(kw) for kw in kept)


def _promote_best_sentence(conn, hits: list[dict[str, Any]], keywords: list[str]) -> list[dict[str, Any]]:
    """同一篇文档里，换上和问句重合最多的那一句。"""
    keys = [kw for kw in keywords if len(kw) >= 2]
    if not hits or not keys:
        return hits
    keys = sorted(keys, key=len, reverse=True)[:6]
    by_doc: dict[Any, dict[str, Any]] = {}
    for hit in hits:
        title = hit.get("doc_title") or ""
        profile = any(k in title for k in (
            "学校简介", "学校章程", "南邮精神", "南邮校史", "单位概况",
            "本馆简介", "学生手册", "招生章程", "教学质量报告", "推荐免试",
        ))
        if hit["document_id"] in by_doc:
            continue
        if len(by_doc) < 4 or profile:
            by_doc[hit["document_id"]] = hit
        if len(by_doc) >= 8:
            break
    probe = [kw for kw in _match_terms(keys) if 2 <= len(kw) <= 8][:5] or keys[:3]
    like = " OR ".join("ru.text LIKE ?" for _ in probe)
    like_params = [f"%{kw}%" for kw in probe]
    best: tuple[float, str, dict[str, Any]] | None = None
    for did, src in by_doc.items():
        try:
            units = conn.execute(
                f"""
                SELECT ru.text FROM retrieval_units ru
                WHERE ru.document_id=? AND ({like})
                LIMIT 24
                """,
                [did, *like_params],
            ).fetchall()
        except Exception:
            continue
        blobs = [src.get("text") or ""]
        blobs.extend((row["text"] or "") for row in units)
        for blob in blobs:
            parts = _clauses(blob)
            for i, sent in enumerate(parts):
                score = _sentence_score(sent, keys)
                chunk = sent
                if i + 1 < len(parts) and len(sent) < 180:
                    chunk = sent + parts[i + 1]
                prefer = "学校简介" in (src.get("doc_title") or "")
                better = best is None or score > best[0] or (
                    score == best[0] and prefer and "学校简介" not in (best[2].get("doc_title") or "")
                )
                if better:
                    best = (score, chunk, src)
    if not best or best[0] < 8:
        return hits
    current = max(
        (_sentence_score(p, keys) for p in _clauses(hits[0].get("text") or "")),
        default=0,
    )
    if best[0] < current:
        return hits
    if best[0] == current and best[2].get("document_id") == hits[0].get("document_id"):
        return hits
    promoted = dict(best[2])
    promoted["text"] = best[1][:480]
    promoted["score"] = (best[2].get("score") or 0) + best[0]
    return [promoted] + [h for h in hits if h.get("id") != promoted.get("id")]


def retrieve(question: str, k: int = 10) -> tuple[QueryPlan, list[dict[str, Any]]]:
    plan = plan_query(question)
    hits = lexical_search(question, plan, k=k)
    return plan, hits
