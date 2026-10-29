"""按问题自己的词，在原文里给数字打分。

离线可用。不按题目编号写答案：先用问题里的内容词收窄到少数文档，
再看哪个数字前面贴着这些词。范围词（初次或年终、研究生或本科、
总体规模或调查说明）也来自问题本身。数字之间的空格会保留，避免
把相邻的两个数粘成一个。
"""

from __future__ import annotations

import re
from functools import lru_cache

from .db import init_db
from .util import focus_text

NUM_RE = re.compile(r"\d+(?:\.\d+)?")
REFUSE = "资料不足，不能确定。"
STOP = {
    "多少", "什么", "如何", "怎么", "报告", "南京", "邮电", "大学", "年度", "其中",
    "分别", "标准", "每人", "每年", "写明", "给出", "列出", "印刷", "直接", "写出",
    "要求", "可以", "申请", "这个", "那个", "正文", "一份", "发布", "落款",
}


def _squash(text: str) -> str:
    # 页码印在句子中间时是「6000 ·208· 元」，先拿掉，避免金额和页码粘在一起。
    text = re.sub(r"·\s*\d{1,4}\s*·", "", text or "")
    kept = re.sub(r"(?<=\d)\s+(?=\d)", "\u0001", text)
    kept = re.sub(r"\s+", "", kept)
    return kept.replace("\u0001", " ")


@lru_cache(maxsize=1)
def _corpus() -> tuple[tuple[int, str, str], ...]:
    conn = init_db()
    grouped: dict[int, list[str]] = {}
    titles: dict[int, str] = {}
    for row in conn.execute(
        """SELECT d.id, d.title, b.text
           FROM documents d
           LEFT JOIN blocks b ON b.document_id=d.id
           ORDER BY d.id, b.ordinal, b.id"""
    ):
        titles[int(row[0])] = row[1] or ""
        grouped.setdefault(int(row[0]), []).append(row[2] or "")
    conn.close()
    return tuple(
        (doc_id, titles[doc_id], _squash(titles[doc_id] + "。" + "".join(grouped[doc_id])))
        for doc_id in titles
    )


def _plan_documents(college: str = "") -> list[tuple[str, str]]:
    """本科培养方案。学院名写在正文「所属学院」时，标题里不一定有。"""
    conn = init_db()
    rows = conn.execute(
        """
        SELECT d.id, d.title FROM documents d
        WHERE d.title LIKE '%培养方案%'
          AND d.title NOT LIKE '%研究生%'
        """
    ).fetchall()
    found: list[tuple[str, str]] = []
    for row in rows:
        blocks = conn.execute(
            "SELECT text FROM blocks WHERE document_id=? ORDER BY id",
            (row["id"],),
        ).fetchall()
        text = "\n".join(b["text"] or "" for b in blocks)
        blob = f"{row['title']}\n{text}"
        if "总学分" not in text and "本科" not in blob and "毕业" not in text[:500]:
            continue
        if college and college not in row["title"] and not re.search(
            rf"所属学院[\s\S]{{0,40}}{re.escape(college)}", blob
        ):
            continue
        found.append((row["title"], text))
    conn.close()
    return found


def _clean_major(name: str) -> str:
    name = re.split(r"[的里，]", name)[-1]
    name = re.sub(r"^.*学院", "", name)
    name = re.sub(r"^(?:20\d{2})?级", "", name)
    name = re.sub(r"专业$", "", name).strip()
    if name in {"本科", "毕业", "培养", "方案"} or "培养方案" in name or "学院" in name:
        return ""
    return name if len(name) >= 2 else ""


def program_credit(question: str) -> str | None:
    """从本科培养方案的毕业学分表读取总学分。比较题留给后面拆开再算。"""
    if "绩点" in question:
        return None
    if re.search(r"比.{0,30}(多|少|大|高|低)(?:多少|几)|相差", question):
        return None
    asks_credit = any(k in question for k in ("毕业总学分", "毕业学分", "总学分")) or (
        "多少学分" in question and re.search(r"专业|英才班|强化班", question)
    )
    if not asks_credit:
        return None
    focus = focus_text(question)
    major = ""
    found = None
    for found in re.finditer(
        r"(?:学院|的|里|，)([\u4e00-\u9fffA-Za-z0-9]{2,18})(?:专业|毕业学分|的总学分|要修)",
        focus,
    ):
        pass
    if found:
        major = _clean_major(found.group(1))
    if not major:
        lead = re.search(r"([\u4e00-\u9fff]{2,18})专业", focus)
        if lead:
            major = _clean_major(lead.group(1))
    if not major:
        clazz = re.search(r"([\u4e00-\u9fff]{2,12}(?:英才班|强化班))", focus)
        if clazz:
            major = _clean_major(clazz.group(1)) or clazz.group(1)
    college_m = re.search(r"([\u4e00-\u9fff]{2,16}学院)", focus)
    college = college_m.group(1) if college_m else ""
    docs = _plan_documents(college) if college else []
    pool = docs or _plan_documents("")
    if major:
        titled = [item for item in pool if major in item[0]]
        if not titled and not docs:
            titled = [item for item in _plan_documents("") if major in item[0]]
        if not titled:
            titled = [item for item in pool if major in (item[0] + item[1][:300])]
        docs = titled
    elif not docs:
        return None
    years = re.findall(r"20\d{2}", focus)
    hits: list[tuple[int, str]] = []
    for title, text in docs:
        blob = f"{title}\n{text}"
        value_m = re.search(r"总学分\s*[/／]\s*比例\s*(\d+(?:\.\d+)?)", text)
        if not value_m:
            value_m = re.search(r"(?<!平均)毕业总学分\s*[:：为]?\s*(\d+(?:\.\d+)?)", text)
        if not value_m:
            continue
        year_m = re.search(r"适用年级[\s:：]*?(20\d{2})", blob) or re.search(r"(20\d{2})\s*级", title + text[:200])
        year = int(year_m.group(1)) if year_m else 0
        hits.append((year, value_m.group(1)))
    if years:
        scoped = [item for item in hits if str(item[0]) in years]
        if scoped:
            hits = scoped
    if not hits:
        return None
    hits.sort()
    year, value = hits[-1]
    grade = f"{year}级" if year else ""
    others = [f"{y}级为{v}" for y, v in hits if y and not (y == year and v == value)]
    extra = f"另有{'，'.join(others)}。" if others else ""
    return f"{value}。{grade}本科培养方案毕业总学分。{extra}"


def policy_revision(question: str) -> str | None:
    """办法、细则问修订日期时，读标题或正文里写明的日期。

    「按……修订的办法，退役后何时入学」里的修订只是在点文件，不是在问日期。
    """
    asked = re.sub(r"按[^，。？?]{2,48}[，,]?", "", question, count=1)
    if not any(k in asked for k in ("哪一天", "哪天", "何时", "什么时候")):
        return None
    if not any(k in asked for k in ("修订", "通过", "印发", "发布")):
        return None
    named = re.search(r"([\u4e00-\u9fff]{2,8})(管理办法|实施细则|暂行规定|办法|细则|规定)", question)
    if not named:
        return None
    # 取名称末四字加文种。六字会把「本科生选读课程」切成标题里没有的「科生选读课程」。
    label = named.group(1)[-4:] + named.group(2)
    conn = init_db()
    rows = conn.execute(
        """
        SELECT id, title FROM documents
        WHERE parse_status='parsed' AND title LIKE ?
        """,
        (f"%{label}%",),
    ).fetchall()
    date_re = re.compile(r"(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
    found: list[tuple[int, str]] = []
    for row in rows:
        text = "\n".join(
            b["text"] or ""
            for b in conn.execute(
                "SELECT text FROM blocks WHERE document_id=? ORDER BY id LIMIT 2",
                (row["id"],),
            )
        )
        blob = _glue_spaced_year(f"{row['title']}\n{text[:500]}")
        near = None
        for cue in ("修订", "通过", "印发", "发布"):
            pos = blob.find(cue)
            if pos < 0:
                continue
            window = blob[max(0, pos - 24) : pos + 8]
            near = date_re.search(window)
            if near:
                break
        hit = near or date_re.search(blob)
        if not hit:
            continue
        y, mo, d = (int(hit.group(i)) for i in (1, 2, 3))
        found.append((y * 10000 + mo * 100 + d, f"{y}年{mo}月{d}日"))
    conn.close()
    if not found:
        return None
    found.sort()
    return f"{found[-1][1]}。"


def _glue_spaced_year(text: str) -> str:
    """正文有时把年份印成「20 2 6 年」，先拼回 2026 年再取日期。"""
    text = re.sub(
        r"(?<!\d)(\d)\s+(\d)\s+(\d)\s+(\d)\s*年",
        lambda m: "".join(m.groups()) + "年",
        text,
    )
    return re.sub(
        r"(?<!\d)(\d{2})\s+(\d)\s+(\d)\s*年",
        lambda m: m.group(1) + m.group(2) + m.group(3) + "年",
        text,
    )


def policy_meeting(question: str) -> str | None:
    """问的是哪个会议通过，不拿网页主办单位或录取表来顶。"""
    if not any(k in question for k in ("哪个会议", "什么会议", "哪个会")):
        return None
    named = re.search(r"([\u4e00-\u9fff]{2,8})(管理办法|实施细则|暂行规定|办法|细则|规定)", question)
    if not named:
        return None
    label = named.group(1)[-4:] + named.group(2)
    meetings = ("校长办公会", "党委常委会", "校学位评定委员会", "校长办公会议")
    for _, title, text in _corpus():
        if label not in title and label not in text[:120]:
            continue
        window = text[:180]
        for name in meetings:
            if name in window or name in title:
                return f"{name}。"
    return None


def exam_minutes(question: str) -> str | None:
    """开考前多久入场。页码和条款号不是分钟数。"""
    if "分钟" not in question or "考试" not in question or "多少" not in question:
        return None
    focus = focus_text(question)
    for _, title, text in _corpus():
        if "规则" in focus and "规则" not in title:
            continue
        if "考试" not in title and "考试规则" not in text[:40]:
            continue
        found = re.search(r"提前\s*(\d+)\s*分钟", text)
        if found:
            return f"{found.group(1)}。"
    return None


_AWARD_PAIR = re.compile(
    r"(本科生|研究生|硕士研究生)[:：](\d+)(名|人)，(?:每学年奖学金标准为)?(\d+)元[/／]人"
)


def _award_pairs(bodies: list[str]) -> tuple[tuple[str, str, str, str], ...] | None:
    parsed = []
    for body in bodies:
        found = tuple(_AWARD_PAIR.findall(body))
        if found:
            parsed.append(found)
    if not parsed or len(set(parsed)) != 1:
        return None
    return parsed[0]


def _award_quota_lines(bodies: list[str]) -> str | None:
    """本科和研究生各只有一档时，才把两档名额和金额一起报出。"""
    pairs = _award_pairs(bodies)
    if not pairs:
        return None
    under = [item for item in pairs if item[0] == "本科生"]
    grad = [item for item in pairs if item[0] in {"研究生", "硕士研究生"}]
    if len(under) != 1 or len(grad) != 1:
        return None
    _, un, uu, um = under[0]
    _, gn, gu, gm = grad[0]
    return f"本科生{un}{uu}、每人{um}元；研究生{gn}{gu}、每人{gm}元。"


def _award_roll(bodies: list[str], focus: str) -> str | None:
    """各档名额乘以单人金额，只在问合计人数或发放总额时相加。"""
    pairs = _award_pairs(bodies)
    if not pairs:
        return None
    people = sum(int(item[1]) for item in pairs)
    money = sum(int(item[1]) * int(item[3]) for item in pairs)
    if "总额" in focus:
        if money % 10000 == 0 and money >= 100000:
            return f"{money // 10000}万元"
        return f"{money}元"
    if "名额" in focus and any(k in focus for k in ("合计", "一共", "全部")):
        return f"{people}人"
    return None


def _precise_award(question: str) -> str | None:
    """办法标题被问句完整点名时，才把各档名额相加或并列报出。"""
    focus = focus_text(question)
    wants_split = (
        "本科" in focus
        and ("研究生" in focus or "硕士" in focus)
        and any(k in focus for k in ("分别", "名额", "金额"))
    )
    wants_money = "总额" in focus
    wants_people = "名额" in focus and any(k in focus for k in ("合计", "一共", "全部"))
    if not wants_split and not wants_money and not wants_people:
        return None
    conn = init_db()
    rows = conn.execute(
        """
        SELECT id, title FROM documents
        WHERE parse_status='parsed' AND (title LIKE '%奖学金%' OR title LIKE '%奖助学金%')
        """
    ).fetchall()
    found: list[tuple[int, int, str]] = []
    for row in rows:
        title = re.sub(r"\s+", "", row["title"] or "")
        matched = re.findall(r"[\u4e00-\u9fff]{2,12}奖助学金|[\u4e00-\u9fff]{2,12}奖学金", title)
        for name in matched:
            if name.startswith("国家") or name not in question:
                continue
            found.append((len(name), row["id"], name))
    if not found:
        conn.close()
        return None
    best = max(item[0] for item in found)
    bodies = []
    for length, doc_id, _name in found:
        if length != best:
            continue
        body = re.sub(
            r"\s+",
            "",
            "".join(
                b["text"] or ""
                for b in conn.execute("SELECT text FROM blocks WHERE document_id=? ORDER BY id", (doc_id,))
            ),
        )
        if body:
            bodies.append(body)
    conn.close()
    if wants_split:
        quoted = _award_quota_lines(bodies)
        if quoted:
            return quoted
    if wants_money or wants_people:
        return _award_roll(bodies, focus)
    return None


def named_award(question: str) -> str | None:
    """点名的校内奖学金。国家奖助学金仍走原文打分，比较题留给拆句。"""
    if re.search(r"比.{0,40}(多|少|大|高|低)(?:多少|几)|相差", question):
        return None
    precise = _precise_award(question)
    if precise:
        return precise
    focus = focus_text(question)
    named = re.search(r"([\u4e00-\u9fff]{2,6}?)(特等奖学金|励志奖学金|奖学金|助学金)", focus)
    if not named or named.group(1).startswith("国家"):
        return None
    name = named.group(1)
    conn = init_db()
    rows = conn.execute(
        """
        SELECT id, title FROM documents
        WHERE parse_status='parsed' AND title LIKE ? AND title LIKE '%奖%'
        """,
        (f"%{name}%",),
    ).fetchall()
    if not rows:
        conn.close()
        return None
    bodies = []
    for row in rows:
        body = re.sub(
            r"\s+",
            "",
            "\n".join(
                b["text"] or ""
                for b in conn.execute("SELECT text FROM blocks WHERE document_id=? ORDER BY id", (row["id"],))
            ),
        )
        if body:
            bodies.append(body)
    conn.close()
    asks_people = any(k in focus for k in ("多少人", "名额", "几人")) and "多少元" not in focus and "金额" not in focus
    if "生均" in focus:
        for body in bodies:
            amount = re.search(r"生均奖励金额为(\d+(?:\.\d+)?)元", body)
            if amount:
                return f"{amount.group(1)}。"
    label = "研究生" if "研究生" in focus else "本科生" if "本科" in focus else ""
    if not label:
        return None
    person = rf"{label}[:：]?(\d+)人[，,、](\d+)元"
    for body in bodies:
        scope = body
        if "特等" in focus:
            headed = re.search(rf"特等奖学金[:：][\s\S]{{0,120}}?{person}", body)
            if headed:
                found = headed
                return f"{found.group(1) if asks_people else found.group(2)}。"
        found = re.search(person, scope)
        if found:
            return f"{found.group(1) if asks_people else found.group(2)}。"
    return None


def refuse_reason(question: str) -> str | None:
    private = (
        "学号", "我考了", "能不能被", "能否录取", "在校生目前", "发放账号",
        "银行卡", "身份证号", "张三", "某本科生", "某同学", "加权平均", "个人成绩",
    )
    if any(k in question for k in private):
        return REFUSE + "公开资料不包含个人成绩、账号或录取结论。"
    # 「不及格怎么处理」「校园卡去哪充」是公开制度。问到本人账户才拒答。
    if any(k in question for k in ("校园卡", "教务系统", "不及格")) and any(
        k in question for k in ("余额", "我的", "我本", "卡号", "登录后", "目前的")
    ):
        return REFUSE + "公开资料不包含个人成绩、账号或录取结论。"
    college = re.search(r"([\u4e00-\u9fff]{2,20}学院)", question)
    if college and "培养方案" in question and "学分" in question:
        name = college.group(1)
        # 学院名要在标题里，或紧挨「所属学院」。别的方案正文里的专业目录不算。
        if not _plan_documents(name):
            return REFUSE + "没有检索到该学院的本科培养方案，不能用研究生方案或招生计划代替学分。"
    return None


@lru_cache(maxsize=1)
def _leaders() -> list[tuple[str, str]]:
    """官网“现任领导”页：按行解析「南京邮电大学职务 姓名」。"""
    conn = init_db()
    row = conn.execute(
        "SELECT d.id FROM documents d WHERE d.url='http://www.njupt.edu.cn/17226/list.htm'"
    ).fetchone()
    if not row:
        conn.close()
        return []
    blocks = conn.execute(
        "SELECT text FROM blocks WHERE document_id=? ORDER BY id", (row[0],)
    ).fetchall()
    conn.close()
    text = "\n".join(b["text"] or "" for b in blocks)
    for junk in ("南邮概况", "首页", "南邮校歌", "南邮校史", "学校简介", "学校章程",
                 "南邮精神", "校标校训", "现任领导", "视频展播", "校园实景漫游", "校园景色",
                 "校区地图"):
        text = text.replace(junk, " ")
    text = re.sub(r"(?<=[\u4e00-\u9fff]) (?=[\u4e00-\u9fff])", "", text)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith("南京邮电大学") and not line.startswith("省纪委监委"):
            i += 1
            continue
        m = re.search(r"南京邮电大学(.+)$", line)
        if not m:
            i += 1
            continue
        rest = m.group(1).rstrip("，, ")
        if "南京邮电大学" in rest or "校区" in rest or not (2 <= len(rest) <= 24):
            i += 1
            continue
        # 下一行若是 2-3 个汉字，且是「南京邮电大学职务」重复行，则本行 rest = 职务+姓名。
        if i + 1 < len(lines) and lines[i + 1].startswith("南京邮电大学"):
            m2 = re.search(r"南京邮电大学(.+)$", lines[i + 1])
            if m2 and m2.group(1).rstrip("，, ") == rest[: len(m2.group(1).rstrip("，, "))]:
                name = rest[len(m2.group(1).rstrip("，, ")) :]
                role = m2.group(1).rstrip("，, ")
                if re.fullmatch(r"[\u4e00-\u9fff]{2,3}", name):
                    pairs.append((role, name))
                    i += 2
                    continue
        # 下一行若是纯人名，本行 rest 可能是职务（姜小舜式）或职务+姓名（刘青山式）。
        if i + 1 < len(lines) and re.fullmatch(r"[\u4e00-\u9fff]{2,3}", lines[i + 1]):
            next_name = lines[i + 1]
            if rest.endswith(next_name):
                role = rest[: -len(next_name)]
                if role and 2 <= len(role) <= 24:
                    pairs.append((role, next_name))
                    i += 2
                    continue
            pairs.append((rest, next_name))
            i += 2
            continue
        i += 1
    # 人名去重，保持出现顺序。
    found: list[tuple[str, str]] = []
    seen_names = set()
    for role, name in pairs:
        if name not in seen_names:
            seen_names.add(name)
            found.append((role, name))
    return found


def current_leader(question: str) -> str | None:
    """问现任校领导（校长/书记/副校长/某人是干什么的）。"""
    leaders = _leaders()
    if not leaders:
        return None
    focus = focus_text(question)
    # 问具体某个人：从现任名单里点名。
    for role, name in leaders:
        if name and name in focus:
            return f"{name}现任{role}。"
    if not any(k in question for k in ("校长", "副校长", "党委书记", "校领导", "现任领导", "领导")):
        return None
    asks_president = "校长" in focus and "副校长" not in focus
    asks_secretary = "党委书记" in focus and "副书记" not in focus
    if "校长" in focus and "党委书记" in focus and "副校长" not in focus:
        asks_president = asks_secretary = True
    pres = next((n for r, n in leaders if "校长" in r and "副校长" not in r), None)
    sec = next((n for r, n in leaders if r == "党委书记" or (r.endswith("党委书记") and "副" not in r)), None)
    if asks_president and asks_secretary:
        if pres and sec:
            return f"现任校长是{pres}，党委书记是{sec}。"
        if pres:
            return f"现任校长是{pres}。"
        if sec:
            return f"现任党委书记是{sec}。"
        return None
    if asks_president:
        if pres:
            return f"现任校长是{pres}。"
        return None
    if asks_secretary:
        if sec:
            return f"现任党委书记是{sec}。"
        return None
    if "副校长" in focus:
        vps = [n for r, n in leaders if "副校长" in r]
        if vps:
            return "、".join(vps)
        return None
    # 「现任领导有哪些」这类宽问题：列主要职务。
    if any(k in focus for k in ("现任领导", "校领导", "领导有哪些", "领导是谁")):
        head = [f"{n}（{r}）" for r, n in leaders[:8]]
        return "；".join(head) + "。"
    return None


def teacher_count(question: str) -> str | None:
    """问全校教师规模（多少老师/多少教师/教职工多少人/师资情况）。"""
    if not any(k in question for k in ("多少老师", "多少教师", "多少名教师", "多少位教师", "教师人数",
                                       "专任教师", "教职工", "师资", "多少教职工", "有多少教师")):
        return None
    # 「教职工借期/图书」不是在问教师人数。这个捷径若先返回，
    # 会挡住后面按槽位组装的证据。
    if any(k in question for k in ("借期", "借阅", "图书", "册")) and not any(
        k in question for k in ("多少人", "人数", "多少名")
    ):
        return None
    # 「A比B多多少」要走两个指标相减，不能在这里先吐出一个人数。
    if re.search(r"比.{1,30}(?:多|少|高|低)(?:多少|几)", question):
        return None
    # 「反推系数」「复算生师比」是计算题，不能在这里提前答人数。
    if any(k in question for k in ("反推", "复算", "折算系数", "系数", "生师比")):
        return None
    # 问的是职称、学位或年龄结构，不是专任教师总人数。
    if any(k in question for k in ("正高", "副高", "中级", "学位", "学缘", "年龄结构")):
        return None
    # 点名单个学院时不抢答校级口径，避免拿全校数去答某学院。
    if re.search(r"[\u4e00-\u9fff]{2,20}学院", question) and "学校" not in question and "全校" not in question and "南邮" not in question:
        return None
    conn = init_db()
    rows = conn.execute(
        """
        SELECT f.metric, f.value, f.year, f.source_locator, d.title, d.url
        FROM facts f JOIN documents d ON d.id=f.source_document_id
        WHERE f.metric IN ('fulltime','adjunct','equivalent','ratio')
          AND f.scope='全校' AND d.title LIKE '%本科教学质量报告%'
        ORDER BY f.year DESC, f.id
        """
    ).fetchall()
    conn.close()
    if not rows:
        return None
    # 报告正文是「截止2023-2024学年末」，facts 表里 fulltime 记成了 2022。取最大学年作为口径年。
    years = [int(r["year"]) for r in rows if r["year"]]
    if not years:
        return None
    best_year = max(years)
    ft = next((r for r in rows if r["metric"] == "fulltime"), None)
    adj = next((r for r in rows if r["metric"] == "adjunct"), None)
    eq = next((r for r in rows if r["metric"] == "equivalent"), None)
    if "专任教师" in question or "师资" in question or "多少老师" in question or "多少教师" in question:
        if ft:
            return f"截止{best_year}-{best_year+1}学年末，学校有专任教师{int(float(ft['value']))}人。"
    if "教职工" in question or "教师" in question or "老师" in question:
        parts = []
        if ft:
            parts.append(f"专任教师{int(float(ft['value']))}人")
        if adj:
            parts.append(f"聘请校外教师{int(float(adj['value']))}人")
        if eq:
            parts.append(f"折合教师总数{int(float(eq['value']))}人")
        if parts:
            return f"截止{best_year}-{best_year+1}学年末，学校" + "，".join(parts) + "。"
    return None


def _terms(question: str) -> list[str]:
    """从连续汉字里切出 3 到 8 字的片段。长句不会被一次吞成一个匹配不到的词。"""
    out: list[str] = []
    for run in re.findall(r"[\u4e00-\u9fff]{2,}", question):
        if run in STOP:
            continue
        if len(run) <= 8:
            out.append(run)
            continue
        for size in range(8, 2, -1):
            for i in range(0, len(run) - size + 1):
                frag = run[i : i + size]
                if frag not in STOP:
                    out.append(frag)
    return list(dict.fromkeys(out))


def _prefilter(question: str) -> list[tuple[int, str, str]]:
    docs = list(_corpus())
    dates = re.findall(r"20\d{2}年\d{1,2}月\d{1,2}日", _squash(question))
    if dates:
        hit = [d for d in docs if any(dt in d[2] for dt in dates)]
        if hit:
            return hit
    editions = re.findall(r"20\d{2}版", focus_text(question))
    if editions:
        hit = [d for d in docs if any(ed in d[1] for ed in editions)]
        if hit:
            return hit
    years = re.findall(r"20\d{2}", question)
    if years and any(k in question for k in ("就业", "毕业生", "落实率")):
        hit = [d for d in docs if "就业" in d[1] and any(y in d[1] for y in years)]
        if hit:
            return hit
    rules = (
        (("就业", "毕业生", "落实率"), ("就业",)),
        (("教学质量", "生均", "藏书", "仪器", "立项", "购置", "实习", "实验经费"), ("教学质量",)),
        (("奖学金", "助学金", "绩点", "学位", "运动员", "重修", "手册"), ("手册", "学位", "简介")),
        (("章程", "校区", "结算", "住宿", "三牌楼", "锁金", "仙林"), ("章程",)),
        (("末位", "同分"), ("综合评价", "简章")),
        (("传真", "电话", "联系方式"), ("联系", "招生")),
    )
    for cues, title_keys in rules:
        if any(k in question for k in cues):
            hit = [d for d in docs if any(k in d[1] for k in title_keys)]
            if hit:
                return hit
    return docs


def _pool(question: str) -> list[tuple[int, str, str]]:
    base = _prefilter(question)
    terms = [t for t in _terms(question) if len(t) >= 3]
    if not terms:
        return base[:4]
    ranked = []
    for doc in base:
        score = sum(len(t) ** 2 for t in terms if t in doc[2])
        if score:
            ranked.append((score, doc))
    if not ranked:
        return []
    ranked.sort(key=lambda item: -item[0])
    best = ranked[0][0]
    return [doc for score, doc in ranked if score >= best * 0.72][:4]


def _nearest_label(before: str) -> str:
    # “博士毕业生”里“毕业生”更靠右，但问的就是博士。
    if "博士" in before[-8:]:
        return "博士"
    if "研究生" in before[-8:]:
        return "研究生"
    if "本科" in before[-8:]:
        return "本科"
    labels = ("总人数", "总数", "总体规模", "毕业生", "学校")
    best, pos = "", -1
    for label in labels:
        i = before.rfind(label)
        if i >= pos:
            pos, best = i, label
    return best


def _address(question: str, docs: list[tuple[int, str, str]]) -> str | None:
    if not any(k in question for k in ("地址", "校区", "三牌楼", "锁金", "仙林")):
        return None
    found = []
    for _, title, text in docs:
        for name in ("仙林", "三牌楼", "锁金村"):
            if name not in question:
                continue
            m = re.search(
                name + r".{0,48}?([\u4e00-\u9fff]{2,12}路\d+号)(?:.{0,16}?邮编[：:]?(\d{6}))?",
                text,
            )
            if not m:
                continue
            span = m.group(0)
            if "邮编" in question and "邮编" not in span:
                continue
            if span not in found:
                found.append(span)
    return "；".join(found) if found else None


def _phone(question: str, docs: list[tuple[int, str, str]]) -> str | None:
    """电话取“电话：”后面的号码。问传真时不拿电话，反过来也不拿传真。"""
    focus = focus_text(question)
    if "传真" not in focus and "电话" not in focus and "联系方式" not in focus:
        return None
    want_fax = "传真" in focus and "电话" not in focus
    ordered = docs
    if "招生" in question or "联系方式" in question:
        titled = [d for d in docs if "联系" in d[1]]
        if titled:
            ordered = titled
    for _, title, text in ordered:
        if want_fax:
            fax = re.search(r"传真[：:]?.{0,12}?((?:0\d{2,3}-)?\d{7,8})", text)
            if fax:
                return fax.group(1)
            continue
        m = re.search(r"电话[：:]?(0\d{2,3}-\d{7,8})(?:[，,、](\d{7,8}))?", text)
        if not m:
            continue
        nums = [m.group(1)]
        if m.group(2):
            nums.append(m.group(2))
        if "传真" not in focus:
            fax = re.search(r"传真[：:]?.{0,12}?((?:0\d{2,3}-)?\d{7,8})", text)
            if fax:
                nums = [n for n in nums if n != fax.group(1)]
        if nums:
            return "，".join(nums)
    return None


METRICS = (
    "平均学分绩点",
    "国家励志奖学金",
    "国家助学金",
    "国家奖学金",
    "生均本科实习经费",
    "生均本科实验经费",
    "生均教学科研仪器设备",
    "教学科研仪器设备总价值",
    "纸质藏书",
    "新增图书",
    "立项项目",
    "参加学生",
    "购置费",
    "绩点",
)


def _metric(question: str) -> str | None:
    focus = focus_text(question)
    hits = [m for m in METRICS if m in focus]
    return max(hits, key=len) if hits else None


def _asked_pop(question: str) -> str | None:
    """“多少”前面最近的人群才是要的数。后半句里的“博士包含在内”不算。"""
    focus = focus_text(question)
    idx = focus.rfind("多少")
    if idx < 0:
        idx = focus.rfind("百分之")
    if idx < 0:
        return None
    window = focus[max(0, idx - 18) : idx]
    best, pos = "", -1
    for label in ("博士毕业生", "博士", "研究生", "本科生", "本科"):
        i = window.rfind(label)
        if i > pos:
            pos, best = i, label
    if best.startswith("博士"):
        return "博士"
    if best == "研究生":
        return "研究生"
    if best in {"本科生", "本科"}:
        return "本科"
    return None


def _sentence_at(text: str, pos: int) -> str:
    """一句或一条。按“第X条”切开，避免后一条的“重修”连坐前一条。"""
    left = max(text.rfind("。", 0, pos), text.rfind("；", 0, pos))
    marks = list(re.finditer(r"第[一二三四五六七八九十0-9]{1,3}条", text[: pos + 1]))
    if marks:
        left = max(left, marks[-1].start() - 1)
    right_period = text.find("。", pos)
    nxt = re.search(r"第[一二三四五六七八九十0-9]{1,3}条", text[pos + 1 :])
    rights = []
    if right_period >= 0:
        rights.append(right_period)
    if nxt:
        rights.append(pos + 1 + nxt.start())
    right = min(rights) if rights else min(len(text), pos + 90)
    return text[left + 1 : right]


def _column_hit(text: str, pos: int, metric: str) -> bool:
    """表头被排成一行时，按“数”切开的列序对上后面的数字。正文里的“人”不会走这条。"""
    left = max(0, pos - 36)
    head = text[left:pos]
    cut = max(head.rfind("。"), head.rfind("；"))
    if cut >= 0:
        head = head[cut + 1 :]
    if "数" not in head or re.search(r"[人%％]", head):
        return False
    head = re.sub(r"20\d{2}[-－]20\d{2}学年$", "", head)
    parts = [p for p in re.split(r"数", head) if re.search(r"[\u4e00-\u9fff]{2,}", p)]
    if len(parts) < 2 or not any(metric in p or p in metric for p in parts):
        return False
    idx = next(i for i, p in enumerate(parts) if metric in p or p in metric)
    start = text.rfind(parts[0], left, pos + 1)
    if start < 0:
        return False
    values = []
    for match in NUM_RE.finditer(text[start : pos + 16]):
        raw = match.group()
        if raw.isdigit() and len(raw) == 4 and raw.startswith("20"):
            continue
        values.append((start + match.start(), raw))
    return idx < len(values) and values[idx][0] == pos


def _anchored(question: str, docs: list[tuple[int, str, str]], terms: list[str]) -> list[tuple[float, str, str]]:
    """每个数字按它前面贴着的问法打分。页码、图号和另一类人群直接丢掉。"""
    metric = _metric(question)
    pop = _asked_pop(question)
    fq = focus_text(question)
    wants_people = any(k in question for k in ("多少人", "多少名", "总数", "总人数"))
    wants_rate = any(k in question for k in ("就业率", "落实率", "百分之"))
    broad = any(k in question for k in ("一共", "总共", "总数", "总人数", "总体规模")) and pop is None and "脚注" not in fq
    schoolwide = any(k in question for k in ("全校", "不要分", "统一的"))
    remedial = any(k in question for k in ("重修", "离校"))
    named = {
        num
        for a, b in re.findall(r"(\d+(?:\.\d+)?)[分人元个]?(?:与|和)(\d+(?:\.\d+)?)", question)
        for num in (a, b)
    }
    found: list[tuple[float, str, str]] = []
    for _, title, text in docs:
        for match in NUM_RE.finditer(text):
            raw = match.group()
            if raw.isdigit() and 1990 <= int(raw) <= 2035:
                continue
            before = text[max(0, match.start() - 42) : match.start()]
            after = text[match.end() : match.end() + 2]
            if before.endswith("·") and (after.startswith("·") or after[:1] not in "人名元%％"):
                continue
            if re.search(r"(报告|图表|图|表|附录|第)$", before[-6:]) and raw.isdigit() and after[:1] not in "人名元%％":
                continue
            if wants_people and after[:1] not in {"人", "名"}:
                continue
            if ("多少元" in question or "每人每年" in question) and "元" not in text[match.end() : match.end() + 8] and "每人每年" not in before:
                continue
            if metric and metric not in before and raw not in named:
                continue
            label = _nearest_label(before[-20:])
            if pop == "博士" and label != "博士":
                continue
            if pop == "研究生" and label != "研究生":
                continue
            if pop == "本科" and label not in {"本科", "本科生"}:
                continue
            if (broad or schoolwide) and label in {"研究生", "本科", "本科生", "博士"}:
                continue
            if broad and not re.search(r"(共|总数|总人数|总体规模|总毕业生)", before[-12:]):
                continue
            clause = _sentence_at(text, match.start())
            nearby = text[max(0, match.start() - 12) : match.start() + 100]
            if remedial and not any(k in nearby for k in ("重修", "离校")):
                continue
            # 第六条在学位办法里是补授，在奖助办法里是奖学金正文，只在问绩点时避开。
            if (
                clause.startswith("第六条")
                and "绩点" in question
                and "重修" not in question
                and "离校" not in question
                and "奖学金" not in question
                and "助学金" not in question
            ):
                continue
            if "免修" in clause and "免修" not in question:
                continue
            if "奖学金" in clause and "奖学金" not in question and "助学金" not in question and "励志" not in question:
                continue
            if "初次" in fq and "年终" not in fq:
                if "初次" not in before or before.rfind("年终") > before.rfind("初次"):
                    continue
            if "年终" in fq and "初次" not in fq and ("年终" not in before or before.rfind("初次") > before.rfind("年终")):
                continue
            score = 0.0
            for term in terms:
                if len(term) < 2:
                    continue
                pos = before.rfind(term)
                if pos < 0:
                    continue
                dist = len(before) - (pos + len(term))
                score += (len(term) ** 2) * (6 / (1 + dist))
            if metric:
                dist = len(before) - (before.rfind(metric) + len(metric))
                score += (len(metric) ** 2) * (8 / (1 + dist))
            if metric and _column_hit(text, match.start(), metric):
                score += 160
            if wants_people and after[:1] in {"人", "名"}:
                score += 24
            if "元" in question and "元" in after:
                score += 16
            if wants_rate and after[:1] in {"%", "％"}:
                score += 28
            if wants_rate and (float(raw) > 100 or ("%" not in clause and "％" not in clause)):
                score -= 30
            if "大约" in question or "约" in question:
                score += 36 if "约" in clause else -24
            asks_athlete = "运动员" in fq and not re.search(r"(还有|以及|例外).{0,12}运动员", fq)
            if "运动员" in before and not asks_athlete:
                continue
            if asks_athlete and "运动员" not in before:
                continue
            if "一般" in fq and "运动员" in clause:
                continue
            if ("高于" in fq or "下限" in fq) and before.endswith("低于"):
                score -= 24
            if ("高于" in fq or "下限" in fq) and "高于" in before[-4:]:
                score += 42
            if broad and any(k in before[-6:] for k in ("男生", "女生", "性别")):
                continue
            if remedial:
                score += 20
            if any(k in clause for k in ("免修", "先修")):
                score -= 48
            if any(k in question for k in ("不低于", "至少")) and "及以上" in clause:
                score += 28
            if "授予" in question and "学位" in question and "授予" in clause and "学位" in clause:
                score += 24
            if "低于" in question and before.endswith("低于") and "下限" not in fq and "高于" not in fq:
                score += 36
            if "优秀" in question and "优秀" in clause:
                score += 18
            for band in ("一档", "二档", "三档"):
                if band not in question:
                    continue
                if band in before[-10:]:
                    score += 90
                else:
                    score -= 35
            if "第五条" in question:
                score += 16 if "第五条" in clause else -12
            if "第二条" in question and "第五条" not in question:
                score += 12 if "第二条" in clause else 0
            if ("脚注" in fq or "专转本" in fq) and not any(
                k in clause for k in ("专转本", "分段培养", "脚注")
            ):
                continue
            if ("正文" in fq or "分析口径" in fq) and "脚注" not in fq and any(
                k in clause for k in ("专转本", "分段培养", "脚注", "目录")
            ):
                continue
            if "正文" in fq and ("调查" in clause or "调研" in clause) and "调查" not in fq and "调研" not in fq:
                continue
            elif "总人数" in question or "总体规模" in question:
                if any(k in clause for k in ("专转本", "分段培养")):
                    score -= 28
            if "总体规模" in fq and "调查" not in fq and "调研" not in fq:
                score += 22 if "总体规模" in clause or "总毕业生数" in clause else -16
                if "调查" in clause or "调研" in clause:
                    score -= 24
            if any(k in fq for k in ("调查", "调研")) and "总体规模" not in fq and "正文" not in fq:
                score += 22 if ("调查" in clause or "调研" in clause) else -18
            if any(k in before[-6:] for k in ("奖励", "西部")):
                score -= 40
            if score <= 0 and raw not in named:
                continue
            if raw in named:
                score += 1
            found.append((score, raw, clause))
    asked = re.search(r"第[一二三四五六七八九十0-9]{1,3}条", question)
    if asked:
        headed = [item for item in found if asked.group() in item[2][:12]]
        if headed:
            return headed
    # 问句用「59分与60分」点了两个数时，只留同时写着这两个数的那一句，并先报问句里较前的那个。
    if len(named) >= 2:
        both = [item for item in found if all(n in item[2] for n in named)]
        if both:
            first = min(named, key=lambda n: question.find(n))
            headed = [item for item in both if item[1] == first]
            if headed:
                return headed
    return found


def ground_answer(question: str) -> str | None:
    docs = _pool(question)
    if "末位" in question:
        for _, title, text in docs:
            i = text.find("末位同分")
            if i < 0:
                i = text.find("末位")
            if i >= 0 and ("入围" in text[i : i + 16] or "同分" in text[i : i + 12]):
                return text[i : i + 16]
    located = _phone(question, docs) or _address(question, docs)
    if located:
        return located
    years = re.findall(r"20\d{2}", question)
    employment = any(k in question for k in ("就业", "落实率", "届"))
    if not docs:
        if employment and years:
            titled = [d for d in _corpus() if "就业" in d[1] and any(y in d[1] for y in years)]
            if not titled:
                return REFUSE + "已入库的就业质量报告里没有这一届。"
        return None
    named_band = [b for b in ("一档", "二档", "三档") if b in question]
    asks_span = any(k in question for k in ("区间", "分别", "两份", "各档", "三个档"))
    if len(named_band) == 1 and not asks_span:
        band = named_band[0]
        for _, _, text in docs:
            start = 0
            while True:
                pos = text.find(band, start)
                if pos < 0:
                    break
                start = pos + len(band)
                if text[max(0, pos - 2) : pos] == "分为":
                    continue
                window = text[pos + len(band) : pos + len(band) + 14]
                found = NUM_RE.search(window)
                if not found or "为" not in window:
                    continue
                raw = found.group()
                if raw.isdigit() and 1990 <= int(raw) <= 2035:
                    continue
                return f"{raw}。"
    terms = _terms(question)
    wants_rate = any(k in question for k in ("就业率", "落实率", "百分之"))
    schoolwide = any(k in question for k in ("全校", "不要分", "统一的"))
    wants_many = any(k in question for k in ("分别", "两份", "各写", "不一致", "档", "区间"))
    ranked = _anchored(question, docs, terms)
    if not ranked:
        if schoolwide and wants_rate:
            return REFUSE + "报告没有给出不分学历的全校口径。"
        return None
    ranked.sort(key=lambda item: -item[0])
    best_score, best_num, best_sent = ranked[0]
    if any(k in question for k in ("上限", "最高", "下限", "最低")):
        nums = [
            n for n in NUM_RE.findall(best_sent)
            if not (n.isdigit() and 1990 <= int(float(n)) <= 2035)
        ]
        if nums:
            pick = max if any(k in question for k in ("上限", "最高")) else min
            best_num = pick(nums, key=float)
    if schoolwide and wants_rate and (
        "全校" in question and "全校" not in best_sent
        or not any(k in best_sent for k in ("学校", "总就业", "全校", "总体"))
        or any(k in best_sent for k in ("困难毕业生", "研究生为", "本科生为"))
    ):
        return REFUSE + "报告没有给出不分学历的全校口径。"
    if wants_many and "总体规模" in question and any(k in question for k in ("调查", "调研")):
        picked = []
        for key in ("总体规模", "总毕业生", "调查", "调研"):
            for _, num, sent in ranked:
                if key in sent and num not in picked:
                    picked.append(num)
                    break
        if len(picked) >= 2:
            return f"{'、'.join(picked[:4])}。{best_sent}。"
    named_band = [b for b in ("一档", "二档", "三档") if b in question]
    if wants_many and any(k in question for k in ("档", "区间")) and len(named_band) != 1:
        sent_nums = [n for n in NUM_RE.findall(best_sent) if not (n.isdigit() and 1990 <= int(float(n)) <= 2035)]
        if len(sent_nums) >= 2:
            return f"{best_sent}。"
    if wants_many:
        others = []
        for score, num, _ in ranked:
            if num != best_num and num not in others and score >= best_score * 0.55:
                others.append(num)
            if len(others) >= 3:
                break
        if others:
            return f"{best_num}。{best_sent}。另一处为{'、'.join(others)}。"
    return f"{best_num}。{best_sent}。"
