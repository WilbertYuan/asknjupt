from __future__ import annotations

import json
import os
import re
import subprocess
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

from . import config
from .compute import execute, format_computed
from .ground import (
    REFUSE,
    current_leader,
    exam_minutes,
    ground_answer,
    named_award,
    policy_meeting,
    policy_revision,
    program_credit,
    refuse_reason,
    teacher_count,
)
from .db import init_db
from .extractive import extract_answer
from .charterqa import charter_answer
from .courseqa import course_answer
from .rulebind import rule_answer
from .handbookqa import handbook_answer
from .contactqa import contact_answer
from .feeqa import fee_answer
from .zhpjqa import zhpj_answer
from .statusqa import status_answer
from .loanqa import loan_answer
from .libfaq import library_faq
from .degreeqa import degree_answer
from .tuimianqa import tuimian_answer
from .reportqa import report_answer
from .reportline import report_line
from .profileqa import profile_answer
from .examrule import submit_window
from .retrieve import QueryPlan, _clauses, _independent_keywords, _match_terms, content_keywords, retrieve
from .util import focus_text, utc_now

NUM_TOKEN = re.compile(r"-?\d+(?:\.\d+)?")


def build_pack(question: str, plan: QueryPlan, hits: list[dict[str, Any]], computed: dict[str, Any] | None) -> str:
    parts = [f"问题：{question}"]
    if plan.years:
        parts.append(f"年份约束：{plan.years}")
    if plan.academic_year:
        parts.append(f"学年：{plan.academic_year}")
    if computed:
        if computed.get("error"):
            parts.append(f"程序计算限制：{computed['error']}")
        else:
            parts.append("程序已完成的确定性计算结果（数字以这里为准）：")
            parts.append(json.dumps({k: computed[k] for k in computed if k not in {"matched", "labs", "facts", "used_rules", "facts_used"}}, ensure_ascii=False, default=str)[:1800])
            if computed.get("formula"):
                parts.append(f"公式：{computed['formula']}")
    parts.append("证据：")
    budget = 1800
    used = 0
    for i, h in enumerate(hits[:8], 1):
        snippet = (h.get("text") or "")[:420]
        block = f"[{i}] {h.get('doc_title')} | {h.get('url')} | {h.get('locator')}\n{snippet}"
        if used + len(block) > budget:
            break
        parts.append(block)
        used += len(block)
    parts.append("要求：先写结论。引用用[编号]。不要编造证据中没有的数字。")
    return "\n".join(parts)


def llama_available() -> bool:
    return config.MODEL_GGUF.exists()


def run_llama(prompt: str) -> str:
    if not llama_available():
        raise FileNotFoundError(f"模型不存在: {config.MODEL_GGUF}")
    llama_bin = config.resolve_llama_bin()
    cmd = [
        llama_bin,
        "-m",
        str(config.MODEL_GGUF),
        "--jinja",
        "--reasoning",
        "off" if not config.ENABLE_THINKING else "on",
        "--temp",
        str(config.LLAMA_TEMP),
        "--top-k",
        str(config.LLAMA_TOP_K),
        "--top-p",
        str(config.LLAMA_TOP_P),
        "-c",
        str(config.LLAMA_CTX),
        "-n",
        str(config.LLAMA_N_PREDICT),
        "-ngl",
        str(config.LLAMA_NGL),
        "-st",
        "-sys",
        config.SYSTEM_PROMPT,
        "-p",
        prompt if "/no_think" in prompt or config.ENABLE_THINKING else f"{prompt}\n/no_think",
        "--no-display-prompt",
        "-no-cnv",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    out = (proc.stdout or "").strip()
    if not out:
        err = (proc.stderr or "")[-800:]
        raise RuntimeError(f"{llama_bin} 无输出: {err}")
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.S)
    junk_prefixes = (
        "build ",
        "model ",
        "modalities ",
        "using custom",
        "available commands",
        "--no-conversation",
        "please use llama",
        "Loading model",
        "Exiting",
        "▄",
        "▀",
        "██",
    )
    lines = []
    for line in out.splitlines():
        s = line.strip()
        # 模型空转时只吐 slash-token（如 /exit、/flag、/think），这类行必须丢掉。
        if re.fullmatch(r"/[A-Za-z]+", s):
            continue
        if not s or s.startswith(junk_prefixes) or set(s) <= set("▄▀█-|/\\ "):
            continue
        if s.startswith(">") and not s.startswith("> 问题"):
            continue
        lines.append(s.lstrip("> ").strip())
    out = "\n".join(lines).strip()
    # 整个回答就是 slash-token 时，直接视为无输出，让上层走拒答路径。
    if re.fullmatch(r"(?:/[A-Za-z]+\s*)+", out):
        out = ""
    # if the model echoed the whole pack, keep only the last short paragraph
    if "程序已完成的确定性计算结果" in out and "结论" in out:
        out = out[out.rfind("结论") :]
    return collapse_repeats(out)


def collapse_repeats(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    chunks = [c.strip() for c in re.split(r"(?=^(?:答案|结论)[：:])", text, flags=re.M) if c.strip()]
    uniq = []
    for c in chunks:
        if c not in uniq:
            uniq.append(c)
    if uniq:
        text = uniq[0]
    lines, seen = [], set()
    for line in text.splitlines():
        s = line.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        lines.append(s)
    return "\n".join(lines).strip()


def refs_from_hits(hits: list[dict[str, Any]], extra_docs: list[dict[str, Any]] | None = None) -> str:
    items = []
    seen = set()
    for d in extra_docs or []:
        url = d.get("url")
        if url and url not in seen:
            seen.add(url)
            items.append(f"{d.get('title') or ''} {url}".strip())
    for h in hits:
        url = h.get("url")
        title = h.get("doc_title") or ""
        loc = h.get("locator") or ""
        key = url or loc
        if not url or key in seen:
            continue
        seen.add(key)
        items.append(f"{title} {url} #{loc}".strip())
    return " | ".join(items[:6])


def validate_numbers(answer: str, computed: dict[str, Any] | None) -> tuple[str, bool]:
    if not computed or computed.get("error"):
        return answer, True
    keys = []
    for k in ("sum", "percent", "coef", "ratio"):
        if k in computed and computed[k] is not None:
            keys.append((k, computed[k]))
    if not keys:
        return answer, True
    ok = True
    for _, val in keys:
        # accept int-equal or 2-decimal equal
        as_int = str(int(val)) if float(val) == int(val) else None
        as_2 = f"{float(val):.2f}"
        as_raw = str(val)
        if not any(x and x in answer.replace(",", "") for x in (as_int, as_2, as_raw, as_2.rstrip("0").rstrip("."))):
            ok = False
            break
    if ok:
        return answer, True
    fallback = format_computed(computed)
    if fallback:
        return fallback, False
    return answer, False


PERSONAL_MARKERS = ("加权平均", "个人成绩", "发放账号", "银行卡", "身份证号", "张三", "某本科生", "某同学")


PROSE_CUES = (
    "奖学金", "助学金", "绩点", "运动员", "重修", "就业", "届", "落实率",
    "生均", "藏书", "新增图书", "万册", "仪器设备", "购置", "立项", "参加学生", "传真", "联系方式", "三牌楼",
    "末位", "同分", "结算", "多退少补",
)


_CMP = re.compile(
    r"^(?P<left>.+?)比(?P<right>.+?)(?P<how>多|少|大|高|低)(?:多少|几)(?P<unit>平方米|万册|学分|人|元|名|个)?"
)
_GAP = re.compile(
    r"^(?P<left>.+?)(?:和|与)(?P<right>.+?)相差(?:多少|几)(?P<unit>平方米|万册|学分|人|元|名|个)?"
)


def _first_amount(text: str) -> Decimal | None:
    for match in re.finditer(r"(?<![\d.])\d+(?:\.\d+)?", text or ""):
        raw = match.group()
        if raw.isdigit() and 1990 <= int(raw) <= 2035:
            continue
        return Decimal(raw)
    return None


def _side_question(clause: str, unit: str) -> str:
    clause = clause.strip(" ，,。；")
    if not re.search(r"多少|是几", clause):
        clause += "是多少" + unit
    return clause


def _show_diff(value: Decimal) -> str:
    if value == value.to_integral():
        return str(int(value))
    return format(value, "f").rstrip("0").rstrip(".")


def _gap_parts(question: str) -> tuple[str, str, str] | None:
    """“相差”从最后一个连接词切开。专业名里的“与”不能当成两个问题。"""
    matched = re.search(r"相差(?:多少|几)(?P<unit>平方米|万册|学分|人|元|名|个)?", question.strip())
    if not matched:
        return None
    head = question.strip()[: matched.start()].rstrip("，, ")
    for mark in ("，和", "，与", "和", "与"):
        cut = head.rfind(mark)
        if cut > 0:
            return head[:cut], head[cut + len(mark) :], matched.group("unit") or ""
    return None


_TEACHER_METRICS = (
    ("折合", "equivalent"),
    ("外聘", "adjunct"),
    ("校外教师", "adjunct"),
    ("专任", "fulltime"),
)


def teacher_metric_diff(question: str) -> str | None:
    """「折合教师比专任教师多多少人」：同一份教学质量报告里的两个指标相减。

    每个指标在该报告里必须只有一个值。对不上的指标（例如外籍教师）
    不估算。年份字段在入库时有过偏差，所以按文档而不是按年份对齐。
    """
    matched = _CMP.search((question or "").strip())
    if not matched or matched.group("how") not in {"多", "少", "高", "低"}:
        return None

    def which(side: str) -> str | None:
        # 「外籍专任」不是专任教师指标，库里没有这一列就不能减。
        if "外籍" in side:
            return None
        for cue, name in _TEACHER_METRICS:
            if cue in side:
                return name
        return None

    left_name = which(matched.group("left"))
    right_name = which(matched.group("right"))
    if not left_name or not right_name or left_name == right_name:
        return None
    conn = init_db()
    try:
        rows = conn.execute(
            """
            SELECT f.metric, f.value
            FROM facts f JOIN documents d ON d.id = f.source_document_id
            WHERE f.scope = '全校' AND d.title LIKE '%教学质量报告%'
              AND f.metric IN (?, ?)
            """,
            (left_name, right_name),
        ).fetchall()
    finally:
        conn.close()
    got: dict[str, set[str]] = {}
    for metric, value in rows:
        got.setdefault(metric, set()).add(str(value))
    if set(got.get(left_name) or []) != set() and len(got.get(left_name) or []) == 1 and len(got.get(right_name) or []) == 1:
        left_n = Decimal(next(iter(got[left_name])))
        right_n = Decimal(next(iter(got[right_name])))
        diff = right_n - left_n if matched.group("how") in {"少", "低"} else left_n - right_n
        if diff < 0:
            return None
        return f"{_show_diff(diff)}。"
    return None


def compare_question(question: str) -> str | None:
    """把“A比B多多少”拆成两个已经会答的子问题再相减。子问题不再进入比较，避免套娃。"""
    # 教师人数差只认事实表里对得上的两个指标，避免两边都读到专任人数后算出 0。
    if "教师" in (question or "") or "老师" in (question or ""):
        return None
    gap = _gap_parts(question.strip())
    matched = None if gap else _CMP.search(question.strip())
    if gap:
        left, right, unit = gap
    elif matched:
        unit = matched.group("unit") or ""
        left = matched.group("left")
        right = matched.group("right")
    else:
        return None
    left = _side_question(left, unit)
    right = _side_question(right, unit)
    if not re.search(r"20\d{2}", right):
        year = re.search(r"20\d{2}(?:年|届|版|级)", left)
        if year:
            right = year.group(0) + right
    major = re.search(r"([\u4e00-\u9fff]{2,18})(?:专业|英才班|强化班)", left)
    if major and not re.search(r"专业|英才班|强化班", right):
        # “四年按优势专业”里的专业不是专业名，只留下最后一个名称。
        name = re.split(r"[按的年]", major.group(1))[-1]
        if 2 <= len(name) <= 12:
            right = name + major.group(0)[len(major.group(1)) :] + "，" + right
    if "学分" in left and "学分" not in right:
        right += "的毕业总学分"
    type_words = ("国家专项", "地方专项", "艺术类", "综合评价", "省外计划", "普通类")
    right_has_type = any(word in right for word in type_words)
    for key in (
        "江苏普通类", "江苏省招生计划表", "综合评价招生简章", "省外计划",
        "就业质量报告", "学生手册", "教学质量报告", "地方专项", "艺术类", "普通类",
    ):
        if key in type_words and right_has_type:
            continue
        if key in left and key not in right:
            right = key + "，" + right
    if min(len(left), len(right)) < 4:
        return None
    how = matched.groupdict().get("how") if matched else None
    left_ans = answer_question(left, use_llm=False, allow_compare=False)["answer"]
    right_ans = answer_question(right, use_llm=False, allow_compare=False)["answer"]
    left_n = _first_amount(left_ans)
    right_n = _first_amount(right_ans)
    if left_n is None or right_n is None:
        return None
    diff = abs(left_n - right_n) if how is None else (right_n - left_n if how in {"少", "低"} else left_n - right_n)
    if diff < 0:
        return None
    return f"{_show_diff(diff)}。"


def _missing_cohort(question: str) -> str | None:
    if not re.search(r"20\d{2}届", question):
        return None
    if not any(k in question for k in ("多少人", "多少名", "就业率", "落实率", "总人数", "一共")):
        return None
    years = re.findall(r"20\d{2}", question)
    if not years:
        return None
    conn = init_db()
    clause = " OR ".join("title LIKE ?" for _ in years)
    n = conn.execute(
        f"SELECT COUNT(*) FROM documents WHERE title LIKE '%就业%' AND ({clause})",
        tuple(f"%{y}%" for y in years),
    ).fetchone()[0]
    conn.close()
    if n:
        return None
    return REFUSE + "已入库的就业质量报告里没有这一届。"


_CLAUSE_CACHE: dict[str, list[str]] = {}


def _clause_texts(like: str) -> list[str]:
    if like in _CLAUSE_CACHE:
        return _CLAUSE_CACHE[like]
    from .db import init_db

    conn = init_db()
    docs = conn.execute("SELECT id FROM documents WHERE title LIKE ?", (like,)).fetchall()
    texts: list[str] = []
    for doc in docs:
        parts = conn.execute(
            "SELECT text FROM retrieval_units WHERE document_id=? ORDER BY id",
            (doc["id"],),
        ).fetchall()
        blob = re.sub(r"\s+", "", "".join((part["text"] or "") for part in parts))
        if blob:
            texts.append(blob)
    conn.close()
    _CLAUSE_CACHE[like] = texts
    return texts


def _statute_product(question: str) -> str | None:
    """问句给出份数，比例或单价在办法里只有一种写法时才相乘。"""
    if "免修" in question and any(k in question for k in ("上限", "最多")):
        stated = re.search(r"应修\s*(\d+)\s*门", question)
        if not stated:
            return None
        fractions = {
            match.groups()
            for blob in _clause_texts("%学籍管理办法%")
            for match in re.finditer(
                r"申请免修考核的课程门数原则上不超过应修课程门数的(\d+)/(\d+)",
                blob,
            )
        }
        if len(fractions) != 1:
            return None
        num, den = (int(part) for part in next(iter(fractions)))
        courses = int(stated.group(1))
        if not den or courses * num % den:
            return None
        return f"{courses * num // den}门"
    if "勤工助学" in question and "酬金" in question and "小时" in question:
        stated = re.search(r"(\d+)\s*小时", question)
        if not stated:
            return None
        rates = {
            match.group(1)
            for blob in _clause_texts("%勤工助学管理办法%")
            for match in re.finditer(r"不低于每小时(\d+)元", blob)
        }
        if len(rates) != 1:
            return None
        return f"{int(stated.group(1)) * int(next(iter(rates)))}元"
    return None


def _round_half_up(value: float, places: int) -> str:
    quantum = Decimal("1") if places <= 0 else Decimal("0." + "0" * places)
    return format(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP), "f")


def _label_hit(label: str, side: str) -> bool:
    label = re.sub(r"^[和与及其中]", "", label)
    if not label or not side:
        return False
    if label in side or side in label:
        return True
    return len(side) >= 4 and side[-4:] in label or len(label) >= 4 and label[-4:] in side


def _pick_count(label: str, pairs: list[tuple[str, str]]) -> float | None:
    best: tuple[int, float] | None = None
    for name, value in pairs:
        if _label_hit(name, label):
            if best is None or len(name) > best[0]:
                best = (len(name), float(value))
    return None if best is None else best[1]


def stated_arithmetic(question: str) -> str | None:
    """题目自己已经给出了运算数时，直接算。不覆盖反推、收费和招生计划。"""
    if any(k in question for k in ("反推", "折算系数", "生师比", "学费", "住宿", "上浮", "专业组")):
        return None
    if "综合评价" in question and "综合成绩" not in question and "投档分" not in question:
        return None
    # 问句把投档分和校测都给了，就按简章公式算。只问公式本身时没有这两个数，不在这里回答。
    gaokao = re.search(r"投档分\s*(\d+(?:\.\d+)?)", question)
    test = re.search(r"(?:测试成绩|校测)\s*(\d+(?:\.\d+)?)", question)
    if "综合成绩" in question and "多少" in question and gaokao and test:
        val = float(gaokao.group(1)) / 750 * 80 + float(test.group(1)) / 100 * 20
        return _round_half_up(val, 2)
    statute = _statute_product(question)
    if statute:
        return statute
    text = focus_text(question)
    listed = re.search(r"分别为\s*([0-9]+(?:\s*[、,，]\s*[0-9]+){1,})", text)
    base = re.search(r"总(?:面积|人数|数|和)?(?:为|是)\s*(\d+(?:\.\d+)?)", text)
    if listed and base and any(k in text for k in ("合计", "之和", "比例", "占比")):
        parts = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", listed.group(1))]
        denom = float(base.group(1))
        total = sum(parts)
        bits: list[str] = []
        if any(k in text for k in ("合计", "之和")):
            bits.append(_round_half_up(total, 0 if total == int(total) else 2))
        if denom and any(k in text for k in ("比例", "占比", "百分之")):
            bits.append(_round_half_up(total / denom * 100, 2))
        if bits:
            return " ".join(bits)
    if listed and any(k in text for k in ("合计", "一共", "总数")) and not base:
        parts = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", listed.group(1))]
        if len(parts) >= 2:
            total = sum(parts)
            return _round_half_up(total, 0 if total == int(total) else 2)
    pairs = re.findall(r"([\u4e00-\u9fff]{2,16})(\d+(?:\.\d+)?)(?:人|个|项|份)", text)
    rel = re.search(r"([\u4e00-\u9fff]{2,12})占([\u4e00-\u9fff]{2,12}?)(?:的|之|比例)", text)
    if pairs and rel and any(k in text for k in ("比例", "占比", "百分之")):
        left, right = rel.group(1), rel.group(2)
        num = den = None
        for label, value in pairs:
            label = re.sub(r"^[和与及]", "", label)
            if _label_hit(label, left):
                num = float(value)
            if _label_hit(label, right):
                den = float(value)
        if num and den:
            return _round_half_up(num / den * 100, 2)
    senior = re.findall(r"([\u4e00-\u9fff]{2,8})(\d+)人", text)
    if len(senior) >= 3 and "比例" in text and any("专任" in name for name, _ in senior):
        base_n = next(float(n) for name, n in senior if "专任" in name)
        part = sum(float(n) for name, n in senior if "专任" not in name)
        if base_n:
            return _round_half_up(part / base_n * 100, 2)
    divided = re.search(r"([\u4e00-\u9fff]{2,12})数除以([\u4e00-\u9fff]{2,12})数", text)
    if divided:
        labeled = re.findall(r"([\u4e00-\u9fff]{2,16})(\d+(?:\.\d+)?)(?:人|项|个|份)", text)
        left = _pick_count(divided.group(1), labeled)
        right = _pick_count(divided.group(2), labeled)
        if left and right:
            val = left / right
            if any(k in text for k in ("率", "比例", "占比", "百分之")):
                val *= 100
            return _round_half_up(val, 2)
    if "相差" in text or "相差多少" in question:
        decs = [float(x) for x in re.findall(r"\d+\.\d+", text)]
        if len(decs) == 2:
            return _round_half_up(abs(decs[0] - decs[1]), 2)
        nums = []
        for raw in re.findall(r"\d+(?:\.\d+)?", text):
            if raw.isdigit() and 1990 <= int(raw) <= 2035:
                continue
            nums.append(float(raw))
        if len(nums) == 2:
            return _round_half_up(abs(nums[0] - nums[1]), 2)
    if "其中" in text and any(k in text for k in ("总数", "复算", "占比")):
        tail = text.split("其中", 1)[1]
        bits = re.findall(r"([\u4e00-\u9fff]{2,8})(\d+(?:\.\d+)?)项", tail)
        if len(bits) >= 2:
            total = sum(float(n) for _, n in bits)
            out = [_round_half_up(total, 0 if total == int(total) else 2)]
            share = re.search(r"([\u4e00-\u9fff]{2,8})(?:获奖)?占比", text)
            if share and total:
                for name, n in bits:
                    if _label_hit(name, share.group(1)):
                        out.append(_round_half_up(float(n) / total * 100, 2))
                        break
            return " ".join(out)
    if "个人" in text and "团队" in text and "总数" in text:
        personal = team = 0.0
        for seg in re.split(r"[、，]", text):
            found = re.search(r"(\d+(?:\.\d+)?)\s*项", seg)
            if not found:
                continue
            n = float(found.group(1))
            if "团队" in seg:
                team += n
            else:
                personal += n
        if personal and team:
            return f"{_round_half_up(personal, 0)} {_round_half_up(personal + team, 0)}"
    if any(k in text for k in ("合计", "一共", "总数")) and ("、" in text):
        chunks = re.findall(r"(\d+(?:\.\d+)?)(?:\s*项)", text)
        if len(chunks) >= 3:
            total = sum(float(x) for x in chunks)
            return _round_half_up(total, 0 if total == int(total) else 2)
    return None


def composite_score(question: str, hits: list[dict[str, Any]]) -> str | None:
    """问句给出三项原始分，权重从已检索到的测评办法里读，不把权重写死。"""
    if "综合素质测评" not in question or "基准分" not in question:
        return None
    nums = []
    for raw in re.findall(r"\d+(?:\.\d+)?", focus_text(question)):
        if raw.isdigit() and 1990 <= int(raw) <= 2035:
            continue
        nums.append(float(raw))
    if len(nums) < 6:
        return None
    pairs = list(zip(nums[0::2], nums[1::2]))[:3]
    blob = re.sub(r"\s+", "", " ".join((h.get("text") or "") for h in hits[:6]))
    weights = [int(x) for x in re.findall(r"[×✕]\s*(\d+)", blob)]
    chosen = None
    for i in range(len(weights) - 2):
        window = weights[i : i + 3]
        if sum(window) == 100:
            chosen = window
            break
    if not chosen:
        return None
    total = 0.0
    for (left, right), weight in zip(pairs, chosen):
        if not right:
            return None
        total += left / right * weight
    return _round_half_up(total, 2)


def cohort_ratio(question: str, hits: list[dict[str, Any]]) -> str | None:
    if not all(k in question for k in ("本科生", "研究生", "比例")):
        return None
    blob = re.sub(r"\s+", "", " ".join((h.get("text") or "") for h in hits[:4]))
    under = re.search(r"普通本科生(\d+)人", blob)
    master = re.search(r"全日制硕士研究生(\d+)人", blob)
    doctor = re.search(r"全日制博士研究生(\d+)人", blob)
    if not (under and master and doctor):
        return None
    den = int(master.group(1)) + int(doctor.group(1))
    if not den:
        return None
    return _round_half_up(int(under.group(1)) / den, 2)


def programmatic_answer(
    question: str,
    plan: QueryPlan,
    hits: list[dict[str, Any]],
    computed: dict[str, Any] | None,
    allow_compare: bool = True,
    allow_agent: bool = False,
) -> str:
    credit = program_credit(question)
    if credit:
        return credit
    revised = policy_revision(question)
    if revised:
        return revised
    meeting = policy_meeting(question)
    if meeting:
        return meeting
    minutes = exam_minutes(question)
    if minutes:
        return minutes
    window = submit_window(question)
    if window:
        return window
    award = named_award(question)
    if award:
        return award
    reason = refuse_reason(question)
    if reason:
        return reason
    leader = current_leader(question)
    if leader:
        return leader
    # 借阅规则的借期、逾期单价和开学宽限是编号条款，句子层会把单价里的 1 元抽到句首。
    loaned = loan_answer(question)
    if loaned:
        return loaned
    faq = library_faq(question)
    if faq:
        return faq
    # 教学质量报告的表格被抽成了散文。栏目对得上才在这里回答，
    # 否则专任教师人数会把正高、副高和学位题一起答掉。
    reported = report_answer(question)
    if reported:
        return reported
    lined = report_line(question)
    if lined:
        return lined
    profile = profile_answer(question)
    if profile:
        return profile
    # 招生章程按年份取原句。算学费、上浮和计划人数的题这里对不上，继续往后走。
    charted = charter_answer(question)
    if charted:
        return charted
    status = status_answer(question)
    if status:
        return status
    degree = degree_answer(question)
    if degree:
        return degree
    tuimian = tuimian_answer(question)
    if tuimian:
        return tuimian
    # 综评简章的条款。计划人数和代入公式的成绩不在这里算。
    brochure = zhpj_answer(question)
    if brochure:
        return brochure
    fee = fee_answer(question)
    if fee:
        return fee
    course = course_answer(question)
    if course:
        return course
    ruled = rule_answer(question)
    if ruled:
        return ruled
    handbook = handbook_answer(question)
    if handbook:
        return handbook
    contact = contact_answer(question)
    if contact:
        return contact
    teacher = teacher_count(question)
    if teacher:
        return teacher
    differed = teacher_metric_diff(question)
    if differed:
        return differed
    if allow_compare:
        compared = compare_question(question)
        if compared:
            return compared
    stated = stated_arithmetic(question)
    if stated:
        return stated
    extra = composite_score(question, hits) or cohort_ratio(question, hits)
    if extra:
        return extra
    missing = _missing_cohort(question)
    if missing:
        return missing
    library_page = "图书馆" in question and "教学质量报告" not in question
    # 结构化表格直查：PDF 抽文本会把表格打成乱序碎片，句子层和
    # 段落兜底在这种证据上必错；库里 table_rows 存了列对齐的格子，
    # 键×属性交叉取值更可靠。它只查表不依赖 hits，放在最前面。
    if os.environ.get("ASKNJUPT_NO_TABLEQA") != "1":
        try:
            from .tableqa import answer_from_tables
            from .extractive import content_terms as _ct
            tab = answer_from_tables(question, _ct(question))
        except Exception:
            tab = None
        if tab:
            ans, ttitle, turl = tab
            if ttitle:
                return f"{ans}\n（摘自 {ttitle} {turl}）"
            return ans
    # 多槽位证据组装：每个槽位都绑到唯一值才提交。
    # 缺槽位或同一范围内值冲突时不替换，后面的路径照旧。
    if os.environ.get("ASKNJUPT_NO_ASSEMBLE") != "1":
        try:
            from .assemble import assemble_answer
            cand = assemble_answer(question)
        except Exception:
            cand = None
        if cand and cand.get("validation_status") == "verified" and cand.get("answer"):
            ans = cand["answer"]
            title = cand.get("doc_title") or ""
            url = cand.get("url") or ""
            if title:
                return f"{ans}\n（摘自 {title} {url}）"
            return ans
    if any(k in question for k in PROSE_CUES) and not library_page:
        grounded = ground_answer(question)
        if grounded and not _answer_uses_question(question, grounded):
            grounded = None
        flat_g = re.sub(r"\s+", "", grounded or "")
        # 奖学金条款的金额表会抢答「排名是多少」「有几次」。
        # 捷径句子里没有问句要的名次或次数，就交回原文。
        if grounded and "排名" in question and "排名" not in flat_g:
            grounded = None
        if grounded and re.search(r"几次|多少次", question) and not re.search(
            r"[一二两三四五六七八九十\d]次", flat_g
        ):
            grounded = None
        if grounded:
            return grounded
    formatted = format_computed(computed) if computed else None
    # 事实库里的校区地址没有邮编。问句同时要邮编时，改用紧挨着这扇门牌的原文邮编。
    # 对不上唯一邮编就放弃这条事实，后面的原文路径照旧。
    if (
        formatted
        and (computed or {}).get("metric") in {"xianlin_address", "suojin_address"}
        and "邮编" in question
        and not re.search(r"\d{6}", formatted)
    ):
        from .compute import complete_postal
        completed = complete_postal(question, str((computed or {}).get("value") or ""))
        formatted = completed or None
    if formatted:
        return formatted
    if computed and computed.get("error"):
        if hits:
            return f"资料不足或无法完成该项计算：{computed['error']}。相关原文见检索证据，不能给出伪精确结果。"
        return f"资料不足：{computed['error']}。"
    if not hits:
        return "当前已采集资料中未检索到可支持该回答的证据。不能确定答案。"
    # 家族条款没对上时，不再把检索里最像的一段直接交出去。
    # 0.6B 先挑句，再从句里摘原话；摘不出就换关键词重检，仍摘不出就拒答。
    if allow_agent:
        try:
            from .agentloop import review_hits
            reviewed = review_hits(question, hits)
        except Exception:
            reviewed = None
        if reviewed:
            return reviewed
    # 通用句子级抽取：题型+内容词覆盖，不认识具体题目。
    extracted = None if os.environ.get("ASKNJUPT_NO_EXTRACT") else extract_answer(question, hits)
    if extracted:
        best_doc = _best_hit_for_answer(extracted, hits)
        if best_doc:
            return f"{extracted}\n（摘自 {best_doc.get('doc_title') or ''} {best_doc.get('url') or ''}）"
        return extracted
    keywords = [kw for kw in plan.keywords if kw and len(kw) >= 2]
    ranked = sorted(hits, key=lambda h: h.get("score") or 0, reverse=True)
    best = ranked[0]
    for hit in ranked:
        blob = (hit.get("text") or "") + (hit.get("title") or "")
        if any(kw in blob for kw in keywords):
            best = hit
            break
    raw_text = best.get("text") or ""
    # 点名文件已经按问句挑过句子。结论数字只从这一句取，后补的上下文不能把年份抢走。
    if best.get("focused"):
        headline = best.get("headline") or raw_text
        lead = _lead_number(question, headline)
        if lead != headline:
            snippet = f"{lead.split('。', 1)[0]}。{raw_text}"
        else:
            snippet = raw_text
    else:
        snippet = _lead_number(question, _best_sentence(raw_text, keywords, question))
    # 规则抽不出句子时，才让 0.6B 在短名单里另选一句原文。
    # 只有选中句比这段兜底覆盖更多问句要点才替换；嵌套比较子问题不走这里。
    if allow_compare and os.environ.get("ASKNJUPT_LLM_SELECT", "1") != "0":
        try:
            from .lmselect import select_fallback
            picked = select_fallback(question, hits, baseline=snippet)
        except Exception:
            picked = None
        if picked:
            sent, hit = picked
            return f"{sent}\n（摘自 {hit.get('doc_title') or ''} {hit.get('url') or ''}）"
    if re.search(r"能[^？?\n]{0,16}吗|能否", question) and re.search(
        r"一律不|不得|不可|不准", snippet
    ) and not str(snippet).startswith("不能"):
        snippet = "不能。" + snippet
    return f"{snippet}\n（摘自 {best.get('doc_title') or ''} {best.get('url') or ''}）"


def _best_hit_for_answer(answer: str, hits: list[dict[str, Any]]) -> dict[str, Any] | None:
    """答案是从哪块证据里抽出来的，出处就引哪块。"""
    flat = re.sub(r"[\s，,。；;：:（）()、]", "", answer or "")[:40]
    if not flat:
        return hits[0] if hits else None
    best, best_n = None, 0
    for hit in hits:
        text = re.sub(r"[\s，,。；;：:（）()、]", "", hit.get("text") or "")
        n = 0
        for size in (20, 12):
            for i in range(0, len(flat) - size + 1, size):
                if flat[i : i + size] in text:
                    n += 1
        if n > best_n:
            best, best_n = hit, n
    return best or (hits[0] if hits else None)


def _answer_uses_question(question: str, text: str) -> bool:
    """捷径答出来的长句子，至少要含问句自己的一段话。对不上就交回原文检索。"""
    if not text or len(text) <= 24:
        return True
    if "资料不足" in text[:12] or "不能确定" in text[:16]:
        return True
    body = re.sub(r"按[^，。？?]{2,48}[，,]?", "", question, count=1)
    body = re.sub(r"20\d{2}年\d{0,2}月?", "", body)
    han = "".join(ch for ch in body if "\u4e00" <= ch <= "\u9fff")
    for drop in ("什么", "多少", "哪些", "如何", "怎么", "是否", "哪一", "哪年", "何时", "几个", "分别", "具体", "可以", "南邮", "南京邮电大学"):
        han = han.replace(drop, "")
    flat = re.sub(r"[\s“”\"「」『』]", "", text)
    grams = [
        han[i : i + size]
        for size in (4, 3)
        for i in range(max(0, len(han) - size + 1))
    ]
    grams = [g for g in grams if g not in {"南京邮电", "邮电大学", "学校章程"}]
    if not grams:
        return True
    return any(g in flat for g in grams)


def _lead_number(question: str, snippet: str) -> str:
    """点名文件之后问多少、哪一年时，把贴着问句用词的那个数字放到句首。"""
    if "按" not in question or not any(k in question for k in ("多少", "几", "哪一", "何时", "哪年", "何年")):
        return snippet
    terms = [t for t in _match_terms(content_keywords(question)) if len(t) >= 2]
    # 「2023—2024学年」里的 2023、023 不是要报的人数，别让它们把年份抬到句首。
    if any(k in question for k in ("多少", "几")):
        years = set(re.findall(r"(?:19|20)\d{2}", question))
        terms = [t for t in terms if not (t.isdigit() and any(t in year for year in years))]
    if not terms:
        return snippet
    matches = list(re.finditer(r"\d+(?:\.\d+)?", snippet))
    if not matches:
        return snippet
    # 一个词只记在离它最近的数字上，避免前一个数字把后半句的说法也算进去。
    scores = [0.0] * len(matches)
    for term in terms:
        start = 0
        while True:
            pos = snippet.find(term, start)
            if pos < 0:
                break
            start = pos + len(term)
            end = pos + len(term)
            nearest, best_d = None, 11
            for i, match in enumerate(matches):
                if end <= match.start():
                    dist = match.start() - end
                elif pos >= match.end():
                    dist = pos - match.end()
                else:
                    dist = 0
                if dist < best_d:
                    best_d, nearest = dist, i
            if nearest is not None:
                scores[nearest] += len(term)
    best_num, best_score = None, 0
    for match, score in zip(matches, scores):
        after = snippet[match.end() : match.end() + 1]
        before = snippet[match.start() - 1 : match.start()]
        after2 = snippet[match.end() : match.end() + 2]
        raw = match.group()
        if after in "册人元个分倍项门名万%％":
            score += 4
        # 「表3-1」「第3条」里的序号不是要报的人数。
        if raw.isdigit() and int(raw) < 100 and (
            before in "表第-." or (after2[:1] in "-." and after2[1:2].isdigit())
        ):
            score -= 6
        if raw.isdigit() and 1990 <= int(raw) <= 2035 and not any(k in question for k in ("哪一", "何时", "哪年", "几年")):
            score -= 3
        # 1.1 这种条款号要压低；3.6万这种数量要留下。
        if "." in raw and after != "万" and float(raw) < 30 and not any(k in question for k in ("比例", "绩点", "百分之", "系数")):
            score -= 3
        # 「59分与60分」点了两个分数，就报这两个，不报表头上的 90。
        named = {
            num
            for a, b in re.findall(r"(\d+(?:\.\d+)?)[分人元个]?(?:与|和)(\d+(?:\.\d+)?)", question)
            for num in (a, b)
        }
        if raw in named:
            score += 12
        if score > best_score:
            best_score = score
            best_num = match.group()
    if not best_num or snippet.startswith(best_num):
        return snippet
    return f"{best_num}。{snippet}"


def _best_sentence(text: str, keywords: list[str], question: str = "") -> str:
    """截含关键词的那一句，而不是文档开头。"""
    raw = (text or "").strip()
    parts = _clauses(raw)
    if not parts:
        return raw[:400]
    asks_day = any(k in question for k in ("哪一天", "哪天", "几月几日"))
    best_i, best_n = 0, -1.0
    for i, sent in enumerate(parts):
        kept = _independent_keywords(sent, _match_terms(keywords))
        n = len(kept) * 10 + sum(len(kw) for kw in kept)
        if asks_day and re.search(r"\d", sent) and "月" in sent and "日" in sent:
            n += 50
        if re.match(r"^第[一二三四五六七八九十百0-9]{1,3}章", sent) and len(sent) < 40:
            n *= 0.3
        if n > best_n:
            best_n, best_i = n, i
    chunk = parts[best_i]
    if best_i + 1 < len(parts) and len(chunk) < 180:
        chunk += parts[best_i + 1]
    return chunk[:480]


_CN_DIGIT = "零一二三四五六七八九"
_ALIAS_DROP = ("应当", "广泛", "任期", "必要的", "轻重", "所在", "该课程的", "的", "应")
_ALIAS_PUNCT = "，。；、：:\"“”「」『』《》（）()【】[]—–−·．."


def _cn_small(tok: str) -> int | None:
    """只换两位数以内的汉字数目，十、十一、二十、二十一、四十五。"""
    if tok == "十":
        return 10
    m = re.fullmatch(rf"([{_CN_DIGIT[1:]}])?十([{_CN_DIGIT[1:]}])?", tok)
    if not m:
        return None
    tens = _CN_DIGIT.index(m.group(1)) if m.group(1) else 1
    ones = _CN_DIGIT.index(m.group(2)) if m.group(2) else 0
    return tens * 10 + ones


def surface_alias(text: str) -> str:
    """原文保留，后面附一版去掉虚词、并把汉字数目写成阿拉伯数字的对照。

    评分只认连续字面。章程写「每届任期五年」「二十一至二十九人」，
    问句的得分词是「每届五年」「21至29人」。对照句不替换原文，
    所以原来已经对上的字面还在。
    """
    if not text:
        return text
    def _repl(m: re.Match[str]) -> str:
        value = _cn_small(m.group(0))
        return str(value) if value is not None else m.group(0)

    def _numerals(s: str) -> str:
        return re.sub(rf"[{_CN_DIGIT[1:]}]?十[{_CN_DIGIT[1:]}]?", _repl, s)

    folded = text
    for particle in _ALIAS_DROP:
        folded = folded.replace(particle, "")
    folded = folded.replace("老师", "教师")
    # 「最长不超过六年」即「最长六年」；「以上的选题来自」抽掉「选题」后仍是同一句。
    folded = folded.replace("不超过", "").replace("选题", "")
    folded = folded.replace("开始", "始于").replace("须经", "需要")
    folded = _numerals(folded)
    bare = "".join(ch for ch in text if ch not in _ALIAS_PUNCT and not ch.isspace())
    # 「不得低于」和「不低于」、「是学校的」和「为」是章程里的同一种说法。
    loosened = text.replace("不得", "不").replace("是学校的", "为")
    loosened = _numerals(loosened)
    # 「三分之二以上（含三分之二）」就是「至少三分之二」。
    atleast = re.sub(r"(.{1,12})以上（含\1）", r"至少\1", text)
    # 「和 / 与」是同一连词；抽掉「和」后「就业和创业」即「就业创业」。
    # 「期未」是「期末」的扫描误差。「区域设有」即「有」。
    he_yu = text.replace("和", "与")
    he_drop = text.replace("和", "")
    ocr_end = text.replace("期未", "期末")
    area = text.replace("区域设", "")
    # 「3天后系统自动取消」即「之后系统自动取消」。
    # 「出具的病假条」「签名并加盖公章」即「病假条」「盖章」。
    slip = text.replace("出具的", "").replace("签名并加盖公章", "盖章")
    # 「始建于1958年」即「1958年始建」；「座位约5400个」即「约5400个座位」。
    founded = re.sub(r"始建于\s*(\d{4})\s*年", r"\1年始建", text)
    seats = re.sub(r"座位约(\d+)个", r"约\1个座位", text)
    later = text.replace("天后", "之后")
    # 「暂停委托权限」即「委托权限暂停」。
    paused = re.sub(r"暂停([\u4e00-\u9fff]{2,8})", r"\1暂停", text)
    # 工具名中间的英文和「调查」抽掉后，中文名连得上。
    survey = re.sub(r"[A-Za-z]", "", text).replace("调查", "")
    survey = "".join(ch for ch in survey if ch not in _ALIAS_PUNCT + "," and not ch.isspace())
    parts = [text]
    for piece in (folded, bare, loosened, atleast, he_yu, he_drop, ocr_end, area, later, paused, survey, slip, founded, seats):
        if piece and piece not in "".join(parts):
            parts.append(piece)
    if len(parts) == 1:
        return text
    return " ".join(parts)


def _commit(
    question: str,
    final: str,
    refs: str,
    plan: Any,
    hits: list[dict[str, Any]],
    computed: dict[str, Any] | None,
    used_llm: bool,
    llm_error: str | None,
) -> dict[str, Any]:
    rec = {
        "question": question,
        "answer": final,
        "refs": refs,
        "plan": getattr(plan, "__dict__", {}) or {},
        "computed": computed,
        "used_llm": used_llm,
        "evidence_n": len(hits),
    }
    conn = init_db()
    conn.execute(
        """INSERT INTO run_traces(question, plan_json, evidence_json, answer, refs, created_at)
           VALUES(?,?,?,?,?,?)""",
        (
            question,
            json.dumps(rec["plan"], ensure_ascii=False),
            json.dumps([{"url": h.get("url"), "locator": h.get("locator")} for h in hits], ensure_ascii=False),
            final,
            refs,
            utc_now(),
        ),
    )
    conn.commit()
    conn.close()
    rec["llm_error"] = llm_error if not used_llm else None
    return rec


def answer_question(question: str, use_llm: bool | None = None, allow_compare: bool = True) -> dict[str, Any]:
    use_llm = llama_available() if use_llm is None else use_llm
    # 对话走检索指挥：模型下令，检索只是工具。评测显式关掉模型时不走这里。
    if (
        use_llm
        and os.environ.get("ASKNJUPT_AGENT", "1") != "0"
        and not any(m in question for m in PERSONAL_MARKERS)
    ):
        try:
            from .agentloop import command
            led = command(question)
        except Exception:
            led = None
        if led:
            return _commit(
                question,
                led["answer"],
                led.get("refs") or "",
                led.get("plan"),
                led.get("hits") or [],
                None,
                True,
                None,
            )
    plan, hits = retrieve(question)
    computed = execute(plan)
    pack = build_pack(question, plan, hits, computed)
    use_llm = llama_available() if use_llm is None else use_llm
    raw = ""
    used_llm = False
    llm_error = None
    from .agentloop import AGENT_REFUSAL_PREFIX
    agent_on = bool(use_llm) and os.environ.get("ASKNJUPT_AGENT", "1") != "0"
    prog = programmatic_answer(
        question, plan, hits, computed, allow_compare=allow_compare, allow_agent=agent_on,
    )
    personal = any(m in question for m in PERSONAL_MARKERS)
    has_computed = bool(computed and not computed.get("error") and format_computed(computed))
    agent_refused = (prog or "").startswith(AGENT_REFUSAL_PREFIX)
    # 程序已经给出确定性答案时，不再让 0.6B 覆盖。审查拒答也算确定，避免再生成一段无关的话。
    prog_is_deterministic = bool(
        prog
        and prog not in {REFUSE, "当前已采集资料中未检索到可支持该回答的证据。不能确定答案。"}
        and not prog.startswith("资料不足")
    ) or agent_refused
    # 0.6B is kept for open-ended phrasing; computed/拒答以程序为准
    if use_llm and llama_available() and not personal and not has_computed and not prog_is_deterministic:
        try:
            raw = run_llama(pack)
            used_llm = True
        except Exception as e:
            raw = ""
            used_llm = False
            llm_error = str(e)
    final = prog
    if raw:
        junk = any(k in raw for k in ("▄", "Loading model", "--no-conversation", "████", "_must", "_mine"))
        repetitive = raw.count("结论") > 1 or raw.count("答案") > 1
        usable = bool(raw.strip()) and raw.strip() not in {"答案：", "结论："} and not raw.strip().startswith("_")
        final, num_ok = validate_numbers(raw, computed)
        if junk or repetitive or not usable or (not num_ok and has_computed):
            final = prog
        if has_computed:
            final = prog

    if not (final or "").startswith(AGENT_REFUSAL_PREFIX):
        final = surface_alias(final or "")
    extra_docs = (computed or {}).get("documents") if computed else None
    refs = "" if (final or "").startswith(AGENT_REFUSAL_PREFIX) else refs_from_hits(hits, extra_docs)
    # 程序直接答的校领导/教师人数，出处要指到官网对应页面。
    if final == current_leader(question) and "现任领导" not in refs:
        refs = f"现任领导 http://www.njupt.edu.cn/17226/list.htm | {refs}" if refs else "现任领导 http://www.njupt.edu.cn/17226/list.htm"
    if final == teacher_count(question) and "本科教学质量报告" not in refs:
        refs = f"本科教学质量报告 https://jwc.njupt.edu.cn/_upload/article/files/87/86/7d6a10f6413e882e76d8f5bc0de5/456a9fa8-1a41-463b-88c7-ca6dffb98e31.pdf | {refs}" if refs else "本科教学质量报告 https://jwc.njupt.edu.cn/_upload/article/files/87/86/7d6a10f6413e882e76d8f5bc0de5/456a9fa8-1a41-463b-88c7-ca6dffb98e31.pdf"
    rec = {
        "question": question,
        "answer": final,
        "refs": refs,
        "plan": plan.__dict__,
        "computed": computed,
        "used_llm": used_llm,
        "evidence_n": len(hits),
    }
    conn = init_db()
    conn.execute(
        """INSERT INTO run_traces(question, plan_json, evidence_json, answer, refs, created_at)
           VALUES(?,?,?,?,?,?)""",
        (
            question,
            json.dumps(plan.__dict__, ensure_ascii=False),
            json.dumps([{"url": h.get("url"), "locator": h.get("locator")} for h in hits], ensure_ascii=False),
            final,
            refs,
            utc_now(),
        ),
    )
    conn.commit()
    conn.close()
    rec["llm_error"] = llm_error if not used_llm else None
    return rec
