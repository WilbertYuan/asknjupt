"""让 Qwen3-0.6B 只做证据选择：从已经检索到的句子里指出哪一句直接回答问题。

模型的输出只被当成编号。答案文本永远是原文句子，不采用模型改写。
编号对不上、覆盖不上问句内容词、或者它同意规则排序的第一名时，返回 None，
调用方继续走原来的兜底。这样它不能覆盖表格、组装和已经算出的数字。
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

_CN = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8}


def _endpoint() -> str:
    port = os.environ.get("ASKNJUPT_LLM_PORT", "8791")
    return os.environ.get("ASKNJUPT_LLM_URL", f"http://127.0.0.1:{port}/v1/chat/completions")


def _terms(question: str) -> list[str]:
    from .extractive import content_terms

    out = []
    for term in content_terms(question):
        t = re.sub(r"\s+", "", term)
        if len(t) >= 2 and t not in out:
            out.append(t)
    return out


def _repeated(question: str) -> list[str]:
    """问句里出现两次的二字词，通常才是真正要核对的对象。"""
    body = re.sub(r"[^\u4e00-\u9fff]", "", _asked(question))
    counts: dict[str, int] = {}
    for i in range(len(body) - 1):
        gram = body[i : i + 2]
        counts[gram] = counts.get(gram, 0) + 1
    return [gram for gram, n in counts.items() if n >= 2]


def _asked(question: str) -> str:
    """去掉「按某文件」。文件名只说明去哪一份，不是答案里必须出现的词。"""
    return re.sub(r"^按[^，。？?]{2,48}[，,]?", "", question or "", count=1)


def candidate_sentences(question: str, hits: list[dict[str, Any]], limit: int = 6):
    """沿用抽取层的句子排序，取短名单。这里不再放宽抽取门槛。"""
    from .extractive import (
        content_terms, question_pieces, question_type, split_sentences, _gather, _is_noise,
    )

    asked = _asked(question)
    grams = _repeated(question)
    ranked = _gather(
        hits,
        content_terms(asked),
        question_type(question),
        asked,
        max_sent=3,
        pieces=question_pieces(asked),
    )
    out: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()

    def _take(text: str, hit: dict[str, Any], into: list) -> None:
        text = re.sub(r"\s+", "", text or "").strip("，,。；; ")
        if len(text) < 8 or "=" in text or _is_noise(text):
            return
        if (hit.get("unit_type") or "") == "table_row":
            return
        key = text[:40]
        if key in seen or len(into) >= limit:
            return
        seen.add(key)
        into.append((text, hit))

    for _score, sent, hit in ranked:
        _take(sent, hit, out)
    # 词族对不上时抽取层会把整句打成 0。问句里反复出现的二字
    # （「当天…当天」）仍可能在原文里，只在排序结果完全没有这个词时补入。
    if grams and not any(g in sent for sent, _hit in out for g in grams):
        extra: list[tuple[str, dict[str, Any]]] = []
        for hit in hits[:4]:
            for sent in split_sentences(hit.get("text") or ""):
                if any(g in sent for g in grams):
                    _take(sent, hit, extra)
                if len(extra) >= 2:
                    break
            if len(extra) >= 2:
                break
        out = (extra + out)[:limit]
    # 问「多少、排名」时，正确句常常只有「前20%」「仅有一次」，
    # 话题词对不上，抽取层不会把它排进来。补到名单末尾，编号题里末位比首位稳。
    if re.search(r"多少|几|排名|比例", question) and len(out) < limit:
        terms = [t for t in content_terms(asked) if len(t) >= 2]
        extra_q: list[tuple[str, dict[str, Any]]] = []
        for hit in hits[:4]:
            for sent in split_sentences(hit.get("text") or ""):
                flat = re.sub(r"\s+", "", sent)
                if not re.search(r"前\s*\d+\s*%|有且仅有一次|仅有一次", flat):
                    continue
                # 硕士/研究生、成绩排名这类说法对不齐整词，用三字重合判断是不是同一件事。
                body = re.sub(r"[^\u4e00-\u9fff]", "", asked)
                tri = {body[i:i + 3] for i in range(len(body) - 2)}
                if sum(g in flat for g in tri) < 2:
                    continue
                _take(sent, hit, extra_q)
                if len(extra_q) >= 2:
                    break
            if len(extra_q) >= 2:
                break
        for item in extra_q:
            if len(out) >= limit:
                break
            if item not in out:
                out.append(item)
    return out


def _ask(question: str, sentences: list[str]) -> int | None:
    lines = []
    for i, sent in enumerate(sentences, start=1):
        lines.append(f"{i}. {sent[:160]}")
    prompt = (
        f"问题：{question}\n"
        "候选句子：\n"
        + "\n".join(lines)
        + "\n哪一句直接回答了这个问题？只输出编号。\n/no_think"
    )
    body = json.dumps(
        {
            "temperature": 0,
            "max_tokens": 8,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode()
    req = urllib.request.Request(
        _endpoint(),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            payload = json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    content = ""
    try:
        content = payload["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        return None
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
    m = re.search(r"[1-8]", content)
    if m:
        return int(m.group(0))
    for ch, n in _CN.items():
        if ch in content:
            return n
    return None


def _long_terms(question: str) -> list[str]:
    """只拿较长的内容词做覆盖比较。短碎片在长段落里几乎总会命中，比不出差别。"""
    long = [t for t in _terms(question) if len(t) >= 4]
    return long or _terms(question)


# 「一般在每年」这种语法片段能对上另一件事的句子，不能算新证据。
_GLUE = ("一般", "通常", "每年", "什么", "如何", "进行", "开展", "采用", "可以", "是否")


def _substantive(term: str) -> bool:
    text = term
    for glue in _GLUE:
        text = text.replace(glue, "")
    han = "".join(ch for ch in text if "\u4e00" <= ch <= "\u9fff")
    return len(han) >= 4


def _substantive_hits(terms: list[str], text: str) -> set[str]:
    compact = re.sub(r"\s+", "", text)
    return {t for t in terms if _substantive(t) and t in compact}


def select_fallback(
    question: str,
    hits: list[dict[str, Any]],
    baseline: str = "",
) -> tuple[str, dict[str, Any]] | None:
    """模型选中的句子必须比当前兜底覆盖更多问句要点，才替换。

    列举和「分别」要绑多句，单句选择做不到，直接放弃。
    """
    if os.environ.get("ASKNJUPT_LLM_SELECT", "1") == "0":
        return None
    if not hits or not baseline:
        return None
    if any(k in question for k in ("分别", "依次", "哪些", "哪几")):
        return None
    cands = candidate_sentences(question, hits)
    grams = _repeated(question)
    if not cands:
        return None
    if len(cands) == 1:
        # 只有一句含题眼、而且当前兜底里没有这个词，才不用模型编号。
        # 兜底已经含题眼时，单句替换没有对照，保持原句。
        sent0 = cands[0][0]
        base_c = re.sub(r"\s+", "", baseline)
        count_only = (
            bool(re.search(r"几次|多少次", question))
            and "仅有一次" in sent0
            and "仅有一次" not in base_c
        )
        if not count_only and (
            not grams
            or not any(g in sent0 for g in grams)
            or any(g in base_c for g in grams)
        ):
            return None
        idx = 1
    else:
        idx = _ask(question, [s for s, _ in cands])
    if idx is None or not (1 <= idx <= len(cands)):
        return None
    sent, hit = cands[idx - 1]
    asked = _asked(question)
    terms = _long_terms(asked)
    gained = _substantive_hits(terms, sent) - _substantive_hits(terms, baseline)
    lost = _substantive_hits(terms, baseline) - _substantive_hits(terms, sent)
    # 新句子必须多出实质要点，并且不能丢掉兜底里已经对上的要点。
    # 兜底和问句完全对不上时，改选一句至少带上一个问句里的词。
    short = [t for t in _terms(asked) if 2 <= len(t) <= 4]
    grams = _repeated(question)
    short += [g for g in grams if g not in short]
    # 两字的普通片段（「书刊」）太容易碰到旁边的句子。弱接受只认
    # 三字以上的问句词，或者问句里反复出现的那一个二字。
    useful = [t for t in short if len(t) >= 3 or t in grams]
    base_c = re.sub(r"\s+", "", baseline)
    sent_c = re.sub(r"\s+", "", sent)
    # 弱接受只用于「问句里同一个词出现两次，兜底却完全没沾到」。
    # 没有这种题眼时，多出来的三字碎片会把本来对的句子换成旁边的句子。
    weak = (
        bool(grams)
        and not _substantive_hits(terms, baseline)
        and not lost
        and any(g in sent_c for g in grams)
        and not any(g in base_c for g in grams)
        and sum(t in sent_c for t in useful) > sum(t in base_c for t in useful)
    )
    # 「几次」对上「仅有一次」时，问句里的「推免」和原文「推荐免试」对不齐，
    # 实质词增益会是空的。这种次数句本身就是答案，不再要求多出一个长词。
    count_hit = (
        bool(re.search(r"几次|多少次", question))
        and "仅有一次" in sent_c
        and "仅有一次" not in base_c
    )
    # 问硕士排名时，原文写「研究生…前20%」。长词对不齐，但这句就是名次。
    rank_hit = (
        "排名" in question
        and "排名" in sent_c
        and re.search(r"前\d+%", sent_c) is not None
        and re.search(r"前\d+%", base_c) is None
    )
    if ("硕士" in question or "研究生" in question) and not any(
        w in sent_c for w in ("硕士", "研究生")
    ):
        rank_hit = False
    if rank_hit:
        # 「小米特等奖」只是问句里的项目名，名次句常常不再重复一遍。
        lost = {t for t in lost if "排名" in t}
    if lost or not (gained or weak or count_hit or rank_hit):
        return None
    # 问句里点名的编号、年份、认证号，兜底里已经有了就不能在换句时丢掉。
    # 「1945年发展为战邮学校」不能换成只复述了「战邮干训班」的简介。
    marks = re.findall(r"[A-Za-z0-9]{2,}", asked)
    if any(m in base_c and m not in sent_c for m in marks):
        return None
    if re.search(r"能[^？?\n]{0,16}吗|能否", question) and re.search(
        r"一律不|不得|不可|不准", sent
    ) and not sent.startswith("不能"):
        sent = "不能。" + sent
    if len(sent) > 220:
        cut = sent[:220]
        stop = max(cut.rfind("。"), cut.rfind("；"))
        sent = cut[: stop + 1] if stop >= 40 else cut
    if not sent.endswith(("。", "；")):
        sent += "。"
    return sent, hit
