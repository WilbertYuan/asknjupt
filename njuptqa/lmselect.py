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


def candidate_sentences(question: str, hits: list[dict[str, Any]], limit: int = 6):
    """沿用抽取层的句子排序，取短名单。这里不再放宽抽取门槛。"""
    from .extractive import content_terms, question_pieces, question_type, _gather

    ranked = _gather(
        hits,
        content_terms(question),
        question_type(question),
        question,
        max_sent=3,
        pieces=question_pieces(question),
    )
    out: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for _score, sent, hit in ranked:
        text = re.sub(r"\s+", "", (sent or "")).strip("，,。；; ")
        if len(text) < 8:
            continue
        key = text[:40]
        if key in seen:
            continue
        seen.add(key)
        out.append((text, hit))
        if len(out) >= limit:
            break
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
    if len(cands) < 2:
        return None
    idx = _ask(question, [s for s, _ in cands])
    if idx is None or idx == 1 or not (1 <= idx <= len(cands)):
        return None
    sent, hit = cands[idx - 1]
    terms = _long_terms(question)
    gained = _substantive_hits(terms, sent) - _substantive_hits(terms, baseline)
    lost = _substantive_hits(terms, baseline) - _substantive_hits(terms, sent)
    # 新句子必须多出实质要点，并且不能丢掉兜底里已经对上的要点。
    # 否则总则、另一件事的「每年」会把原来答对的句子换掉。
    if not gained or lost:
        return None
    if len(sent) > 220:
        cut = sent[:220]
        stop = max(cut.rfind("。"), cut.rfind("；"))
        sent = cut[: stop + 1] if stop >= 40 else cut
    if not sent.endswith(("。", "；")):
        sent += "。"
    return sent, hit
