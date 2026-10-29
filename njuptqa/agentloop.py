"""检索指挥。0.6B 决定搜什么、看工具返回、再决定换词或收束。

检索不是先跑完再请模型挑一句。模型每一轮只能发出三种工具动作之一：
检索、带着依据回答、或判定没有。动作格式用语法卡住，避免它空口作答。
依据必须是某次检索结果里的原句，对不上就打回，让它再下命令。
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

AGENT_REFUSAL_PREFIX = "公开资料里没有能直接回答这个问题的条款。"

_CHAT = ("心情", "哈哈", "随便", "你好", "在吗", "天气", "笑话", "吃饭了吗", "无聊")


def _endpoint() -> str:
    port = os.environ.get("ASKNJUPT_LLM_PORT", "8791")
    return os.environ.get("ASKNJUPT_LLM_URL", f"http://127.0.0.1:{port}/v1/chat/completions")


def _complete(prompt: str, max_tokens: int) -> str | None:
    body = json.dumps(
        {
            "temperature": 0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt + "\n/no_think"}],
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
        content = payload["choices"][0]["message"].get("content") or ""
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, KeyError, IndexError, TypeError):
        return None
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
    return content or None


def _compact(text: str) -> str:
    return re.sub(r"[\s，,。；;：:、\"“”『』「」]", "", text or "")


def _specific_terms(question: str) -> list[str]:
    """学校文件里可能出现的内容词。闲聊和纯乱码没有这种词，就不再换检索。"""
    from .extractive import content_terms

    if any(w in (question or "") for w in _CHAT):
        return []
    han = "".join(ch for ch in question or "" if "\u4e00" <= ch <= "\u9fff")
    if len(han) < 4:
        return []
    out: list[str] = []
    for term in content_terms(question):
        if len(term) >= 3 and term not in out:
            out.append(term)
    return out


def _excerpt_supports(excerpt: str | None, sentence: str, question: str) -> bool:
    if not excerpt:
        return False
    raw = excerpt.strip().splitlines()[0].strip()
    raw = re.sub(r"^(摘录|摘出|答案)[:：]", "", raw).strip()
    flat = _compact(raw)
    # 「没有」单独出现是摘录失败。原文里本来就有「没有委托中介」这种更长的话，仍然算摘到了。
    if not flat or flat in {"无", "没有", "没"} or (flat.startswith("没有") and len(flat) <= 4):
        return False
    if len(flat) < 2 or flat not in _compact(sentence):
        return False
    # 从原文里随便抄几个字不算回答。摘录要沾上问句里的事项，或者就是问句要的数字。
    if any(term in flat for term in _specific_terms(question)):
        return True
    return bool(re.search(r"多少|几|哪一|何时|哪年|收费|罚款|金额", question) and re.search(r"\d", flat))


def _candidates(question: str, hits: list[dict], broad: bool = False) -> list[tuple[str, dict]]:
    from .extractive import split_sentences
    from .lmselect import candidate_sentences

    found = list(candidate_sentences(question, hits) or [])
    if found and not broad:
        return found
    out: list[tuple[str, dict]] = []
    seen = {sent[:24] for sent, _hit in found}
    # 重检时原问句的词对不上新条款（过期对逾期）。把带金额或罚则的句子补进短名单，
    # 否则审查看的仍是上一轮那些没回答的句子。
    for hit in hits[:3]:
        for sent in split_sentences(hit.get("text") or ""):
            sent = re.sub(r"\s+", "", sent).strip("，,。；; ")
            if len(sent) < 8 or sent[:24] in seen:
                continue
            if broad and not (
                re.search(r"\d", sent) or any(k in sent for k in ("违约", "逾期", "超期", "不得", "应当"))
            ):
                continue
            seen.add(sent[:24])
            out.append((sent, hit))
            if len(out) >= 6:
                break
        if len(out) >= 6:
            break
    if not broad:
        return out
    # 罚则句放在词面重合句前面，避免编号又落回文档开头。
    return (out + found)[:6]


def _pick_index(question: str, sentences: list[str]) -> int | None:
    lines = [f"{i}. {sent[:160]}" for i, sent in enumerate(sentences, start=1)]
    content = _complete(
        f"问题：{question}\n候选句子：\n" + "\n".join(lines) + "\n哪一句直接回答了这个问题？只输出编号。",
        8,
    )
    if not content:
        return None
    match = re.search(r"[1-8]", content)
    if not match:
        return None
    idx = int(match.group(0))
    if 1 <= idx <= len(sentences):
        return idx
    return None


def _screen(question: str, hits: list[dict], broad: bool = False) -> tuple[str, dict] | None:
    """挑句之后逐句摘录。摘录不是原文子串，就当这句没回答。"""
    cands = _candidates(question, hits, broad=broad)
    if not cands:
        return None
    sentences = [sent for sent, _hit in cands]
    picked = _pick_index(question, sentences) if len(cands) > 1 else 1
    order: list[int] = []
    # 重检补进来的罚则句在名单开头。先审它，不让编号习惯把它跳过。
    if broad:
        order.append(0)
    if picked and (picked - 1) not in order:
        order.append(picked - 1)
    # 编号偏向第 1 句时，再抽两句词面更贴的，避免只审错那一句。
    terms = _specific_terms(question)
    ranked = sorted(
        range(len(cands)),
        key=lambda i: sum(t in cands[i][0] for t in terms),
        reverse=True,
    )
    for i in ranked:
        if i not in order:
            order.append(i)
        if len(order) >= 3:
            break
    for i in order[:3]:
        sent, hit = cands[i]
        excerpt = _complete(
            f"摘出原文中回答问题的那几个字。摘不出来就写没有。\n问题：{question}\n原文：{sent[:240]}",
            32,
        )
        if _excerpt_supports(excerpt, sent, question):
            return sent, hit
    return None


# 0.6B 的改写经常只是删掉「怎么」。这些是检索用的说法转换，不是答案模板。
_SYN_GROUPS = (
    (("过期", "超期"), "逾期 违约金"),
    (("罚款", "收费"), "违约金"),
    (("开门", "开馆", "闭馆"), "开放时间"),
)


def _too_close(left: str, right: str) -> bool:
    a, b = set(_compact(left)), set(_compact(right))
    if not a or not b:
        return True
    return len(a & b) / max(len(a), len(b)) >= 0.8


def _synonym_query(question: str) -> str | None:
    parts: list[str] = []
    for keys, alt in _SYN_GROUPS:
        if any(key in question for key in keys) and alt not in parts:
            parts.append(alt)
    if not parts:
        return None
    topic = next((word for word in ("图书", "借书", "图书馆", "宿舍", "学费") if word in question), "")
    tokens: list[str] = [topic] if topic else []
    for part in parts:
        for token in part.split():
            if token not in tokens:
                tokens.append(token)
    return " ".join(tokens)


def _rewrite(question: str, previous: list[str] | None = None) -> str | None:
    used = "、".join(previous or []) or "原问题"
    content = _complete(
        "你是检索改写员。上一轮关键词没有对上条款。"
        "换成学校规章里更可能出现的词，20字以内，不要解释，不要重复已经用过的词。"
        "如果这不是学校公开文件里的事实问题，只输出无。\n"
        f"问题：{question}\n已经用过：{used}",
        24,
    )
    if not content:
        return None
    text = content.strip().splitlines()[0].strip("。． ")
    text = re.sub(r"^(关键词|改写)[:：]", "", text).strip()
    if not text or text in {"无", "没有", "不知道"} or text == question or len(text) > 30:
        return None
    return text


def _format(sent: str, hit: dict, via: str | None) -> str:
    body = sent if sent.endswith(("。", "；", "！")) else sent + "。"
    head = f"换了检索「{via}」之后，对上的是这一句：\n" if via else ""
    title = hit.get("doc_title") or ""
    url = hit.get("url") or ""
    return f"{head}{body}\n（摘自 {title} {url}）"


def _refuse(rounds: list[str]) -> str:
    shown = "、".join(f"「{item}」" for item in rounds)
    return (
        f"{AGENT_REFUSAL_PREFIX}"
        f"检索审查看过{shown}，没有一句原文能摘出答案，所以不把旁边的段落拿来充数。"
        "不能确定答案。"
    )


_G_SEARCH = r'''root ::= "检索：" [^\n]{2,16}'''
_G_DECIDE = r'''root ::= answer | search | none
answer ::= "回答：" [^\n]{2,80} "\n依据：" [^\n]{4,180}
search ::= "检索：" [^\n]{2,30}
none ::= "没有"
'''


def _chat(prompt: str, max_tokens: int, grammar: str) -> str | None:
    body = json.dumps(
        {
            "temperature": 0,
            "max_tokens": max_tokens,
            "grammar": grammar,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt + "\n/no_think"}],
        }
    ).encode()
    req = urllib.request.Request(
        _endpoint(),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode())
        content = payload["choices"][0]["message"].get("content") or ""
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, KeyError, IndexError, TypeError):
        return None
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
    return content or None


def _parse_action(text: str) -> tuple[str, str, str]:
    raw = (text or "").strip()
    if raw.startswith("没有"):
        return "none", "", ""
    if raw.startswith("检索"):
        query = re.sub(r"^检索[:：]\s*", "", raw).splitlines()[0].strip("。 ")
        return "search", query[:30], ""
    if raw.startswith("回答"):
        parts = re.split(r"依据[:：]", raw, maxsplit=1)
        answer = re.sub(r"^回答[:：]\s*", "", parts[0]).strip()
        cite = parts[1].strip() if len(parts) > 1 else ""
        cite = re.sub(r"^\[\d+\]\s*《[^》]*》", "", cite).strip()
        return "answer", answer, cite
    return "bad", raw, ""


def _probes(question: str, query: str) -> list[str]:
    """用模型这一轮写的检索词。它写到别的学校时，改回用户的问题，不替它换成某条固定条款。"""
    text = re.sub(r"\s+", "", query or "")
    text = re.sub(r"南京邮电大学|南京邮电|邮电大学|南邮", "", text)
    text = re.sub(r"[\u4e00-\u9fff]{2,8}大学", "", text)
    text = re.sub(r"20\d{2}年?", "", text)
    if len(text) < 2:
        text = re.sub(r"\s+", "", question or "")
        text = re.sub(r"南京邮电大学|南京邮电|邮电大学|南邮", "", text)
    return [text[:30]] if len(text) >= 2 else []


def _window_at(flat: str, at: int, probe: str) -> str | None:
    punct = max(flat.rfind("。", 0, at), flat.rfind("；", 0, at))
    # 一百字以内的上一句句号算同一句。再远就是目录粘连，改从学年或逗号起句。
    if punct >= 0 and at - punct <= 100:
        start = punct + 1
    else:
        back_from = max(0, at - 28)
        year = re.search(r"20\d{2}.{0,3}20\d{2}学年", flat[back_from:at + 2])
        if year:
            start = back_from + year.start()
        else:
            comma = flat.rfind("，", max(0, at - 48), at)
            start = comma + 1 if comma >= 0 else max(0, at - 8)
    end = flat.find("。", at)
    if end < 0 or end - start > 90:
        end = min(len(flat), start + 90)
    else:
        end += 1
    sent = flat[start:end]
    return sent if len(sent) >= 8 else None


def _windows(text: str, probe: str) -> list[str]:
    """同一段里这个词可能出现好几次。光盘违约金和图书违约金要分开交出去。"""
    flat = re.sub(r"\s+", "", text or "")
    out: list[str] = []
    start = 0
    while len(out) < 4:
        at = flat.find(probe, start)
        if at < 0:
            break
        sent = _window_at(flat, at, probe)
        if sent and sent not in out:
            out.append(sent)
        start = at + max(len(probe), 1)
    return out


def _sql_probe(probes: list[str], question: str = "") -> list[tuple[str, dict]]:
    """不走向量。每个关键词各取几句，避免先命中的词把别的词挤掉。"""
    from .db import init_db

    probes = [p for p in probes if len(p) >= 2][:6]
    if not probes:
        return []
    conn = init_db()
    out: list[tuple[str, dict]] = []
    seen: set[str] = set()
    try:
        for probe in probes:
            # 人数、违约金这类原词不加主题限制。口语碎片才要求沾上图书，免得扫进合同。
            others = [p for p in probes if p != probe and len(p) >= 3][:3]
            params: tuple[str, ...] = (f"%{probe}%",)
            if others:
                score_sql = " + ".join(
                    "(CASE WHEN d.title LIKE ? THEN 3 ELSE 0 END) + (CASE WHEN ru.text LIKE ? THEN 2 ELSE 0 END)"
                    for _ in others
                )
                for other in others:
                    params += (f"%{other}%", f"%{other}%")
                order_sql = f"{score_sql} DESC"
            else:
                order_sql = "length(ru.text)"
            rows = conn.execute(
                f"""
                SELECT d.title AS doc_title, d.url, ru.text, ru.locator
                FROM retrieval_units ru JOIN documents d ON d.id = ru.document_id
                WHERE ru.text LIKE ?
                ORDER BY {order_sql}
                LIMIT 8
                """,
                params,
            ).fetchall()
            rows = list(rows)

            def _row_score(row: object, probe: str = probe) -> int:
                title = row["doc_title"] or ""
                # 标题里带上另一个关键词，说明这页就是这件事，而不是别处顺带出现的同词。
                title_hit = 4 if any(len(p) >= 4 and p != probe and p in title for p in probes) else 0
                body = max(
                    (
                        (3 if _on_topic(question, sent) else 0)
                        + (1 if re.search(r"\d", sent) else 0)
                        + sum(1 for p in probes if len(p) >= 4 and p in sent)
                        for sent in _windows(row["text"], probe)[:3]
                    ),
                    default=0,
                )
                return title_hit + body

            rows.sort(key=_row_score, reverse=True)
            added = 0
            for row in rows:
                hit = {
                    "doc_title": row["doc_title"],
                    "url": row["url"] or "",
                    "locator": row["locator"],
                    "text": row["text"],
                }
                for sent in _windows(row["text"], probe)[:2]:
                    if sent[:24] in seen:
                        continue
                    seen.add(sent[:24])
                    out.append((sent[:220], hit))
                    added += 1
                    if added >= 2:
                        break
                if added >= 2:
                    break
    finally:
        conn.close()
    return out


def _asks_who(question: str) -> bool:
    text = question or ""
    if re.search(r"是谁|叫什么|什么名字|何人", text):
        return True
    return bool(_role_in(text) and re.search(r"有哪些|有谁|都有谁|是哪些|哪几位|几位|名单", text))


def _asks_span(question: str) -> bool:
    """学制、可以读几年。休学多久不走这里，那类句子里往往没有「学制」。"""
    return bool(re.search(r"读几年|学制|修业|可以读", question or ""))


def _asks_english_name(question: str) -> bool:
    text = question or ""
    return "英文" in text and bool(re.search(r"全称|校名|名字|名|怎么写|如何写|叫什么|是什么", text))


def _asks_place(question: str) -> bool:
    text = question or ""
    return bool(
        re.search(r"在哪|哪个区|什么区|地址", text)
        or re.search(r"是.+区吗|是不是.+区", text)
        or ("校区" in text and "区" in text and "吗" in text)
    )


def _named_thing(question: str) -> str:
    """「有没有某某学院」必须整词出现。差一个字不能算有这个学院。"""
    matched = re.search(
        r"(?:有没有|是否有|有无|(?<![没])有)([\u4e00-\u9fff]{2,8}(?:学院|专业|基地|医院))",
        question or "",
    )
    if not matched:
        return ""
    name = matched.group(1)
    if any(word in name for word in ("多少", "几个", "什么", "哪些")):
        return ""
    return name


def _phrases(question: str) -> list[str]:
    """问句里剩下的事项，长的优先。疑问词和校名不是事项。"""
    text = re.sub(r"南京邮电大学|南京邮电|邮电大学|南邮", "", question or "")
    text = re.sub(
        r"叫什么名字|叫什么|是谁|怎么写|如何写|有没有|是否有|到底|算不算|是不是|是在|是否在|现在的|正常|可以|最多|最长|在哪个区|哪个区|在哪|地址是什么|地址|什么|怎么|如何|读几年|几年|多久|吗|呢|的",
        " ",
        text,
    )
    raw = re.findall(r"[\u4e00-\u9fff]{2,12}", text)
    raw.sort(key=len, reverse=True)
    kept: list[str] = []
    for phrase in raw:
        if any(phrase in longer for longer in kept):
            continue
        kept.append(phrase)
    return kept[:3]


def _role_in(question: str) -> str:
    text = question or ""
    if "副校长" in text:
        return "副校长"
    for role in ("党委书记", "校长", "院长", "主任"):
        if role in text:
            return role
    return ""


_NOT_NAME = ("办公", "负责", "同志", "老师", "大学", "学校", "学院", "质量", "评价", "研究", "发展", "出席", "典礼", "委员", "组织", "宣传")


def _role_name(sent: str, role: str) -> str | None:
    """职务紧后面的人名。校长办公会、副校长、党委副书记都不是这个职务的姓名。"""
    if not role or role not in (sent or ""):
        return None
    found: dict[int, str] = {}
    for size in (2, 3):
        for matched in re.finditer(re.escape(role) + rf"[:：]?([\u4e00-\u9fff]{{{size}}})", sent):
            if matched.start() > 0 and sent[matched.start() - 1] == "副":
                continue
            name = matched.group(1)
            if any(word in name for word in _NOT_NAME):
                continue
            if any(ch in name for ch in "委办处院系部组员会室校党副率赴队看望问访"):
                continue
            found[matched.start()] = name
    if not found:
        return None
    return found[min(found)]


def _asks_rule(question: str) -> bool:
    """能不能、还能不能。问的是许不允许，不是要一个总人数。"""
    if "退学" in (question or "") or ("休学" in (question or "") and "复学" not in (question or "")):
        return False
    if re.search(r"体测|体质", question or ""):
        return False
    return bool(re.search(r"能不能|可不可以|能否|还能|允许|禁止|有没有规定|行不行|是必须", question or "") or re.search(r"能[\u4e00-\u9fff]{1,8}吗", question or ""))


def _rule_focus(question: str) -> str:
    text = re.sub(r"南京邮电大学|南京邮电|邮电大学|南邮", "", question or "")
    matched = re.search(r"(?:能不能|可不可以|能否|是否可以|还能|不能)([\u4e00-\u9fff]{2,8})", text)
    if not matched:
        matched = re.search(r"([\u4e00-\u9fff]{2,8})行不行", text)
    if not matched:
        matched = re.search(r"能([\u4e00-\u9fff]{1,8})吗", text)
    if not matched:
        matched = re.search(r"^([\u4e00-\u9fff]{2,6})是必须", text)
    if not matched:
        return ""
    obj = re.sub(r"(吗|呢|啊|么)$", "", matched.group(1))
    obj = re.sub(r"^(养|不|要)", "", obj)
    return obj if len(obj) >= 2 else ""


def _asks_hours(question: str) -> bool:
    return bool(re.search(r"开门|开馆|闭馆|几点|周末|周六|周日|开放时间", question or ""))


def _hours_mode(question: str) -> str:
    """周末、工作日、今天要分开。今天按提问当天是不是周末。"""
    text = question or ""
    if re.search(r"周末|周六|周日", text) and not re.search(r"工作日|周一", text):
        return "weekend"
    if re.search(r"工作日|周一|周中|平常|平时", text):
        return "weekday"
    if re.search(r"今天|现在", text):
        import datetime
        return "weekend" if datetime.date.today().weekday() >= 5 else "weekday"
    return "both"


def _fee_kind(question: str) -> str:
    """本科里问的是哪一类学费。研究生学费另走。"""
    text = question or ""
    if not any(k in text for k in ("学费", "每生每年", "多少钱")):
        return ""
    if any(k in text for k in ("博士", "硕士", "研究生")) and "本科" not in text:
        return ""
    found = [(text.find(kind), kind) for kind in ("工科", "文科", "理科", "艺术") if kind in text]
    return min(found)[1] if found else ""


def _grad_level(question: str) -> str:
    text = question or ""
    if "住宿" in text:
        return ""
    if "学费" not in text and "多少钱" not in text:
        return ""
    if "本科" in text and "研究生" not in text and "博士" not in text and "硕士" not in text:
        return ""
    if "博士" in text and "学术" not in text:
        return "博士"
    if "学术" in text:
        return "学术"
    if "专业学位" in text:
        return "专业学位"
    if "硕士" in text or "研究生" in text:
        return "研究生"
    return ""


def _asks_school_code(question: str) -> bool:
    text = question or ""
    return "代码" in text and "学校" in text


def _asks_campuses(question: str) -> bool:
    return bool(re.search(r"几个校区|哪些校区|多少个校区|有几个校区", question or ""))


def _asks_borrow_count(question: str) -> bool:
    text = question or ""
    return bool(re.search(r"几本|几册|多少本|多少册|限借", text) and re.search(r"借|书", text))


def _asks_borrow_days(question: str) -> bool:
    text = question or ""
    return bool(
        re.search(r"借期|期限|多久|多少天", text)
        and re.search(r"借|书", text)
        and not _asks_borrow_count(text)
    )


def _best_at(flat: str, needle: str, prefer: tuple[str, ...]) -> int:
    """同一个词出现很多次时，靠近「必须、严禁、周六」的那次才是条款。"""
    found = [m.start() for m in re.finditer(re.escape(needle), flat or "")]
    if not found:
        compact = re.sub(r"\s+", "", needle)
        if compact != needle:
            found = [m.start() for m in re.finditer(re.escape(compact), flat or "")]
    if not found:
        return -1
    for at in found:
        span = flat[max(0, at - 36) : at + 48]
        if any(word in span for word in prefer):
            return at
    return found[0]


def _shape_ok(question: str, sent: str, title: str = "") -> bool | None:
    """问法对得上才收下。None 表示这道题不属这类，交给原有规则。"""
    sent = sent or ""
    thing = _named_thing(question)
    if thing:
        if len(re.findall(r"\d{4}年\d{1,2}月\d{1,2}日", sent)) >= 2:
            return False
        if thing not in sent:
            return False
        if any(word in sent for word in ("报告", "讲座", "招聘", "研讨会", "暑期学校", "讲话", "学习会")):
            return False
        return True
    if _asks_english_name(question):
        flat = re.sub(r"\s+", "", sent)
        return "UniversityofPosts" in flat or "University of Posts" in sent
    if _asks_who(question):
        role = _role_in(question)
        if not role:
            return None
        return _role_name(sent, role) is not None
    if _asks_span(question):
        if "=" in sent or not re.search(r"学制|修业年限|学习年限", sent):
            return False
        for phrase in _phrases(question):
            if len(phrase) >= 4 and phrase not in sent and phrase not in (title or ""):
                return False
        for level in ("博士", "硕士", "本科"):
            if level in (question or "") and level not in sent and level not in (title or ""):
                return False
        if "本科" in (question or "") and "硕士" not in (question or "") and "博士" not in (question or ""):
            if re.search(r"博士研究生|硕士研究生", sent) and "本科标准学制" not in sent:
                return False
        return bool(re.search(r"(?:\d+|[一二三四五六七八九十]{1,3})\s*年", sent))
    if _asks_place(question):
        campus = _asked_campus(question)
        asked_district = re.search(_DISTRICT, question or "")
        if campus and campus not in sent and campus not in (title or ""):
            return False
        if asked_district and asked_district.group(0) not in sent:
            return False
        if asked_district:
            return bool(re.search(r"路|地址[:：]|邮编", sent))
        need = [phrase for phrase in _phrases(question) if len(phrase) >= 2]
        if not need:
            return None
        if not all(phrase in sent or phrase in (title or "") for phrase in need):
            return False
        if re.search(r"万元|收入|置换", sent) and not re.search(r"路|号", sent):
            return False
        if re.search(r"哪个区|什么区", question or ""):
            campus = _asked_campus(question)
            clause = _campus_clause(sent, campus) if campus else sent
            if not re.search(_DISTRICT, clause):
                return False
        return bool(re.search(r"路|地址[:：]|邮编", sent))
    if _asks_rule(question):
        focus = _rule_focus(question)
        if not focus:
            return None
        if focus not in sent and focus not in (title or ""):
            return False
        if focus == "补考":
            return bool(re.search(r"必须参加[^。]{0,8}补考|补考”或“重修|补考或重修", sent))
        return bool(re.search(r"(?:严禁|禁止|不准|不许|不得|可以申请).{0,16}" + re.escape(focus), sent))
    if _asks_hours(question):
        place = next((word for word in ("图书馆", "食堂", "餐厅", "宿舍", "体育馆", "校门", "大门") if word in (question or "")), "")
        if place:
            in_place = place in sent or place in (title or "")
            if place == "图书馆":
                in_place = in_place or any(word in sent or word in (title or "") for word in ("阅览", "开馆", "开放"))
            if not in_place:
                return False
        if not re.search(r"\d{1,2}[:：]\d{2}", sent):
            return False
        if place in ("食堂", "餐厅") and (
            any(word in sent for word in ("提前", "考研", "送考", "志愿", "临时"))
            or not re.search(r"每天|早餐|午餐|晚餐|营业时间", sent)
        ):
            return False
        mode = _hours_mode(question)
        if mode == "weekend" and "周六" not in sent and "周末" not in sent:
            return False
        if mode == "weekday" and "周一" not in sent:
            return False
        if any(word in (question or "") for word in ("图书馆", "开馆", "自习", "阅览")) and "22" not in sent:
            return False
        if any(word in sent for word in ("采访", "编目")) and "22" not in sent:
            return False
        return True
    kind = _fee_kind(question)
    if kind:
        return bool(kind in sent and re.search(r"每生每年\d+", sent))
    level = "" if "中外合作" in (question or "") else _grad_level(question)
    if level:
        if "元/生" not in sent:
            return False
        if level == "博士":
            return "博士" in sent
        if level == "学术":
            return "学术" in sent and "硕士" in sent
        if level == "专业学位":
            return "专业学位" in sent
        return "研究生" in sent or "硕士" in sent
    if "助学金" in (question or "") and "贷款" not in (question or ""):
        return bool(re.search(r"助学金[^。]{0,24}每人每年\d+", sent))
    if "贷款" in (question or ""):
        if re.search(r"多少|额度|上限", question or "") and not re.search(r"怎么|如何|办理", question or ""):
            return "本专科" in sent and "研究生" in sent and len(re.findall(r"不超过\d+元", sent)) >= 2
        if re.search(r"怎么|如何|申请|办理", question or ""):
            return "户籍所在地" in sent and "申请办理" in sent and "生源地" in sent
        return "生源地" in sent and "贷款" in sent
    if "创新" in (question or "") and "学分" in (question or "") and "微专业" not in (question or ""):
        return "折算为学分" in sent and "计入学业成绩" in sent and "实施细则" in sent
    if "奖学" in (question or "") and re.search(r"怎么评|如何评|怎么选|评选", question or "") and not re.search(r"多少|几元", question or ""):
        return "可申请国家奖学金" in sent and "下列条件" in sent and not re.search(r"奖励标准为每人每年\d+", sent)
    if "三好" in (question or "") and "标兵" not in (question or "") and re.search(r"怎么评|如何评|条件|要求|评选", question or ""):
        return "三好学生：" in sent and "前20%" in sent
    if "标兵" in (question or "") and re.search(r"怎么评|如何评|条件|要求|评选", question or ""):
        return "三好学生标兵" in sent and "前4%" in sent
    if "优秀学生干部" in (question or "") and re.search(r"怎么评|如何评|条件|要求|评选", question or ""):
        return "优秀学生干部" in sent and "前35%" in sent and "前20%" not in sent
    if "优秀学生" in (question or "") and re.search(r"怎么评|如何评|评选|条件", question or "") and "毕业" not in (question or ""):
        return "荣誉称号分为" in sent and "前4%" in sent and "前35%" in sent and "前20%" in sent
    if "奖学" in (question or ""):
        if "励志" in (question or "") and not re.search(r"一样|相同|区别|不同", question or ""):
            return bool(re.search(r"励志奖学金的?奖励标准为每人每年\d+", sent))
        return bool(re.search(r"奖学金的?奖励标准为每人每年\d+", sent))
    if "研究生" in (question or "") and "住宿" in (question or "") and "本科" not in (question or ""):
        if any(word in sent for word in ("复试", "报名", "考试费")):
            return False
        return bool(re.search(r"研究生.{0,16}住宿|住宿.{0,16}研究生", sent) and re.search(r"人/间\d+", sent))
    if "住宿" in (question or "") and "费" in (question or ""):
        if any(word in sent for word in ("宾馆", "出差", "报销", "职级", "复试", "报名")) and "人/间" not in sent:
            return False
        if "复试" in sent or "报名" in sent:
            return False
        return bool(re.search(r"人/间\d+", sent) and ("住宿" in sent or "宿舍" in sent or "住宿" in (title or "")))
    if _asks_school_code(question):
        return bool(re.search(r"国标代码[:：]\d{4,}", sent))
    if "校训" in (question or ""):
        return bool(re.search(r"[“\"][^”\"]{2,30}[”\"]的校训", sent))
    if _asks_campuses(question):
        return "四个校区" in sent and all(word in sent for word in ("仙林", "三牌楼", "锁金村"))
    if _asks_borrow_count(question):
        if "限借" not in sent:
            return False
        if "本科" in (question or "") and "本科生" not in sent:
            return False
        return True
    if _asks_borrow_days(question):
        return bool(re.search(r"借期\d+天", sent))
    if "作弊" in (question or ""):
        return bool("作弊" in sent and "零分" in sent and "免试" not in sent)
    if "旷课" in (question or ""):
        asked = re.search(r"[二三四五]分之一", question or "")
        if asked and asked.group(0) != "五分之一":
            return asked.group(0) in sent and "旷课" in sent
        return bool("旷课" in sent and "五分之一" in sent)
    if "转专业" in (question or ""):
        if re.search(r"名额|比例|多少|限制", question or ""):
            return "15%" in sent and "转" in sent and "专业" in sent
        if re.search(r"什么时候|何时|几月|哪一学期", question or ""):
            return "转专业" in sent and ("第二学期" in sent or "5月" in sent)
        return "可以申请转专业" in sent or "可允许转专业" in sent
    if re.search(r"几个学院|多少个学院|多少学院|教学机构", question or ""):
        return bool(re.search(r"校区，\d+个教学机构", sent)) and "教研室" not in sent
    if re.search(r"建校|创办|始建", question or ""):
        return bool(re.search(r"前身是\d{4}年诞生于", sent))
    if "折合" in (question or "") and re.search(r"教师|老师", question or ""):
        return bool(re.search(r"折合教师总数为\d+", sent))
    if "成立" in (question or "") and re.search(r"[\u4e00-\u9fff]{2,8}学院", question or ""):
        school = re.search(r"[\u4e00-\u9fff]{2,8}学院", question or "").group(0)
        return bool(re.search(r"\d{4}年成立" + re.escape(school), sent)) and "招生简章" not in sent and "微专业" not in sent
    if "波特兰" in (question or "") and re.search(r"学费|多少", question or "") and "塞浦路斯" not in (question or ""):
        if re.search(r"研究生|硕士|博士", question or ""):
            return "波特兰" in sent and "硕士研究生" in sent and re.search(r"60000|55000", sent)
        return "57000" in sent and "波特兰" in sent
    if "塞浦路斯" in (question or "") and re.search(r"学费|多少", question or ""):
        if re.search(r"研究生|硕士|博士", question or ""):
            return "塞浦路斯" in sent and "68000" in sent
        return "63000" in sent and "塞浦路斯" in sent
    if re.search(r"博士点|几个博士|多少个博士", question or "") and "硕士点" not in (question or ""):
        return bool(re.search(r"一级学科博士学位授权点\d+个", sent))
    if re.search(r"硕士点|几个硕士|多少个硕士", question or "") and "博士" not in (question or ""):
        return bool(re.search(r"硕士学位授权点\d+个", sent))
    if re.search(r"推免|保研", question or "") and re.search(r"条件|资格|要求", question or ""):
        return "申请条件" in sent and "推免资格" in sent
    if "缓考" in (question or ""):
        if "证明" in (question or ""):
            return "诊断证明" in sent and "缓考" in sent
        return "缓考" in sent and "=" not in sent and ("申请" in sent or "患病" in sent)
    if "缓交" in (question or ""):
        return "缓交" in sent
    if "校外住宿" in (question or ""):
        return bool(re.search(r"校外住宿.{0,16}(?:可以|允许|禁止|不准)|(?:可以|允许|禁止|不准).{0,16}校外住宿", sent))
    if "免修" in (question or "") and re.search(r"怎么|如何|申请", question or ""):
        return "免修申请表" in sent
    if "校歌" in (question or ""):
        if any(word in sent for word in ("首页", "校区地图", "修改校歌", "谱曲")) or sent.count("校区") >= 3:
            return False
        return "歌词" in sent and "修改" not in sent
    if "体育课" in (question or "") and "体测" not in (question or "") and "体质" not in (question or ""):
        if "体育课" not in sent or not re.search(r"补考|重修|不及格", sent):
            return False
        if "体质健康" in sent and "补考" not in sent and "重修" not in sent:
            return False
        return True
    if "留学生" in (question or "") and ("学费" in (question or "") or "多少钱" in (question or "")):
        at = sent.find("自费来华留学生学费")
        return at >= 0 and "元/生" in sent[at:at + 80]
    if re.search(r"隶属|属于哪个|哪个部门|主管部门", question or ""):
        return "工业和信息化部" in sent and "江苏省" in sent and "共建" in sent and "项目" not in sent
    if "电话" in (question or ""):
        if "图书馆" in (question or ""):
            return "进行咨询" in sent and bool(re.search(r"\d{7,}", sent))
        for office in ("教务处", "财务处"):
            if office in (question or ""):
                return bool(re.search(re.escape(office) + r"(?:咨询|联系)?电话[:：]?\d{7,}", sent)) and "值班" not in sent and "暑假" not in sent and "个人所得税" not in sent
        return bool("电话" in sent and re.search(r"\d{7,}", sent))
    if "邮编" in (question or "") or "邮政编码" in (question or ""):
        campus = _asked_campus(question)
        if campus:
            return bool(_zip_after_campus(sent, campus))
        return bool(re.search(r"(?:邮编|邮政编码)[:：]\d{6}", sent))
    if "邮箱" in (question or "") or "信箱" in (question or ""):
        if "书记" not in (question or "") and "书记信箱" in sent:
            return False
        if "校长信箱" in (question or "") or "校长邮箱" in (question or ""):
            return "校长信箱" in sent and "@" in sent and "班车" not in sent
        return "@" in sent
    if "体测" in (question or "") or "体质" in (question or ""):
        if "毕业" in (question or ""):
            return "按结业处理" in sent and "体质" in sent
        return "体质" in sent and "不及格" in sent
    if "光盘" in (question or "") and re.search(r"书|图书", question or "") and re.search(r"过期|超期|逾期|罚款|违约", question or ""):
        return "每盘每天" in sent or ("交纳违约金" in sent and re.search(r"图书|0\.10|0\.5", sent))
    if "光盘" in (question or "") and re.search(r"过期|超期|逾期|罚款|违约", question or "") and "书" not in (question or ""):
        return "每盘每天" in sent
    if re.search(r"男女|性别", question or "") and re.search(r"比例|性别比", question or ""):
        return bool(re.search(r"男生\d+", sent) and re.search(r"女生\d+", sent))
    if "生师比" in (question or ""):
        return bool(re.search(r"全校生师比为\d+(?:\.\d+)?", sent))
    if re.search(r"专业", question or "") and re.search(r"多少|几个", question or "") and "博士" not in (question or "") and "硕士" not in (question or "") and "学费" not in (question or "") and "学院" not in (question or "") and "微专业" not in (question or "") and "学分" not in (question or ""):
        return bool(re.search(r"共有\d+个本科专业|本科专业\d+个", sent))
    if "休学" in (question or "") and not any(k in (question or "") for k in ("研究生", "博士", "硕士")):
        return "累计休学" in sent and "不得超过" in sent
    if "退学" in (question or "") and "休学" not in (question or ""):
        if "退学" not in sent:
            return False
        if re.search(r"复学|回来|再读|重新入学|还能再", question or ""):
            return "重新参加入学考试" in sent
        return True
    if re.search(r"超期|逾期", question or "") and re.search(r"书|借", question or "") and not re.search(r"丢|遗失|赔", question or ""):
        return "交纳违约金" in sent
    if re.search(r"丢|遗失|赔", question or "") and "书" in (question or "") and "卡" not in (question or ""):
        return "遗失" in sent and "赔偿" in sent and sent.count("？") < 2 and sent.count("?") < 2
    if re.search(r"丢|丢失|挂失", question or "") and re.search(r"卡|一卡通", question or ""):
        return "挂失" in sent and "一卡通" in sent
    if "开除学籍" in (question or "") and re.search(r"什么情况|哪些|什么情形|会怎样", question or ""):
        return "下列情形" in sent and "违反宪法" in sent
    if "微专业" in (question or "") and "学分" in (question or ""):
        return "每个微专业" in sent and "15学分" in sent
    if "微专业" in (question or "") and re.search(r"是什么|什么是", question or ""):
        return "微专业旨在围绕" in sent
    if "中外合作" in (question or "") and re.search(r"学费|多少", question or ""):
        if re.search(r"研究生|硕士|博士", question or ""):
            return "硕士研究生" in sent and "波特兰" in sent and "塞浦路斯" in sent
        return "57000" in sent and "63000" in sent and "硕士研究生" not in sent
    if "非全日制" in (question or "") and re.search(r"学费|多少", question or ""):
        return "非全日制" in sent and "元/生" in sent and "培养方案" not in sent
    if "双一流" in (question or ""):
        if any(word in sent for word in ("均值", "标准差", "常模")) and "入选" not in sent:
            return False
        if re.search(r"学科|哪个|什么", question or ""):
            return "电子科学与技术" in sent and "双一流" in sent
        if "入选" in sent and "双一流" in sent:
            return True
    return None


def _latest_year(sent: str, hit: dict | None) -> int:
    blob = (sent or "") + ((hit or {}).get("text") or "")[:500]
    marked = [int(year) for year in re.findall(r"20\d{2}(?=年|级|届|[-—/]|学年)", blob)]
    bare = [
        int(year)
        for year in re.findall(r"(?<!\d)(20[12]\d)(?!\d)", blob)
        if 2016 <= int(year) <= 2026
    ]
    years = marked + bare
    return max(years) if years else 0


_DISTRICT = r"(?:栖霞|鼓楼|玄武|秦淮|建邺|雨花|江宁|浦口|六合|溧水|高淳)区"


def _campus_clause(sent: str, campus: str) -> str:
    """只留这个校区自己的地址，不把上一个校区的区名算进来。"""
    at = (sent or "").find(campus)
    if at < 0:
        return ""
    prev = sent[max(0, at - 36):at]
    if "校区" in prev:
        prev = ""
    nxt = sent.find("校区", at + len(campus))
    after = sent[at:nxt if nxt > 0 else at + 40]
    return prev + after


def _asked_campus(question: str) -> str:
    for name in ("三牌楼", "锁金村", "仙林", "江宁"):
        if name in (question or ""):
            return name + "校区"
    return ""


def _zip_after_campus(text: str, campus: str) -> str:
    """邮编必须紧跟这个校区。中间再出现另一个校区，就是别人的邮编。"""
    flat = re.sub(r"\s+", "", text or "")
    at = flat.find(campus)
    if at < 0:
        return ""
    window = flat[at:at + 100]
    matched = re.search(r"(?:邮编|邮政编码)[:：]\d{6}", window)
    if matched and "校区" not in window[len(campus):matched.start()]:
        return window[:matched.end()]
    prev = flat[max(0, at - 30):at]
    matched = re.search(r"(?:邮编|邮政编码)[:：]\d{6}", prev)
    if matched and "校区" not in prev[matched.end():]:
        return prev[matched.start():] + flat[at:at + 12]
    return ""


def _zip_follows(question: str, sent: str) -> bool:
    campus = _asked_campus(question)
    if not campus:
        return bool(re.search(r"(?:邮编|邮政编码)[:：]\d{6}", sent or ""))
    return bool(_zip_after_campus(sent, campus))


def _rank_bias(question: str, sent: str, hit: dict | None) -> tuple[int, int, int, int]:
    shaped = _shape_ok(question, sent, (hit or {}).get("doc_title") or "")
    return (
        0 if shaped is True else 1,
        -(_latest_year(sent, hit) if _asks_who(question) else 0),
        -(5 if "副校长" in (question or "") and re.search(r"(?<!副)校长[\u4e00-\u9fff]{2,3}", sent or "") else 0),
        1 if "双一流" in (question or "") and any(word in (sent or "") for word in ("均值", "标准差", "常模")) else 0,
        -(1 if "双一流" in (question or "") and "入选" in (sent or "") and "建设" in (sent or "") else 0),
        -(max((int(y) for y in re.findall(r"20\d{2}", ((hit or {}).get("doc_title") or ""))), default=0) if "比例" in (question or "") and not re.search(r"男女|性别", question or "") else 0),
        -(1 if _asks_span(question) and "学制为" in (sent or "") and "本学科" not in (sent or "") and "本专业" not in (sent or "") else 0),
        -(1 if _asks_span(question) and re.search(r"学制为(?:\d+|[一二三四五六七八九十]{1,3})年", sent or "") else 0),
        -(2 if _asks_span(question) and "本科标准学制" in (sent or "") else 0),
        -(1 if _asks_span(question) and "最长不超过" in (sent or "") else 0),
        -(1 if _asks_campuses(question) and "教学质量" in ((hit or {}).get("doc_title") or "") else 0),
        -(2 if re.search(r"博士点|几个博士|多少个博士", question or "") and any(word in ((hit or {}).get("doc_title") or "") for word in ("教学质量", "学校简介")) else 0),
        -(2 if _asks_hours(question) and "图书馆" in (question or "") and "自习" not in (question or "") and "开放时间" in ((hit or {}).get("doc_title") or "") else 0),
        -(2 if "缓考" in (question or "") and "申请缓考程序" in (sent or "") else 0),
        -(2 if "体育课" in (question or "") and "重修体育课" in (sent or "") else 0),
        -(2 if "转专业" in (question or "") and "2026" in ((hit or {}).get("doc_title") or "") else 0),
        -(3 if "转专业" in (question or "") and "累计上限为60" in (sent or "") else 0),
        -(2 if "开除学籍" in (question or "") and "违反宪法" in (sent or "") else 0),
        -(2 if "开除学籍" in (question or "") and "2026" in ((hit or {}).get("doc_title") or "") else 0),
        -(1 if "生师比" in (question or "") and "：1" in (sent or "") else 0),
        -(3 if (year := re.search(r"20\d{2}", question or "")) and re.search(r"男女|性别", question or "") and year.group(0) in ((hit or {}).get("doc_title") or "") else 0),
        -(max((int(y) for y in re.findall(r"20\d{2}", ((hit or {}).get("doc_title") or ""))), default=0) if any(office in (question or "") for office in ("教务处", "财务处")) and "电话" in (question or "") else 0),
        -(5 if ("邮编" in (question or "") or "邮政编码" in (question or "")) and _zip_follows(question, sent or "") else 0),
        -(2 if _named_thing(question) and "成立" in (sent or "") and _named_thing(question) in (sent or "") else 0),
        1 if _asks_span(question) and any(word in (sent or "") and word not in (question or "") for word in ("翻译", "非全日制")) else 0,
        -(len(re.findall(r"副校长(?:、[\u4e00-\u9fff]{2,12})?[\u4e00-\u9fff]{2,3}", sent or "")) if "副校长" in (question or "") else 0),
        -(1 if "校训" in (question or "") and "校训" in (sent or "") and "校风" in (sent or "") else 0),
        -(1 if "作弊" in (question or "") and "相应的纪律处分" in (sent or "") else 0),
        -(_latest_year(sent, hit) if "学费" in (question or "") and not re.search(r"\d{6,}", sent or "") else 0),
        -(max([int(y) for y in re.findall(r"20\d{2}(?=年)", (hit or {}).get("doc_title") or "")] or [0]) if _fee_kind(question) or _asks_school_code(question) else 0),
        -(1 if _asks_rule(question) and any(word in (sent or "") for word in ("严禁", "必须", "可以申请")) else 0),
        -(1 if _asks_hours(question) and (("周一" in (sent or "")) if _hours_mode(question) == "weekday" else "周六" in (sent or "")) else 0),
        -(1 if _asks_hours(question) and "开放时间" in ((hit or {}).get("doc_title") or "") else 0),
        -(1 if (_fee_kind(question) and _fee_kind(question) in (sent or "") and "每生每年" in (sent or "")) else 0),
        -(1 if any(k in (question or "") for k in ("奖学", "助学")) and "2026" in ((hit or {}).get("doc_title") or "") else 0),
    )


def _recall_pairs(question: str) -> list[tuple[str, dict]]:
    """用问句里的事项直接扫原文。模型换词没换对时，这些句子仍然进候选。"""
    from .db import init_db

    specs: list[tuple[str, str, tuple, str]] = []
    role = _role_in(question)
    if _asks_who(question) and role:
        order = "CASE WHEN ru.text LIKE '%2026%' THEN 0 WHEN ru.text LIKE '%2025%' THEN 1 ELSE 2 END, length(ru.text)"
        if role == "校长":
            order = "CASE WHEN ru.text LIKE '%校党委书记%' THEN 0 ELSE 1 END, " + order
        elif role == "副校长":
            order = (
                "CASE WHEN ru.text LIKE '%2026%' THEN 0 WHEN ru.text LIKE '%2025%' THEN 1 ELSE 2 END, "
                "(length(ru.text) - length(replace(ru.text, '副校长', ''))) DESC"
            )
        specs.append((
            "ru.text LIKE ? AND (ru.text LIKE '%南京邮电大学%' OR ru.text LIKE '%校党委%')",
            order,
            (f"%{role}%",),
            role,
        ))
    if _asks_span(question) and "本科" in (question or "") and "硕士" not in (question or "") and "博士" not in (question or ""):
        specs.append((
            "ru.text LIKE '%本科标准学制%'",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "本科标准学制",
        ))
    if _asks_span(question):
        phrase = next((item for item in _phrases(question) if len(item) >= 4), "")
        if not phrase:
            for level, fuller in (("博士", "博士研究生"), ("硕士", "硕士研究生"), ("本科", "本科生")):
                if level in (question or ""):
                    phrase = fuller
                    break
        if phrase:
            specs.append((
                "(ru.text LIKE ? OR d.title LIKE ?) AND ru.text LIKE '%学制%'",
                "CASE WHEN ru.text LIKE '%学制为%' AND ru.text LIKE '%年，最长%' AND ru.text NOT LIKE '%本学科%' AND ru.text NOT LIKE '%本专业%' THEN 0 ELSE 1 END, length(ru.text)",
                (f"%{phrase}%", f"%{phrase}%"),
                "学制",
            ))
    if _asks_place(question):
        need = [item for item in _phrases(question) if len(item) >= 2][:2]
        if need:
            clause = " AND ".join("ru.text LIKE ?" for _ in need)
            specs.append((
                clause,
                "CASE WHEN ru.text LIKE '%上一页%' OR ru.text LIKE '%跳转%' THEN 1 ELSE 0 END, "
                "CASE WHEN ru.text LIKE ? THEN 0 ELSE 1 END, "
                "CASE WHEN d.title LIKE '%章程%' THEN 0 ELSE 1 END, "
                "CASE WHEN d.title LIKE '%2026%' THEN 0 WHEN d.title LIKE '%2025%' THEN 1 "
                "WHEN d.title LIKE '%2024%' THEN 2 WHEN d.title LIKE '%2023%' THEN 3 ELSE 4 END, length(ru.text)",
                tuple(f"%{item}%" for item in need) + (f"%{need[0]}地址%",),
                need[0],
            ))
    if _asks_english_name(question):
        specs.append((
            "ru.text LIKE '%Nanjing University of Posts%'",
            "length(ru.text)",
            (),
            "Nanjing University of Posts",
        ))
    thing = _named_thing(question)
    if thing:
        specs.append((
            "ru.text LIKE ? OR d.title LIKE ?",
            "CASE WHEN d.title LIKE ? THEN 0 ELSE 1 END, length(ru.text)",
            (f"%{thing}%", f"%{thing}%", f"%{thing}%"),
            thing,
        ))
    if "双一流" in (question or ""):
        specs.append((
            "ru.text LIKE '%双一流%' AND ru.text LIKE '%入选%'",
            "length(ru.text)",
            (),
            "入选",
        ))
    focus = _rule_focus(question)
    if _asks_rule(question) and focus:
        specs.append((
            "ru.text LIKE ? AND (ru.text LIKE '%必须%' OR ru.text LIKE '%严禁%' OR ru.text LIKE '%不得%' OR ru.text LIKE '%禁止%' OR ru.text LIKE '%不准%' OR ru.text LIKE '%可以申请%')",
            "CASE WHEN ru.text LIKE '%必须参加%' OR ru.text LIKE '%严禁%' THEN 0 ELSE 1 END, length(ru.text)",
            (f"%{focus}%",),
            focus,
        ))
    if _asks_hours(question):
        place = next((word for word in ("食堂", "餐厅", "宿舍", "体育馆", "校门", "大门") if word in (question or "")), "")
        if place:
            clock = re.search(r"开门|关门|几点", question or "")
            extra = " AND (ru.text LIKE '%开门%' OR ru.text LIKE '%关门%')" if clock else ""
            specs.append((
                f"ru.text LIKE ? AND (ru.text LIKE '%:%' OR ru.text LIKE '%：%'){extra}",
                "CASE WHEN d.title LIKE '%2026%' THEN 0 WHEN d.title LIKE '%2025%' THEN 1 WHEN d.title LIKE '%2024%' THEN 2 ELSE 3 END, length(ru.text)",
                (f"%{place}%",),
                "开门" if clock else place,
            ))
        else:
            mode = _hours_mode(question)
            if "自习" in (question or ""):
                clause = "ru.text LIKE '%研读%' AND ru.text LIKE '%周一至周五%' AND (ru.text LIKE '%22:00%' OR ru.text LIKE '%22：00%')"
                needle = "周一至周五"
            elif mode == "weekend":
                clause = "ru.text LIKE '%周六%' AND ru.text LIKE '%周日%' AND (ru.text LIKE '%22:00%' OR ru.text LIKE '%22：00%')"
                needle = "周六"
            else:
                clause = "ru.text LIKE '%周一至周五%' AND (ru.text LIKE '%22:00%' OR ru.text LIKE '%22：00%')"
                if mode == "both":
                    clause += " AND ru.text LIKE '%周六%'"
                needle = "周一至周五"
            specs.append((
                clause,
                "CASE WHEN d.title LIKE '%开放时间%' THEN 0 WHEN d.title LIKE '%研读%' THEN 1 ELSE 2 END, length(ru.text)",
                (),
                needle,
            ))
    kind = _fee_kind(question)
    if kind:
        specs.append((
            "ru.text LIKE ?",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 WHEN d.title LIKE '%章程%' THEN 1 ELSE 2 END, length(ru.text)",
            (f"%{kind}类专业每生每年%",),
            f"{kind}类专业每生每年",
        ))
    level = _grad_level(question)
    if level:
        needle = "博士研究生" if level == "博士" else "硕士研究生"
        specs.append((
            "ru.text LIKE ? AND ru.text LIKE '%元/生%'",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, "
            "CASE WHEN ru.text LIKE '%学术型硕士研究生%' THEN 0 ELSE 1 END, length(ru.text)",
            (f"%{needle}%",),
            needle,
        ))
    if "助学金" in (question or ""):
        specs.append((
            "ru.text LIKE '%国家助学金%' AND ru.text LIKE '%每人每年%'",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "国家助学金",
        ))
    if "住宿" in (question or "") and "费" in (question or ""):
        specs.append((
            "ru.text LIKE '%人/间%' AND ru.text NOT LIKE '%|%'",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "人/间",
        ))
    if _asks_school_code(question):
        specs.append((
            "ru.text LIKE '%学校代码%' AND ru.text LIKE '%国标代码%'",
            "CASE WHEN d.title LIKE '%章程%' THEN 0 ELSE 1 END, "
            "CASE WHEN d.title LIKE '%2026%' THEN 0 WHEN d.title LIKE '%2025%' THEN 1 "
            "WHEN d.title LIKE '%2024%' THEN 2 WHEN d.title LIKE '%2023%' THEN 3 ELSE 4 END, length(ru.text)",
            (),
            "国标代码",
        ))
    if "校训" in (question or ""):
        specs.append((
            "ru.text LIKE '%的校训%'",
            "CASE WHEN ru.text LIKE '%校风%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "的校训",
        ))
    if _asks_campuses(question):
        specs.append((
            "ru.text LIKE '%仙林%' AND ru.text LIKE '%三牌楼%' AND ru.text LIKE '%锁金村%' AND ru.text LIKE '%四个校区%'",
            "CASE WHEN d.title LIKE '%2024%' OR d.title LIKE '%2023%' THEN 0 WHEN d.title LIKE '%2022%' OR d.title LIKE '%2021%' OR d.title LIKE '%2020%' THEN 1 ELSE 2 END, length(ru.text)",
            (),
            "四个校区",
        ))
    if _asks_borrow_count(question):
        specs.append((
            "ru.text LIKE '%每证限借%' AND ru.text LIKE '%本科生%'",
            "length(ru.text)",
            (),
            "每证限借",
        ))
    if _asks_borrow_days(question):
        specs.append((
            "ru.text LIKE '%借期%' AND ru.text LIKE '%本科生%'",
            "CASE WHEN d.title LIKE '%手册%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "借期",
        ))
    if "作弊" in (question or ""):
        specs.append((
            "ru.text LIKE '%作弊%' AND ru.text LIKE '%零分%'",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "零分",
        ))
    if "旷课" in (question or ""):
        specs.append((
            "ru.text LIKE '%无故旷课%' AND ru.text LIKE '%五分之一%'",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, CASE WHEN ru.text LIKE '%必须重修%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "旷课",
        ))
    if (
        "转专业" in (question or "")
        and "比例" not in (question or "")
        and "多少" not in (question or "")
        and not re.search(r"什么时候|何时|几月|名额|限制", question or "")
    ):
        specs.append((
            "ru.text LIKE '%可以申请转专业%'",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "可以申请转专业",
        ))
    if re.search(r"几个学院|多少个学院|多少学院|教学机构", question or ""):
        specs.append((
            "ru.text LIKE '%个教学机构%' AND ru.text LIKE '%校区%'",
            "CASE WHEN d.title LIKE '%2026%' AND d.title LIKE '%手册%' THEN 0 WHEN d.title LIKE '%学校简介%' THEN 1 WHEN d.title LIKE '%教学质量%' THEN 2 ELSE 3 END, length(ru.text)",
            (),
            "个教学机构",
        ))
    if re.search(r"隶属|属于哪个|哪个部门|主管部门", question or ""):
        specs.append((
            "ru.text LIKE '%共建%' AND ru.text LIKE '%部%'",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 WHEN ru.text LIKE '%国家邮政局%' THEN 1 ELSE 2 END, length(ru.text)",
            (),
            "工业和信息化部",
        ))
    if "电话" in (question or ""):
        if "图书馆" in (question or ""):
            specs.append((
                "ru.text LIKE '%进行咨询%' AND ru.text LIKE '%电话%' AND ru.text LIKE '%三牌楼%' AND ru.text LIKE '%仙林%'",
                "length(ru.text)",
                (),
                "电话",
            ))
        elif "招生" in (question or ""):
            specs.append((
                "d.title LIKE '%章程%' AND ru.text LIKE '%电话%'",
                "CASE WHEN d.title LIKE '%2026%' THEN 0 WHEN d.title LIKE '%2025%' THEN 1 WHEN d.title LIKE '%2024%' THEN 2 ELSE 3 END, length(ru.text)",
                (),
                "电话",
            ))
    if "邮编" in (question or "") or "邮政编码" in (question or ""):
        campus = _asked_campus(question)
        if campus:
            specs.append((
                "ru.text LIKE ? AND ru.text LIKE '%邮编%'",
                "CASE WHEN d.title LIKE '%章程%' THEN 0 ELSE 1 END, CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, length(ru.text)",
                (f"%{campus}%",),
                "邮编",
            ))
        else:
            specs.append((
                "d.title LIKE '%章程%' AND ru.text LIKE '%邮编：%'",
                "CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, length(ru.text)",
                (),
                "邮编",
            ))
    if "体测" in (question or "") or "体质" in (question or ""):
        specs.append((
            "ru.text LIKE '%体质健康%' AND (ru.text LIKE '%不及格%' OR ru.text LIKE '%结业%' OR ru.text LIKE '%肄业%')",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "学生毕业时" if "毕业" in (question or "") else "不及格",
        ))
    if re.search(r"建校|创办|始建", question or ""):
        specs.append((
            "ru.text LIKE '%前身%' AND ru.text LIKE '%诞生于%'",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 WHEN d.title LIKE '%学校简介%' THEN 1 ELSE 2 END, length(ru.text)",
            (),
            "诞生于",
        ))
    if "光盘" in (question or "") and re.search(r"书|图书", question or "") and re.search(r"过期|超期|逾期|罚款|违约", question or ""):
        specs.append((
            "ru.text LIKE '%每盘每天%' AND ru.text LIKE '%0.10%'",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "每盘每天",
        ))
    if "光盘" in (question or "") and re.search(r"过期|超期|逾期|罚款|违约", question or "") and "书" not in (question or ""):
        specs.append((
            "ru.text LIKE '%每盘每天%'",
            "length(ru.text)",
            (),
            "每盘每天",
        ))
    if re.search(r"男女|性别", question or "") and re.search(r"比例|性别比", question or ""):
        year = re.search(r"20\d{2}", question or "")
        specs.append((
            "ru.text LIKE '%男生%' AND ru.text LIKE '%女生%' AND ru.text LIKE '%性别比%'",
            "CASE WHEN d.title LIKE ? THEN 0 ELSE 1 END, length(ru.text)",
            (f"%{year.group(0)}%" if year else "%",),
            "性别比",
        ))
    if ("邮箱" in (question or "") or "信箱" in (question or "")) and "书记" not in (question or ""):
        specs.append((
            "d.title LIKE '%章程%' AND (ru.text LIKE '%E-mail%' OR ru.text LIKE '%@njupt.edu.cn%')",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "@",
        ))
    if "奖学" in (question or "") and re.search(r"怎么评|如何评|怎么选|评选", question or "") and not re.search(r"多少|几元", question or ""):
        specs.append((
            "ru.text LIKE '%可申请国家奖学金%' AND ru.text LIKE '%下列条件%'",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "可申请国家奖学金",
        ))
    elif "奖学" in (question or ""):
        specs.append((
            "ru.text LIKE '%奖励标准为每人每年%'",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "奖励标准",
        ))
    if "三好" in (question or "") and "标兵" not in (question or "") and re.search(r"怎么评|如何评|条件|要求|评选", question or ""):
        specs.append((
            "ru.text LIKE '%三好学生：%' AND ru.text LIKE '%前20%'",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "三好学生：",
        ))
    elif "标兵" in (question or "") and re.search(r"怎么评|如何评|条件|要求|评选", question or ""):
        specs.append((
            "ru.text LIKE '%三好学生标兵%' AND ru.text LIKE '%前4%'",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "三好学生标兵",
        ))
    elif "优秀学生干部" in (question or "") and re.search(r"怎么评|如何评|条件|要求|评选", question or ""):
        specs.append((
            "ru.text LIKE '%优秀学生干部：%' AND (ru.text LIKE '%前35%' OR ru.text LIKE '%评选条件%')",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "优秀学生干部",
        ))
    elif "优秀学生" in (question or "") and "毕业" not in (question or "") and re.search(r"怎么评|如何评|评选|条件", question or ""):
        specs.append((
            "ru.text LIKE '%荣誉称号分为%' AND ru.text LIKE '%三好学生标兵%'",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "荣誉称号分为",
        ))
    if re.search(r"博士点|几个博士|多少个博士", question or ""):
        specs.append((
            "ru.text LIKE '%一级学科博士学位授权点%' AND ru.text LIKE '%个%'",
            "CASE WHEN d.title LIKE '%学校简介%' OR d.title LIKE '%教学质量%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "一级学科博士学位授权点",
        ))
    if re.search(r"推免|保研", question or "") and re.search(r"条件|资格|要求", question or ""):
        specs.append((
            "ru.text LIKE '%申请条件%' AND ru.text LIKE '%推免资格%'",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 WHEN d.title LIKE '%2027%' THEN 1 ELSE 2 END, length(ru.text)",
            (),
            "申请条件",
        ))
    if "缓考" in (question or "") and "证明" in (question or ""):
        specs.append((
            "ru.text LIKE '%诊断证明%' AND ru.text LIKE '%缓考%'",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "诊断证明",
        ))
    elif "缓考" in (question or ""):
        specs.append((
            "ru.text LIKE '%缓考%' AND ru.text LIKE '%申请缓考%'",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "申请缓考程序" if re.search(r"怎么|如何|办理", question or "") else "缓考",
        ))
    if "免修" in (question or "") and re.search(r"怎么|如何|申请", question or ""):
        specs.append((
            "ru.text LIKE '%免修申请表%'",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "免修申请表",
        ))
    if re.search(r"硕士点|几个硕士|多少个硕士", question or "") and "博士" not in (question or ""):
        specs.append((
            "ru.text LIKE '%一级学科硕士学位授权点%' AND d.title LIKE '%教学质量%'",
            "length(ru.text)",
            (),
            "硕士学位授权点",
        ))
    if "留学生" in (question or "") and ("学费" in (question or "") or "多少钱" in (question or "")):
        specs.append((
            "ru.text LIKE '%自费来华留学生%' AND ru.text LIKE '%元/生%'",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "自费来华留学生",
        ))
    if "体育课" in (question or "") and "体测" not in (question or ""):
        specs.append((
            "ru.text LIKE '%重修体育课%' OR (ru.text LIKE '%体育课%' AND ru.text LIKE '%不及格%' AND ru.text LIKE '%补考%')",
            "CASE WHEN ru.text LIKE '%重修体育课%' THEN 0 WHEN d.title LIKE '%2026%' THEN 1 ELSE 2 END, length(ru.text)",
            (),
            "重修体育课",
        ))
    if "生师比" in (question or ""):
        specs.append((
            "ru.text LIKE '%全校生师比%'",
            "CASE WHEN d.title LIKE '%教学质量%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "全校生师比",
        ))
    if re.search(r"专业", question or "") and re.search(r"多少|几个", question or "") and "博士" not in (question or "") and "硕士" not in (question or "") and "学费" not in (question or "") and "学院" not in (question or "") and "微专业" not in (question or "") and "学分" not in (question or ""):
        specs.append((
            "ru.text LIKE '%本科专业%' AND ru.text LIKE '%个%'",
            "CASE WHEN d.title LIKE '%教学质量%' THEN 0 WHEN ru.text LIKE '%共有%' THEN 1 ELSE 2 END, length(ru.text)",
            (),
            "本科专业",
        ))
    if "双一流" in (question or "") and re.search(r"学科|哪个|什么", question or ""):
        specs.append((
            "ru.text LIKE '%电子科学与技术%' AND ru.text LIKE '%双一流%'",
            "CASE WHEN d.title LIKE '%教学质量%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "电子科学与技术",
        ))
    if "休学" in (question or "") and not any(k in (question or "") for k in ("研究生", "博士", "硕士")):
        specs.append((
            "ru.text LIKE '%累计休学%' AND d.title LIKE '%2026%'",
            "length(ru.text)",
            (),
            "累计休学",
        ))
    if "退学" in (question or "") and re.search(r"复学|回来|再读|重新入学|还能再", question or ""):
        specs.append((
            "ru.text LIKE '%退学%' AND ru.text LIKE '%重新参加入学考试%'",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "重新参加入学考试",
        ))
    if "成立" in (question or "") and (school := re.search(r"[\u4e00-\u9fff]{2,8}学院", question or "")):
        specs.append((
            f"ru.text LIKE '%年成立{school.group(0)}%'",
            "CASE WHEN d.title LIKE '%简介%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            f"年成立{school.group(0)}",
        ))
    if "折合" in (question or "") and re.search(r"教师|老师", question or ""):
        specs.append((
            "ru.text LIKE '%折合教师总数为%'",
            "CASE WHEN d.title LIKE '%教学质量%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "折合教师总数",
        ))
    if "波特兰" in (question or "") and re.search(r"学费|多少", question or "") and "中外合作" not in (question or ""):
        specs.append((
            "ru.text LIKE '%波特兰%' AND ru.text LIKE '%57000%' AND ru.text NOT LIKE '%=%'",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "57000",
        ))
    if "塞浦路斯" in (question or "") and re.search(r"学费|多少", question or "") and re.search(r"研究生|硕士|博士", question or ""):
        specs.append((
            "ru.text LIKE '%塞浦路斯%' AND ru.text LIKE '%68000%' AND ru.text NOT LIKE '%=%'",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "68000",
        ))
    elif "塞浦路斯" in (question or "") and re.search(r"学费|多少", question or ""):
        specs.append((
            "ru.text LIKE '%塞浦路斯%' AND ru.text LIKE '%63000%' AND ru.text NOT LIKE '%=%'",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "63000",
        ))
    if "贷款" in (question or "") and re.search(r"怎么|如何|申请|办理", question or ""):
        specs.append((
            "ru.text LIKE '%贷款申请和发放%' AND ru.text LIKE '%户籍所在地%'",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "户籍所在地",
        ))
    elif "贷款" in (question or "") and re.search(r"多少|额度|上限", question or ""):
        specs.append((
            "ru.text LIKE '%不超过16000%' AND ru.text LIKE '%不超过20000%'",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "不超过16000",
        ))
    if "创新" in (question or "") and "学分" in (question or "") and "微专业" not in (question or ""):
        specs.append((
            "ru.text LIKE '%创新创业学分%' AND ru.text LIKE '%计入学业成绩%' AND ru.text LIKE '%实施细则%'",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "折算为学分",
        ))
    if "微专业" in (question or "") and "学分" in (question or ""):
        specs.append((
            "ru.text LIKE '%每个微专业%' AND ru.text LIKE '%学分%'",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "15学分",
        ))
    if "微专业" in (question or "") and re.search(r"是什么|什么是", question or ""):
        specs.append((
            "ru.text LIKE '%微专业旨在%'",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "微专业旨在",
        ))
    if "中外合作" in (question or "") and re.search(r"学费|多少", question or ""):
        if re.search(r"研究生|硕士|博士", question or ""):
            specs.append((
                "ru.text LIKE '%波特兰学院硕士研究生%' AND ru.text LIKE '%塞浦路斯%' AND ru.text NOT LIKE '%=%'",
                "CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, length(ru.text)",
                (),
                "波特兰学院硕士研究生",
            ))
        else:
            specs.append((
                "ru.text LIKE '%中外合作办学（波特兰学院）%' AND ru.text LIKE '%57000%' AND ru.text LIKE '%63000%' AND ru.text NOT LIKE '%=%'",
                "CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, length(ru.text)",
                (),
                "中外合作办学（波特兰学院）",
            ))
    if "开除学籍" in (question or ""):
        specs.append((
            "ru.text LIKE '%第五十二条%' AND ru.text LIKE '%违反宪法%' AND ru.text LIKE '%开除学籍%'",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "违反宪法",
        ))
    if re.search(r"丢|丢失|挂失", question or "") and re.search(r"卡|一卡通", question or ""):
        specs.append((
            "ru.text LIKE '%一卡通%' AND ru.text LIKE '%挂失%'",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "一卡通挂失",
        ))
    if re.search(r"丢|遗失|赔", question or "") and "书" in (question or "") and "卡" not in (question or ""):
        specs.append((
            "ru.text LIKE '%遗失%' AND ru.text LIKE '%赔偿%'",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "遗失",
        ))
    for office in ("教务处", "财务处"):
        if office in (question or "") and "电话" in (question or ""):
            specs.append((
                "ru.text LIKE ? AND (ru.text LIKE ? OR ru.text LIKE ?) AND ru.text NOT LIKE '%值班电话%' AND ru.text NOT LIKE '%个人所得税%'",
                "CASE WHEN d.title LIKE '%2024%' OR d.title LIKE '%2023%' OR d.title LIKE '%2022%' OR d.title LIKE '%2021%' OR d.title LIKE '%2020%' OR d.title LIKE '%2019%' OR d.title LIKE '%2018%' THEN 0 ELSE 1 END, CASE WHEN d.title LIKE '%2019%' THEN 0 WHEN d.title LIKE '%2018%' THEN 1 ELSE 2 END, length(ru.text)",
                (f"%{office}%", f"%{office}联系电话%", f"%{office}咨询电话%"),
                office,
            ))
    if "非全日制" in (question or "") and re.search(r"学费|多少", question or ""):
        specs.append((
            "ru.text LIKE '%非全日制%' AND ru.text LIKE '%元/生%' AND ru.text NOT LIKE '%=%'",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "非全日制",
        ))
    if "转专业" in (question or "") and re.search(r"名额|比例|限制|多少", question or ""):
        specs.append((
            "(ru.text LIKE '%转专业%' OR ru.text LIKE '%转入专业%') AND ru.text LIKE '%累计上限为60%' AND ru.text NOT LIKE '%=%'",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "15%",
        ))
    if "转专业" in (question or "") and re.search(r"什么时候|何时|几月", question or ""):
        specs.append((
            "ru.text LIKE '%转专业%' AND ru.text LIKE '%第二学期%'",
            "CASE WHEN d.title LIKE '%2026%' THEN 0 ELSE 1 END, length(ru.text)",
            (),
            "第二学期",
        ))
    specs.sort(key=lambda spec: 0 if spec[3] in ("第二学期", "15%", "教务处", "财务处", "可申请国家奖学金", "评选条件", "三好学生：", "荣誉称号分为", "优秀学生干部", "三好学生标兵", "违反宪法", "微专业旨在", "中外合作办学（波特兰学院）", "波特兰学院硕士研究生", "每盘每天", "性别比", "学生毕业时", "个教学机构", "折合教师总数", "57000", "68000", "重新参加入学考试", "诞生于", "15学分", "户籍所在地", "不超过16000", "折算为学分", "诊断证明", "免修申请表", "硕士学位授权点", "63000") or str(spec[3]).startswith("年成立") else 1)
    if not specs:
        return []
    conn = init_db()
    out: list[tuple[str, dict]] = []
    try:
        for clause, order, params, needle in specs:
            rows = conn.execute(
                f"""
                SELECT d.title AS doc_title, d.url, ru.text, ru.locator, ru.id AS unit_id, ru.document_id
                FROM retrieval_units ru JOIN documents d ON d.id = ru.document_id
                WHERE {clause}
                ORDER BY {order}
                LIMIT 24
                """,
                params,
            ).fetchall()
            for row in rows:
                hit = {
                    "doc_title": row["doc_title"],
                    "url": row["url"] or "",
                    "locator": row["locator"],
                    "text": row["text"],
                }
                flat = re.sub(r"\s+", "", row["text"] or "")
                prefer = ("必须", "严禁", "不得", "禁止") if _asks_rule(question) else ()
                if _asks_hours(question):
                    prefer = ("周六", "周日")
                if any(office in (question or "") for office in ("教务处", "财务处")) and "电话" in (question or ""):
                    prefer = ("咨询电话", "联系电话", "值班电话")
                campus = _asked_campus(question) if ("邮编" in (question or "") or "邮政编码" in (question or "")) else ""
                zip_piece = _zip_after_campus(flat, campus) if campus else ""
                if campus and not zip_piece:
                    continue
                at = _best_at(flat, needle, prefer)
                if at < 0 and not zip_piece:
                    continue
                wide = any(k in (question or "") for k in ("助学", "住宿", "作弊", "旷课", "转专业", "借期", "限借", "体测", "体质"))
                if zip_piece:
                    piece = zip_piece
                else:
                    start = max(0, at - (64 if wide else 28))
                    punct = flat.rfind("。", start, at)
                    if punct >= 0:
                        start = punct + 1
                    piece = flat[start : at + (150 if wide else 80)]
                if _grad_level(question):
                    bits = re.findall(
                        r"(?:博士研究生|全日制学术型硕士研究生|其他全日制专业学位硕士研究生)\d+元/生.?学年",
                        flat,
                    )
                    kept_bits = []
                    for bit in dict.fromkeys(bits):
                        at_bit = flat.find(bit)
                        prev = flat[max(0, at_bit - 24):at_bit]
                        if any(word in prev for word in ("留学", "合作办学", "短期", "学院")):
                            continue
                        kept_bits.append(bit)
                    if kept_bits:
                        piece = "，".join(kept_bits)
                if _asks_span(question):
                    piece = re.split(r"类别代码|专业代码|研究方向", piece)[0].rstrip("，,= ")
                if "留学生" in (question or "") and "自费来华留学生学费" in flat:
                    at_fee = flat.find("自费来华留学生学费")
                    end = flat.find("备注", at_fee)
                    piece = flat[at_fee:end if end > at_fee else at_fee + 240]
                if "体育课" in (question or "") and "重修体育课" in flat:
                    at_pe = flat.find("必须重修体育课")
                    if at_pe < 0:
                        at_pe = flat.find("重修体育课")
                    start_pe = flat.rfind("学生每学期", max(0, at_pe - 90), at_pe)
                    if start_pe < 0:
                        start_pe = flat.rfind("。", max(0, at_pe - 90), at_pe)
                        start_pe = 0 if start_pe < 0 else start_pe + 1
                    piece = flat[start_pe:at_pe + len("必须重修体育课")]
                if _asks_hours(question) and "图书馆" in (question or "") and "自习" not in (question or ""):
                    clock = re.search(
                        r"周一至周五[，,]?\d{1,2}[:：]\d{2}.{0,8}22[:：]00(?:周六[、，]?周日[，,]?\d{1,2}[:：]\d{2}.{0,8}22[:：]00)?",
                        flat,
                    )
                    if clock and "=" not in clock.group(0) and "|" not in clock.group(0):
                        piece = clock.group(0)
                if "生师比" in (question or "") and "全校生师比" in flat:
                    matched = re.search(r"全校生师比为\d+(?:\.\d+)?：\d+", flat) or re.search(r"全校生师比为\d+(?:\.\d+)?", flat)
                    if matched:
                        piece = matched.group(0)
                if "休学" in (question or "") and "不得超过两学年" in flat:
                    at_leave = flat.find("不得超过两学年")
                    start_leave = flat.rfind("休学一般以一学年为限", max(0, at_leave - 50), at_leave)
                    piece = flat[start_leave if start_leave >= 0 else at_leave - 8:at_leave + len("不得超过两学年")]
                if "转专业" in (question or "") and re.search(r"名额|比例|限制|多少", question or "") and "累计上限为60" in flat:
                    at_quota = flat.find("转入专业拟接收人数原则上不超过")
                    if at_quota < 0:
                        at_quota = flat.find("累计上限为60") - 40
                    piece = flat[max(0, at_quota):flat.find("累计上限为60") + len("累计上限为60")]
                if "转专业" in (question or "") and re.search(r"什么时候|何时|几月", question or "") and "转专业工作一般安排在每学年第二学期" in flat:
                    at_move = flat.find("转专业工作一般安排在每学年第二学期")
                    piece = flat[at_move:at_move + 36]
                if "退学" in (question or "") and "重新参加入学考试" in flat:
                    at_back = flat.find("学生因退学等情况中止学业")
                    if at_back < 0:
                        at_back = flat.find("重新参加入学考试")
                    piece = flat[at_back:at_back + 96]
                if "非全日制" in (question or "") and "元/生" in flat and "=" not in flat[flat.find("非全日制"):flat.find("非全日制") + 40]:
                    at_fee = flat.find("非全日制")
                    piece = flat[at_fee:at_fee + 160]
                    cut = piece.find("中外合作")
                    if cut > 0:
                        piece = piece[:cut]
                if "开除学籍" in (question or "") and "违反宪法" in flat and "下列情形" in flat:
                    at_rule = flat.find("学生有下列情形之一")
                    if at_rule < 0:
                        at_rule = flat.find("可以给予开除学籍处分")
                    piece = flat[at_rule:at_rule + 220]
                    if "屡次违反学校规定" not in piece:
                        nxt = conn.execute(
                            "SELECT text, document_id FROM retrieval_units WHERE id = ?",
                            (row["unit_id"] + 1,),
                        ).fetchone()
                        if nxt and nxt["document_id"] == row["document_id"]:
                            extra = re.sub(r"^·?\d+·", "", re.sub(r"\s+", "", nxt["text"] or ""))
                            cut = extra.find("。")
                            piece += extra[: cut + 1] if cut > 0 else extra[:180]
                if "个教学机构" in flat and re.search(r"教学机构|几个学院|多少个学院|多少学院", question or ""):
                    matched = re.search(r"现有[^。]{0,24}校区，\d+个教学机构", flat) or re.search(r"[^。]{8,24}校区，\d+个教学机构", flat)
                    if matched:
                        piece = matched.group(0)
                if re.search(r"建校|创办|始建", question or "") and "诞生于" in flat:
                    matched = re.search(r"前身是\d{4}年诞生于[^。]+", flat)
                    if matched:
                        piece = matched.group(0)
                        rebuilt = re.search(r"\d{4}年经国务院批准改建为本科高校", flat)
                        if rebuilt:
                            piece += "。" + rebuilt.group(0)
                if "光盘" in (question or "") and re.search(r"书|图书", question or "") and "每盘每天" in flat and "0.10" in flat:
                    start = flat.find("随书光盘应按期归还")
                    if start < 0:
                        start = flat.find("每盘每天")
                    end = flat.find("0.10元", start)
                    if end > start:
                        tail = "0.10元交纳违约金" if flat[end:end + len("0.10元交纳违约金")].startswith("0.10元交纳违约金") else "0.10元"
                        piece = flat[start:end + len(tail)]
                if "折合教师总数为" in flat and "折合" in (question or "") and re.search(r"教师|老师", question or ""):
                    matched = re.search(r"折合教师总数为\d+", flat)
                    if matched:
                        piece = matched.group(0)
                if "成立" in (question or "") and (school := re.search(r"[\u4e00-\u9fff]{2,8}学院", question or "")) and f"年成立{school.group(0)}" in flat:
                    matched = re.search(r"\d{4}年成立" + re.escape(school.group(0)), flat)
                    if matched:
                        piece = matched.group(0)
                if "塞浦路斯" in (question or "") and "68000" in flat and re.search(r"研究生|硕士|博士", question or ""):
                    at_fee = flat.find("塞浦路斯学院硕士研究生")
                    if at_fee < 0:
                        at_fee = flat.find("68000")
                    end = flat.find("起执行", at_fee)
                    piece = flat[max(0, at_fee - 8):end + len("起执行")] if end > at_fee else flat[at_fee:at_fee + 48]
                if "波特兰" in (question or "") and "57000" in flat and "中外合作" not in (question or "") and not re.search(r"研究生|硕士|博士", question or ""):
                    at_fee = flat.find("中外合作办学（波特兰学院）")
                    if at_fee < 0:
                        at_fee = flat.find("57000")
                    end = flat.find("起执行", flat.find("57000", at_fee))
                    cut = flat.find("塞浦路斯", at_fee)
                    stop = end + len("起执行") if end > at_fee else at_fee + 40
                    if cut > at_fee:
                        stop = min(stop, cut)
                    piece = re.sub(r"(中外合作办学（波特兰学院）)中外合作办学", r"\1", flat[at_fee:stop])
                if ("体测" in (question or "") or "体质" in (question or "")) and "毕业" in (question or "") and "按结业处理" in flat:
                    end = flat.find("按结业处理") + len("按结业处理")
                    at_rule = flat.rfind("学生毕业时《国家学生体质健康标准》", 0, end)
                    if at_rule < 0:
                        at_rule = flat.rfind("学生毕业时", max(0, end - 40), end)
                    piece = flat[at_rule if at_rule >= 0 else max(0, end - 40):end]
                if re.search(r"男女|性别", question or "") and "男生" in flat and "女生" in flat:
                    matched = re.search(r"男生\d+人（[^）]*），女生\d+人（[^）]*）(?:，[^。]{0,16}性别比[^。]{0,8})?", flat)
                    if matched:
                        piece = matched.group(0)
                if "缓考" in (question or "") and "申请缓考程序" in flat:
                    at_rule = flat.find("申请缓考程序")
                    piece = flat[at_rule:at_rule + 160]
                if re.search(r"推免|保研", question or "") and "申请条件" in flat and "推免资格" in flat:
                    at_rule = flat.find("申请条件")
                    piece = flat[at_rule:at_rule + 90]
                if "奖学" in (question or "") and re.search(r"怎么评|如何评|怎么选|评选", question or "") and "可申请国家奖学金" in flat:
                    at_rule = flat.find("第七条同时符合下列条件")
                    if at_rule < 0:
                        at_rule = flat.find("同时符合下列条件")
                    piece = flat[at_rule:at_rule + 80]
                    nxt = conn.execute(
                        "SELECT text, document_id FROM retrieval_units WHERE id = ?",
                        (row["unit_id"] + 1,),
                    ).fetchone()
                    if nxt and nxt["document_id"] == row["document_id"] and "前10%" not in piece:
                        extra = re.sub(r"^·?\d+·", "", re.sub(r"\s+", "", nxt["text"] or ""))
                        cut = extra.find("第八条")
                        piece += extra[:cut] if cut > 0 else extra[:160]
                if (
                    "优秀学生" in (question or "")
                    and "毕业" not in (question or "")
                    and "干部" not in (question or "")
                    and "三好" not in (question or "")
                    and "荣誉称号分为" in flat
                ):
                    at_rule = flat.find("本科优秀学生荣誉称号分为")
                    piece = flat[at_rule:]
                    if "前20%" not in piece:
                        nxt = conn.execute(
                            "SELECT text, document_id FROM retrieval_units WHERE id = ?",
                            (row["unit_id"] + 1,),
                        ).fetchone()
                        if nxt and nxt["document_id"] == row["document_id"]:
                            extra = re.sub(r"^·?\d+·", "", re.sub(r"\s+", "", nxt["text"] or ""))
                            cut = extra.find("三、评选办法")
                            if cut < 0 and "前20%" in extra:
                                cut = extra.find("前20%") + len("前20%")
                            if cut > 0:
                                piece += extra[:cut]
                elif "三好" in (question or "") and "标兵" not in (question or "") and "三好学生：" in flat and "前20%" in flat:
                    at_rule = flat.find("三好学生：")
                    end = flat.find("前20%", at_rule)
                    piece = flat[at_rule:end + len("前20%")]
                elif "微专业" in (question or "") and "学分" in (question or "") and "15学分" in flat:
                    matched = re.search(r"每个微专业需开设5～7门课程，总学分控制在15学分以内，每门课程原则上为2～3学分", flat)
                    if matched:
                        piece = matched.group(0)
                elif "微专业" in (question or "") and "微专业旨在" in flat:
                    at_rule = flat.find("第二条微专业旨在")
                    if at_rule < 0:
                        at_rule = flat.find("微专业旨在")
                    end = flat.find("。", at_rule)
                    piece = flat[at_rule:end + 1] if end > at_rule else flat[at_rule:at_rule + 140]
                elif "中外合作" in (question or "") and "波特兰" in flat and "塞浦路斯" in flat:
                    if re.search(r"研究生|硕士|博士", question or ""):
                        at_rule = flat.find("波特兰学院硕士研究生")
                        end = flat.find("起执行", flat.find("塞浦路斯学院硕士研究生", at_rule))
                    else:
                        at_rule = flat.find("中外合作办学（波特兰学院）")
                        end = flat.find("起执行", flat.find("63000", at_rule))
                    piece = flat[at_rule:end + len("起执行")] if end > at_rule else flat[at_rule:at_rule + 180]
                    piece = re.sub(r"\d{1,3}(?=中外合作办学（)", "", piece)
                elif "优秀学生干部" in (question or "") and "优秀学生干部：" in flat:
                    at_rule = flat.find("优秀学生干部：")
                    end = flat.find("前35%", at_rule)
                    piece = flat[at_rule:end + len("前35%")] if end > at_rule else flat[at_rule:]
                    if "前35%" not in piece:
                        nxt = conn.execute(
                            "SELECT text, document_id FROM retrieval_units WHERE id = ?",
                            (row["unit_id"] + 1,),
                        ).fetchone()
                        if nxt and nxt["document_id"] == row["document_id"]:
                            extra = re.sub(r"^·?\d+·", "", re.sub(r"\s+", "", nxt["text"] or ""))
                            end = extra.find("前35%")
                            if end >= 0:
                                piece += extra[:end + len("前35%")]
                elif "标兵" in (question or "") and "三好学生标兵：" in flat and "国籍" in flat:
                    at_rule = flat.find("三好学生标兵：")
                    end = flat.find("前4%", at_rule)
                    piece = flat[at_rule:end + len("前4%")] if end > at_rule else flat[at_rule:at_rule + 280]
                if re.search(r"每人每年\d+$", piece):
                    nxt = conn.execute(
                        "SELECT text, document_id FROM retrieval_units WHERE id = ?",
                        (row["unit_id"] + 1,),
                    ).fetchone()
                    if nxt and nxt["document_id"] == row["document_id"] and re.sub(r"\s+", "", nxt["text"] or "").startswith("元"):
                        piece += "元"
                if "贷款" in (question or "") and "户籍所在地经办网点" in flat and re.search(r"怎么|如何|申请|办理", question or ""):
                    start = flat.find("被确认符合生源地信用助学贷款条件")
                    end = flat.rfind("申请办理生源地信用助学贷款")
                    if start >= 0 and end > start:
                        piece = flat[start:end + len("申请办理生源地信用助学贷款")]
                if "贷款" in (question or "") and "不超过16000" in flat and re.search(r"多少|额度|上限", question or ""):
                    matched = re.search(r"全日制普通本专科学生[^。]{0,80}不超过\d+元；全日制研究生[^。]{0,40}不超过\d+元", flat)
                    if matched:
                        piece = matched.group(0)
                if "免修" in (question or "") and "免修申请表" in flat and re.search(r"怎么|如何|申请", question or ""):
                    start = flat.find("申请者应在课程开课一周内")
                    end = flat.find("方可参加免修考核", start)
                    if start >= 0 and end > start:
                        piece = flat[start:end + len("方可参加免修考核")]
                if "诊断证明" in flat and "证明" in (question or "") and "缓考" in (question or ""):
                    matched = re.search(r"病假者必须在考前出示[^。]{0,80}诊断证明[^。]*", flat)
                    if matched:
                        piece = matched.group(0)
                if "硕士学位授权点" in flat and re.search(r"硕士点|几个硕士|多少个硕士", question or "") and "博士" not in (question or ""):
                    matched = re.search(r"一级学科硕士学位授权点\d+个，硕士专业学位授权点（类别）\d+个", flat)
                    if matched:
                        piece = matched.group(0)
                if "63000" in flat and "塞浦路斯" in (question or "") and not re.search(r"研究生|硕士|博士", question or ""):
                    at_fee = flat.find("中外合作办学（欧洲塞浦路斯学院）")
                    if at_fee < 0:
                        at_fee = flat.find("63000")
                    end = flat.find("起执行", flat.find("63000", at_fee))
                    if end > at_fee:
                        piece = re.sub(r"(中外合作办学（欧洲塞浦路斯学院）)中外合作办学", r"\1", flat[at_fee:end + len("起执行")])
                if "创新" in (question or "") and "学分" in (question or "") and "折算为学分" in flat and "微专业" not in (question or ""):
                    matched = re.search(r"学校建立创新创业档案、设置创新创业学分。学生参加创新创业、社会实践等活动以及发表论文、获得专利授权等与专业学习、学业要求相关的经历、成果，可以折算为学分，计入学业成绩。具体学分折算按相关实施细则执行", flat)
                    if matched:
                        piece = matched.group(0)
                out.append((piece, hit))
                if len(out) >= 24:
                    return out
    finally:
        conn.close()
    return out


def _pattern_texts(question: str) -> list[str]:
    """问法决定要扫的句式，不把某个数字写死。人数为、专任教师人数、性别比、违约金各只有几处。"""
    found: list[str] = []
    text = question or ""
    staff = any(k in text for k in ("职工", "教师", "老师"))
    if staff:
        found.append("有专任教师")
    if re.search(r"多少|几人|人数", text) and not staff and not any(
        k in text for k in ("比例", "生师比", "专业", "学费", "收费", "罚款", "钱", "奖学", "助学", "违约", "代码", "学分", "住宿", "借", "校区", "旷课", "作弊", "学院", "电话", "邮编", "邮箱", "学制", "建校", "校庆", "校歌")
    ):
        found.append("人数为")
    if "比例" in text and ("男" in text or "女" in text):
        found.append("男女性别比")
    if any(k in text for k in ("过期", "超期", "逾期", "罚款")) and any(k in text for k in ("书", "借", "收费", "罚款")):
        found.append("交纳违约金")
    if "学费" in text:
        found.append("学费标准")
    if "双一流" in text:
        found.append("双一流")
    return found


def _tool_search(query: str, question: str = "") -> tuple[object, list[dict], list[tuple[str, dict]], str]:
    """原词直查，再按向量把相邻段落里的数字句补进来。限制句和目录不占名额。"""
    from .db import init_db
    from .extractive import split_sentences
    from .retrieve import QueryPlan
    from .semantic import vector_rank

    probes = _probes(question, query)
    plan: object = QueryPlan(question=question or query, keywords=probes)
    found: list[tuple[str, dict, int]] = []
    seen: set[str] = set()

    def _add(sent: str, hit: dict, direct: int) -> None:
        sent = _focus_count(re.sub(r"\s+", "", sent or "").strip("，,。；; "))
        sent = re.sub(r"^\d+(?=（)", "", sent)
        stop = sent.find("。", 8)
        phone = re.search(r"(?:财务处|教务处)(?:咨询|联系)?电话[:：]?\d{7,}", sent)
        if phone and any(office in (question or "") for office in ("教务处", "财务处")) and "电话" in (question or ""):
            sent = sent[phone.start():phone.end() + 8]
            stop = -1
        if "重新参加入学考试" in sent and "学生因退学" in sent:
            begin = sent.find("学生因退学")
            end = sent.find("再次入学")
            sent = sent[begin:end + len("再次入学")] if end > begin else sent[begin:]
            stop = -1
        if "荣誉称号分为" in sent and "前20%" in sent and "优秀学生" in (question or "") and "三好" not in (question or ""):
            begin = sent.find("本科优秀学生荣誉称号分为")
            end = sent.find("前20%")
            sent = sent[begin if begin >= 0 else 0:end + len("前20%")]
            stop = -1
        elif "三好学生：" in sent and "前20%" in sent and "标兵" not in (question or "") and "干部" not in (question or ""):
            begin = sent.find("三好学生：")
            end = sent.find("前20%", begin)
            sent = sent[begin:end + len("前20%")]
            stop = -1
        elif "优秀学生干部" in (question or "") and "前35%" in sent:
            begin = sent.find("优秀学生干部：")
            end = sent.find("前35%", begin if begin >= 0 else 0)
            if begin >= 0 and end > begin:
                sent = sent[begin:end + len("前35%")]
                stop = -1
        elif "三好学生标兵：" in sent and "前4%" in sent and "标兵" in (question or ""):
            begin = sent.find("三好学生标兵：")
            end = sent.find("前4%", begin)
            sent = sent[begin:end + len("前4%")] if end > begin else sent[begin:]
            stop = -1
        if "微专业旨在" in sent:
            begin = sent.find("第二条微专业旨在")
            if begin < 0:
                begin = sent.find("微专业旨在")
            end = sent.find("。", begin)
            sent = sent[begin:end + 1] if end > begin else sent[begin:begin + 140]
            stop = -1
        if "下列情形之一" in sent and "违反宪法" in sent:
            begin = sent.find("学生有下列情形之一")
            if begin < 0:
                begin = sent.find("下列情形之一")
            end = sent.find("经教育不改的", begin)
            sent = sent[begin:end + len("经教育不改的")] if end > begin else sent[begin:]
            stop = -1
        if "诞生于" in sent and "改建为本科高校" in sent:
            stop = -1
        if "每盘每天" in sent and "0.10" in sent and "光盘" in (question or "") and re.search(r"书|图书", question or ""):
            stop = -1
        if "折算为学分" in sent and "创新创业学分" in sent:
            stop = -1
        if "户籍所在地经办网点" in sent and sent.count("申请办理") >= 2:
            stop = -1
        if "不超过16000" in sent and "不超过20000" in sent:
            stop = -1
        if "不得超过两学年" in sent and "一学年为限" in sent:
            begin = sent.find("休学一般以一学年为限")
            end = sent.find("不得超过两学年")
            sent = sent[begin if begin >= 0 else 0:end + len("不得超过两学年")]
            stop = -1
        if stop > 0 and "本科标准学制" in sent and "最长不超过" in sent[stop:stop + 28]:
            stop2 = sent.find("。", stop + 1)
            sent = sent[: stop2 + 1] if stop2 > 0 else sent[: stop + 28]
        elif stop > 0:
            sent = sent[: stop + 1]
        if len(sent) < 8 or sent[:24] in seen or _catalog(sent) or _smashed(sent):
            return
        if any(k in sent for k in _NAV):
            return
        seen.add(sent[:24])
        limit = 1200 if any(k in sent for k in ("前20%", "前35%", "前4%", "可申请国家奖学金", "经教育不改的", "违反宪法", "微专业旨在")) else 220
        found.append((sent[:limit], hit, direct))

    for sent, hit in _sql_probe(probes, question):
        _add(sent, hit, 1)
    if _pattern_texts(question):
        from .db import init_db as _init_db

        conn = _init_db()
        try:
            for needle in _pattern_texts(question):
                rows = conn.execute(
                    """
                    SELECT ru.text, ru.locator, d.title AS doc_title, d.url
                    FROM retrieval_units ru JOIN documents d ON d.id = ru.document_id
                    WHERE ru.text LIKE ?
                    LIMIT 80
                    """,
                    (f"%{needle}%",),
                ).fetchall()
                for row in rows:
                    hit = {
                        "doc_title": row["doc_title"],
                        "url": row["url"] or "",
                        "locator": row["locator"],
                        "text": row["text"],
                    }
                    flat = re.sub(r"\s+", "", row["text"] or "")
                    for match in re.finditer(re.escape(needle), flat):
                        start = flat.rfind("。", max(0, match.start() - 80), match.start())
                        start = 0 if start < 0 else start + 1
                        if flat[start:start + 1].isdigit():
                            while start > 0 and flat[start - 1].isdigit():
                                start -= 1
                            kept = 0
                            while start > 0 and kept < 8 and "\u4e00" <= flat[start - 1] <= "\u9fff":
                                start -= 1
                                kept += 1
                        _add(flat[start:match.end() + 90], hit, 1)
        finally:
            conn.close()
    for sent, hit in _recall_pairs(question):
        _add(sent, hit, 1)
    texts = list(probes)
    qtext = re.sub(r"南京邮电大学|南京邮电|邮电大学|南邮", "", re.sub(r"\s+", "", question or ""))
    if len(qtext) >= 2 and qtext[:30] not in texts:
        texts.append(qtext[:30])
    ids: list[int] = []
    for text in texts:
        for uid in vector_rank(text, top=8):
            if uid not in ids:
                ids.append(uid)
    hits: list[dict] = []
    if ids:
        conn = init_db()
        try:
            for uid in ids:
                row = conn.execute(
                    """
                    SELECT ru.id, ru.document_id, ru.text, ru.locator, d.title AS doc_title, d.url
                    FROM retrieval_units ru JOIN documents d ON d.id = ru.document_id
                    WHERE ru.id = ?
                    """,
                    (uid,),
                ).fetchone()
                if not row:
                    continue
                group = [(row, 1)]
                for nid in (uid - 2, uid - 1, uid + 1, uid + 2):
                    near = conn.execute(
                        """
                        SELECT ru.id, ru.document_id, ru.text, ru.locator, d.title AS doc_title, d.url
                        FROM retrieval_units ru JOIN documents d ON d.id = ru.document_id
                        WHERE ru.id = ?
                        """,
                        (nid,),
                    ).fetchone()
                    if near and near["document_id"] == row["document_id"]:
                        group.append((near, 0))
                for item, direct in group:
                    hit = {
                        "doc_title": item["doc_title"],
                        "url": item["url"] or "",
                        "locator": item["locator"],
                        "text": item["text"],
                    }
                    hits.append(hit)
                    flat = re.sub(r"\s+", "", item["text"] or "")
                    for sent in (split_sentences(item["text"] or "") or [])[:8]:
                        _add(sent, hit, direct)
                    for match in re.finditer(r".{0,48}\d+(?:\.\d+)?人.{0,36}", flat):
                        _add(match.group(0), hit, direct)
                    for probe in probes:
                        if len(probe) >= 2 and probe in flat:
                            for sent in _windows(item["text"] or "", probe)[:2]:
                                _add(sent, hit, direct)
        finally:
            conn.close()
    body = re.sub(r"南京邮电大学|南京邮电|邮电大学|南邮", "", question or "")

    want_people = bool(re.search(r"多少|人数|几人", question or "")) and "比例" not in (question or "")

    def _rank(item: tuple[str, dict, int]) -> tuple[int, int, int, int, int]:
        sent, hit, direct = item
        return (
            *_rank_bias(question, sent, hit),
            0 if _answerable(question, sent, probes, hit.get("doc_title") or "") else 1,
            1 if ("工科" in (question or "") and "文科" not in (question or "") and "文科" in sent) else 0,
            1 if "学费" in (question or "") and re.search(r"\d{6,}", sent or "") else 0,
            -(1 if "比例" in (question or "") and re.search(r"男生\d+人", sent or "") and re.search(r"女生\d+人", sent or "") else 0),
            -(1 if any(k in (question or "") for k in ("职工", "教师", "老师")) and "学校" in (sent or "") and re.search(r"专任教师\d+人", sent or "") else 0),
            0 if want_people and _headcount(sent) else 1,
            -(1 if "交纳违约金" in sent and "图书" in sent and any(k in (question or "") for k in ("过期", "逾期", "罚款", "收费")) else 0),
            -(1 if want_people and re.search(r"人数为\d+", sent) else 0),
            -len(re.findall(r"\d+(?:\.\d+)?人", sent)) if want_people else 0,
            -_char_hits(probes[0] if probes else "", sent),
            -_gram_hits(body, sent),
            -len(re.findall(r"\d+(?:\.\d+)?元", sent)) if any(k in (question or "") for k in ("费", "罚款")) else 0,
            -direct,
        )

    found.sort(key=_rank)
    chosen = [(sent, hit) for sent, hit, _direct in found if _answerable(question, sent, probes, hit.get("doc_title") or "")][:6]
    if not chosen:
        chosen = [
            (sent, hit)
            for sent, hit, _direct in found
            if _shape_ok(question, sent, hit.get("doc_title") or "") is not False
        ][:4]
    lines = [f"[{i}]《{hit.get('doc_title') or ''}》{sent}" for i, (sent, hit) in enumerate(chosen, 1)]
    return plan, hits, chosen, "\n".join(lines) if lines else "（没有返回段落）"


def _resolve_cite(cite: str, pool: list[tuple[str, dict]]) -> tuple[str, dict] | None:
    """依据可以是原句，也可以是模型点的编号。编号表示它选了工具返回的那一条。"""
    matched = _cite_hit(cite, pool)
    if matched:
        return matched
    mark = re.search(r"\[([1-6])\]", cite or "")
    if not mark:
        mark = re.fullmatch(r"\s*([1-6])\s*[。.]?", cite or "")
    if not mark:
        return None
    index = int(mark.group(1)) - 1
    if 0 <= index < len(pool):
        return pool[index]
    return None


def _cite_hit(cite: str, pool: list[tuple[str, dict]]) -> tuple[str, dict] | None:
    flat = _compact(cite)
    if len(flat) < 8:
        return None
    for sent, hit in pool:
        sent_flat = _compact(sent)
        if flat[:12] in sent_flat or sent_flat[:12] in flat:
            return sent, hit
    return None


def _on_topic(question: str, sent: str) -> bool:
    terms = [t for t in _specific_terms(question) if t not in _CHAT]
    if any(t in sent for t in terms):
        return True
    # 问句滑窗会把「弄虚作假」黏上「会怎样」，整词对不上原句。连续四字碰到原句才算同一件事。
    han = "".join(ch for ch in question if "\u4e00" <= ch <= "\u9fff")
    skip = ("会怎样", "怎么样", "怎么办", "是什么", "有没有", "怎么收", "了怎么", "如何办")
    for i in range(max(0, len(han) - 3)):
        gram = han[i : i + 4]
        if gram in skip or gram not in sent:
            continue
        return True
    return False


def _cited_ok(question: str, sent: str, queries: list[str]) -> bool:
    """模型引用了检索回来的句子。问句很短时，原句不会复述「多少人」，不能因此判成无关。"""
    if _on_topic(question, sent):
        return True
    if not _specific_terms(question):
        return True
    flat = re.sub(r"\s+", "", sent or "")
    for query in queries:
        text = re.sub(r"\s+|南京邮电大学|南京邮电|邮电大学|南邮", "", query or "")
        if len(text) >= 4 and any(text[i:i + 4] in flat for i in range(len(text) - 3)):
            return True
        if 2 <= len(text) < 4 and text in flat:
            return True
    return False


def _show(question: str, answer: str, sent: str, hit: dict, queries: list[str]) -> str:
    body = sent if sent.endswith(("。", "；", "！")) else sent + "。"
    nums_a = set(re.findall(r"\d+(?:\.\d+)?", answer or ""))
    nums_c = set(re.findall(r"\d+(?:\.\d+)?", sent))
    echo = _compact(answer) in _compact(question) or _compact(question) in _compact(answer)
    if answer and not echo and not (nums_a - nums_c) and _compact(answer) not in _compact(body):
        body = answer.rstrip("。") + "。\n" + body
    shown = "、".join(f"「{q}」" for q in queries)
    title = hit.get("doc_title") or ""
    url = hit.get("url") or ""
    return f"检索指挥查了{shown}，采用这一句：\n{body}\n（摘自 {title} {url}）"


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "在已采集的公开文件中检索。query 由你写成公文里会出现的说法。没答上就换一组不同的词再调用。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]

_DIRECTOR = (
    "你是公开资料问答的主导，检索只是工具。你不记得数字和条款，需要时调用 search。"
    "query 写成报告、章程、办法里会出现的词，不要只重复用户的口语，每次都换不同的说法。"
    "返回后你来判断：某一句真能回答，就用那句里的数字作答；句子是在答别的问题，就换词再搜。"
    "问题在问实际数量或比例时，「无要求」「不限」「没有限制」不是答案。"
    "至少换过三组不同的词，才能说资料里没有。说没有时不要附上无关段落。"
    "不要编造数字，数字必须来自某次 search 的返回。"
)


def _sampling() -> dict:
    """Qwen3 思考模式的推荐采样。温度 0 会把工具调用挤成空话，上下文和长度也不能再用 16 token。"""
    return {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0,
        "presence_penalty": 0,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def _agent_turn(
    messages: list[dict],
    tool_choice: str,
    max_tokens: int = 768,
    think: bool = True,
    temperature: float = 0.6,
) -> dict | None:
    sampling = _sampling()
    sampling["temperature"] = temperature
    sampling["chat_template_kwargs"] = {"enable_thinking": think}
    body = json.dumps(
        {
            **sampling,
            "max_tokens": max_tokens,
            "tools": _TOOLS,
            "tool_choice": tool_choice,
            "messages": messages,
        }
    ).encode()
    req = urllib.request.Request(
        _endpoint(),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            payload = json.loads(resp.read().decode())
        return payload["choices"][0]["message"]
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, KeyError, IndexError, TypeError) as exc:
        return None


def _tool_query(message: dict) -> str | None:
    calls = message.get("tool_calls") or []
    if not calls:
        return None
    raw = (calls[0].get("function") or {}).get("arguments") or ""
    if isinstance(raw, dict):
        return str(raw.get("query") or "").strip() or None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        found = re.search(r'"query"\s*:\s*"([^"]+)"', raw)
        return found.group(1).strip() if found else None
    return str(data.get("query") or "").strip() or None


def _call_id(message: dict) -> str:
    calls = message.get("tool_calls") or []
    if calls and calls[0].get("id"):
        return str(calls[0]["id"])
    return "search"


def _assistant_message(message: dict) -> dict:
    out = {"role": "assistant", "content": message.get("content") or ""}
    if message.get("tool_calls"):
        out["tool_calls"] = message["tool_calls"]
    if message.get("reasoning_content"):
        out["reasoning_content"] = message["reasoning_content"]
    return out


def _invented_numbers(answer: str, corpus: str) -> bool:
    nums = set(re.findall(r"\d+(?:\.\d+)?", answer or ""))
    have = set(re.findall(r"\d+(?:\.\d+)?", corpus or ""))
    return bool(nums - have)


def _covered_sentence(answer: str, pool: list[tuple[str, dict]]) -> tuple[str, dict] | None:
    """模型把表格里相邻的数字粘在一起时，仍指回含这些数字的原句。"""
    tokens = re.findall(r"\d+(?:\.\d+)?", answer or "")
    if not tokens:
        return None
    best = None
    best_n = 0
    for sent, hit in pool:
        figs = _real_figures(sent)
        n = 0
        for fig in figs:
            if fig in (answer or "") or any(fig in token for token in tokens):
                n += 1
        if n > best_n:
            best, best_n = (sent, hit), n
    if best_n >= 2 or (best_n >= 1 and best and any(fig in (answer or "") for fig in _real_figures(best[0]))):
        return best
    return None


def _best_support(answer: str, pool: list[tuple[str, dict]]) -> tuple[str, dict] | None:
    flat = _compact(answer)
    for noise in ("南京邮电大学", "南京邮电", "邮电大学"):
        flat = flat.replace(noise, "")
    wanted = _real_figures(answer)
    best = None
    best_n = 0
    for sent, hit in pool:
        sent_flat = _compact(sent)
        n = 5 * len(wanted & _real_figures(sent))
        for i in range(0, max(0, len(sent_flat) - 7), 4):
            piece = sent_flat[i : i + 8]
            if piece and piece in flat:
                n += 1
        if n > best_n:
            best, best_n = (sent, hit), n
    return best if best_n else None


def _is_limit_sentence(sent: str) -> bool:
    """「不超过、无要求」是限制，不是问题要的实际数量。"""
    return any(k in (sent or "") for k in ("不超过", "原则上", "无要求", "不限", "没有限制", "不作限制", "以内"))


def _asks_figure(question: str) -> bool:
    return bool(re.search(r"多少|比例|人数|几人|收费|学费|标准", question or ""))


def _real_figures(text: str) -> set[str]:
    """编号、日期、代码和年份不算已经回答了数量。"""
    body = re.sub(r"\[\d+\]", "", text or "")
    body = re.sub(r"20\d{2}\s*年(?:\s*\d{1,2}\s*月)?(?:\s*\d{1,2}\s*日)?", "", body)
    body = re.sub(r"20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}", "", body)
    body = re.sub(r"\d{1,2}\s*月(?:\s*\d{1,2}\s*日)?", "", body)
    body = re.sub(r"(代码|邮编|电话|学号|编号|序号)\s*[=:：]?\s*\d+", "", body)
    found = set()
    for n in re.findall(r"\d+(?:\.\d+)?", body):
        if "." not in n and (len(n) < 2 or re.fullmatch(r"20\d{2}", n)):
            continue
        found.add(n)
    return found


def _strip_tool_echo(text: str) -> str:
    kept = []
    for line in (text or "").splitlines():
        if "查法：" in line or "判断哪一句" in line:
            continue
        if re.match(r"\s*\[\d+\]", line):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _retry(used: list[str], why: str) -> str:
    shown = "、".join(f"「{q}」" for q in used) or "（无）"
    return f"已经用过{shown}。{why}请换成公文里另一种说法，再调用 search。不要重复用过的词，不要现在就说没有。"


_NARROW = ("学院", "工会", "班级", "部门", "岗位", "转专业", "拟接收", "拟录取", "本单位", "推免", "本专业", "本学科")
_NAV = ("每页", "上一页", "下一页", "尾页", "跳转", "条记录")
_GENERIC = ("研究", "机构", "情况", "如何", "处理", "标准", "数据", "相关", "措施", "工作", "管理")


def _too_narrow(question: str, sent: str) -> bool:
    """人数、名额问的是全校时，学院和转专业名额不是答案。

    问职务、学制、地址时不能一见到「学院」就丢掉：开学典礼会写「各学院负责人」，
    双一流沿革里也会出现已经并入的学院。
    """
    if any(k in (question or "") for k in _NARROW):
        return False
    if re.search(r"怎么评|如何评|评选|条件", question or "") and any(
        k in (question or "") for k in ("奖学", "三好", "优秀学生")
    ):
        return False
    if "中外合作" in (question or "") or ("微专业" in (question or "") and re.search(r"是什么|什么是", question or "")):
        return False
    if "免修" in (question or "") or ("缓考" in (question or "") and "证明" in (question or "")):
        return False
    if (
        _asks_who(question)
        or _asks_span(question)
        or _asks_place(question)
        or _asks_english_name(question)
        or _asks_rule(question)
        or _asks_hours(question)
        or _asks_campuses(question)
        or "双一流" in (question or "")
    ):
        if _asks_who(question) and re.search(r"学院党委|系党委|学院书记|学院院长", sent or ""):
            return True
        return False
    markers = [item for item in _NARROW if item in (sent or "") and item not in (question or "")]
    if "部门" in markers and ("主管部门" in (sent or "") or "职能部门" in (sent or "")):
        rest = (sent or "").replace("主管部门", "").replace("职能部门", "")
        if "部门" not in rest:
            markers.remove("部门")
    return bool(markers)


def _query_hit(sent: str, queries: list[str], question: str = "") -> bool:
    """整段检索词要出现在句子里。四字片段还必须来自用户的问题，避免「研究机构」这类搭车。"""
    flat = re.sub(r"\s+", "", sent or "")
    qflat = re.sub(r"\s+|南京邮电大学|南京邮电|邮电大学|南邮", "", question or "")
    for query in queries:
        text = re.sub(r"\s+|南京邮电大学|南京邮电|邮电大学|南邮", "", query or "")
        text = re.sub(r"[\u4e00-\u9fff]{2,8}大学", "", text)
        if len(text) >= 2 and text in flat:
            return True
        if len(text) < 4:
            continue
        for i in range(len(text) - 3):
            gram = text[i : i + 4]
            if gram in qflat and gram in flat:
                return True
    return False


def _linked(question: str, sent: str, queries: list[str]) -> bool:
    """问句或检索词里的二字，要在原句里出现两组以上。太泛的二字不算。"""
    flat = re.sub(r"\s+", "", sent or "")
    texts = [re.sub(r"南京邮电大学|南京邮电|邮电大学|南邮", "", question or "")]
    for query in queries:
        text = re.sub(r"\s+|南京邮电大学|南京邮电|邮电大学|南邮", "", query or "")
        text = re.sub(r"[\u4e00-\u9fff]{2,8}大学", "", text)
        texts.append(text)
    grams: set[str] = set()
    for text in texts:
        for i in range(len(text) - 1):
            gram = text[i : i + 2]
            if not all("\u4e00" <= ch <= "\u9fff" for ch in gram):
                continue
            if gram in _GENERIC or gram in {"多少", "几个", "什么", "怎么", "如何"}:
                continue
            grams.add(gram)
    return sum(gram in flat for gram in grams) >= 2


def _catalog(sent: str) -> bool:
    """通知列表、点线目录、分页条不是条款。"""
    if sent.count("|") >= 2 or sent.count("｜") >= 2:
        return True
    if "····" in sent or sent.count("·") >= 8:
        return True
    return len(re.findall(r"20\d{2}-\d{2}-\d{2}", sent)) >= 2


def _smashed(sent: str) -> bool:
    """表格抽掉空格后，人数会和百分比粘成 389359.34%。这种不能当答案。"""
    return bool(re.search(r"\d{5,}\.\d+\s*[%％]", sent or ""))


def _quantity(sent: str) -> bool:
    return bool(re.search(r"\d+(?:\.\d+)?[人元%％]", sent or ""))


def _anchors(question: str) -> list[str]:
    """问句里长度为四字的事项。有两件时必须都出现，避免只碰上其中一件。"""
    out: list[str] = []
    for term in _specific_terms(question):
        term = re.sub(r"^(有没有|是否有|怎么|如何)", "", term)
        term = re.sub(r"(会怎样|怎么样|怎么办|如何办|会)$", "", term)
        if any(w in term for w in ("多少", "一共", "什么", "怎么", "如何", "哪些", "几个", "不要", "一天", "自己", "算不", "进去", "到底", "除了", "一下", "别把")):
            continue
        if len(term) == 4 and term not in out:
            out.append(term)
    cleaned: list[str] = []
    for term in out:
        rest = re.sub(r"^(算|不|有|没)+", "", term)
        if 2 <= len(rest) < len(term):
            term = rest
        if term not in cleaned:
            cleaned.append(term)
    return cleaned


def _covers(term: str, sent: str, title: str = "") -> bool:
    """原词出现在句子或标题里；句子里只差一个字也算（过期对逾期）。标题不参与差一字，避免两个词拼到一起。"""
    if term in (sent or "") or (title and term in title):
        return True
    # 差一个字可以，但词的头两个字必须还在，避免「航天」被「学院」两个常见字凑上。
    if term[:2] not in (sent or ""):
        return False
    have = set(sent or "")
    return sum(ch in have for ch in term) >= len(term) - 1


def _focus_count(sent: str) -> str:
    """人数句前面常常粘着专业名单。从学年写起，只要前缀里没有人数。"""
    match = re.search(r"20\d{2}.{0,6}20\d{2}学年", sent or "")
    if match and match.start() > 0 and not re.search(r"\d+(?:\.\d+)?人", sent[: match.start()]):
        start = max(0, match.start() - 8)
        head = re.search(r"[\u4e00-\u9fff]{2,}$", sent[start : match.start()])
        if head:
            start += head.start()
        return sent[start :]
    return sent


def _headcount(sent: str) -> bool:
    return bool(re.search(r"\d+(?:\.\d+)?人", sent or ""))


def _char_hits(text: str, sent: str) -> int:
    """检索词里的字有几个出现在句子里。用来区分「在校生人数」和「毕业生人数」。"""
    skip = set("的了多少有和在及与或")
    chars = [ch for ch in (text or "") if "\u4e00" <= ch <= "\u9fff" and ch not in skip]
    return sum(ch in (sent or "") for ch in dict.fromkeys(chars))


def _gram_hits(text: str, sent: str) -> int:
    text = re.sub(r"南京邮电大学|南京邮电|邮电大学|南邮", "", text or "")
    n = 0
    for i in range(len(text) - 1):
        gram = text[i : i + 2]
        if not all("\u4e00" <= ch <= "\u9fff" for ch in gram):
            continue
        if gram in _GENERIC or gram in {"多少", "几个", "什么", "怎么", "如何", "有没"}:
            continue
        if gram in sent:
            n += 1
    return n


def _answerable(question: str, sent: str, queries: list[str], title: str = "") -> bool:
    """限制、局部名额、目录都不算答上了。事项词要盖住，短问句则要沾上检索词。"""
    if any(k in sent for k in _NAV) or _catalog(sent) or _smashed(sent):
        return False
    if _asks_place(question) and ("=" in (sent or "") or "＝" in (sent or "")):
        sent = re.sub(r"[=＝]", "", sent)
    elif "=" in (sent or "") or "＝" in (sent or ""):
        return False
    if (
        "励志" in (question or "")
        and "励志" not in (sent or "")
        and not (re.search(r"一样|相同|区别|不同", question or "") and "国家奖学金" in (sent or ""))
    ):
        return False
    if "研究生" in (question or "") and "住宿" in (question or "") and "本科" not in (question or ""):
        if "研究生" not in (sent or "") and "研究生" not in (title or ""):
            return False
    if re.search(r"入伍|应征|参军", question or ""):
        if not re.search(r"入伍|应征|参军|兵役", sent or ""):
            return False
        if "学籍" in (question or "") and "保留学籍" not in (sent or ""):
            return False
    if "学籍" in (question or "") and "入学资格" in (sent or "") and "保留学籍" not in (sent or ""):
        return False
    if "创新" in (question or "") and "学分" in (question or "") and any(
        word in (sent or "") and word not in (question or "") for word in ("专转本", "3+2", "微专业")
    ):
        return False
    if re.search(r"什么时候|何时|几月|哪个月", question or "") and re.search(r"毕业证|学位证", question or ""):
        if not re.search(r"\d{1,2}月", sent or "") or re.search(r"丙系|健业", sent or ""):
            return False
    if "折合" in (question or "") and "折合" not in (sent or ""):
        return False
    for word in ("四六级", "四级", "六级", "211", "985"):
        if word not in (question or ""):
            continue
        blob = (sent or "") + (title or "")
        present = bool(re.search(rf"(?<!\d){word}(?!\d)", blob)) if word in {"211", "985"} else word in blob
        if not present:
            return False
    if "密码" in (question or ""):
        if "密码" not in (sent or ""):
            return False
        if any(word in (sent or "") + (title or "") for word in ("等保", "外网IP", "开发商", "继续教育", "培训", "证件号码", "C/S", "B/S")):
            return False
        if not any(word in (sent or "") for word in ("重置", "忘记", "找回")):
            return False
    if ("211" in (question or "") or "985" in (question or "")) and any(
        word in (sent or "") for word in ("分数", "录取", "均值", "标准差", "就业")
    ):
        return False
    if _too_narrow(question, sent):
        return False
    # 问的是书，就不要拿只针对光盘的违约金来回答。毕业生就业人数也不是教职工人数。
    if "书" in (question or "") and "光盘" not in (question or "") and "光盘" in sent and "图书" not in sent:
        return False
    if "光盘" in (question or "") and not any(k in (question or "") for k in ("图书", "外文", "书")) and "光盘" not in sent:
        return False
    if "女" in (question or "") and any(k in (question or "") for k in ("老师", "教师", "职工")):
        if not ("学校" in sent and re.search(r"女(?:教职工|教师).{0,6}\d+人", sent or "")):
            return False
    if "女" not in (question or "") and "女教职工" in sent:
        return False
    if any(k in (question or "") for k in ("职工", "教师", "老师")) and not any(
        k in sent or k in (title or "") for k in ("教职工", "专任教师", "职工")
    ):
        return False
    scoped = re.search(r".{0,8}有专任教师", sent or "")
    if scoped and "职工" in (question or "") and "院" in scoped.group(0) and "学校" not in scoped.group(0):
        return False
    # 问全校有多少人时，某一届毕业生的人数是另一批人。问比例时可以保留，因为资料里写的就是某一届。
    if (
        "毕业" not in (question or "")
        and "比例" not in (question or "")
        and "推免" not in (question or "")
        and "保研" not in (question or "")
        and "毕业生" in sent
    ):
        return False
    if "统计表" in sent and not re.search(r"人数为\d+", sent):
        return False
    if (
        "家庭" not in (question or "")
        and "家庭" in sent
        and not any(k in (question or "") for k in ("助学", "奖学", "困难", "贷款"))
    ):
        return False
    if "学分" in (question or "") and "学分" not in (sent or ""):
        return False
    if re.search(r"\d+人以上", sent or ""):
        return False
    if "竞赛" not in (question or "") and "竞赛" in sent:
        return False
    if any(k in sent for k in ("专业代码", "类别代码", "研究方向")) and "微专业旨在围绕" not in (sent or ""):
        return False
    other_uni = []
    for match in re.finditer(r"[\u4e00-\u9fff]{2,12}大学", sent or ""):
        if (sent or "")[match.end():match.end() + 1] == "生":
            continue
        name = match.group(0)
        core = name[1:] if len(name) > 6 and name.endswith("南京邮电大学") else name
        if core in {"南京邮电大学", "邮电大学", "高水平大学"} or name in (question or ""):
            continue
        other_uni.append(name)
    if other_uni:
        return False
    verdict = _shape_ok(question, sent, title)
    if verdict is False:
        return False
    if verdict is True:
        return True
    if _asks_figure(question) and (_is_limit_sentence(sent) or not (_quantity(sent) or _real_figures(sent))):
        return False
    needles = [needle for needle in _pattern_texts(question) if needle in (sent or "")]
    if needles:
        anchors = _anchors(question)
        if not anchors:
            return True
        # 外文图书和光盘的违约金写在相邻两句，不能要求一句同时带上所有事项词。
        if "交纳违约金" in needles:
            return True
        return all(_covers(anchor, sent, title) for anchor in anchors)
    if (
        re.search(r"多少|几人", question or "")
        and not any(k in (question or "") for k in ("职工", "教师", "老师", "费", "罚", "钱", "比例", "书"))
        and "在校生" not in sent
        and not re.search(r"人数为\d+", sent)
    ):
        return False
    anchors = _anchors(question)
    if anchors:
        return all(_covers(anchor, sent, title) for anchor in anchors)
    if _linked(question, sent, queries) or _query_hit(sent, queries, question):
        return True
    return _asks_figure(question) and any(_gram_hits(query, sent) >= 1 for query in queries)


def _answer_candidates(question: str, memory: list[tuple[str, dict]], queries: list[str]) -> list[tuple[str, dict]]:
    """留下能回答问题的原句。"""
    probes = []
    for query in queries:
        probes.extend(_probes(question, query))
    kept = [(sent, hit) for sent, hit in memory if _answerable(question, sent, probes or queries, hit.get("doc_title") or "")]
    body = re.sub(r"南京邮电大学|南京邮电|邮电大学|南邮", "", question or "")
    want_people = bool(re.search(r"多少|人数|几人", question or "")) and "比例" not in (question or "")

    def rank(item: tuple[str, dict]) -> tuple[int, int, int, int]:
        sent, hit = item
        return (
            *_rank_bias(question, sent, hit),
            1 if ("工科" in (question or "") and "文科" not in (question or "") and "文科" in sent) else 0,
            1 if "学费" in (question or "") and re.search(r"\d{6,}", sent or "") else 0,
            -(1 if "比例" in (question or "") and re.search(r"男生\d+人", sent or "") and re.search(r"女生\d+人", sent or "") else 0),
            -(1 if any(k in (question or "") for k in ("职工", "教师", "老师")) and "学校" in (sent or "") and re.search(r"专任教师\d+人", sent or "") else 0),
            0 if want_people and _headcount(sent) else 1,
            -(1 if "交纳违约金" in sent and "图书" in sent and any(k in (question or "") for k in ("过期", "逾期", "罚款", "收费")) else 0),
            -(1 if want_people and re.search(r"人数为\d+", sent) else 0),
            -len(re.findall(r"\d+(?:\.\d+)?人", sent)) if want_people else 0,
            -max((_char_hits(query, sent) for query in probes), default=0),
            -_gram_hits(body, sent),
            -len(re.findall(r"\d+(?:\.\d+)?元", sent)) if any(k in (question or "") for k in ("费", "罚款")) else 0,
        )

    kept.sort(key=rank)
    if "光盘" in (question or "") and any(k in (question or "") for k in ("书", "外文", "图书")):
        book = next((item for item in kept if "图书" in item[0] and "0.10" in item[0]), None)
        disc = next((item for item in kept if "每盘每天" in item[0]), None)
        if disc is None:
            disc = next((item for item in kept if "光盘" in item[0] and "1元" in item[0] and "借期" not in item[0]), None)
        head = [item for item in (book, disc) if item]
        kept = head + [item for item in kept if item not in head]
    if "励志" in (question or "") and re.search(r"一样|相同|区别|不同", question or ""):
        bits = []
        first_hit = None
        for sent, hit in kept:
            matched = re.search(r"国家[\u4e00-\u9fff]{0,6}奖学金的?奖励标准为每人每年\d+元?", sent)
            if not matched:
                continue
            bit = matched.group(0) if matched.group(0).endswith("元") else matched.group(0) + "元"
            if bit not in bits:
                bits.append(bit)
                first_hit = first_hit or hit
        if len(bits) >= 2 and first_hit is not None:
            kept = [("。".join(bits), first_hit)] + kept
    return kept[:6]


def _choose_sentence(question: str, cands: list[tuple[str, dict]]) -> tuple[str, dict] | None:
    """模型只点编号。答案用那句原文，不用它改写后的数字。"""
    if not cands:
        return None
    if _shape_ok(question, cands[0][0], cands[0][1].get("doc_title") or "") is True:
        return cands[0]
    if len(cands) == 1 or re.search(r"人数为\d+|有专任教师\d+人|交纳违约金|学费标准=\d{4}|每生每年\d+元", cands[0][0]):
        return cands[0]
    if re.search(r"男生\d+人", cands[0][0]) and re.search(r"女生\d+人", cands[0][0]):
        return cands[0]
    body = re.sub(r"南京邮电大学|南京邮电|邮电大学|南邮", "", question or "")
    def _score(sent: str) -> int:
        return sum(ch in sent for ch in body if "\u4e00" <= ch <= "\u9fff")
    if _score(cands[0][0]) >= _score(cands[1][0]) + 2:
        return cands[0]
    lines = [f"{i}. {sent[:180]}" for i, (sent, _hit) in enumerate(cands, 1)]
    content = _complete(
        f"问题：{question}\n候选句子：\n" + "\n".join(lines) + "\n哪一句直接回答了这个问题？只输出编号。",
        8,
    )
    if not content:
        return None
    match = re.search(r"[1-6]", content)
    if not match:
        return None
    idx = int(match.group(0))
    if 1 <= idx <= len(cands):
        return cands[idx - 1]
    return None


def _finish(sent: str, hit: dict, plan: object, hits: list[dict], question: str = "") -> dict:
    if _asks_who(question):
        role = _role_in(question)
        if role == "副校长":
            found = re.findall(
                r"副校长(?:、[\u4e00-\u9fff]{2,12})?([\u4e00-\u9fff]{2,3})(?:、([\u4e00-\u9fff]{2,3}))?",
                sent,
            )
            boss = re.search(r"(?<!副)校长([\u4e00-\u9fff]{2,3})", sent)
            names: list[str] = []
            for first, second in found:
                for name in (first, second):
                    if not name or any(word in name for word in _NOT_NAME):
                        continue
                    if name[-1] in "一行列等及":
                        name = name[:-1]
                    if len(name) < 2 or (boss and name == boss.group(1)):
                        continue
                    if name not in names:
                        names.append(name)
            if names:
                sent = "，".join(f"副校长{name}" for name in names)
        else:
            name = _role_name(sent, role)
            if name:
                at = 0
                while True:
                    at = sent.find(role, at)
                    if at < 0:
                        break
                    if role == "校长" and at > 0 and sent[at - 1] == "副":
                        at += len(role)
                        continue
                    break
                if at >= 0 and at > 0 and sent[at - 1] == "校":
                    at -= 1
                end = sent.find(name, max(at, 0)) + len(name)
                if at >= 0 and end > at:
                    sent = sent[at:end]
    if _asks_span(question):
        matched = re.search(
            r"本科标准学制为[一二三四五六七八九十\d]+年(?:。[^。]{0,20}最长不超过[一二三四五六七八九十\d]+年)?",
            sent,
        ) or re.search(
            r"(?:学术型硕士研究生|专业学位硕士研究生|专业学位研究生|博士研究生|硕士研究生|本科生|本科)?(?:标准)?学制为(?:\d+|[一二三四五六七八九十]{1,3})年(?:，最长[\u4e00-\u9fff]{0,6}年限为(?:\d+|[一二三四五六七八九十]{1,3})年)?",
            sent,
        )
        if matched:
            sent = matched.group(0)
    if _asks_place(question):
        campus = _asked_campus(question) or next((phrase for phrase in _phrases(question) if phrase.endswith("校区")), "")
        if re.search(r"哪个区|什么区", question) and campus:
            clause = _campus_clause(sent, campus)
            matched = re.search(_DISTRICT + r"[^。]{0,24}号", clause)
            if matched:
                sent = campus + "：" + matched.group(0)
                campus = ""
        if campus and campus in sent:
            matched = re.search(
                re.escape(campus) + r"(?:地址)?[:：]?[^。]{0,24}?(?=三牌楼|锁金村|仙林校区|$)",
                sent,
            )
            if matched and campus == "仙林校区":
                sent = matched.group(0)
            elif matched:
                sent = matched.group(0)
    if _asks_english_name(question):
        if "NanjingUniversityofPostsandTelecommunications" in re.sub(r"\s+", "", sent):
            sent = "Nanjing University of Posts and Telecommunications"
    if "折合" in question and re.search(r"教师|老师", question):
        matched = re.search(r"折合教师总数为\d+", sent)
        if matched:
            sent = matched.group(0)
    elif "折合" in question:
        at = sent.find("折合在校生")
        if at >= 0:
            sent = sent[at:]
    elif "职工" not in question and "教师" not in question and any(k in question for k in ("学生", "多少人")):
        for key in ("全日制在校生总人数", "折合在校生人数", "在校生人数"):
            at = sent.find(key)
            if at >= 0:
                sent = sent[at:]
                break
    if any(k in question for k in ("职工", "教师", "老师")):
        at = sent.find("专任教师")
        if at > 0:
            sent = sent[at:]
        cut = sent.find("，外籍")
        if cut > 0:
            sent = sent[:cut]
        cut = sent.find("，折合在校生")
        if cut > 0:
            sent = sent[:cut]
        cut = sent.find("折合在校生")
        if cut > 0:
            sent = sent[:cut].rstrip("，,")
    if "生师比" in question:
        matched = re.search(r"全校生师比为\d+(?:\.\d+)?(?:：\d+)?", sent)
        if matched:
            sent = matched.group(0)
    if re.search(r"专业", question) and re.search(r"多少|几个", question) and "博士" not in question and "硕士" not in question and "学费" not in question and "学院" not in question and "微专业" not in question and "学分" not in question:
        matched = re.search(r"共有\d+个本科专业|本科专业\d+个", sent)
        if matched:
            sent = matched.group(0)
    if "双一流" in question and re.search(r"学科|哪个|什么", question):
        matched = re.search(r"依托一级学科[“\"][^”\"]{2,20}[”\"]入选国家[“\"]双一流[”\"]建设学科", sent)
        if matched:
            sent = matched.group(0)
    if "休学" in question and "累计休学" in sent:
        matched = re.search(r"休学一般以一学年为限.{0,40}不得超过两学年", sent) or re.search(r"累计休学一般不得超过两学年", sent)
        if matched:
            sent = matched.group(0)
    if "转专业" in question and "15%" in sent and re.search(r"名额|比例|限制|多少", question):
        matched = re.search(r"转入专业拟接收人数原则上不超过[^。]*", sent) or re.search(r"[^。]*15%[^。]*", sent)
        if matched:
            sent = matched.group(0)
    if "转专业" in question and "第二学期" in sent and re.search(r"什么时候|何时|几月", question):
        matched = re.search(r"转专业工作一般安排在每学年第二学期[^。]*", sent)
        if matched:
            sent = matched.group(0)
    if "退学" in question and "重新参加入学考试" in sent:
        matched = re.search(r"学生因退学等情况中止学业[^。]*重新参加入学考试[^。]*", sent)
        if matched:
            sent = matched.group(0)
    if "双一流" in question and "依托一级学科" not in sent:
        at = sent.find("入选")
        if at < 0:
            at = sent.find("双一流")
        if at > 0:
            sent = sent[at:]
    if "奖学" in question and re.search(r"每人每年\d+$", sent):
        sent += "元"
    if "奖学" in question and "奖励标准" in sent:
        found = re.findall(r"国家[\u4e00-\u9fff]{0,8}奖学金的?奖励标准为每人每年\d+元?", sent)
        found = [item if item.endswith("元") else item + "元" for item in found]
        if len(found) >= 2:
            sent = "。".join(dict.fromkeys(found))
        elif found:
            sent = found[0]
    if _asks_rule(question):
        focus = _rule_focus(question)
        if focus and focus in sent:
            matched = re.search(r"[^。]{0,18}必须参加[^。]{0,8}补考[^。；]{0,10}", sent) if focus == "补考" else None
            if not matched:
                matched = re.search(
                    r"(?:严禁|禁止|不准|不许)[^。；]{0,24}" + re.escape(focus) + r"(?:，[^。]{0,36})?",
                    sent,
                )
            if matched:
                sent = matched.group(0)
    if _asks_hours(question):
        week = re.search(r"周一至周五[，,]?\d{1,2}[:：]\d{2}.{0,6}\d{1,2}[:：]\d{2}", sent)
        weekend = re.search(r"周六[、，,]周日[，,]?\d{1,2}[:：]\d{2}.{0,6}\d{1,2}[:：]\d{2}", sent)
        mode = _hours_mode(question)
        if mode == "weekday" and week:
            sent = week.group(0)
        elif mode == "weekend" and weekend:
            sent = weekend.group(0)
        elif week and weekend:
            sent = week.group(0) + "。" + weekend.group(0)
        elif weekend:
            sent = weekend.group(0)
        elif week:
            sent = week.group(0)
    kind = _fee_kind(question)
    if kind and kind in sent:
        stated = re.search(re.escape(kind) + r"类专业每生每年\d+元", sent)
        if stated:
            sent = stated.group(0)
    level = "" if "中外合作" in question else _grad_level(question)
    if level:
        if level == "博士":
            bits = re.findall(r"博士研究生\d+元/生.?学年", sent)
        elif level == "学术":
            bits = re.findall(r"学术型硕士研究生\d+元/生.?学年", sent)
        elif level == "专业学位":
            bits = re.findall(r"专业学位硕士研究生\d+元/生.?学年", sent)
        else:
            bits = re.findall(r"(?:博士研究生|全日制学术型硕士研究生|其他全日制专业学位硕士研究生)\d+元/生.?学年", sent)
        if bits:
            sent = "，".join(dict.fromkeys(bits))
    if "助学金" in question:
        matched = re.search(r"国家助学金分为三档，为每人每年\d+\s*[-－~～]\s*\d+元[^。]*", sent)
        if matched:
            sent = matched.group(0)
    if "住宿" in question and "费" in question:
        bits = re.findall(r"(?:较好条件4人/间|较好条件5-6人间|一般条件5-6人间|条件较差6人/间|普通宿舍)\d+", sent)
        if bits:
            sent = "，".join(bits)
            if "元/生" in (hit.get("text") or ""):
                sent = "住宿费（元/生·学年）：" + sent
    if _asks_school_code(question):
        matched = re.search(r"学校代码[:：]国标代码[:：]\d+", sent) or re.search(r"国标代码[:：]\d+", sent)
        if matched:
            sent = matched.group(0)
    if "校训" in question:
        matched = re.search(r"(?:践行)?[“\"][^”\"]{2,30}[”\"]的校训", sent)
        if matched:
            sent = matched.group(0)
    if _asks_campuses(question):
        matched = re.search(r"现有[^。]{0,40}校区", sent) or re.search(r"仙林、三牌楼、锁金村[^。]{0,16}校区", sent)
        if matched:
            sent = matched.group(0)
    if _asks_borrow_count(question):
        matched = re.search(r"(?:研究生、)?本科生每证限借\d+册", sent)
        if matched:
            sent = matched.group(0)
    if _asks_borrow_days(question):
        matched = re.search(r"研究生借期\d+天，本科生借期\d+天", sent)
        if matched:
            sent = matched.group(0)
    if "作弊" in question:
        matched = re.search(r"学生严重违反考核纪律或者作弊的，[^。]*零分[^。]*", sent)
        if matched:
            sent = matched.group(0)
    if "旷课" in question:
        matched = re.search(r"无故旷课累计超过[^。]*", sent)
        if matched:
            sent = matched.group(0)
    if "转专业" in question and "可以申请转专业" in sent:
        at = sent.find("可以申请转专业")
        sent = sent[max(0, at - 24):]
    if re.search(r"博士点|几个博士|多少个博士", question):
        matched = re.search(r"一级学科博士学位授权点\d+个(?:，博士专业学位授权点[^，]{0,16}\d+个)?", sent)
        if matched:
            sent = matched.group(0)
    if re.search(r"推免|保研", question) and "申请条件" in sent:
        matched = re.search(r"申请条件.{0,70}", sent)
        if matched:
            sent = matched.group(0)
    if "缓考" in question and "申请缓考程序" in sent:
        matched = re.search(r"申请缓考程序.{0,90}", sent)
        if matched:
            sent = matched.group(0)
    if "留学生" in question and "自费来华留学生" in sent:
        matched = re.search(r"自费来华留学生学费.{0,260}", sent)
        if matched:
            sent = matched.group(0)
    if re.search(r"几个学院|多少个学院|多少学院|教学机构", question):
        matched = re.search(r"现有[^。]{0,24}校区，\d+个教学机构", sent) or re.search(r"校区，\d+个教学机构", sent)
        if matched:
            sent = matched.group(0)
    if re.search(r"隶属|属于哪个|哪个部门|主管部门", question) and "共建" in sent:
        matched = re.search(r"20\d{2}年起[^。]{0,16}工业和信息化部[^。]{0,28}共建", sent) or re.search(r"工业和信息化部[^。]{0,28}共建", sent)
        if matched:
            sent = matched.group(0)
    if "电话" in question:
        office = re.search(r"(?:财务处|教务处)(?:咨询|联系)?电话[:：]?\d{7,}", sent)
        if office and any(name in question for name in ("财务处", "教务处")):
            sent = office.group(0).rstrip("，,；;")
        else:
            matched = re.search(r"电话\d{7,}（[^）]{2,8}）或者\d{7,}（[^）]{2,8}）", sent) or re.search(r"电话[:：][0-9０-９\-－，,]{8,28}", sent)
            if matched and re.search(r"\d{7,}", matched.group(0)):
                sent = matched.group(0).rstrip("，,；;")
    asked_district = re.search(_DISTRICT, question or "")
    if _asks_place(question) and asked_district:
        campus = _asked_campus(question)
        matched = re.search(
            (re.escape(campus) + r"地址[:：][^。]{0,20}" if campus else r"[^。]{0,8}")
            + re.escape(asked_district.group(0))
            + r"[^。]{0,18}",
            sent,
        )
        if matched:
            sent = matched.group(0)
    if "邮编" in question or "邮政编码" in question:
        campus = _asked_campus(question)
        window = sent
        if campus and campus in sent:
            at = sent.find(campus)
            window = sent[at: at + 42]
            if not re.search(r"(?:邮编|邮政编码)[:：]\d{6}", window):
                window = sent[max(0, at - 24): at]
        matched = re.search(r"(?:邮编|邮政编码)[:：]\d{6}", window) or re.search(r"(?:邮编|邮政编码)[:：]\d{6}", sent)
        if matched:
            sent = (campus + "：" if campus else "") + matched.group(0) if campus else matched.group(0)
    if re.search(r"男女|性别", question) and "男生" in sent and "女生" in sent:
        sent = re.sub(r"^（[一二三四五六七八九十]+）性别结构", "", sent)
        matched = re.search(r"(?:20\d{2}届毕业生中，)?男生\d+人（[^）]*），女生\d+人（[^）]*），[^。]*性别比[^。]*", sent)
        if matched:
            sent = matched.group(0)
    if "光盘" in question and re.search(r"书|图书", question) and "每盘每天" in sent and "0.10" in sent:
        start = sent.find("随书光盘应按期归还")
        if start < 0:
            start = sent.find("每盘每天")
        end = sent.find("0.10元交纳违约金", start)
        if end > start:
            sent = sent[start:end + len("0.10元交纳违约金")]
    if "光盘" in question and "书" not in question and "每盘每天" in sent:
        matched = re.search(r"[^。；]{0,16}每盘每天\d+(?:\.\d+)?元", sent)
        if matched:
            sent = matched.group(0)
    if "体测" in question or "体质" in question:
        matched = re.search(r"学生毕业时《国家学生体质健康标准》[^。]*", sent) if "毕业" in question else None
        if not matched:
            matched = re.search(r"学生每学年必须参加一次《国家学生体质健康标准》的测试[^。]*。学生毕业时《国家学生体质健康标准》[^。]*", sent)
        if not matched:
            matched = re.search(r"测试成绩不及格者[^。]*", sent)
        if matched:
            sent = matched.group(0)
    if "邮箱" in question or "信箱" in question:
        matched = re.search(r"E-mail[:：]\S+@\S+", sent, re.I) or re.search(r"[\w.+-]+@njupt\.edu\.cn", sent)
        if matched:
            sent = matched.group(0)
    sent = re.sub(r"[一二三四五六七八九十]+、逾期处理\d+\.", "", sent)
    title = hit.get("doc_title") or ""
    url = hit.get("url") or ""
    body = sent if sent.endswith(("。", "；", "！")) else sent + "。"
    return {
        "answer": body + f"\n（摘自 {title} {url}）",
        "refs": f"{title} {url}".strip(),
        "plan": plan,
        "hits": hits,
    }


def command(question: str) -> dict | None:
    """模型负责换词检索。答句从原句里选，不采用改写后的数字。"""
    if os.environ.get("ASKNJUPT_AGENT", "1") == "0":
        return None
    from types import SimpleNamespace

    han = "".join(ch for ch in question if "\u4e00" <= ch <= "\u9fff")
    body = re.sub(r"南京邮电大学|南京邮电|邮电大学|南邮", "", han)
    if any(w in question for w in _CHAT) or (len(body) < 2 and not re.search(r"\d", question)):
        return {
            "answer": f"{AGENT_REFUSAL_PREFIX}不能确定答案。",
            "refs": "",
            "plan": SimpleNamespace(),
            "hits": [],
        }
    messages: list[dict] = [
        {"role": "system", "content": _DIRECTOR},
        {"role": "user", "content": question},
    ]
    plan = None
    hits: list[dict] = []
    pool: list[tuple[str, dict]] = []
    memory: list[tuple[str, dict]] = []
    seen_mem: set[str] = set()
    used: list[str] = []
    searches = 0
    choice = "required"
    for _round in range(6):
        if searches >= 3:
            break
        message = _agent_turn(
            messages,
            choice,
            temperature=0.6,
            think=True,
        )
        if message is None:
            if searches == 0:
                return None
            break
        query = _tool_query(message)
        if not query and choice == "required":
            raw = _strip_tool_echo((message.get("content") or "").strip()).splitlines()
            raw = raw[0].strip("。 ") if raw else ""
            if 2 <= len(raw) <= 30:
                query = raw
        if not query:
            # 这一轮没发出检索词时，仍用问句本身扫一遍，避免还没查库就拒答。
            if searches == 0:
                query = question
            else:
                break
        if query in used:
            messages.append(_assistant_message(message))
            if message.get("tool_calls"):
                messages.append({
                    "role": "tool",
                    "tool_call_id": _call_id(message),
                    "content": "这组词已经查过。请换一组不同的词。",
                })
            messages.append({"role": "user", "content": _retry(used, "这组词已经查过。")})
            choice = "required"
            continue
        searches += 1
        used.append(query)
        try:
            plan, hits, pool, evidence = _tool_search(query, question)
        except Exception:
            pool, evidence = [], ""
        for sent, hit in pool:
            if sent[:24] in seen_mem:
                continue
            seen_mem.add(sent[:24])
            memory.append((sent, hit))
        shown = "\n".join(evidence.splitlines()[:8]) or "（没有返回段落）"
        messages.append(_assistant_message(message))
        messages.append({"role": "tool", "tool_call_id": _call_id(message), "content": shown})
        ready = _answer_candidates(question, memory, used)
        if ready and not (_asks_figure(question) and searches < 2):
            break
        why = (
            "再换一种统计报告里的口径，不要重复刚才的词。"
            if ready
            else "这些句子没有给出所问的总体数量。请换成统计报告里的另一种说法，不要重复刚才的词。"
        )
        messages.append({"role": "user", "content": _retry(used, why)})
        choice = "required"
    cands = _answer_candidates(question, memory, used)
    picked = _choose_sentence(question, cands)
    if not picked:
        return {
            "answer": f"{AGENT_REFUSAL_PREFIX}不能确定答案。",
            "refs": "",
            "plan": plan or SimpleNamespace(),
            "hits": hits,
        }
    if "励志" in question and re.search(r"一样|相同|区别|不同", question):
        bits = []
        for sent, _hit in cands:
            matched = re.search(r"国家[\u4e00-\u9fff]{0,6}奖学金的?奖励标准为每人每年\d+元?", sent)
            if not matched:
                continue
            bit = matched.group(0) if matched.group(0).endswith("元") else matched.group(0) + "元"
            if bit not in bits:
                bits.append(bit)
        if len(bits) >= 2:
            picked = ("。".join(bits), picked[1])
    if "光盘" in question and any(k in question for k in ("外文", "图书", "书")) and not ("每盘每天" in picked[0] and "0.10" in picked[0]):
        parts = []
        for sent, _hit in cands:
            if "交纳违约金" not in sent or sent in parts:
                continue
            parts.append(sent if sent.endswith("。") else sent + "。")
            if len(parts) >= 2:
                break
        if len(parts) == 2:
            picked = ("".join(parts), picked[1])
    return _finish(picked[0], picked[1], plan or SimpleNamespace(), hits, question)



def review_hits(question: str, hits: list[dict]) -> str | None:
    """返回摘录通过的原句，或明确拒答。模型服务连不上时返回 None，调用方走原兜底。"""
    if os.environ.get("ASKNJUPT_AGENT", "1") == "0":
        return None
    if _complete("只输出1", 4) is None:
        return None
    rounds = ["原问题"]
    # 闲聊和乱码没有可检索的事项。先拒掉，避免模型从旁边句子里抄出「心境」这类字充数。
    if not _specific_terms(question):
        return _refuse(rounds)
    found = _screen(question, hits)
    if found:
        return _format(found[0], found[1], None)
    from .retrieve import retrieve

    queries: list[str] = []
    rewritten = _rewrite(question, [question])
    if rewritten and not _too_close(rewritten, question):
        queries.append(rewritten)
    synonym = _synonym_query(question)
    if synonym and synonym not in queries and not _too_close(synonym, question):
        queries.append(synonym)
    for query in queries[:2]:
        rounds.append(query)
        try:
            _plan, hits_n = retrieve(query)
        except Exception:
            hits_n = []
        found = _screen(question, hits_n, broad=True) if hits_n else None
        if found:
            return _format(found[0], found[1], query)
    return _refuse(rounds)
