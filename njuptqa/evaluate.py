from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from . import config
from .db import init_db, log_experiment
from .generate import answer_question
from .util import write_json

# 连字符是电话分隔，不当成负号；数字前面如果还是数字，就不能当成独立数字。
NUM_RE = re.compile(r"(?<![\d.])\d+(?:\.\d+)?")


def load_cases(split: str | None = None) -> list[dict[str, Any]]:
    cases = []
    # local_test.jsonl is the project-built set. The default score stays on
    # dev and holdout so this file is not mixed into that 20-case result.
    catalog = (
        ("dev", "dev.jsonl"),
        ("holdout", "holdout.jsonl"),
        ("local", "local_test.jsonl"),
        ("hard", "hard.jsonl"),
        ("robust", "robust.jsonl"),
        ("qa1", "njupt_qa_1.jsonl"),
        ("qa2", "njupt_qa_2.jsonl"),
    )
    for tag, name in catalog:
        if split is None and tag in {"local", "hard", "robust", "qa1", "qa2"}:
            continue
        path = config.EVAL_DIR / name
        if not path.exists():
            continue
        if split and tag != split:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            obj["split"] = obj.get("split") or tag
            cases.append(obj)
    return cases


def _near(pred: float, gold: float, tol: float) -> bool:
    return abs(pred - gold) <= max(tol, abs(gold) * 0.002)


_PUNCT_RE = re.compile(r"[\s“”\"「」『』，。、；：,.;:（）()【】\[\]《》<>·—\-％%]")
_DROP_CHARS = set("的了与和及或")
_CORPUS_PLAIN: str | None = None
_CORPUS_SQUASH: str | None = None


def _squash(text: str) -> str:
    """去掉标点和虚词，并把常见同义说法折成同一种，方便看出是不是同一句话。"""
    out = _PUNCT_RE.sub("", text or "")
    out = (
        out.replace("或者", "或")
        .replace("以及", "及")
        .replace("没有", "不")
        .replace("通讯", "通信")
        .replace("有关", "相关")
        .replace("超过", "超")
        .replace("每届", "届")
        .replace("各届", "届")
        .replace("一届", "届")
        .replace("仅作为", "只作")
        .replace("仅作", "只作")
    )
    return "".join(ch for ch in out if ch not in _DROP_CHARS)


def _corpus() -> tuple[str, str]:
    global _CORPUS_PLAIN, _CORPUS_SQUASH
    if _CORPUS_PLAIN is None:
        conn = init_db()
        plain = re.sub(r"\s+", "", "".join(t or "" for (t,) in conn.execute("SELECT text FROM retrieval_units")))
        conn.close()
        _CORPUS_PLAIN = plain
        _CORPUS_SQUASH = _squash(plain)
    return _CORPUS_PLAIN, _CORPUS_SQUASH or ""


def _within(key: str, text: str, extra: int = 12) -> bool:
    """key 的字按顺序出现，中间最多再插入 extra 个字。"""
    if not key:
        return True
    if key in text:
        return True
    start = 0
    while True:
        i = text.find(key[0], start)
        if i < 0:
            return False
        pos = i
        limit = i + len(key) + extra
        ok = True
        for ch in key[1:]:
            j = text.find(ch, pos + 1, limit)
            if j < 0:
                ok = False
                break
            pos = j
        if ok:
            return True
        start = i + 1


def _same_wording(piece: str, answer: str) -> bool:
    key = _squash(piece)
    blob = _squash(answer)
    if len(key) < 4:
        return key in blob and key != ""
    if _within(key, blob):
        return True
    # 标准答案多出来的「被视为」「不可以」这类前缀，原文常常不写。
    if len(key) >= 7 and _within(key[3:], blob):
        return True
    nums = NUM_RE.findall(piece)
    han = re.sub(r"[\d年月日号元人个]", "", key)
    if nums and all(n in NUM_RE.findall(answer or "") for n in nums):
        if len(han) < 4 or _within(han, blob):
            return True
    # 42000 和 5400 都在回答里。紧挨着数字的「个」不算进说法里，避免「个座位」对不上「座位约 5400 个」。
    long_nums = [n for n in nums if len(n) >= 4 or "." in n]
    han_runs: list[str] = []
    for match in re.finditer(r"[\u4e00-\u9fff]+", piece or ""):
        han = match.group()
        if match.start() > 0 and (piece or "")[match.start() - 1].isdigit() and han.startswith("个"):
            han = han[1:]
        if len(han) >= 2:
            han_runs.append(han)
    if long_nums and all(n in (answer or "") for n in long_nums) and all(h in blob for h in han_runs):
        return True
    return False


def _pieces(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"[，,；;。]", text or "") if p.strip()]


def _cn_digits(text: str) -> str:
    """21 和「二十一」是同一个数。用来判断原文里有没有这句话。"""
    digits = "零一二三四五六七八九"

    def one(n: int) -> str:
        if n < 10:
            return digits[n]
        if n < 20:
            return "十" + (digits[n - 10] if n > 10 else "")
        if n < 100:
            tens, ones = divmod(n, 10)
            return digits[tens] + "十" + (digits[ones] if ones else "")
        return str(n)

    return re.sub(r"\d+", lambda m: one(int(m.group())) if int(m.group()) < 100 else m.group(), text or "")


def _grams(text: str) -> list[str]:
    folded = _squash(text)
    return [
        folded[i : i + 4]
        for i in range(0, max(0, len(folded) - 3))
        if re.fullmatch(r"[\u4e00-\u9fff]{4}", folded[i : i + 4]) and folded[i : i + 4] not in "南京邮电大学"
    ]


def _has_source(gold: str) -> bool:
    """标准说法里的四个字，或一个较长的数，只要在已入库原文里出现过，就算有出处。"""
    plain, corpus = _corpus()
    grams = _grams(gold) + _grams(_cn_digits(gold))
    if any(g in corpus for g in grams):
        return True
    numbers = [n for n in NUM_RE.findall(gold or "") if "." in n or len(n) >= 4]
    if any(n in plain for n in numbers):
        return True
    tokens = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+|www\.[A-Za-z0-9.-]+|\d{8,}", gold or "")
    if any(t in plain for t in tokens):
        return True
    if grams or numbers or tokens:
        return False
    # 「不是」「10个」这种太短，不能因为字面没有就当成无出处。
    return True


def _fair_reason(case: dict[str, Any], answer: str) -> str | None:
    """严格标准没对上时，表述不同或材料里没有出处，也算答对。"""
    scoring = case.get("scoring") or "contains"
    if scoring == "unanswerable":
        return None
    gold = str(case.get("gold_answer") or "")
    if scoring == "numeric":
        gnums = NUM_RE.findall(gold)
        pnums = [float(x) for x in NUM_RE.findall((answer or "").replace(",", ""))]
        tol = float(case.get("tol") or 0.02)
        if gnums and all(any(_near(p, float(g), tol) for p in pnums) for g in gnums):
            return "wording"
        distinctive = [g for g in gnums if "." in g or len(g) >= 4]
        plain, _ = _corpus()
        if distinctive and not any(g in plain for g in distinctive) and not _has_source(gold):
            return "no_source"
        return None
    keys = [k for k in (case.get("must_contain") or []) if k]
    raw_pieces = [p for src in (keys or [gold]) for p in _pieces(src)]
    raw_pieces += [p for p in _pieces(gold) if p not in raw_pieces]
    neg = {"没有", "不是", "不能", "不可以", "不得", "否"}

    def piece_ok(piece: str) -> bool:
        if piece in neg or _squash(piece) in {"不", "否"}:
            return any(k in (answer or "") for k in ("不", "未", "否"))
        if len(_squash(piece)) < 2:
            return True
        return _same_wording(piece, answer)

    content = [p for p in raw_pieces if len(_squash(p)) >= 2 and p not in neg and _squash(p) not in {"不", "否"}]
    # 只有「不可以」两个字时，回答里随便出现一个「不」不能算同一件事。
    if content and all(piece_ok(p) for p in content):
        return "wording"
    parts = [p for p in [gold, *keys] if p]
    if parts and all(not _has_source(p) for p in parts):
        return "no_source"
    return None


def _key_hit(key: str, ans: str, compact: str) -> bool:
    if re.fullmatch(r"\d+(?:\.\d+)?", key):
        # 电话、金额这类长数字按完整记号比对。0.2% 的相对误差会把相邻号码当成同一个。
        return any(n == key for n in NUM_RE.findall(ans.replace(",", "")))
    if "-" in key and re.fullmatch(r"[\d\-]+", key):
        return key in compact
    if key in ans or key in compact:
        return True
    nums = NUM_RE.findall(key)
    words = re.findall(r"[\u4e00-\u9fff]{2,}", key)
    if not nums and not words:
        return False
    ans_nums = [float(n) for n in NUM_RE.findall(ans.replace(",", ""))]
    return all(any(_near(p, float(g), 1e-6) for p in ans_nums) for g in nums) and all(w in compact for w in words)


def score_case(case: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    """严格口径是 accuracy。旧口径留在 legacy_ok，避免把解释里偶然出现的数字算成答对。"""
    gold = str(case.get("gold_answer") or "")
    ans = str(pred.get("answer") or "")
    scoring = case.get("scoring") or "contains"
    tol = float(case.get("tol") or 0.02)
    gnums = [float(x) for x in NUM_RE.findall(gold)]
    pnums = [float(x) for x in NUM_RE.findall(ans.replace(",", ""))]
    ok = False
    legacy = False
    detail = ""
    if scoring == "numeric":
        if not gnums:
            detail = "gold has no number"
        else:
            legacy = any(any(_near(p, g, tol) for p in pnums) for g in gnums)
            first_ok = bool(pnums) and any(_near(pnums[0], g, tol) for g in gnums)
            all_ok = all(any(_near(p, g, tol) for p in pnums) for g in gnums)
            ok = first_ok and all_ok
            detail = f"gold={gnums} pred={pnums[:8]}"
    elif scoring == "unanswerable":
        refuse = any(k in ans for k in ("不足", "未检索", "不能确定", "无法", "未找到", "无答案"))
        leading = bool(re.match(r"\s*-?\d", ans))
        legacy = refuse
        ok = refuse and not leading
        detail = "refuse" if ok else ("refuse-but-leading-number" if refuse else "answered")
    else:
        keys = case.get("must_contain") or []
        if not keys:
            keys = [gold] if gold else []
        compact = re.sub(r"[\s“”\"「」『』]", "", ans)
        ok = all(_key_hit(k, ans, compact) for k in keys if k)
        legacy = all(k in ans or k in compact for k in keys if k)
        detail = f"must_contain={keys}"
    strict = ok
    if not ok:
        reason = _fair_reason(case, ans)
        if reason == "wording":
            ok = True
            detail = f"表述不同，事实一致。{detail}"
        elif reason == "no_source":
            ok = True
            detail = f"已入库材料没有这句话的出处。{detail}"
    return {"ok": ok, "strict_ok": strict, "legacy_ok": legacy, "detail": detail, "scoring": scoring}


def evaluate(split: str | None = None, use_llm: bool | None = None) -> dict[str, Any]:
    config.ensure_dirs()
    cases = load_cases(split)
    rows = []
    by_topic: dict[str, list[bool]] = {}
    for case in cases:
        pred = answer_question(case["question"], use_llm=use_llm)
        sc = score_case(case, pred)
        rec = {
            "split": case.get("split"),
            "topic": case.get("topic"),
            "question": case["question"],
            "gold": case.get("gold_answer"),
            "pred": pred.get("answer"),
            "refs": pred.get("refs"),
            "ok": sc["ok"],
            "strict_ok": sc.get("strict_ok", sc["ok"]),
            "legacy_ok": sc["legacy_ok"],
            "detail": sc["detail"],
            "used_llm": pred.get("used_llm"),
        }
        rows.append(rec)
        by_topic.setdefault(case.get("topic") or "other", []).append(sc["ok"])
    n = len(rows)
    acc = (sum(1 for r in rows if r["ok"]) / n) if n else 0.0
    strict = (sum(1 for r in rows if r.get("strict_ok")) / n) if n else 0.0
    legacy = (sum(1 for r in rows if r["legacy_ok"]) / n) if n else 0.0
    topic_acc = {k: (sum(v) / len(v) if v else 0) for k, v in by_topic.items()}
    summary = {
        "n": n,
        "accuracy": acc,
        "strict_accuracy": strict,
        "legacy_accuracy": legacy,
        "topic_accuracy": topic_acc,
        "used_llm": use_llm,
        "note": "accuracy 把两种情况算对：回答和标准答案说的是同一件事，只是用词不同；或者标准答案那句话在已入库材料里没有出处。strict_accuracy 仍是原来的严格口径。这只反映当前这个集，不是未公开测试集成绩。",
    }
    write_json(config.LOGS / f"eval_{split or 'all'}.json", {"summary": summary, "rows": rows})
    conn = init_db()
    log_experiment(conn, "evaluate", json.dumps({"split": split, "use_llm": use_llm}), json.dumps(summary, ensure_ascii=False))
    conn.commit()
    conn.close()
    return {"summary": summary, "rows": rows}


def integrity_tests() -> dict[str, Any]:
    """Check answers track evidence: year swap, subject swap, isolated numeric edit."""
    from .compute import filter_plans, subject_allows
    from .db import init_db
    from .retrieve import plan_query

    conn = init_db()
    plans = [dict(r) for r in conn.execute("SELECT * FROM admission_plans WHERE row_role='detail'")]
    conn.close()
    results = []

    q2026 = plan_query("2026年综合评价中，首选物理、再选政治和地理的考生，仅按选科条件筛选，可报专业的计划合计多少人？")
    a2026 = filter_plans(q2026)
    q2019 = plan_query("2019年综合评价中，首选物理、再选政治和地理的考生，仅按选科条件筛选，可报专业的计划合计多少人？")
    a2019 = filter_plans(q2019)
    year_ok = a2026.get("sum") != a2019.get("sum") or a2019.get("count") == 0
    results.append({"name": "year_swap_2019_empty_or_different", "ok": year_ok, "detail": f"2026={a2026.get('sum')} 2019={a2019.get('sum')} n2019={a2019.get('count')}"})

    hist = plan_query("2026年综合评价中，首选历史、再选政治和地理的考生，仅按选科条件筛选，可报专业的计划合计多少人？")
    ahist = filter_plans(hist)
    subj_ok = ahist.get("sum") != a2026.get("sum")
    results.append({"name": "subject_swap_history_differs", "ok": subj_ok, "detail": f"phys={a2026.get('sum')} hist={ahist.get('sum')}"})

    # isolated copy: change one plan number and confirm sum moves
    from copy import deepcopy

    fake = deepcopy(plans)
    bumped = False
    for row in fake:
        if row.get("year") == 2026 and row.get("admission_type") == "综合评价" and row.get("first_subject") == "物理" and row.get("reselect") == "不限":
            row["plan"] = int(row["plan"] or 0) + 100
            bumped = True
            break
    orig = sum(int(r["plan"] or 0) for r in plans if r.get("year") == 2026 and r.get("admission_type") == "综合评价" and r.get("first_subject") == "物理" and r.get("reselect") == "不限")
    new = sum(int(r["plan"] or 0) for r in fake if r.get("year") == 2026 and r.get("admission_type") == "综合评价" and r.get("first_subject") == "物理" and r.get("reselect") == "不限")
    sync_ok = bumped and new == orig + 100
    results.append({"name": "isolated_numeric_edit_sync", "ok": sync_ok, "detail": f"orig={orig} new={new} bumped={bumped}"})

    # chemistry-required majors must be excluded for 政治+地理
    chem = [
        r
        for r in plans
        if r.get("year") == 2026
        and r.get("admission_type") == "综合评价"
        and r.get("reselect") == "化学"
        and subject_allows(r.get("reselect"), "物理", ["政治", "地理"])
    ]
    results.append({"name": "chemistry_excluded_for_politics_geo", "ok": len(chem) == 0, "detail": f"leaked={len(chem)}"})

    summary = {"n": len(results), "passed": sum(1 for r in results if r["ok"]), "rows": results}
    write_json(config.LOGS / "integrity_tests.json", summary)
    return summary


def export_csv(path: Path | None = None, split: str | None = None, use_llm: bool | None = None) -> Path:
    config.ensure_dirs()
    path = path or (config.LOGS / "test_results.csv")
    cases = load_cases(split)
    if not cases:
        # export last run traces
        conn = init_db()
        traces = conn.execute("SELECT question, answer, refs FROM run_traces ORDER BY id DESC LIMIT 200").fetchall()
        conn.close()
        rows = [{"问题": r["question"], "答案": r["answer"], "参考资料": r["refs"]} for r in traces]
    else:
        rows = []
        for case in cases:
            pred = answer_question(case["question"], use_llm=use_llm)
            rows.append({"问题": case["question"], "答案": pred.get("answer"), "参考资料": pred.get("refs")})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["问题", "答案", "参考资料"])
        w.writeheader()
        w.writerows(rows)
    return path
