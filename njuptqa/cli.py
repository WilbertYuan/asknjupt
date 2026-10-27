from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import config
from .crawl import crawl, discover, ingest_local
from .hosts import ingest_subdomains
from .db import init_db, log_experiment
from .evaluate import evaluate, export_csv
from .extract import extract_all, validate
from .generate import answer_question, llama_available
from .ocr import backfill_low_quality
from .parse import parse_all
from .util import utc_now, write_json


def cmd_discover(_args: argparse.Namespace) -> None:
    print(json.dumps(discover(), ensure_ascii=False, indent=2))


def cmd_crawl(args: argparse.Namespace) -> None:
    print(json.dumps(crawl(max_pages=args.max_pages), ensure_ascii=False, indent=2))


def cmd_parse(args: argparse.Namespace) -> None:
    redo = None
    ocr_bf = None
    if getattr(args, "reocr", False):
        ocr_bf = backfill_low_quality(limit=args.reocr_limit)
        redo = set(ocr_bf.get("hashes") or [])
    stats = parse_all(limit=args.limit, redo_hashes=redo)
    ext = extract_all()
    print(json.dumps({"ocr_backfill": ocr_bf, "parse": stats, "extract": ext}, ensure_ascii=False, indent=2))


def cmd_validate(_args: argparse.Namespace) -> None:
    print(json.dumps(validate(), ensure_ascii=False, indent=2, default=str))


def cmd_index(_args: argparse.Namespace) -> None:
    # FTS is updated during parse; rebuild + extract again for safety
    from .db import rebuild_fts, session

    ext = extract_all()
    with session() as conn:
        rebuild_fts(conn)
        log_experiment(conn, "index", "{}", json.dumps(ext, ensure_ascii=False), "rebuild fts")
    print(json.dumps({"indexed": True, "extract": ext}, ensure_ascii=False, indent=2))


def cmd_answer(args: argparse.Namespace) -> None:
    rec = answer_question(args.question, use_llm=None if args.llm == "auto" else args.llm == "on")
    print(json.dumps({"问题": rec["question"], "答案": rec["answer"], "参考资料": rec["refs"], "used_llm": rec["used_llm"]}, ensure_ascii=False, indent=2))


def cmd_evaluate(args: argparse.Namespace) -> None:
    use_llm = None if args.llm == "auto" else args.llm == "on"
    rec = evaluate(split=args.split, use_llm=use_llm)
    print(json.dumps(rec["summary"], ensure_ascii=False, indent=2))


def cmd_export(args: argparse.Namespace) -> None:
    use_llm = None if args.llm == "auto" else args.llm == "on"
    path = export_csv(Path(args.out) if args.out else None, split=args.split, use_llm=use_llm)
    print(path)


def cmd_update(args: argparse.Namespace) -> None:
    d = discover()
    c = crawl(max_pages=args.max_pages)
    p = parse_all()
    e = extract_all()
    print(json.dumps({"discover": d, "crawl": c, "parse": p, "extract": e}, ensure_ascii=False, indent=2))


def cmd_pipeline(args: argparse.Namespace) -> None:
    config.ensure_dirs()
    init_db()
    out = {
        "started_at": utc_now(),
        "discover": discover(),
        "crawl": crawl(max_pages=args.max_pages),
        "parse": parse_all(),
        "extract": extract_all(),
        "validate": validate(),
        "model_ready": llama_available(),
    }
    write_json(config.LOGS / "pipeline_last.json", out)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


def cmd_cleanup(_args: argparse.Namespace) -> None:
    from .cleanup import cleanup

    print(json.dumps(cleanup(), ensure_ascii=False, indent=2))


def cmd_status(_args: argparse.Namespace) -> None:
    conn = init_db()
    tables = [
        "sources",
        "crawl_manifest",
        "documents",
        "blocks",
        "tables",
        "admission_plans",
        "fee_rules",
        "lab_centers",
        "facts",
        "retrieval_units",
        "quality_queue",
    ]
    stats = {t: conn.execute(f"SELECT COUNT(*) c FROM {t}").fetchone()["c"] for t in tables}
    stats["model"] = str(config.MODEL_GGUF) if llama_available() else "missing"
    conn.close()
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="南京邮电大学公开资料离线问答")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("discover").set_defaults(func=cmd_discover)
    pc = sub.add_parser("crawl")
    pc.add_argument("--max-pages", type=int, default=config.CRAWL_MAX_PAGES)
    pc.set_defaults(func=cmd_crawl)
    pp = sub.add_parser("parse")
    pp.add_argument("--limit", type=int, default=None)
    pp.add_argument("--reocr", action="store_true")
    pp.add_argument("--reocr-limit", type=int, default=40)
    pp.set_defaults(func=cmd_parse)
    sub.add_parser("validate").set_defaults(func=cmd_validate)
    sub.add_parser("index").set_defaults(func=cmd_index)
    pa = sub.add_parser("answer")
    pa.add_argument("question")
    pa.add_argument("--llm", choices=["auto", "on", "off"], default="auto")
    pa.set_defaults(func=cmd_answer)
    pe = sub.add_parser("evaluate")
    pe.add_argument("--split", choices=["dev", "holdout", "local", "hard", "robust", "qa1", "qa2"], default=None)
    pe.add_argument("--llm", choices=["auto", "on", "off"], default="off")
    pe.set_defaults(func=cmd_evaluate)
    px = sub.add_parser("export")
    px.add_argument("--out", default="")
    px.add_argument("--split", default=None)
    px.add_argument("--llm", choices=["auto", "on", "off"], default="off")
    px.set_defaults(func=cmd_export)
    pu = sub.add_parser("update")
    pu.add_argument("--max-pages", type=int, default=400)
    pu.set_defaults(func=cmd_update)
    pl = sub.add_parser("pipeline")
    pl.add_argument("--max-pages", type=int, default=800)
    pl.set_defaults(func=cmd_pipeline)
    sub.add_parser("ingest-hosts").set_defaults(func=lambda _a: print(json.dumps(ingest_subdomains(), ensure_ascii=False, indent=2)))
    il = sub.add_parser("ingest-local")
    il.add_argument("--dir", default="")
    il.set_defaults(
        func=lambda a: print(
            json.dumps(ingest_local(Path(a.dir) if a.dir else None), ensure_ascii=False, indent=2)
        )
    )
    sub.add_parser("cleanup").set_defaults(func=cmd_cleanup)
    sub.add_parser("status").set_defaults(func=cmd_status)
    pi = sub.add_parser("integrity")
    pi.set_defaults(func=lambda _a: print(json.dumps(__import__("njuptqa.evaluate", fromlist=["integrity_tests"]).integrity_tests(), ensure_ascii=False, indent=2)))
    return p


def main(argv: list[str] | None = None) -> None:
    import sys

    sys.stdout.reconfigure(line_buffering=True)
    config.ensure_dirs()
    args = build_parser().parse_args(argv)
    args.func(args)
