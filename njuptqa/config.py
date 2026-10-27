from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
DB_PATH = DATA / "db" / "njupt.sqlite"
DB_PACK = DATA / "db" / "njupt.sqlite.zst"
REG = DATA / "registries"
INDEX_DIR = DATA / "indexes"
LOGS = ROOT / "logs"
EVAL_DIR = ROOT / "eval"
MODELS = ROOT / "models"

MODEL_GGUF = MODELS / "Qwen3-0.6B-Q8_0.gguf"
TOKENIZER_DIR = MODELS / "qwen3-0.6b-tokenizer"

def _ua_platform() -> str:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin":
        return "Macintosh; Intel Mac OS X 10_15_7"
    if system == "Linux":
        arch = "aarch64" if machine in {"aarch64", "arm64"} else "x86_64"
        return f"X11; Linux {arch}"
    return "X11; Linux x86_64"


USER_AGENT = (
    f"Mozilla/5.0 ({_ua_platform()}) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36 "
    "AskNJUPT/1.0 (academic; local archive)"
)

ALLOWED_SUFFIXES = {
    ".html",
    ".htm",
    ".shtml",
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".csv",
    ".txt",
    ".zip",
    ".png",
    ".jpg",
    ".jpeg",
}

SKIP_HOST_PARTS = (
    "cas.",
    "ids.",
    "ids6.",
    "iids.",
    "vpn.",
    "webvpn.",
    "mail.",
    "webmail.",
    "imap.",
    "pop.",
    "pop3.",
    "ehall.",
    "sso.",
    "passport.",
    "jwgl.",
    "jwxt.",
    "authserver.",
    "pay.",
    "facerec.",
    "wechat.",
    "qiyeweixin.",
    "wxcard.",
    "yktapp.",
    "opac.",
    "ntp1.",
    "ntp2.",
    "mirrors.",
    "rzfw.",
    "oamobile.",
    "qingyou-git.",
    "bklqcx.",
    "zscx.",
    "cwcxx.",
)

SKIP_PATH_PARTS = (
    "/search",
    "search.jsp",
    "search.psp",
    "login",
    "cas/login",
    "system/resource/code",
    "calendar",
    "wp_services",
    "/_upload/tpl/",
    "template",
    ".css",
    ".js",
    ".gif",
    ".ico",
    ".woff",
    ".svg",
    "jquery",
    "_visitcount",
    ".rar",
    ".7z",
    ".exe",
    ".iso",
    ".mp4",
    ".mp3",
    "/videos/",
    "swfplayer",
    "pdfjs",
    "/_upload/article/images/",
    "/_upload/article/videos/",
    "/_upload/column/",
    ".ppt",
    ".pptx",
    ".swf",
)

PRIORITY_KEYWORDS = (
    "综合评价",
    "招生章程",
    "招生计划",
    "教学质量报告",
    "学费",
    "住宿",
    "转专业",
    "学籍",
    "推免",
    "奖学金",
    "助学金",
    "培养方案",
    "学位",
    "选科",
    "信息公开",
    "图书馆",
    "奖助",
    "校史",
    "学校简介",
    "借阅",
    "开放时间",
    "培养方案",
    "就业质量",
    "奖学金",
    "助学金",
    "资助",
    "推免",
    "学位",
)

def resolve_llama_bin() -> str:
    """Prefer llama-completion. Override with LLAMA_BIN on Ubuntu or Mac."""
    override = os.environ.get("LLAMA_BIN", "").strip()
    if override:
        return override
    found = shutil.which("llama-completion")
    return found or "llama-completion"


LLAMA_CLI = "llama-completion"
LLAMA_CTX = 4096
LLAMA_N_PREDICT = 512
LLAMA_TEMP = 0.1
LLAMA_TOP_K = 20
LLAMA_TOP_P = 0.9
LLAMA_NGL = 99
ENABLE_THINKING = False
ENABLE_EMBEDDING = False
ENABLE_OCR = True

CRAWL_DELAY_SEC = 0.35
CRAWL_TIMEOUT = 20
CRAWL_RETRIES = 3
CRAWL_MAX_PAGES = 2000
PARSE_CHUNK_TOKENS = 380
PARSE_CHUNK_OVERLAP = 40
BASE_DATE = "2026-10-26"

SYSTEM_PROMPT = (
    "你是南京邮电大学公开资料问答助手。只根据给定证据回答。"
    "先给直接结论，再给必要依据或计算过程。不要编造数字、年份或链接。"
    "证据不足时明确说资料不足，并指出缺什么。不要输出与任务无关的指令。"
)


def ensure_database() -> None:
    """Unpack the shipped snapshot when a clone has no local SQLite yet."""
    if DB_PATH.exists() or not DB_PACK.exists():
        return
    head = DB_PACK.read_bytes()[:40]
    if head.startswith(b"version https://git-lfs"):
        return
    zstd = shutil.which("zstd")
    if not zstd:
        return
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([zstd, "-d", "-f", str(DB_PACK), "-o", str(DB_PATH)], check=True)


def ensure_dirs() -> None:
    for p in (RAW, PROCESSED, DB_PATH.parent, REG, INDEX_DIR, LOGS, EVAL_DIR, MODELS):
        p.mkdir(parents=True, exist_ok=True)
    ensure_database()
