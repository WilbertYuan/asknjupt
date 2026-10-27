from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urldefrag, urljoin, urlparse

from . import config

SPACE_RE = re.compile(r"[\s\u00a0\u3000]+")
YEAR_RE = re.compile(r"(20\d{2})")
ACADEMIC_YEAR_RE = re.compile(r"(20\d{2})\s*[-—–至到]\s*(20\d{2})\s*学?年")
PUNCT_RE = re.compile(r"[，。！？；：、,.!?;:()\[\]{}<>《》【】\"'‘’“”…—\-_/\\|]+")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def normalize_ws(text: str) -> str:
    return SPACE_RE.sub(" ", (text or "").replace("\xa0", " ")).strip()


def canonicalize_url(url: str, base: str | None = None) -> str:
    if base:
        url = urljoin(base, url)
    url, _ = urldefrag(url)
    parsed = urlparse(url)
    scheme = parsed.scheme.lower() or "http"
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    query = parsed.query
    # drop cache-busting
    if query and any(k in query.lower() for k in ("t=", "timestamp=", "_=")):
        parts = [p for p in query.split("&") if not re.match(r"(?i)(t|timestamp|_)=", p)]
        query = "&".join(parts)
    rebuilt = f"{scheme}://{netloc}{path}"
    if query:
        rebuilt += f"?{query}"
    return rebuilt


def host_of(url: str) -> str:
    return urlparse(url).netloc.lower()


def is_njupt(url: str) -> bool:
    host = host_of(url)
    return host.endswith("njupt.edu.cn") or host.endswith("njupt.edu.cn:443")


IMAGE_KEEP_MARKERS = ("_upload/article", "_upload/file", "_upload/cms", "/attach", "/download")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


def should_skip_url(url: str) -> bool:
    host = host_of(url)
    if host in {
        "i.njupt.edu.cn",
        "my.njupt.edu.cn",
        "my2.njupt.edu.cn",
        "admin.njupt.edu.cn",
        "forms.njupt.edu.cn",
        "p.njupt.edu.cn",
        "bysj.njupt.edu.cn",
        "dmtzk.njupt.edu.cn",
        "shiziyinjin.njupt.edu.cn",
        "tycj.njupt.edu.cn",
        "c.njupt.edu.cn",
    }:
        return True
    if any(part in host for part in config.SKIP_HOST_PARTS):
        return True
    low = url.lower()
    if any(p in low for p in config.SKIP_PATH_PARTS):
        return True
    path = urlparse(url).path.lower()
    if path.endswith(IMAGE_EXTS):
        if any(k in low for k in IMAGE_KEEP_MARKERS) and "/_upload/article/images/" not in low:
            return False
        return True
    m = re.search(r"/list(\d+)\.(htm|html|psp)$", path)
    if m and int(m.group(1)) > 12:
        return True
    return False


def guess_ext(url: str, content_type: str = "") -> str:
    path = urlparse(url).path.lower()
    for ext in config.ALLOWED_SUFFIXES:
        if path.endswith(ext):
            return ext
    ct = (content_type or "").lower()
    if "pdf" in ct:
        return ".pdf"
    if "html" in ct:
        return ".html"
    if "word" in ct or "officedocument.word" in ct:
        return ".docx"
    if "excel" in ct or "spreadsheet" in ct:
        return ".xlsx"
    if "msword" in ct:
        return ".doc"
    return ".bin"


def raw_path_for(content_hash: str, ext: str) -> Path:
    return config.RAW / content_hash[:2] / f"{content_hash}{ext}"


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_years(text: str) -> list[int]:
    years = [int(y) for y in YEAR_RE.findall(text or "")]
    # keep document order, unique
    out: list[int] = []
    for y in years:
        if y not in out:
            out.append(y)
    return out


def primary_year(title: str, text: str = "") -> int | None:
    """Prefer a year that appears in the title, not a citation year in the body."""
    title_years = [y for y in extract_years(title) if 2000 <= y <= 2030]
    if title_years:
        return title_years[0]
    body_years = [y for y in extract_years(text[:400]) if 2018 <= y <= 2030]
    return body_years[0] if body_years else None


def extract_academic_year(text: str) -> str | None:
    m = ACADEMIC_YEAR_RE.search(text or "")
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}"


def chinese_ngrams(text: str, n: int = 2) -> str:
    """Build a space-separated bigram string for FTS5."""
    text = normalize_ws(text)
    if not text:
        return ""
    parts: list[str] = []
    buf = []
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            buf.append(ch)
        else:
            if len(buf) >= n:
                parts.extend("".join(buf[i : i + n]) for i in range(len(buf) - n + 1))
            elif buf:
                parts.append("".join(buf))
            buf = []
            if ch.isalnum():
                parts.append(ch.lower())
    if len(buf) >= n:
        parts.extend("".join(buf[i : i + n]) for i in range(len(buf) - n + 1))
    elif buf:
        parts.append("".join(buf))
    # keep original tokens for exact phrases
    for tok in PUNCT_RE.split(text):
        tok = tok.strip()
        if len(tok) >= 2:
            parts.append(tok.lower())
    return " ".join(parts)


PRIORITY_HOST_SCORE = {
    "zs.njupt.edu.cn": 55,
    "xxgk.njupt.edu.cn": 50,
    "xsc.njupt.edu.cn": 50,
    "zsjyc.njupt.edu.cn": 48,
    "yzb.njupt.edu.cn": 42,
    "jwc.njupt.edu.cn": 40,
    "pg.njupt.edu.cn": 38,
    "cwc.njupt.edu.cn": 32,
    "lib.njupt.edu.cn": 10,
}

LOW_VALUE_HOST_PARTS = (
    "youth.",
    "quantum.",
    "tyb.",
    "wicom.",
    "gzzd.",
    "ac.njupt",
    "news.",
    "ladzz.",
)


def priority_score(url: str, title: str = "") -> int:
    blob = f"{url} {title}"
    score = 0
    seen_kw: set[str] = set()
    for i, kw in enumerate(config.PRIORITY_KEYWORDS):
        if kw in blob and kw not in seen_kw:
            seen_kw.add(kw)
            score += 45 - min(i, 20)
    host = host_of(url)
    score += PRIORITY_HOST_SCORE.get(host, 0)
    if any(p in host for p in LOW_VALUE_HOST_PARTS):
        score -= 25
    low = url.lower()
    is_att = low.split("?")[0].endswith((".pdf", ".docx", ".xlsx", ".xls", ".doc")) or "_upload" in low or "/upfiles/" in low
    if is_att:
        score += 35 if host in PRIORITY_HOST_SCORE else 4
    if "list.htm" in low or "list.psp" in low:
        score += 4 if host in PRIORITY_HOST_SCORE else 0
    return score


def looks_like_login(html: str, url: str) -> bool:
    text = html[:4000].lower()
    markers = ("统一身份认证", "cas/login", "请输入密码", "ids.njupt", "sso login")
    return any(m.lower() in text for m in markers)


def parse_publish_date(text: str) -> str | None:
    m = re.search(r"(20\d{2})[年\-\./](\d{1,2})[月\-\./](\d{1,2})", text or "")
    if not m:
        return None
    return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"


def sleep_politely(seconds: float | None = None) -> None:
    time.sleep(seconds if seconds is not None else config.CRAWL_DELAY_SEC)


_CUT = "。？?；，,"


def distractor_spans(question: str) -> list[tuple[int, int]]:
    """“不要回答/不要用/有人算成”盖住的是干扰，不是查询目标。"""
    q = question or ""
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"(?:不要|别|不能)(?:回答|用|报|停在|改用|改成|拿|按)", q):
        back = max((q.rfind(mark, 0, match.start()) for mark in _CUT), default=-1)
        begin = back + 1 if back >= 0 else match.start()
        end = match.end()
        while end < len(q) and q[end] not in "。？?；":
            end += 1
            if end - match.end() > 40:
                break
        spans.append((begin, end))
    for match in re.finditer(r"有人", q):
        end = match.end()
        while end < len(q) and q[end] not in _CUT:
            end += 1
            if end - match.end() > 40:
                break
        spans.append((match.start(), end))
    return spans


def constraint_spans(question: str) -> list[tuple[int, int]]:
    """“不要把/不含/除了”后面的对象要排除，不能反过来当成要查的对象。"""
    q = question or ""
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"(?:不要把|不把|别把|不含|除了|不算)", q):
        end = match.end()
        while end < len(q) and q[end] not in "。？?；":
            end += 1
            if end - match.end() > 28:
                break
        spans.append((match.end(), end))
    return spans


def focus_text(question: str) -> str:
    q = question or ""
    spans = distractor_spans(q) + constraint_spans(q)
    if not spans:
        return q
    chars = list(q)
    for start, end in spans:
        for i in range(max(0, start), min(len(chars), end)):
            chars[i] = " "
    return "".join(chars)


def in_focus(question: str, name: str) -> bool:
    if not name:
        return False
    return re.sub(r"\s+", "", name) in re.sub(r"\s+", "", focus_text(question))


def iter_unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
