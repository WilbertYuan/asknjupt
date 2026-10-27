from __future__ import annotations

import heapq
import re
from html.parser import HTMLParser
from typing import Iterable
from urllib.parse import quote, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from . import config
from .db import add_quality, init_db, session, upsert_source
from .util import (
    append_jsonl,
    canonicalize_url,
    guess_ext,
    host_of,
    is_njupt,
    looks_like_login,
    parse_publish_date,
    priority_score,
    raw_path_for,
    read_json,
    sha256_bytes,
    should_skip_url,
    sleep_politely,
    utc_now,
    write_bytes,
    write_json,
)

HREF_RE = re.compile(
    r"""(?:href|src|data-url|data-href|pdfsrc)\s*=\s*['"]([^'"]+)['"]""",
    re.I,
)
BARE_FILE_RE = re.compile(
    r"""https?://[^\s"'<>]+?\.(?:pdf|docx?|xlsx?|xls|zip|pptx?)(?:\?[^\s"'<>]*)?""",
    re.I,
)
REL_FILE_RE = re.compile(
    r"""(?:\.\./|\./|/)?_upload/[^'"\s<>]+\.(?:pdf|docx?|xlsx?|xls|zip)""",
    re.I,
)


class _HrefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        ad = {k: v for k, v in attrs}
        for key in ("href", "src", "data-url", "data-href"):
            if ad.get(key):
                self.hrefs.append(ad[key] or "")


def extract_links(html: str, base: str) -> list[str]:
    urls: list[str] = []
    try:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup.find_all(["a", "area", "iframe", "embed", "source"]):
            for attr in ("href", "src", "data-url", "data-href", "pdfsrc"):
                val = tag.get(attr)
                if val:
                    urls.append(val)
        for tag in soup.find_all(onclick=True):
            m = re.findall(r"""['"](https?://[^'"]+|/_upload/[^'"]+)['"]""", tag.get("onclick") or "")
            urls.extend(m)
    except Exception:
        parser = _HrefParser()
        try:
            parser.feed(html)
            urls.extend(parser.hrefs)
        except Exception:
            pass
    urls.extend(HREF_RE.findall(html))
    urls.extend(BARE_FILE_RE.findall(html))
    urls.extend(REL_FILE_RE.findall(html))
    out: list[str] = []
    seen: set[str] = set()
    for u in urls:
        u = u.strip()
        if not u or u.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        cu = canonicalize_url(u, base)
        if cu not in seen:
            seen.add(cu)
            out.append(cu)
    return out


TOPIC_WORDS = (
    "综合评价",
    "招生章程",
    "招生计划",
    "教学质量",
    "学费",
    "培养方案",
    "就业质量",
    "奖学金",
    "助学金",
    "奖助",
    "资助",
    "推免",
    "保研",
    "学生手册",
    "学位",
    "学籍",
    "转专业",
    "毕业",
    "图书馆",
)


def infer_topics(text: str) -> str | None:
    hits = [w for w in TOPIC_WORDS if w in (text or "")]
    return ",".join(dict.fromkeys(hits)) if hits else None


def backfill_attachment_topics(conn) -> int:
    rows = conn.execute(
        """
        SELECT s.id, s.url, s.discovered_from, p.topics AS ptopics, p.notes AS pnotes, d.title
        FROM sources s
        LEFT JOIN sources p ON p.url = s.discovered_from
        LEFT JOIN documents d ON d.url = s.discovered_from
        WHERE (s.topics IS NULL OR s.topics='')
          AND (s.url LIKE '%/_upload/%' OR s.url LIKE '%/upfiles/%')
        """
    ).fetchall()
    n = 0
    for r in rows:
        blob = " ".join(x for x in (r["ptopics"], r["pnotes"], r["title"]) if x)
        topics = infer_topics(blob)
        notes = (r["title"] or r["pnotes"] or "")[:160] or None
        if not topics and not notes:
            continue
        conn.execute(
            "UPDATE sources SET topics=COALESCE(NULLIF(topics,''), ?), notes=COALESCE(NULLIF(notes,''), ?) WHERE id=?",
            (topics, notes, r["id"]),
        )
        n += 1
    return n


def detect_org(url: str, html: str = "") -> str:
    host = host_of(url)
    mapping = {
        "zs.njupt.edu.cn": "本科招生网",
        "jwc.njupt.edu.cn": "教务处/本科生院",
        "xxgk.njupt.edu.cn": "信息公开网",
        "yzb.njupt.edu.cn": "研究生招生网",
        "cwc.njupt.edu.cn": "财务处",
        "sys.njupt.edu.cn": "实验室与设备管理处",
        "www.njupt.edu.cn": "学校官网",
        "lib.njupt.edu.cn": "图书馆",
        "xsc.njupt.edu.cn": "学生工作处",
        "zsjyc.njupt.edu.cn": "招生就业处",
        "hqc.njupt.edu.cn": "后勤管理处",
        "xxb.njupt.edu.cn": "信息化建设与管理办公室",
        "pg.njupt.edu.cn": "研究生院",
        "rsc.njupt.edu.cn": "人事处",
        "news.njupt.edu.cn": "新闻网",
        "cs.njupt.edu.cn": "计算机学院",
        "scie.njupt.edu.cn": "通信与信息工程学院",
        "eoe.njupt.edu.cn": "电子与光学工程学院",
    }
    if host in mapping:
        return mapping[host]
    title = ""
    m = re.search(r"<title>(.*?)</title>", html or "", re.I | re.S)
    if m:
        title = re.sub(r"\s+", "", m.group(1))
    return title[:40] or host


class RobotsCache:
    def __init__(self) -> None:
        self.cache: dict[str, RobotFileParser | None] = {}

    def allowed(self, url: str) -> bool:
        host = host_of(url)
        if host not in self.cache:
            rp = RobotFileParser()
            robots_url = f"{urlparse(url).scheme}://{host}/robots.txt"
            try:
                rp.set_url(robots_url)
                rp.read()
                self.cache[host] = rp
            except Exception:
                self.cache[host] = None
        rp = self.cache[host]
        if rp is None:
            return True
        try:
            return rp.can_fetch(config.USER_AGENT, url)
        except Exception:
            return True


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": config.USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
    )
    return s


def _abort_skip_host(resp, **_kwargs):
    loc = resp.headers.get("Location")
    if not loc:
        return
    nxt = canonicalize_url(loc, str(resp.url))
    if should_skip_url(nxt):
        raise requests.exceptions.RequestException("redirect_or_host_skipped")


def fetch(session_http: requests.Session, url: str) -> dict:
    last_err = None
    max_bytes = 12 * 1024 * 1024
    for attempt in range(1, config.CRAWL_RETRIES + 1):
        try:
            ext_hint = guess_ext(url, "")
            max_bytes = 40 * 1024 * 1024 if ext_hint == ".pdf" else 12 * 1024 * 1024
            resp = session_http.get(
                url,
                timeout=config.CRAWL_TIMEOUT,
                allow_redirects=True,
                stream=True,
                hooks={"response": _abort_skip_host},
            )
            final = canonicalize_url(str(resp.url))
            if should_skip_url(final):
                resp.close()
                return {
                    "ok": False,
                    "status": resp.status_code,
                    "final_url": final,
                    "content": b"",
                    "content_type": "",
                    "text": "",
                    "attempt": attempt,
                    "is_login_page": False,
                    "is_empty": True,
                    "error": "redirect_or_host_skipped",
                }
            chunks: list[bytes] = []
            size = 0
            for chunk in resp.iter_content(65536):
                if not chunk:
                    continue
                size += len(chunk)
                if size > max_bytes:
                    resp.close()
                    return {
                        "ok": False,
                        "status": resp.status_code,
                        "final_url": final,
                        "content": b"",
                        "content_type": resp.headers.get("Content-Type", ""),
                        "text": "",
                        "attempt": attempt,
                        "is_login_page": False,
                        "is_empty": True,
                        "error": "too_large",
                    }
                chunks.append(chunk)
            resp.close()
            data = b"".join(chunks)
            ctype = resp.headers.get("Content-Type", "")
            text = ""
            if "html" in ctype.lower() or guess_ext(url, ctype) in {".html", ".htm"}:
                enc = resp.encoding or "utf-8"
                try:
                    text = data.decode(enc, errors="replace")
                except Exception:
                    text = data.decode("utf-8", errors="replace")
            login = bool(text and looks_like_login(text, url))
            empty = len(data) < 80 or (text and len(re.sub(r"\s+", "", text)) < 40)
            return {
                "ok": True,
                "status": resp.status_code,
                "final_url": final,
                "content": data,
                "content_type": ctype,
                "text": text,
                "attempt": attempt,
                "is_login_page": login,
                "is_empty": empty,
                "error": None if resp.status_code < 400 else f"http {resp.status_code}",
            }
        except Exception as e:
            last_err = str(e)
            sleep_politely(1.2 * attempt)
    return {
        "ok": False,
        "status": 0,
        "final_url": url,
        "content": b"",
        "content_type": "",
        "text": "",
        "attempt": config.CRAWL_RETRIES,
        "is_login_page": False,
        "is_empty": True,
        "error": last_err,
    }


def load_seeds() -> list[dict]:
    path = config.REG / "seeds.json"
    data = read_json(path, {"seeds": []})
    return list(data.get("seeds") or [])


def discover(max_extra: int = 200) -> dict:
    """Register seeds and expand one hop from official portals."""
    config.ensure_dirs()
    host_stats = None
    try:
        from .hosts import ingest_subdomains

        if (config.ROOT / "njupt-suburl.txt").exists() and not (config.REG / "subdomains.json").exists():
            host_stats = ingest_subdomains()
    except Exception:
        host_stats = None
    seeds = load_seeds()
    http = make_session()
    robots = RobotsCache()
    found = 0
    blocked = []
    with session() as conn:
        for seed in seeds:
            url = canonicalize_url(seed["url"])
            upsert_source(
                conn,
                {
                    **seed,
                    "url": url,
                    "topics": ",".join(seed.get("topics") or []),
                    "status": "seed",
                    "discovered_from": "seeds.json",
                },
            )
            found += 1
        # one-hop expansion from working html seeds
        extra = 0
        for seed in seeds:
            url = canonicalize_url(seed["url"])
            if seed.get("access") == "public_blocked_waf":
                blocked.append({"url": url, "reason": "marked_blocked"})
                continue
            if should_skip_url(url) or not robots.allowed(url):
                continue
            rec = fetch(http, url)
            sleep_politely()
            if not rec["ok"] or rec["status"] >= 400:
                blocked.append({"url": url, "reason": rec.get("error") or rec["status"]})
                conn.execute(
                    "UPDATE sources SET status=?, notes=?, updated_at=? WHERE url=?",
                    ("blocked", str(rec.get("error")), utc_now(), url),
                )
                continue
            if rec["text"]:
                for link in extract_links(rec["text"], rec["final_url"]):
                    if extra >= max_extra:
                        break
                    if not is_njupt(link) or should_skip_url(link):
                        continue
                    upsert_source(
                        conn,
                        {
                            "url": link,
                            "org": detect_org(link),
                            "topics": ",".join(seed.get("topics") or []),
                            "audience": seed.get("audience"),
                            "access": "public",
                            "status": "discovered",
                            "discovered_from": url,
                        },
                    )
                    extra += 1
                    found += 1
    write_json(
        config.REG / "source_registry.json",
        {"count": found, "blocked": blocked, "updated_at": utc_now(), "hosts": host_stats},
    )
    return {"registered": found, "blocked": blocked, "hosts": host_stats}


def _queue_from_db(conn, extra_urls: Iterable[str] | None = None) -> list[tuple[int, str]]:
    host_n: dict[str, int] = {}
    for r in conn.execute("SELECT url FROM crawl_manifest WHERE http_status BETWEEN 200 AND 399"):
        h = host_of(r["url"])
        host_n[h] = host_n.get(h, 0) + 1
    heap: list[tuple[int, str]] = []
    rows = conn.execute("SELECT url, org, topics FROM sources").fetchall()
    seen = set()
    for r in rows:
        url = r["url"]
        if url in seen or should_skip_url(url):
            continue
        seen.add(url)
        host = host_of(url)
        crawled = host_n.get(host, 0)
        from .util import PRIORITY_HOST_SCORE, LOW_VALUE_HOST_PARTS

        if any(p in host for p in LOW_VALUE_HOST_PARTS):
            continue
        diversity = 0
        if host in PRIORITY_HOST_SCORE and crawled < 40:
            diversity = 12
        heapq.heappush(heap, (-(priority_score(url, f"{r['org']} {r['topics']}") + diversity), url))
    for url in extra_urls or []:
        url = canonicalize_url(url)
        if url not in seen and is_njupt(url) and not should_skip_url(url):
            heapq.heappush(heap, (-(priority_score(url) + 40), url))
    return heap


def priority_pending_urls(conn, limit: int = 900) -> list[str]:
    from .util import PRIORITY_HOST_SCORE

    already = {
        r["url"]
        for r in conn.execute(
            "SELECT DISTINCT url FROM crawl_manifest WHERE http_status BETWEEN 200 AND 399 AND is_empty=0"
        )
    }
    tried = {
        r["url"]
        for r in conn.execute(
            """SELECT DISTINCT url FROM crawl_manifest
               WHERE http_status IN (404, 410)
                  OR IFNULL(error,'') IN ('too_large','redirect_or_host_skipped','robots_disallow')"""
        )
    }
    ranked: list[tuple[int, str]] = []
    for r in conn.execute("SELECT url, org, topics FROM sources"):
        url = r["url"]
        if url in already or url in tried or should_skip_url(url):
            continue
        host = host_of(url)
        theme = f"{r['org'] or ''} {r['topics'] or ''} {r['url']}"
        high_theme = any(k in theme for k in ("培养方案", "就业质量", "奖学金", "助学金", "推免", "学位", "教学质量"))
        if host not in PRIORITY_HOST_SCORE and not high_theme:
            continue
        if host.startswith("lib."):
            continue
        low = url.lower()
        if "/_upload/article/images/" in low:
            continue
        if not (
            "/_upload/article/files/" in low
            or "/upfiles/" in low
            or low.split("?")[0].endswith((".pdf", ".docx", ".xlsx", ".xls", ".doc"))
            or "list.htm" in low
        ):
            continue
        ranked.append((priority_score(url, f"{r['org']} {r['topics']}"), url))
    ranked.sort(reverse=True)
    return [u for _, u in ranked[:limit]]


def ingest_local(directory=None) -> dict:
    """把用户补充的本地公开文件纳入同一套来源链。不把文件名里的数字当成答案。"""
    root = directory or (config.ROOT / "其他资料")
    root = root if hasattr(root, "rglob") else config.ROOT / str(root)
    if not root.exists():
        return {"registered": 0, "error": f"目录不存在: {root}"}
    conn = init_db()
    n = 0
    items = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith(".")):
        data = path.read_bytes()
        if not data:
            continue
        digest = sha256_bytes(data)
        ext = path.suffix.lower() or ".bin"
        raw = raw_path_for(digest, ext)
        if not raw.exists():
            write_bytes(raw, data)
        rel = path.relative_to(config.ROOT).as_posix()
        url = "local://" + quote(rel)
        topics = infer_topics(path.stem) or "用户补充"
        if "学生手册" in path.stem:
            topics = "学籍,奖助,学位,推免,图书馆,学生事务"
        elif "保研" in path.stem or "免试攻读" in path.stem:
            topics = "推免,学位"
        upsert_source(
            conn,
            {
                "url": url,
                "org": "用户补充的公开资料",
                "topics": topics,
                "access": "local_supplement",
                "status": "local",
                "notes": path.stem,
                "discovered_from": str(path),
            },
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO crawl_manifest(
                url, final_url, fetched_at, http_status, content_type, content_hash,
                byte_size, ext, raw_path, parent_url, attempt, error, is_login_page, is_empty)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                url,
                url,
                utc_now(),
                200,
                "application/octet-stream",
                digest,
                len(data),
                ext,
                str(raw),
                str(path),
                1,
                None,
                0,
                0,
            ),
        )
        n += 1
        items.append({"url": url, "bytes": len(data), "name": path.name})
    conn.commit()
    conn.close()
    return {"registered": n, "items": items}


def ingest_job_quality_reports() -> dict:
    """抓取智慧就业平台上南邮公开栏目「就业质量年度报告」中的原文 PDF。

    栏目来自学校招生就业页指向的 njupt.91job.org.cn（学校代码 10293）。
    不收录招聘岗位，也不把接口摘要当成事实。
    """
    http = make_session()
    wz = http.get(
        "https://njupt.91job.org.cn/web/wsjysc/wzsy/getWzid",
        params={"xxdm": "10293", "wzlx": "0"},
        timeout=config.CRAWL_TIMEOUT,
    )
    wz.raise_for_status()
    wzid = (wz.json().get("result") or "").strip()
    if not wzid:
        return {"stored": 0, "error": "未取得就业网站点编号"}
    cols = http.get(
        "https://njupt.91job.org.cn/web/wsjysc/wzsy/getXwlm",
        params={"wzid": wzid},
        timeout=config.CRAWL_TIMEOUT,
    )
    cols.raise_for_status()

    def walk(nodes):
        for node in nodes or []:
            yield node
            yield from walk(node.get("model"))

    column_id = next(
        (n.get("lmid") for n in walk(cols.json().get("result")) if n.get("lmmc") == "就业质量年度报告"),
        None,
    )
    if not column_id:
        return {"stored": 0, "error": "公开栏目中未找到就业质量年度报告"}

    page = 1
    notices = []
    while page <= 8:
        resp = http.post(
            "https://njupt.91job.org.cn/web/wsjysc/lbxq/getTzggPageList",
            json={"current": page, "size": 50, "lmid": column_id},
            timeout=config.CRAWL_TIMEOUT,
        )
        resp.raise_for_status()
        result = resp.json().get("result") or {}
        notices.extend(result.get("records") or [])
        total = int(result.get("total") or 0)
        if page * 50 >= total:
            break
        page += 1

    conn = init_db()
    stored = []
    for notice in notices:
        xwid = notice.get("xwid")
        title = (notice.get("xwbt") or "").strip()
        if not xwid or "就业质量" not in title:
            continue
        detail_url = f"https://njupt.91job.org.cn/sub-station/detail?xwid={xwid}&xxdm=10293"
        detail = http.get(
            f"https://njupt.91job.org.cn/web/wsjysc/lbxq/getTzggxq/{xwid}",
            timeout=config.CRAWL_TIMEOUT,
        )
        detail.raise_for_status()
        html = ((detail.json().get("result") or {}).get("xwnr")) or ""
        pdfs = re.findall(r'href="(https?://[^"]+\.pdf[^"]*)"', html, flags=re.I)
        if not pdfs:
            add_quality(conn, "employment_report_missing_file", f"{title} {detail_url}")
            continue
        for pdf_url in pdfs:
            pdf_url = pdf_url.replace("&amp;", "&")
            file_resp = http.get(pdf_url, timeout=120)
            if file_resp.status_code != 200 or not file_resp.content.startswith(b"%PDF"):
                add_quality(conn, "employment_report_download_failed", f"{title} status={file_resp.status_code}")
                continue
            data = file_resp.content
            digest = sha256_bytes(data)
            raw = raw_path_for(digest, ".pdf")
            if not raw.exists():
                write_bytes(raw, data)
            upsert_source(
                conn,
                {
                    "url": pdf_url,
                    "org": "南京邮电大学就业",
                    "topics": "就业质量",
                    "access": "public",
                    "status": "fetched",
                    "notes": title,
                    "discovered_from": detail_url,
                    "year": int(m.group(1)) if (m := re.search(r"(20\d{2})", title)) else None,
                },
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO crawl_manifest(
                    url, final_url, fetched_at, http_status, content_type, content_hash,
                    byte_size, ext, raw_path, parent_url, attempt, error, is_login_page, is_empty)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    pdf_url,
                    pdf_url.split("?")[0],
                    utc_now(),
                    200,
                    "application/pdf",
                    digest,
                    len(data),
                    ".pdf",
                    str(raw),
                    detail_url,
                    1,
                    None,
                    0,
                    0,
                ),
            )
            stored.append({"title": title, "bytes": len(data), "hash": digest[:12]})
            sleep_politely()
    conn.commit()
    conn.close()
    return {"column_id": column_id, "notices": len(notices), "stored": stored}


# 学院站点「本科教育」栏目。只跟已核实的栏目页，不猜子域名。
COLLEGE_PROGRAM_LISTS = (
    "https://scie.njupt.edu.cn/2801/list.htm",
    "https://eoe.njupt.edu.cn/bksjy/list.htm",
    "https://eoe.njupt.edu.cn/9487/list.htm",
    "https://eoe.njupt.edu.cn/9531/list.htm",
    "https://bhs.njupt.edu.cn/4785/list.htm",
    "https://iam.njupt.edu.cn/10908/list.htm",
    "https://cm.njupt.edu.cn/zyjs/list.htm",
    "https://simp.njupt.edu.cn/7744/list.htm",
)


def ingest_public_program_pdfs() -> dict:
    """把学院公开栏目文章里嵌的培养方案 PDF 按原文入库。

    栏目页本身往往只有专业介绍标题，学分在 pdfsrc 指向的文件里。
    """
    http = make_session()
    conn = init_db()
    stored = []
    seen_pages: set[str] = set()
    for list_url in COLLEGE_PROGRAM_LISTS:
        try:
            listing = http.get(list_url, timeout=config.CRAWL_TIMEOUT)
        except requests.RequestException as exc:
            add_quality(conn, "program_list_failed", f"{list_url} {exc}")
            continue
        if listing.status_code != 200:
            add_quality(conn, "program_list_failed", f"{list_url} status={listing.status_code}")
            continue
        listing.encoding = listing.apparent_encoding or "utf-8"
        pages = re.findall(r"""href=['"]([^'"]+/page\.htm)['"]""", listing.text, flags=re.I)
        # 有的栏目把整份方案直接嵌在列表页，没有再链到文章。
        if re.search(r"""pdfsrc=['"][^'"]+\.pdf['"]""", listing.text, flags=re.I):
            pages.insert(0, list_url)
        for href in dict.fromkeys(pages):
            page_url = canonicalize_url(urljoin(list_url, href))
            if page_url in seen_pages:
                continue
            seen_pages.add(page_url)
            try:
                page = http.get(page_url, timeout=config.CRAWL_TIMEOUT)
            except requests.RequestException as exc:
                add_quality(conn, "program_page_failed", f"{page_url} {exc}")
                continue
            if page.status_code != 200:
                continue
            page.encoding = page.apparent_encoding or "utf-8"
            title_m = re.search(r"<title>([^<]{4,80})</title>", page.text, flags=re.I)
            page_title = re.sub(r"\s+", "", title_m.group(1)) if title_m else page_url
            page_title = re.sub(r"[-_|].*$", "", page_title).strip() or page_url
            pdfs = re.findall(r"""pdfsrc=['"]([^'"]+\.pdf)['"]""", page.text, flags=re.I)
            if not pdfs:
                add_quality(conn, "program_page_missing_pdf", f"{page_title} {page_url}")
                continue
            for href_pdf in dict.fromkeys(pdfs):
                pdf_url = canonicalize_url(urljoin(page_url, href_pdf))
                try:
                    file_resp = http.get(pdf_url, timeout=120)
                except requests.RequestException as exc:
                    add_quality(conn, "program_pdf_download_failed", f"{page_title} {exc}")
                    continue
                if file_resp.status_code != 200 or not file_resp.content.startswith(b"%PDF"):
                    add_quality(conn, "program_pdf_download_failed", f"{page_title} status={file_resp.status_code}")
                    continue
                data = file_resp.content
                digest = sha256_bytes(data)
                raw = raw_path_for(digest, ".pdf")
                if not raw.exists():
                    write_bytes(raw, data)
                note = page_title if "培养方案" in page_title else f"{page_title}培养方案"
                upsert_source(
                    conn,
                    {
                        "url": pdf_url,
                        "org": host_of(pdf_url),
                        "topics": "培养方案",
                        "access": "public",
                        "status": "fetched",
                        "notes": note[:80],
                        "discovered_from": page_url,
                    },
                )
                conn.execute(
                    """
                    INSERT OR IGNORE INTO crawl_manifest(
                        url, final_url, fetched_at, http_status, content_type, content_hash,
                        byte_size, ext, raw_path, parent_url, attempt, error, is_login_page, is_empty)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        pdf_url,
                        pdf_url.split("?")[0],
                        utc_now(),
                        200,
                        "application/pdf",
                        digest,
                        len(data),
                        ".pdf",
                        str(raw),
                        page_url,
                        1,
                        None,
                        0,
                        0,
                    ),
                )
                stored.append({"title": note, "bytes": len(data), "hash": digest[:12], "url": pdf_url})
                sleep_politely()
    conn.commit()
    conn.close()
    return {"pages": len(seen_pages), "stored": stored}


def crawl(max_pages: int | None = None, start_urls: list[str] | None = None, attachment_only: bool = False) -> dict:
    config.ensure_dirs()
    limit = max_pages or config.CRAWL_MAX_PAGES
    http = make_session()
    robots = RobotsCache()
    stats = {"fetched": 0, "stored": 0, "failed": 0, "skipped": 0, "login": 0}

    conn = init_db()
    backfill_attachment_topics(conn)
    conn.commit()
    if start_urls is None:
        start_urls = priority_pending_urls(conn, limit=max(limit, 400))
        heap = _queue_from_db(conn, start_urls)
    else:
        heap = []
        for url in start_urls:
            url = canonicalize_url(url)
            if is_njupt(url) and not should_skip_url(url):
                heapq.heappush(heap, (-(priority_score(url, "就业质量培养方案") + 80), url))
    seen_url: set[str] = set()
    already = {
        r["url"]
        for r in conn.execute(
            "SELECT DISTINCT url FROM crawl_manifest WHERE http_status BETWEEN 200 AND 399 AND is_empty=0"
        )
    }

    while heap and stats["fetched"] < limit:
        _, url = heapq.heappop(heap)
        if url in seen_url:
            continue
        seen_url.add(url)
        if url in already:
            stats["skipped"] += 1
            continue
        if not is_njupt(url) or should_skip_url(url):
            stats["skipped"] += 1
            continue
        if not robots.allowed(url):
            conn.execute(
                """INSERT INTO crawl_manifest(url, final_url, fetched_at, http_status, error, attempt)
                   VALUES(?,?,?,?,?,?)""",
                (url, url, utc_now(), 0, "robots_disallow", 0),
            )
            stats["skipped"] += 1
            continue

        rec = fetch(http, url)
        if rec.get("error") in {"redirect_or_host_skipped", "too_large"}:
            stats["skipped"] += 1
        stats["fetched"] += 1
        content = rec["content"]
        digest = sha256_bytes(content) if content else ""
        ext = guess_ext(url, rec["content_type"])
        path = raw_path_for(digest, ext) if digest else None
        if content and path:
            write_bytes(path, content)
            stats["stored"] += 1
        if rec["is_login_page"]:
            stats["login"] += 1
            add_quality(conn, "login_page", f"疑似登录页 {url}")
        if not rec["ok"] or rec["status"] >= 400 or rec["is_empty"]:
            stats["failed"] += 1

        conn.execute(
            """
            INSERT OR IGNORE INTO crawl_manifest(
                url, final_url, fetched_at, http_status, content_type, content_hash,
                byte_size, ext, raw_path, parent_url, attempt, error, is_login_page, is_empty)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                url,
                rec["final_url"],
                utc_now(),
                rec["status"],
                rec["content_type"],
                digest or None,
                len(content),
                ext,
                str(path) if path else None,
                None,
                rec["attempt"],
                rec["error"],
                1 if rec["is_login_page"] else 0,
                1 if rec["is_empty"] else 0,
            ),
        )
        upsert_source(
            conn,
            {
                "url": url,
                "org": detect_org(url, rec.get("text") or ""),
                "access": "public" if rec["ok"] and rec["status"] < 400 else "blocked",
                "status": "fetched" if rec["ok"] and rec["status"] < 400 else "failed",
            },
        )
        append_jsonl(
            config.LOGS / "crawl_manifest.jsonl",
            {
                "url": url,
                "status": rec["status"],
                "hash": digest,
                "error": rec["error"],
                "at": utc_now(),
            },
        )

        # enqueue children
        if rec["text"] and rec["status"] < 400 and not rec["is_login_page"]:
            parent_title = ""
            m = re.search(r"<title>(.*?)</title>", rec["text"], re.I | re.S)
            if m:
                parent_title = m.group(1)
            for link in extract_links(rec["text"], rec["final_url"]):
                if not is_njupt(link) or should_skip_url(link) or link in seen_url:
                    continue
                if attachment_only:
                    low = link.lower().split("?")[0]
                    if not (
                        low.endswith((".pdf", ".doc", ".docx", ".xls", ".xlsx"))
                        or "/_upload/article/files/" in low
                    ):
                        continue
                topics = infer_topics(parent_title)
                upsert_source(
                    conn,
                    {
                        "url": link,
                        "org": detect_org(link),
                        "status": "discovered",
                        "discovered_from": url,
                        "access": "public",
                        "topics": topics,
                        "notes": (parent_title or "")[:160] or None,
                    },
                )
                heapq.heappush(heap, (-priority_score(link, parent_title), link))

        if stats["fetched"] % 15 == 0:
            conn.commit()
        sleep_politely()

    conn.commit()
    conn.close()
    write_json(config.LOGS / "crawl_stats.json", {**stats, "at": utc_now()})
    return stats
