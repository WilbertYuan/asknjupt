from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests

from . import config
from .util import canonicalize_url, host_of, read_json, utc_now, write_json

HOST_RE = re.compile(r"([a-z0-9][a-z0-9.-]*\.njupt\.edu\.cn|njupt\.edu\.cn)", re.I)

# 身份认证、个人系统、邮件、支付、监控、基础设施：只记录，不采集
SKIP_HOSTS = {
    "cas.njupt.edu.cn",
    "ids.njupt.edu.cn",
    "ids6.njupt.edu.cn",
    "iids.njupt.edu.cn",
    "i.njupt.edu.cn",
    "sso.njupt.edu.cn",
    "authserver.njupt.edu.cn",
    "vpn.njupt.edu.cn",
    "my.webvpn.njupt.edu.cn",
    "client.webvpn.njupt.edu.cn",
    "mail.njupt.edu.cn",
    "imap.njupt.edu.cn",
    "pop.njupt.edu.cn",
    "pop3.njupt.edu.cn",
    "webmail.njupt.edu.cn",
    "my.njupt.edu.cn",
    "my2.njupt.edu.cn",
    "myoa.njupt.edu.cn",
    "oamobile.njupt.edu.cn",
    "ehall.njupt.edu.cn",
    "jwxt.njupt.edu.cn",
    "jwgl.njupt.edu.cn",
    "pay.njupt.edu.cn",
    "facerec.njupt.edu.cn",
    "wechat.njupt.edu.cn",
    "qiyeweixin.njupt.edu.cn",
    "wxcard.njupt.edu.cn",
    "yktapp.njupt.edu.cn",
    "opac.njupt.edu.cn",
    "opac.lib.njupt.edu.cn",
    "forms.njupt.edu.cn",
    "admin.njupt.edu.cn",
    "ntp1.njupt.edu.cn",
    "ntp2.njupt.edu.cn",
    "mirrors.njupt.edu.cn",
    "rzfw.njupt.edu.cn",
    "qingyou-git.njupt.edu.cn",
    "bklqcx.njupt.edu.cn",
    "zscx.njupt.edu.cn",
    "cwcxx.njupt.edu.cn",
    "class.njupt.edu.cn",
    "ip.njupt.edu.cn",
    "p.njupt.edu.cn",
    "bysj.njupt.edu.cn",
    "dmtzk.njupt.edu.cn",
    "shiziyinjin.njupt.edu.cn",
    "tycj.njupt.edu.cn",
    "c.njupt.edu.cn",
}

TOPIC_BY_HOST = {
    "zs.njupt.edu.cn": ["本科招生"],
    "jwc.njupt.edu.cn": ["培养", "学籍"],
    "xxgk.njupt.edu.cn": ["信息公开"],
    "yzb.njupt.edu.cn": ["研究生招生"],
    "lib.njupt.edu.cn": ["图书馆"],
    "xsc.njupt.edu.cn": ["学生事务", "奖助"],
    "zsjyc.njupt.edu.cn": ["就业质量", "就业"],
    "cwc.njupt.edu.cn": ["学费"],
    "hqc.njupt.edu.cn": ["后勤"],
    "xxb.njupt.edu.cn": ["校园网络"],
    "pg.njupt.edu.cn": ["研究生培养"],
    "news.njupt.edu.cn": ["新闻通知"],
    "rsc.njupt.edu.cn": ["教职员工"],
    "exchange.njupt.edu.cn": ["国际交流"],
    "www.njupt.edu.cn": ["学校概况"],
}


def parse_suburl_file(path=None) -> list[str]:
    path = path or (config.ROOT / "njupt-suburl.txt")
    hosts: list[str] = []
    seen: set[str] = set()
    if not path.exists():
        return hosts
    for line in path.read_text(encoding="utf-8").splitlines():
        m = HOST_RE.search(line)
        if not m:
            continue
        host = m.group(1).lower()
        if host not in seen:
            seen.add(host)
            hosts.append(host)
    return hosts


def classify_host(host: str) -> str:
    h = host.lower()
    if h in SKIP_HOSTS or any(h == s or h.endswith("." + s) for s in SKIP_HOSTS):
        return "skip_auth_or_infra"
    if "vpn" in h or h.startswith(("cas.", "ids", "mail.", "imap.", "pop")):
        return "skip_auth_or_infra"
    return "public_candidate"


def _probe_one(host: str) -> dict:
    access = classify_host(host)
    rec = {
        "host": host,
        "class": access,
        "url": None,
        "status": None,
        "error": None,
        "title": None,
        "probed_at": utc_now(),
    }
    if access != "public_candidate":
        rec["error"] = "skipped_by_policy"
        return rec
    http = requests.Session()
    http.headers["User-Agent"] = config.USER_AGENT
    last_err = None
    for scheme in ("https", "http"):
        url = f"{scheme}://{host}/"
        try:
            resp = http.get(url, timeout=8, allow_redirects=True)
            rec["url"] = canonicalize_url(str(resp.url))
            rec["status"] = resp.status_code
            text = ""
            try:
                text = resp.content[:8000].decode(resp.encoding or "utf-8", errors="replace")
            except Exception:
                text = ""
            m = re.search(r"<title>(.*?)</title>", text, re.I | re.S)
            if m:
                rec["title"] = re.sub(r"\s+", " ", m.group(1)).strip()[:80]
            if rec["status"] and rec["status"] < 400:
                final_host = host_of(rec["url"])
                if classify_host(final_host) != "public_candidate":
                    rec["class"] = "skip_auth_or_infra"
                    rec["error"] = f"redirected_to_{final_host}"
                elif any(k in text[:2500] for k in ("统一身份认证", "cas/login", "请输入密码")):
                    rec["class"] = "login_wall"
                    rec["error"] = "login_page"
                else:
                    rec["class"] = "public"
                return rec
            last_err = f"http {resp.status_code}"
        except Exception as e:
            last_err = str(e)
    rec["error"] = last_err
    rec["class"] = "unreachable"
    return rec


def probe_hosts(hosts: list[str] | None = None, workers: int = 10) -> list[dict]:
    hosts = hosts or parse_suburl_file()
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_probe_one, h): h for h in hosts}
        for fut in as_completed(futs):
            rows.append(fut.result())
    rows.sort(key=lambda r: r["host"])
    return rows


def merge_public_seeds(probe_rows: list[dict]) -> int:
    path = config.REG / "seeds.json"
    data = read_json(path, {"note": "", "seeds": []})
    existing = {canonicalize_url(s["url"]) for s in data.get("seeds") or []}
    added = 0
    for row in probe_rows:
        if row.get("class") != "public" or not row.get("url"):
            continue
        url = canonicalize_url(row["url"])
        # prefer site root
        parsed = urlparse(url)
        root = f"{parsed.scheme}://{parsed.netloc}/"
        if root in existing or url in existing:
            continue
        host = parsed.netloc.lower()
        data["seeds"].append(
            {
                "url": root,
                "org": row.get("title") or host,
                "topics": TOPIC_BY_HOST.get(host, ["公开网页"]),
                "audience": "公开",
                "access": "public",
                "discovered_from": "njupt-suburl.txt",
            }
        )
        existing.add(root)
        added += 1
    data["note"] = "种子含任务书优先入口 + njupt-suburl.txt 中探测为公开可访问的站点。不含登录/VPN/个人系统。"
    data["verified_at"] = utc_now()
    write_json(path, data)
    return added


def ingest_subdomains() -> dict:
    config.ensure_dirs()
    hosts = parse_suburl_file()
    rows = probe_hosts(hosts)
    write_json(
        config.REG / "subdomains.json",
        {
            "source": "njupt-suburl.txt",
            "n_unique": len(hosts),
            "probed_at": utc_now(),
            "counts": {
                k: sum(1 for r in rows if r["class"] == k)
                for k in sorted({r["class"] for r in rows})
            },
            "hosts": rows,
        },
    )
    added = merge_public_seeds(rows)
    return {
        "unique_hosts": len(hosts),
        "added_seeds": added,
        "counts": {
            k: sum(1 for r in rows if r["class"] == k)
            for k in sorted({r["class"] for r in rows})
        },
    }
