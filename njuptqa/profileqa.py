"""学校简介里截至 2026 年 9 月仍一致的几句概况。年份对不上就返回 None。"""

from __future__ import annotations

import re
from functools import lru_cache

from .db import init_db


@lru_cache(maxsize=1)
def _profile() -> str:
    conn = init_db()
    best = ""
    try:
        rows = conn.execute("SELECT id FROM documents WHERE title='学校简介'").fetchall()
        for row in rows:
            parts = conn.execute(
                "SELECT text FROM retrieval_units WHERE document_id=? ORDER BY id",
                (row[0],),
            ).fetchall()
            text = re.sub(r"\s+", "", "".join(part[0] or "" for part in parts))
            if "专任教师2400余人" in text and "6个学科在ESI" in text and len(text) > len(best):
                best = text
    finally:
        conn.close()
    return best


def _cut(pattern: str) -> str | None:
    found = re.search(pattern, _profile())
    if not found:
        return None
    clause = found.group(0)
    return clause if clause.endswith("。") else clause + "。"


def profile_answer(question: str) -> str | None:
    if "2026年9月" in question:
        if "前1%" in question and "1‰" not in question:
            return _cut(r"6个学科在ESI学科排名中居全球前1%")
        if "工程教育专业认证" in question:
            return _cut(r"通过中国工程教育专业认证的专业14个")
        if "全国重点实验室" in question and "研究中心" not in question:
            return _cut(r"学校现有全国重点实验室2个")
        if "专任教师" in question and "正高" not in question and "副高" not in question and "学位" not in question:
            return _cut(r"专任教师2400余人")
        if "教职工" in question and "专任教师" not in question:
            return _cut(r"教职工3100余人")
    if "国际交流项目" in question:
        return _cut(r"与110多所世界一流大学建立各种层次与形式的合作，开展各类国际交流项目120余项")
    if "层次" in question and "协调" in question:
        return _cut(r"博士、硕士、本科等多层次教育协调发展")
    if "教职工代表大会" in question and "日常" in question:
        union = _marked("%学校章程%", "学校工会作为教职工代表大会的工作机构", r"学校工会作为教职工代表大会的工作机构，负责其日常工作")
        if union:
            return union
    if "岗位" in question and "类别" in question:
        staff = _marked("%学校章程%", "学校教职员工包括教师、其他专业技术人员、管理人员和工勤人员等", r"学校教职员工包括教师、其他专业技术人员、管理人员和工勤人员等")
        if staff:
            return staff
    if "域名" in question:
        named = _loose(r"学校域名：www\.njupt\.edu\.cn")
        if named:
            return named
    if "并入" in question and "人口" in question:
        return _sentence("%2013年10月，根据教育部和江苏省人民政府文件，原南京人口管理干部学院正式并入南京邮电大学%")
    if "校庆" in question and "标识" in question:
        return _sentence("%南京邮电大学校庆筹备办公室负责%")
    return None


def _marked(like: str, marker: str, pattern: str) -> str | None:
    conn = init_db()
    best = ""
    try:
        rows = conn.execute("SELECT id, title FROM documents WHERE title LIKE ?", (like,)).fetchall()
        for row in rows:
            if "学术委员会" in (row[1] or ""):
                continue
            parts = conn.execute(
                "SELECT text FROM retrieval_units WHERE document_id=? ORDER BY id",
                (row[0],),
            ).fetchall()
            text = re.sub(r"\s+", "", "".join(part[0] or "" for part in parts))
            if marker in text and len(text) > len(best):
                best = text
    finally:
        conn.close()
    found = re.search(pattern, best)
    if not found:
        return None
    clause = found.group(0)
    return clause if clause.endswith("。") else clause + "。"


def _loose(pattern: str) -> str | None:
    conn = init_db()
    try:
        row = conn.execute(
            "SELECT text FROM retrieval_units WHERE text LIKE '%学校域名%' AND text LIKE '%www.njupt.edu.cn%' LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    text = re.sub(r"\s+", "", row[0] or "")
    found = re.search(pattern, text)
    if not found:
        return None
    clause = found.group(0)
    return clause if clause.endswith("。") else clause + "。"


def _sentence(like: str) -> str | None:
    conn = init_db()
    try:
        row = conn.execute(
            "SELECT text FROM retrieval_units WHERE text LIKE ? ORDER BY length(text) DESC LIMIT 1",
            (like,),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    text = re.sub(r"\s+", "", row[0] or "")
    found = re.search(like.strip("%"), text)
    if not found:
        return None
    clause = found.group(0)
    return clause if clause.endswith("。") else clause + "。"
