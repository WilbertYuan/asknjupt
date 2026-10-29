"""学生处联系方式表和退役士兵材料原句。对不上科室和地址就返回 None。"""

from __future__ import annotations

import re

from .db import init_db


def _rows(office: str, addr: str) -> list[str]:
    conn = init_db()
    try:
        found = conn.execute(
            """
            SELECT text FROM retrieval_units
            WHERE text LIKE '%办公电话=%'
              AND text LIKE ?
              AND text LIKE ?
            """,
            (f"%{office}%", f"%{addr}%"),
        ).fetchall()
    finally:
        conn.close()
    return [re.sub(r"\s+", "", row["text"] or "") for row in found]


def _phone(text: str) -> str | None:
    found = re.search(r"办公电话=(\d{3}-\d{8})", text)
    return found.group(1) if found else None


def contact_answer(question: str) -> str | None:
    if "军队材料" in question or "入伍批准书" in question or "退出现役证" in question:
        conn = init_db()
        try:
            rows = conn.execute(
                """
                SELECT text FROM retrieval_units
                WHERE text LIKE '%入伍批准书%' AND text LIKE '%退出现役证%'
                """
            ).fetchall()
        finally:
            conn.close()
        for row in rows:
            text = re.sub(r"\s+", "", row["text"] or "")
            found = re.search(r"考生报考时应当提交本人《入伍批准书》和《退出现役证》", text)
            if found:
                return found.group(0) + "。"
        return None
    if "就业指导" in question and "三牌楼" in question:
        for text in _rows("就业指导中心", "三牌楼科学会堂105室"):
            if "025-83492759" in text and "科学会堂105室" in text:
                return text
    if "就业指导" in question and "303" in question:
        for text in _rows("就业指导中心", "学生事务中心303"):
            phone = _phone(text)
            if phone == "025-85866867":
                return phone + "。" + text
    if "心理健康" in question and "324" in question:
        for text in _rows("心理健康教育与咨询中心", "学生事务中心324"):
            phone = _phone(text)
            if phone == "025-85866359":
                return phone + "。" + text
    if "宿舍管理" in question and ("三牌楼" in question or "学4" in question):
        for text in _rows("学生宿舍管理办公室", "学4-101"):
            phone = _phone(text)
            if phone == "025-83492367" and "学4-101" in text:
                return phone + "。" + text
    return None
