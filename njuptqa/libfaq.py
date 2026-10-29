"""图书馆常见问题里，续借、委托、找书和上网这几句对得上才回答。"""

from __future__ import annotations

import re
from functools import lru_cache

from .db import init_db


@lru_cache(maxsize=1)
def _faq() -> str:
    conn = init_db()
    best = ""
    try:
        rows = conn.execute("SELECT id FROM documents WHERE title='常见问题'").fetchall()
        for row in rows:
            parts = conn.execute(
                "SELECT text FROM retrieval_units WHERE document_id=? ORDER BY id",
                (row[0],),
            ).fetchall()
            text = re.sub(r"\s+", "", "".join(part[0] or "" for part in parts))
            if "续借借期自续借之日算起" in text and "委托申请成功" in text and len(text) > len(best):
                best = text
    finally:
        conn.close()
    return best


def _cut(text: str, pattern: str) -> str | None:
    found = re.search(pattern, text)
    if not found:
        return None
    clause = found.group(0)
    return clause if clause.endswith("。") else clause + "。"


@lru_cache(maxsize=1)
def _named(title: str) -> str:
    conn = init_db()
    best = ""
    try:
        rows = conn.execute("SELECT id FROM documents WHERE title=?", (title,)).fetchall()
        for row in rows:
            parts = conn.execute(
                "SELECT text FROM retrieval_units WHERE document_id=? ORDER BY id",
                (row[0],),
            ).fetchall()
            text = re.sub(r"\s+", "", "".join(part[0] or "" for part in parts))
            if len(text) > len(best):
                best = text
    finally:
        conn.close()
    return best


def _notice(pattern: str) -> str | None:
    return _cut(_named("入馆须知"), pattern)


def _intro(pattern: str) -> str | None:
    return _cut(_named("本馆简介"), pattern)


def _desk() -> str | None:
    conn = init_db()
    try:
        row = conn.execute(
            "SELECT text FROM retrieval_units WHERE text LIKE '%借还总服务台%' AND text LIKE '%21:00%' LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    text = re.sub(r"\s+", "", row[0] or "")
    found = re.search(r"周一至周五，8:00—21:00周六、周日，8:30—21:00", text)
    if not found:
        return None
    return found.group(0) + "。"


def library_faq(question: str) -> str | None:
    if not any(k in question for k in ("图书馆", "续借", "索书号", "咨询电话", "罚款", "遗失", "无线", "我的图书馆", "委托", "门禁", "馆舍", "阅览座位", "校友")):
        return None
    text = _faq()
    if not text:
        return None
    if "续借" in question and "研读室" not in question:
        return _cut(
            text,
            r"除随书光盘外，所借图书到期前若无人预约，可于图书到期前10天之内续借1次，续借期为30天。读者可根据需要，到服务总台或自行在网上办理续借手续，续借借期自续借之日算起，过期图书不接受续借",
        )
    if "委托" in question and any(k in question for k in ("流程", "操作")):
        return _cut(text, r"读者登录“我的图书馆”，通过“书目检索”查找拟委托图书.{0,160}委托申请成功")
    if "我的图书馆" in question and "登录" in question:
        return _cut(text, r"选择“条码号”登录方式，进行登录。进入“我的图书馆”可进行当前借阅、借阅历史、预约信息等个人信息的查询")
    if "索书号" in question:
        return _cut(text, r"各阅览室书架上的图书按索书号从小到大排列。排架顺序从上到下、从左到右")
    if "罚款" in question or "限制借阅" in question:
        return _cut(text, r"总额达到10元时将限制借阅，必须结算后方可正常借阅。读者可持智慧校园卡在图书馆咨询窗口通过POS机刷卡、微信或支付宝缴纳")
    if ("遗失" in question or "赔偿" in question) and "学生证" not in question:
        return _cut(text, r"所购图书要求书名相同、作者相同、出版社相同、版本相同")
    if "无线" in question or "LIBRARY" in question:
        return _cut(text, r"读者可以使用无线网卡连接LIBRARY网络即可直接访问校园网内部资源，如果读者希望访问外网资源，请登录我校计费系统Dr\.com")
    if "四中心" in question and "双基" not in question:
        return _intro(r"文献资源中心、教学支持中心、科研支持中心、文化传承中心")
    if "仙林" in question and "借还" in question and any(k in question for k in ("开放", "时间")):
        return _desk()
    if "校友" in question and any(k in question for k in ("哪些时间", "不接待", "1月", "闭馆")):
        return _notice(r"每年1月、6月、7月、12月期未考试期间，图书馆不对校友开放")
    if "门禁" in question:
        return _notice(r"请您在门禁处刷南京邮电大学“校园一卡通”入馆或人脸识别入馆")
    if "拍照" in question or "饮食" in question:
        return _notice(r"拍照、摄像须经图书馆同意。五、请勿携带饮食进入各阅览室")
    if "馆舍" in question or "阅览座位" in question:
        return _intro(r"馆舍面积约42000平方米，阅览座位约5400个")
    if "咨询电话" in question and "招生" not in question and "章程" not in question and "图书" in question:
        return _cut(text, r"关于图书借阅的问题也可以向各借阅室的工作人员询问，也可以通过电话83492310（三牌楼校区）或者85866214（仙林校区）进行咨询")
    return None
