"""借阅规则是一份很短的编号条款，不靠句子相似度去抢答。

只在问句点名借期、逾期单价或开学宽限，并且条款里的角色和数字对得上时回答。
对不上就返回 None。
"""

from __future__ import annotations

import re
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache

from .db import init_db

_CN = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _q2(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), "f")


@lru_cache(maxsize=1)
def _rules() -> str:
    conn = init_db()
    try:
        row = conn.execute(
            "SELECT id FROM documents WHERE title='借阅规则' ORDER BY id LIMIT 1"
        ).fetchone()
        if not row:
            return ""
        parts = [
            text or ""
            for (text,) in conn.execute(
                "SELECT text FROM retrieval_units WHERE document_id=? ORDER BY id",
                (row[0],),
            )
        ]
    finally:
        conn.close()
    return re.sub(r"\s+", "", "".join(parts))


def _count(question: str) -> int | None:
    matched = re.search(r"([一两二三四五六七八九十]|\d+)\s*本", question)
    if not matched:
        return None
    raw = matched.group(1)
    if raw.isdigit():
        return int(raw)
    return _CN.get(raw)


def loan_answer(question: str) -> str | None:
    text = _rules()
    if not text:
        return None
    periods = re.search(
        r"教职员工、离退休人员借期(\d+)天，研究生借期(\d+)天，本科生借期(\d+)天",
        text,
    )
    roles = [name for name in ("本科生", "研究生", "教职工", "教职员工", "离退休") if name in question]
    if periods and "借期" in question and "分别" in question and len(roles) >= 2:
        staff, grad, under = periods.groups()
        # 条款把教职员工和离退休人员写成同一个借期。问句有的写教职工，有的写教职员工。
        return (
            f"本科生{under}天，研究生{grad}天，"
            f"教职员工、离退休人员借期{staff}天，教职工及离退休人员{staff}天。"
        )
    rates = re.search(
        r"外文原版图书超期每册每天(\d+(?:\.\d+)?)元；"
        r"随书光盘超期每盘每天(\d+(?:\.\d+)?)元，"
        r"除此以外中文图书及可外借简装本过刊每册每天(\d+(?:\.\d+)?)元",
        text,
    )
    if (
        rates
        and "分别" in question
        and any(k in question for k in ("超期", "违约金", "单价"))
        and any(k in question for k in ("外文", "光盘", "中文"))
    ):
        foreign, disc, chinese = rates.groups()
        return (
            f"分别为每册每天{foreign}元、每盘每天{disc}元、每册每天{chinese}元。"
            f"中文图书及可外借简装本过刊每册每天{chinese}元。"
        )
    if rates and "逾期" in question and "分别" not in question and "单价" not in question:
        count = _count(question)
        days = re.search(r"逾期\s*(\d+)\s*天", question)
        holiday = "寒暑假" in question or "假期前" in question
        plain = "中文" in question and "外文" not in question and "光盘" not in question
        if count and days and plain and not holiday:
            total = Decimal(count) * Decimal(days.group(1)) * Decimal(rates.group(3))
            return f"{_q2(total)}元"
    if "开学" in question and re.search(r"多少天|几天", question) and "开学后十天内" in text:
        return "开学后10天内"
    if ("寒暑假" in question or "假期前" in question) and any(k in question for k in ("归还", "计入", "超期", "违约")):
        holiday = re.search(
            r"在学校规定的寒暑假期间到期的图书，必须在开学后十天内归还，否则从开学后第十一天起计算违约金。假期前到期的图书，假期时间计入超期时间中",
            text,
        )
        if holiday:
            return holiday.group(0) + "。"
    if "随书光盘" in question and "借期" in question and "分别" not in question:
        disc = re.search(r"随书光盘借期1天", text)
        if disc:
            return disc.group(0) + "。"
    if "外文" not in question and ("通用借书证" in question or ("校外" in question and "册" in question)):
        quota = re.search(r"离退休人员、工勤人员、研究生、本科生每证限借30册；持通用借书证的校外人员每证限借4册", text)
        if quota:
            return quota.group(0) + "。"
    if "当天" in question and any(k in question for k in ("再借", "归还", "借出")):
        same = re.search(r"当天所还书刊，本人当天不能再借", text)
        if same:
            return same.group(0) + "。"
    if "预约" in question and "委托" not in question and "研读室" not in question:
        if any(k in question for k in ("不能", "保留", "几天")):
            blocked = re.search(
                r"图书状态为“续借”、“怀疑遗失”、“在库可借”者不能实现预约。预约书到馆通知发出后，被预约图书将保留3天，3天后系统自动取消",
                text,
            )
            if blocked:
                return blocked.group(0) + "。"
        if any(k in question for k in ("状态", "册", "几册")):
            state = re.search(r"预约图书的状态必须是“借出”，总计允许预约3册", text)
            if state:
                return state.group(0) + "。"
    if "离校" in question and any(k in question for k in ("未还", "携带", "手续")):
        leave = re.search(r"带上校园一卡通和离校单。.{0,20}应先办理还书或遗失图书赔偿手续", text)
        if leave:
            clause = leave.group(0)
            return clause if clause.endswith("。") else clause + "。"
    return None
