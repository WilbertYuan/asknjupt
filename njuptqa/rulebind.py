"""研读室、转专业、宿舍、评优、补考和综评报考指南里对得上的原句。"""

from __future__ import annotations

import re

from .db import init_db

_CACHE: dict[str, str] = {}


def _doc(like: str, skip: str = "") -> str:
    if like in _CACHE:
        return _CACHE[like]
    conn = init_db()
    best = ""
    for row in conn.execute("SELECT id, title FROM documents WHERE title LIKE ?", (like,)):
        title = row["title"] or ""
        if skip and skip in title:
            continue
        parts = conn.execute(
            "SELECT text FROM retrieval_units WHERE document_id=? ORDER BY id",
            (row["id"],),
        ).fetchall()
        text = re.sub(r"\s+", "", "".join(part["text"] or "" for part in parts))
        if len(text) > len(best):
            best = text
    conn.close()
    _CACHE[like] = best
    return best


def _cut(text: str, pattern: str) -> str | None:
    found = re.search(pattern, text)
    if not found:
        return None
    clause = found.group(0)
    return clause if clause.endswith("。") else clause + "。"


def rule_answer(question: str) -> str | None:
    if "研读室" in question:
        text = _doc("%研读室%")
        if "407" in question:
            return _cut(text, r"407房间供全校老师、研究生召开学术会议使用，限6-20人")
        if any(k in question for k in ("时限", "最长", "几小时", "多少小时")):
            return _cut(text, r"研究生：最长不超过4小时；教师：最长不超过14小时")
        if "违约" in question and "线下" not in question:
            return _cut(text, r"违约责任系统记录在组长身上，违约累计达三次者，系统将暂停其随后30天内的预约权限")
        if "线下" in question:
            return None
        if any(k in question for k in ("多少天", "几天", "2天", "两天")):
            booked = _cut(text, r"可预约2天之内（含当天）开放时间中的任意时段")
            if "刷卡" in question:
                enter = _cut(text, r"所有组员均需在预约时间开始后20分钟内刷卡进入，如未按规定刷卡使用，视为违约一次。如20分钟内无一人刷卡，则自动取消该预约，并视为违约一次")
                if booked and enter:
                    return booked + enter
            return booked
    if "转专业" in question and "学费" not in question:
        text = _doc("%转专业管理办法%")
        if "艺术类" in question and "中外合作" in question:
            art = _cut(text, r"艺术类专业录取的学生仅可在艺术类专业之间申请转专业。")
            joint = _cut(text, r"中外合作办学专业录取的学生仅可在中外合作办学专业之间申请转专业")
            if art and joint:
                return art + joint
        if "申请表" in question or "书面材料" in question or "书面申请" in question:
            return _cut(text, r"填写《南京邮电大学学生转专业申请表》并附家长（或监护人）同意的书面意见")
        if any(k in question for k in ("年级", "一年级", "二年级")) and any(k in question for k in ("课程", "未通过", "通过情况")):
            grade = _cut(text, r"本科一年级或二年级在籍")
            clear = _cut(text, r"无未通过计划内课程")
            if grade and clear:
                return grade + "。" + clear + "。"
    if "宿舍" in question or ("调整" in question and "床位" in question):
        text = _doc("%学生宿舍管理规定%")
        if "报修" in question:
            form = _cut(text, r"学生在宿舍管理员处填写报修单")
            bill = _cut(text, r"如属于自然损坏，免费修理；如属于人为损坏，按价收取维修费用和材料费")
            if form and bill:
                return form + bill
            return form
        if any(k in question for k in ("开门", "关门", "晚归")):
            return _cut(text, r"学生宿舍6：00开门，23：30关门。学生宿舍关门后晚归者，应持有效证件登记并说明原因，得到管理人员许可后方可进入宿舍")
        if "调整" in question and any(k in question for k in ("申请表", "床位", "房间")):
            return _cut(text, r"确需调整者，应填写《学生调整宿舍申请表》经学院审核同意后，报学管办备案")
    if "排名" in question and "比例" not in question and "名额" not in question:
        text = _doc("%优秀学生评选办法%")
        if "三好学生标兵" in question and "优秀学生干部" not in question:
            return _cut(text, r"综合素质测评成绩排名列本专业前4%")
        if "优秀学生干部" in question and "三好学生" in question:
            cadre = _cut(text, r"综合素质测评成绩排名列本专业前35%")
            good = _cut(text, r"综合素质测评成绩排名列本专业前20%")
            if cadre and good:
                return "优秀学生干部" + cadre + "三好学生" + good
    if "补考" in question and any(k in question for k in ("报名", "收费")):
        text = _doc("%缓考、补考和重修%")
        return _cut(text, r"补考由学校统一安排，学生无须报名")
    if "综合素质测评" in question and "每学年" in question and any(k in question for k in ("几次", "何时", "什么时候")):
        text = _doc("%综合素质测评办法%")
        return _cut(text, r"每学年开展一次，一般在每年九月初对上一学年学生在校期间的德育、智育等各项素质进行综合测评")
    if "思想品行" in question:
        text = _doc("%综合评价招生报考指南%")
        return _cut(text, r"教育行政部门颁发的“三好学生”“优秀学生干部”称号；共青团组织颁发的“优秀共青团员”“最美中学生”称号")
    if "往届生" in question and ("综合评价" in question or "综评" in question):
        text = _doc("%综合评价招生报考指南%")
        return _cut(text, r"可以。以学业优秀类报名的往届生，我校认可其原高中阶段的学业成绩，报名申请表及中学公示证明材料等须由原中学负责人签字并加盖中学公章。")
    if "综合评价" in question and any(k in question for k in ("有误", "已确认")) and "志愿" in question:
        text = _doc("%综合评价招生报考指南%")
        return _cut(text, r"可以先取消当前志愿，再重新填报。需要注意的是，已取消志愿对应的申请表将自动失效，考生须重新下载申请表、并请中学签字盖章后上传。")
    return None
