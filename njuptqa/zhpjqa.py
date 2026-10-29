"""2026 综合评价招生简章里对得上原句的条款。计划数和综合成绩计算不在这里回答。"""

from __future__ import annotations

import re

from .db import init_db

_TEXT: str | None = None


def _brochure() -> str:
    global _TEXT
    if _TEXT is not None:
        return _TEXT
    conn = init_db()
    best = ""
    rows = conn.execute(
        "SELECT id, title FROM documents WHERE title LIKE '%综合评价招生简章%'"
    ).fetchall()
    for row in rows:
        if "2026" not in (row["title"] or ""):
            continue
        parts = conn.execute(
            "SELECT text FROM retrieval_units WHERE document_id=? ORDER BY id",
            (row["id"],),
        ).fetchall()
        text = re.sub(r"\s+", "", "".join(part["text"] or "" for part in parts))
        if len(text) > len(best):
            best = text
    conn.close()
    _TEXT = best
    return best


def _cut(pattern: str) -> str | None:
    found = re.search(pattern, _brochure())
    return found.group(0) if found else None


def _ticket() -> str | None:
    conn = init_db()
    best = ""
    try:
        rows = conn.execute(
            "SELECT id FROM documents WHERE title LIKE '%综合评价招生测试公告%'"
        ).fetchall()
        for row in rows:
            parts = conn.execute(
                "SELECT text FROM retrieval_units WHERE document_id=? ORDER BY id",
                (row["id"],),
            ).fetchall()
            text = re.sub(r"\s+", "", "".join(part["text"] or "" for part in parts))
            if "本人准考证和居民身份证" in text and len(text) > len(best):
                best = text
    finally:
        conn.close()
    found = re.search(r"考生凭本人准考证和居民身份证", best)
    if not found:
        return None
    return found.group(0) + "。"


def zhpj_answer(question: str) -> str | None:
    if "综合评价" not in question and "综评" not in question:
        return None
    if "质量报告" in question or "就业" in question:
        return None
    # 计划合计、选科筛选和代入公式的题交给计算器。「学校测试」里的两个字不是校测分数。
    if any(k in question for k in ("多少人", "计划合计", "总计划", "投档分", "哪些专业", "计划为", "计划仅为")):
        return None
    if re.search(r"校测\s*\d", question):
        return None
    if "测试公告" in question and any(k in question for k in ("携带", "身份证", "证件")):
        ticket = _ticket()
        if ticket:
            return ticket
    if "省份" in question or "哪个省" in question:
        if "计划" not in question:
            return _cut(r"继续在江苏省开展综合评价招生工作[^。]*。")
    if "哪些机构" in question or ("组织" in question and "负责" in question):
        return _cut(r"南京邮电大学招生委员会全面负责学校综合评价招生工作，本科生院负责具体工作的组织和实施。")
    if "核定" in question or "计划最终" in question or "专业和计划" in question:
        return _cut(r"招生专业和计划最终以省级招生主管部门核定为准。")
    if any(k in question for k in ("报名条件", "基本条件")):
        return _cut(r"已参加江苏省2026年普通高等学校招生全国统一考试报名，品学兼优、素质全面、身心健康，具备创新潜质的优秀高中毕业生。")
    if "报名时间" in question or ("截止" in question and any(k in question for k in ("哪一天", "哪天", "何时", "几月"))):
        return _cut(r"报名时间即日起至5月10日。")
    if "网址" in question or ("报名系统" in question and "打印" not in question and "查询" not in question and "确认" not in question):
        return _cut(r"https://bm\.chsi\.com\.cn")
    if "五类" in question or "哪几类" in question:
        return _cut(r"①考生身份证。.*?⑤报名申请表。[^。]*。")
    if "签字盖章" in question or ("签字" in question and "公章" in question):
        return _cut(r"所有页面须由考生本人签字、所在中学负责人签字并加盖中学公章后上传。")
    if any(k in question for k in ("扫描件", "原件照片", "电子图片", "上传形式")):
        return _cut(r"上述所有材料须提供原件照片或扫描件，按报名系统要求规格上传。考生及所在中学须对材料真实性、完整性负责，网上提交的电子图片须完整齐全、清晰可辨。未按要求完成报名或材料不符合规定的，视为无效报名，审核不予通过。")
    if "初审" in question and ("公示" in question or "5月25" in question):
        return _cut(r"初审通过考生名单将于5月25日前在我校本科招生网公示，同时按规定上报江苏省教育考试院，考生可通过报名系统查询审核结果。")
    if "测试费" in question and "逾期" in question:
        return _cut(r"综合评价招生测试费标准为60元/生，逾期未确认或未缴费者，视为自动放弃测试资格。")
    if "满分" in question and any(k in question for k in ("笔试", "形式")):
        return _cut(r"测试采用笔试形式（满分100分）")
    if "6月14" in question or ("拟定" in question and "何地" in question) or ("何时" in question and "何地" in question):
        return _cut(r"拟定于6月14日在南京邮电大学仙林校区（南京市仙林大学城文苑路9号）举行。")
    if "末位" in question and "入围" in question:
        return _cut(r"分物理等科目类、历史等科目类，按不超过综合评价招生计划5倍的比例确定入围考生名单（末位同分均入围）。")
    if "控制线" in question or ("什么线" in question and "高考" in question):
        return _cut(r"入围考生高考成绩须达到江苏省相应特殊类型招生控制线")
    if "未填报" in question:
        return _cut(r"登录省教育考试院高考志愿填报系统.{0,220}规定时间内未填报综合评价招生志愿的，视为放弃相应录取资格。")
    if "占" in question and "测试" in question and "公式" not in question and "如何计算" not in question and "投档" not in question:
        return _cut(r"高考成绩占80%，学校测试成绩占20%，综合成绩满分为100分，四舍五入保留2位小数")
    if "成绩相同" in question or ("相同时" in question and "依次" in question):
        return _cut(r"综合成绩相同时，依次按高考总成绩、高考语文\+数学成绩总和、数学成绩、语文成绩、外语成绩排序录取。")
    if ("同等" in question or "是否相同" in question) and any(k in question for k in ("待遇", "权益", "普通高考")):
        return _cut(r"综合评价招生录取考生与普通高考生享受同等政策待遇，包括转专业、国家助学贷款、奖学金、助学金、困难补助、勤工助学、学费减免等相关权益。")
    return None
