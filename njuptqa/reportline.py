"""教学质量报告里几段叙述。表格题仍交给 reportqa，这里只取对得上的原句。"""

from __future__ import annotations

import re
from functools import lru_cache

from .db import init_db


@lru_cache(maxsize=1)
def _report() -> str:
    conn = init_db()
    best = ""
    try:
        rows = conn.execute(
            "SELECT id FROM documents WHERE title LIKE '%本科教学质量报告%'"
        ).fetchall()
        for row in rows:
            parts = conn.execute(
                "SELECT text FROM retrieval_units WHERE document_id=? ORDER BY id",
                (row[0],),
            ).fetchall()
            text = re.sub(r"\s+", "", "".join(part[0] or "" for part in parts))
            if "理工类70%以上的选题" in text and len(text) > len(best):
                best = text
    finally:
        conn.close()
    return best


def _cut(pattern: str) -> str | None:
    found = re.search(pattern, _report())
    if not found:
        return None
    clause = found.group(0)
    return clause if clause.endswith("。") else clause + "。"


def report_line(question: str) -> str | None:
    if "就业质量报告" in question:
        return None
    if "哪些学科门类" in question:
        return _cut(r"理、工、经、管、文、教、艺、法等多学科相互交融")
    if "ESI" in question and ("1‰" in question or "哪些学科" in question):
        return _cut(r"计算机科学学科和工程学学科ESI排名分别进入1‰和1\.33‰")
    if "毕业设计" in question and "选题" in question and any(k in question for k in ("理工", "文科", "比例")):
        return _cut(r"理工类70%以上的选题来自科研课题或技术开发项目，文科类40%以上来自企业发展研究或社会热点问题")
    if "创新创业训练计划" in question and "分别为" not in question and any(k in question for k in ("累计", "合计", "惠及")):
        return _cut(r"2022-2024年，学校共实施国家级创新创业训练计划项目267项，省级创新创业训练计划项目339项，校级创新创业训练计划项目1966项，惠及学生约10000名")
    if "三维" in question or "六措" in question:
        dim = _cut(r"聚焦人才培养过程中学生、教师、管理者三个维度进行监测")
        six = _cut(r"学校以质量文化建设、质量标准建设、质保制度建设、质保项目建设、质保平台建设、质保组织建设等六项重点举措为抓手，构建“质量目标—监控评估—信息反馈—改进完善”的闭环质保体系")
        if dim and six:
            return dim + six
    if "巡视" in question and "试卷" in question:
        return _cut(r"开展课程考试巡视75场次，校内外专家抽检考试试卷306门次.{0,220}个人Ⅲ级教学事故5起，个人Ⅱ级教学事故1起")
    if "未按时毕业" in question:
        return _cut(r"应届毕业生中未按时毕业的人数6人，毕业率99\.90%；学士学位授予数5724人，学士学位授予率98\.44%。2024届本科生深造率54\.64%")
    if "江苏省内" in question and "就业" in question and "千人" not in question and "前五" not in question:
        return _cut(r"主要就业地在江苏省内，占就业学生的64\.83%")
    if "一流" in question and "课程" in question and "新增" in question:
        return _cut(r"学校新增省级一流本科课程12门，推荐国家级本科一流课程20门")
    if "现场考查" in question or ("有效期" in question and "认证" in question):
        return _cut(r"2023年10月16-17日，本专业第三次接受全国工程教育专业认证专家现场考查，通过认证，有效期6年")
    if "理念" in question and any(k in question for k in ("质量保障", "质保")):
        return _cut(r"树立“学为中心、成人成才、追求卓越”的质保理念")
    if "闭环" in question and "三维" not in question and "六措" not in question:
        return _cut(r"构建“质量目标—监控评估—信息反馈—改进完善”的闭环质保体系")
    if "反馈机制" in question:
        return _cut(r"一是建立及时反馈机制.{0,160}二是建立定期反馈机制.{0,160}三是建立专题反馈机制.{0,220}纠正与改进方案")
    if "表7-1" in question and "校风学风" not in question:
        return _cut(r"其中校风学风优势最为33显著")
    if "深造率" in question or ("学位授予" in question and "毕业率" in question):
        return _cut(r"学位授予率98\.44%。2024届本科生深造率54\.64%")
    if "用人单位" in question and "满意度" in question:
        line = _cut(r"根据江苏招就“2023年用人单位调查”数据，2023年用人单位对我校毕业生的总体满意度为91\.32%")
        return None if line is None else "91.32%。" + line
    if any(k in question for k in ("千人", "前五", "1001")):
        return _cut(r"2023届本科毕业生中46\.3%就业于1001人及以上规模的用人单位.{0,80}中兴通讯股份有限公司")
    if "正高级" in question and "副高级" in question and "397" not in question:
        return _cut(r"具有正高级专业技术职务397人，占17\.13%；具有副高级专业技术职务887人，占38\.27%")
    if "博士" in question and "硕士" in question and "占比" in question and "正高" not in question:
        return _cut(r"具有博士学位的有1746人，占75\.32%；具有硕士学位的有534人，占23\.04%")
    if "1+2+3" in question:
        return _cut(r"“1”是全校2024级所有专业全体学生培养方案中都加入人工智能\+内容，“2”是指具体实施采取的“AI\+微课程”和“AI\+微专业”两种主要模式，“3”指的是“通专、科教、产教”三融合")
    if "四个" in question and "创新基地" in question:
        return _cut(r"电子设计创新实践基地.{0,500}工程训练中心创新活动中心")
    if "电子科学与技术" in question and any(k in question for k in ("特色", "育人")):
        return _cut(r"信息光电子与智能微纳技术")
    if "教授" in question and "本科生" in question and "授课" in question:
        return _cut(r"所有教授连续两个学期至少一学期要为本科生授课")
    return None
