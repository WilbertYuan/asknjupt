"""学籍管理办法里对得上的几条：编级、五级制绩点、退学和结业。"""

from __future__ import annotations

import re
from decimal import Decimal, ROUND_HALF_UP

from .db import init_db

_TEXT: str | None = None
_LEVELS = ("优秀", "良好", "中等", "及格", "不及格")


def _rules() -> str:
    global _TEXT
    if _TEXT is not None:
        return _TEXT
    conn = init_db()
    best = ""
    rows = conn.execute(
        "SELECT id FROM documents WHERE title LIKE '%本科生学籍管理办法%'"
    ).fetchall()
    for row in rows:
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


def _grade_point(question: str) -> str | None:
    if "五级制" not in question or "平均" in question or "百分" in question:
        return None
    named = [level for level in _LEVELS if level in question]
    if len(named) != 1:
        return None
    found = re.search(
        r"优秀\|良好\|中等\|及格\|不及格绩点系数\|([0-9.]+)\|([0-9.]+)\|([0-9.]+)\|([0-9.]+)\|([0-9.]+)",
        _rules(),
    )
    if not found:
        return None
    coef = Decimal(found.group(_LEVELS.index(named[0]) + 1))
    credits = re.findall(r"(\d+(?:\.\d+)?)学分", question)
    if not credits:
        shown = format(coef, "f").rstrip("0").rstrip(".")
        return shown
    if len(credits) != 1:
        return None
    product = (coef * Decimal(credits[0])).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    coef_text = format(coef, "f").rstrip("0").rstrip(".")
    product_text = format(product, "f").rstrip("0").rstrip(".")
    return f"系数{coef_text}，课程学分绩点{product_text}"


def _clause(text: str, pattern: str) -> str | None:
    found = re.search(pattern, text)
    if not found:
        return None
    clause = found.group(0)
    return clause if clause.endswith("。") else clause + "。"


def status_answer(question: str) -> str | None:
    graded = _grade_point(question)
    if graded:
        return graded
    cues = (
        "入学", "注册", "学习年限", "考查", "考核", "补考", "重修", "旷课", "免修", "休学",
        "升级", "连续两周", "结业", "学籍",
    )
    if not any(k in question for k in cues):
        return None
    text = _rules()
    if "升级" in question and "编入" in question and "跳级" not in question and "毕业设计" not in question:
        found = _clause(text, r"差距小于30学分者予以升级；等于或大于30学分者，编入下一级学习")
        if found:
            return found
    if "连续两周" in question and "教学活动" in question:
        if "应予退学" in text and "未请假离校连续两周未参加学校规定的教学活动" in text:
            return "应予退学。"
    if "结业" in question and "证书" in question and "颁发" in question and "复印件" not in question:
        if "未达到毕业要求的学生发放结业证书" in text:
            return "未达到毕业要求的学生发放结业证书。"
    if any(k in question for k in ("携带", "录取通知书", "哪些材料", "基本材料")):
        return _clause(text, r"持录取通知书和学校规定的有关文件，按学校有关要求和规定的期限到校办理入学手续")
    if "初步审查" in question or "报到时" in question:
        return _clause(text, r"审查合格的办理入学手续，予以注册学籍")
    if "保留入学资格" in question and "期限" in question and "高考" not in question and "重新" not in question:
        return _clause(text, r"保留入学资格的期限为一学年")
    if "复查" in question or ("入学后" in question and any(k in question for k in ("多长", "多久", "几个月"))):
        return _clause(text, r"学生入学后，学校在3个月内按照国家招生规定进行复查")
    if "家庭经济困难" in question and "注册" in question:
        return _clause(text, r"办理有关手续后注册")
    if "最长" in question and "学习年限" in question and "双学士" not in question:
        return _clause(text, r"学生学习年限一般最长不超过六年（含休学期限，参军入伍除外）")
    if ("考试" in question and "考查" in question) or ("考核" in question and "哪两种" in question):
        return _clause(
            text,
            r"考核分考试与考查两种。课程一般按百分制评定成绩，集中实践性教学环节按“优秀”、“良好”、“中等”、“及格”、“不及格”五级评定成绩",
        )
    if "已经及格" in question and "重修" in question and "档案" not in question:
        filed = _clause(text, r"考核成绩及格但不理想的学生可以申请参加“重修”")
        if filed:
            return filed
    if "补考" in question and any(k in question for k in ("记载", "及格记")):
        return _clause(text, r"补考通过成绩按及格记，未通过按不及格记")
    if "旷课" in question and "考试资格" not in question and any(k in question for k in ("重修", "考核资格", "比例")):
        return _clause(text, r"无故旷课累计超过课程学时数五分之一者，或无故缺交作业、缺做实验超过五分之一者，均不得参加该课程的考核，必须重修")
    if "免修" in question and "绩点" in question:
        return _clause(text, r"前两个学期的平均学分绩点≥2\.5")
    if "休学" in question and any(k in question for k in ("累计", "多久", "多长", "不得超过")) and not any(
        k in question for k in ("创业", "退役", "转专业", "退宿")
    ):
        return _clause(text, r"休学一般以一学年为限.{0,24}累计休学一般不得超过两学年")
    return None
