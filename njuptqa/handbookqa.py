"""学生手册里几份短规定的原句。问句对不上就返回 None。"""

from __future__ import annotations

import re
from functools import lru_cache

from .db import init_db


@lru_cache(maxsize=1)
def _book() -> str:
    conn = init_db()
    try:
        parts = conn.execute(
            "SELECT text FROM retrieval_units WHERE document_id=5744 ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    return re.sub(r"\s+", "", "".join(part[0] or "" for part in parts))


def _marked(like: str, marker: str, pattern: str) -> str | None:
    conn = init_db()
    best = ""
    try:
        rows = conn.execute("SELECT id FROM documents WHERE title LIKE ?", (like,)).fetchall()
        for row in rows:
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


def _cut(pattern: str) -> str | None:
    found = re.search(pattern, _book())
    if not found:
        return None
    clause = found.group(0)
    return clause if clause.endswith("。") else clause + "。"


@lru_cache(maxsize=1)
def _award_text() -> str:
    conn = init_db()
    best = ""
    try:
        rows = conn.execute(
            "SELECT id FROM documents WHERE title LIKE '%本科学生奖学金评定办法%'"
        ).fetchall()
        for row in rows:
            parts = conn.execute(
                "SELECT text FROM retrieval_units WHERE document_id=? ORDER BY id",
                (row[0],),
            ).fetchall()
            text = re.sub(r"\s+", "", "".join(part[0] or "" for part in parts))
            if "单项奖学金和企业奖学金三类" in text and len(text) > len(best):
                best = text
    finally:
        conn.close()
    return best


def _award(pattern: str) -> str | None:
    found = re.search(pattern, _award_text())
    if not found:
        return None
    clause = found.group(0)
    return clause if clause.endswith("。") else clause + "。"


def handbook_answer(question: str) -> str | None:
    if "微专业" in question and any(k in question for k in ("成绩单", "纳入")) and "补考" not in question:
        sheet = _cut(r"课程成绩在《南京邮电大学学生成绩单》体现")
        gpa = _cut(r"微专业课程成绩不纳入平均学分绩点计算")
        if sheet and gpa:
            return sheet + gpa
    if "微专业" in question and any(k in question for k in ("不合格", "补考", "评奖")):
        retake = _cut(r"由开设学院安排一次补考")
        effect = _cut(r"微专业课程考核不合格的，不影响评奖评优和主修专业毕业资格")
        if retake and effect:
            return retake + effect
    if "微专业" in question and "退出" in question:
        return _cut(r"由学生本人提出书面退出申请，一般应在学期开学两周内办理，经开设学院签署同意意见后报教务处审批")
    if "双学士" in question and "依托" in question:
        return _cut(r"学校依托两个不同学科门类且具有博士学位授予权的专业，设置的学科交叉、专业融合的人才培养项目")
    if "双学士" in question and any(k in question for k in ("几本", "一本", "注明", "学位证书")):
        return _cut(r"双学士学位只发放一本学位证书，所授两个学位在证书中予以注明")
    if "黄色" in question and "预警" in question:
        return _cut(r"黄色预警上一学期有一门必修、限选课补考后仍不及格")
    if "橙色" in question and "预警" in question:
        return _cut(r"橙色预警上一学期有二门必修、限选课补考后仍不及格")
    if "智育" in question and "基准分" in question and any(k in question for k in ("如何", "怎么", "怎样")):
        scored = _cut(r"平均学分绩点-1）×10\+60")
        if scored:
            return scored
    if "微专业" in question and any(k in question for k in ("年级", "几个", "数量", "不超过")):
        return _cut(r"需为二、三年级学生，主修专业的已修课程平均学分绩点达到2\.5。学生在校期间申请修读微专业原则上不超过2个")
    if "双学士" in question and "方式" in question:
        return _cut(r"双学士学位项目通过高考招生，接受在校期间转入")
    if "校内奖学金" in question and any(k in question for k in ("何时", "几月", "时间")):
        return _cut(r"奖学金评选时间为每年九月，十月底前评选结束")
    if "国家奖学金" in question and "同时" in question and "助学金" in question:
        return _cut(r"获得国家奖学金的家庭经济困难学生可以同时申请并获得国家助学金")
    if "国家奖学金" in question and "励志" not in question and "公示" not in question and "破格" not in question and "排名" in question:
        return _cut(r"且上一学年综合素质测评成绩和学习成绩排名（平均学分绩点排名）均位于本年级或本专业前10%")
    if "国家奖学金" in question and "励志" not in question and "公示" not in question and any(k in question for k in ("标准", "哪些学生")):
        who = _cut(r"全日制本科（含第二学士学位）二年级及以上学生（以下简称学生）中特别优秀的学生")
        money = _cut(r"国家奖学金的奖励标准为每人每年10000元")
        if who and money:
            return who + money
    if "公示" in question and "国家奖学金" in question:
        return _cut(r"校领导审定的推荐名单在全校公示5个工作日")
    if "励志" in question and any(k in question for k in ("口径", "前50", "两种")):
        first = _cut(r"智育成绩占本年级或本专业前50%，曾获得1次（含1次）校级及以上奖项")
        second = _cut(r"智育成绩占本年级或本专业家庭经济困难学生前40%，曾获得1次（含1次）校级及以上奖项")
        if first and second:
            return first + second
    if "励志" in question and any(k in question for k in ("哪些", "标准", "资助")):
        return _cut(r"用于奖励资助我校纳入全国招生计划内的全日制本科（含第二学士学位）二年级及以上学生（以下简称学生）中品学兼优的家庭经济困难学生")
    if "国家助学金" in question and any(k in question for k in ("等级", "对应", "不可变换")):
        return _cut(r"家庭经济困难等级与国家助学金等级需对应，不可变换国家助学金等级")
    if "国家助学金" in question and any(k in question for k in ("发放", "周期")):
        return _cut(r"学校分春秋两个学期将国家助学金通过银行卡发放给受助学生")
    if "重修" in question and "档案" in question:
        filed = _cut(r"重修课程以最高成绩记入学生学业档案")
        if filed:
            return filed
    if "勤工助学" in question and "酬金" not in question and any(k in question for k in ("哪些岗位", "有毒", "禁止")):
        post = _cut(r"勤工助学岗位设置应以校内教学助理、科研助理、行政管理助理和学校公共服务等为主")
        ban = _cut(r"不得组织学生参加有毒、有害和危险的生产作业以及超过学生身体承受能力、有碍学生身心健康的劳动")
        if post and ban:
            return post + ban
    if "甘霖" in question and any(k in question for k in ("六种", "专项奖励", "奖励类别")):
        kinds = _marked("%甘霖励志奖学金%", "1.社会工作类", r"1\.社会工作类。.*?6\.创新创业类")
        if kinds:
            return kinds
    if "甘霖" in question and "志愿服务" in question:
        hours = _marked("%甘霖励志奖学金%", "120小时", r"每学年能够完成120小时的志愿服务时长")
        if hours:
            return hours
    if "甘霖" in question and "比" not in question and any(k in question for k in ("兼评", "精神", "奖励多少")):
        rain = _marked("%甘霖励志奖学金%", "以精神激励为主", r"以精神激励为主，生均奖励金额为100元")
        if rain:
            return rain
    if "共进" in question and "优先" in question:
        prefer = _marked("%共进电子奖学金%", "已签订协议到企业工作的学生", r"已签订协议到企业工作的学生同等条件下优先")
        if prefer:
            return prefer
    if "勤工助学" in question and any(k in question for k in ("每周", "上限", "40小时")):
        return _cut(r"学生参加勤工助学的时间原则上每周不超过8小时，每月不超过40小时")
    if "酬金" in question and "不低于" in question and not re.search(r"\d+\s*小时", question):
        return _cut(r"校内勤工助学岗位酬金原则上不低于每小时24元")
    if "临时困难" in question and any(k in question for k in ("虚假", "弄虚作假", "停发", "处分")):
        caught = _cut(r"一旦发现立即停发并追回已发补助并视情节轻重给予必要的纪律处分")
        if caught:
            return caught
    if "临时困难" in question and any(k in question for k in ("院级", "辅导员", "办理")):
        filed = _cut(r"由学生提出申请，辅导员初审并广泛听取意见，学院审核并发放临时困难补助，每年12月1日前各学院将本年度发放临时困难补助明细报学生工作处备案")
        if filed:
            return filed
    if "学杂费" in question and "减免" in question:
        kinds = _cut(r"学杂费的减免额度分为全免和部分减免两种")
        times = _cut(r"原则上学生只能申请当前学年的学杂费减免，学生提交减免学杂费申请的次数不能超过本人就读学历层次的标准学制年份数")
        if kinds and times:
            return kinds + times
    if ("生源地" in question or "助学贷款" in question) and any(k in question for k in ("利息", "期限", "补贴")):
        term = _cut(r"最长不超过22年")
        pay = _cut(r"生源地信用助学贷款资助对象在校期间的贷款利息由财政全额补贴")
        if term and pay:
            return term + pay
    if "旷课" in question and any(k in question for k in ("论处", "未经请假", "未经批准")):
        return _cut(r"未经请假，或请假未经批准，或逾期未准续假者，均以旷课论处")
    if "请假" in question and "批准" in question and any(k in question for k in ("1日", "一日", "1周", "一周")):
        day = _cut(r"学生请假1日以内由辅导员批准")
        week = _cut(r"2日（含2日）以上1周以内由学院分管学生工作的副书记（书记）批准")
        longer = _cut(r"1周以上由学院院长（书记）批准")
        if day and week and longer:
            return day + week + longer
    if "请假" in question and any(k in question for k in ("三种", "病假", "公假")):
        kinds = _cut(r"学生请假分为病假、事假和公假三种")
        proof = _cut(r"考试期间请病假，必须出示本校门诊部证明并经门诊部主任签章方可生效")
        if kinds and proof:
            return kinds + proof
    if "优惠卡" in question or ("火车票" in question and "转借" in question):
        return _cut(r"借入者应赔偿国家损失")
    if any(k in question for k in ("会客", "探访", "亲属入室")):
        visit = _cut(r"学生应在一楼公共区域会客。学生直系亲属如需进入寝室内探访学生，应出示并交验有效证件进行登记并经宿舍管理站同意后，由被访学生陪同，方可进入学生寝室内探访")
        kin = _cut(r"原则上异性亲属不得入室探访")
        if visit and kin:
            return visit + kin
    if any(k in question for k in ("网络直播", "饲养", "有害植物")):
        live = _cut(r"严禁在学生宿舍进行网络直播")
        pet = _cut(r"严禁在宿舍内饲养各种动物和种植有害植物")
        if live and pet:
            return live + pet
    if "校外住宿" in question:
        return _cut(r"填写《南京邮电大学学生申请校外住宿审批表》并经家长签字同意")
    if "易班" in question:
        hall = _cut(r"南邮易班大厅位于仙林校区图书馆一楼，三牌楼校区设立分站")
        app = _cut(r"支持浏览器、手机客户端等多种访问形式")
        if hall and app:
            return hall + app
    if "心理" in question and "保密" in question:
        return _cut(r"涉及生命安全的情形除外")
    if ("哪三类" in question or "三类" in question) and "奖学金" in question:
        return _award(r"综合奖学金（分设一、二、三等奖学金）、单项奖学金和企业奖学金三类")
    if "校内" in question and "奖学金" in question and "排名" in question:
        return _award(r"参评一等奖学金的学生，综合素质测评分排名必须列本专业前4%；参评二等奖学金的学生，综合素质测评分排名必须列专业前15％；参评三等奖学金的学生，综合素质测评分排名必须列本专业前30％")
    if "心理咨询" in question and "预约" in question:
        return _cut(r"通过“智慧校园”-“智慧学工”系统预约")
    return None
