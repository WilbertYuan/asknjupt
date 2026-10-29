"""通用句子级抽取问答层。

不认识具体题目。只做三件事：
1. 把问题折成「题型 + 内容词 + 属性分组」；
2. 把证据块切成句子，按内容词覆盖度和题型信号打分；
3. 按题型组装答案：数字绑定、列举聚合、是否判断、默认取句。

这层放在检索和 LLM 之间：检索给出候选块，这里给出可直接当答案的
短句；LLM 只负责润色，且输出必须过覆盖度校验。
"""

from __future__ import annotations

import re
from typing import Any

NUM_RE = re.compile(r"\d+(?:\.\d+)?")
YEAR_RE = re.compile(r"(?:19|20)\d{2}")

# 疑问词和礼貌前缀，从来不是搜索或覆盖度判断的一部分。
_QUESTION_WORDS = (
    "请问", "请教", "是什么", "什么是", "是谁", "什么样", "什么", "怎么样", "怎样",
    "如何", "怎么", "哪些", "哪几", "哪个", "哪种", "哪类", "哪所", "哪里",
    "多少", "几", "何时", "什么时候", "哪年", "何年", "几月几日", "哪天",
    "是否", "能否", "能不能", "可不可以", "可以", "吗", "呢", "？", "?",
    "的", "了", "有", "是", "为", "和", "与", "或", "及", "以及", "分别",
    "各", "具体", "详细", "介绍一下", "告诉我", "说明",
)

# 列举型信号：答案通常不止一句。
_LIST_MARKERS = ("哪些", "哪几", "分别", "几类", "几种", "几方面", "三类", "两类",
                 "两种", "包括", "分为", "覆盖哪些", "有哪些")
_YESNO_MARKERS = ("能否", "能不能", "可不可以", "是否", "可以", "允许")
_NUMBER_MARKERS = ("多少", "几", "数量", "几人", "几个", "几门", "几次", "多少分",
                   "多少年", "比例", "百分比", "上限", "下限", "最长", "最短")
_WHEN_MARKERS = ("何时", "什么时候", "哪年", "何年", "几月几日", "哪天", "什么时间",
                 "日期是", "哪一年")
_WHO_MARKERS = ("是谁", "谁", "什么人")

# 句子切分时的噪声：页眉页脚、条款号、纯数字行。
_NOISE_LINE = re.compile(
    r"^(?:第?\s*\d{1,4}\s*[页·]?|[·\-—\s]*\d{1,4}[·\-—\s]*|目\s*录|摘\s*要)$"
)
# 「第X章/第X节」开头的短行是标题，不是答案。
_HEADING = re.compile(r"^第[一二三四五六七八九十百0-9]{1,4}[章节编][^，。；：]{0,24}$")

# 「多久/何时」类问句：答案句必须带具体时间量词，否则只是
# 命中了话题词，并没有回答问题。「第X周」也算具体时间。
_TIMEWAY = re.compile(
    r"\d+\s*(?:年|月|日|天|周|星期|小时|分钟|周次)|"
    r"第\s*\d+[-–—\d]*\s*周|"
    r"\d+\s*周内|\d+\s*周前|\d+\s*周后|"
    r"提前\s*\d+|之内|以内"
)


# 这些单字是疑问结构词，但也是高频构词字（不及格、行为、或者、所有）。
# 两侧都是汉字时多半在词内部，保护不剥；有一侧是边界（标点/空白/数字）
# 才按结构词剥掉。「和/与」不做保护：枚举连接（A和B）比构词（综合）常见。
_FLANK_PROTECTED = "是为或及有"


def strip_question_words(question: str) -> str:
    """去掉疑问词，留下实体和属性。这是搜索词和覆盖度判断的基础。"""
    text = question or ""
    # 时效约束是筛选条件，不是内容词。
    text = re.sub(r"截至\s*20\d{2}\s*年\s*\d{0,2}月?", " ", text)
    text = re.sub(r"[？?！!。；;]", " ", text)
    for word in sorted(_QUESTION_WORDS, key=len, reverse=True):
        if len(word) == 1 and word in _FLANK_PROTECTED:
            continue
        text = text.replace(word, " ")
    for ch in _FLANK_PROTECTED:
        text = re.sub(f"(?<![一-鿿]){ch}|{ch}(?![一-鿿])", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# 学校名太通用，命中它不能说明句子在回答问题。
GENERIC_TERMS = {"南京邮电大学", "南邮", "南京邮电", "邮电大学", "学校", "我校", "大学"}


_MEASURE_CHARS = "个位名只条款项种"


def question_pieces(question: str) -> list[str]:
    """问句切出的顶层短语（词族）。打分时的「宽度」按词族计。

    长片段在疑问结构词（一般/可以/须…）处再切断：「学生补改选一般安排在
    开学后多久」是一个整体，整句永不命中；切成「补改选/开学」等词族后，
    打分才能区分「补改选在开学后2周内进行」和无关的「补修申请表」。
    """
    body = strip_question_words(question)
    body = re.sub(r"南京邮电大学|南京邮电|邮电大学|南邮(?!精神|校史|校歌|风华|人|学子)", " ", body)
    pieces: list[str] = []
    for piece in re.split(r"[\s，,、：:（）()\"“”‘’]", body):
        piece = piece.strip().lstrip(_MEASURE_CHARS).lstrip("哪")
        if not piece or "\u0001" in piece:
            continue
        if len(piece) >= 2 and not YEAR_RE.fullmatch(piece):
            pieces.append(piece)
            if len(piece) > 3 and piece[-1] in "中里内外":
                pieces.append(piece[:-1])
            # 疑问结构词切断成词族
            for part in re.split(r"一般|通常|可以|需要|应当|应该|将会|能否|可否|必须|须|会|能|不|在|按|是|有|的", piece):
                part = part.strip()
                if len(part) >= 2 and part != piece:
                    pieces.append(part)
                    short = re.sub(r"(多久|何时|怎么办|如何|什么|哪些|哪几|几个|多少|怎么样)$", "", part)
                    if len(short) >= 2 and short != part:
                        pieces.append(short)
    # 同长词必须按字串再排一次：set 的迭代顺序随进程哈希种子变化，
    # 只靠 len 排序会让锚定词选择在每次运行间抖动（评测 flap 的根源）。
    return sorted(set(pieces), key=lambda t: (-len(t), t))


def content_terms(question: str) -> list[str]:
    """从问题里切出 2 字以上的内容词（汉字 n-gram + 数字 + 英文缩写）。

    通用的：不依赖词典，连续汉字按 2-6 字窗口切，去重后按长度排。
    校名先整体拿掉，避免滑窗在校名内部切出碎片词。
    """
    body = strip_question_words(question)
    # 校名是检索词不是内容词，先换成占位符，滑窗不会切进它。
    # 「南邮精神」「南邮人」里的南邮是答案的一部分，不能替换。
    body = re.sub(r"南京邮电大学|南京邮电|邮电大学|南邮(?!精神|校史|校歌|风华|人|学子)", " ", body)
    terms: list[str] = []
    # 完整短语先收进来（按标点/空格切开的长片段）。
    for piece in re.split(r"[\s，,、：:（）()\"“”‘’]", body):
        piece = piece.strip().lstrip(_MEASURE_CHARS).lstrip("哪")
        if not piece or "\u0001" in piece:
            continue
        if len(piece) >= 2 and not YEAR_RE.fullmatch(piece):
            terms.append(piece)
            # 「课程期末考试中」的「中」是问句语法，去尾的也是候选词。
            if len(piece) > 3 and piece[-1] in "中里内外":
                terms.append(piece[:-1])
            # 长片段在疑问结构词（一般/通常/可以/需要/应当/应该/会/能…）
            # 处切断：「学生补改选一般安排在开学后多久」的「补改选」才切得出来。
            cut = re.split(r"一般|通常|可以|需要|应当|应该|将会|能否|可否|必须|须|会|能", piece)
            for part in cut:
                part = part.strip()
                if len(part) >= 2 and part != piece:
                    terms.append(part)
                    # 切断后的片段再剥尾部疑问词（「开学后多久」→「开学」）。
                    short = re.sub(r"(多久|何时|怎么办|如何|什么|哪些|哪几|几个|多少|怎么样)$", "", part)
                    if len(short) >= 2 and short != part:
                        terms.append(short)
        # 滑窗只取词首/词尾两端（「硕士专业学」「学位授权点」），防止长片段
        # 整体匹配不上；中间碎片（「士专业学位」）会被「博士…」近邻蒙混，不生成。
        han = "".join(ch for ch in piece if "\u4e00" <= ch <= "\u9fff")
        for size in (5, 4, 3):
            if len(han) > size:
                for i in (0, len(han) - size):
                    gram = han[i : i + size]
                    if gram[0] in _MEASURE_CHARS or gram[-1] in "的了":
                        continue
                    terms.append(gram)
    for match in re.finditer(r"[A-Za-z][A-Za-z0-9+.-]{1,15}|\d+(?:\.\d+)?%?", body):
        terms.append(match.group())
    # 短核心术语补切：长片段在疑问结构词处切断后，首段（主语侧）
    # 去掉「学生/我校」类泛化前缀，剩下的 2-6 字常是答案术语本身
    # （「学生补改选一般安排在…」→「补改选」），滑窗切不出它。
    for piece in re.split(r"[\s，,、：:（）()\"“”‘’]", body):
        han = "".join(ch for ch in piece if "一" <= ch <= "鿿")
        head = re.split(r"一般|通常|可以|需要|应当|应该|将会|能否|可否|必须|须|会|能", han)[0]
        head = re.sub(r"^(学生|我校|学校|本科生|研究生|读者|考生|教师)", "", head)
        if 2 <= len(head) <= 8 and head != han:
            terms.append(head)
    # 去重、去通用校名。长词和它的碎片都留着：匹配时谁都能中，
    # 覆盖度计算时再按重叠去重，避免重复加分。
    uniq: list[str] = []
    # 同长词按字串定序（见 question_pieces 的说明），保证跨进程确定。
    for t in sorted(set(terms), key=lambda t: (-len(t), t)):
        if len(t) < 2 or t in GENERIC_TERMS:
            continue
        uniq.append(t)
    return uniq[:40]


def has_recency_constraint(question: str) -> bool:
    return bool(re.search(r"截至|目前|现在|现有|最新|当今", question or ""))


def question_type(question: str) -> str:
    """粗粒度题型，只决定组装策略，不影响检索。"""
    q = question or ""
    if any(k in q for k in _LIST_MARKERS) or re.search(r"哪[一二三四五六七八九十\d]+[个种类方面]", q):
        return "LIST"
    if any(k in q for k in _YESNO_MARKERS):
        return "YESNO"
    if any(k in q for k in _WHEN_MARKERS):
        return "WHEN"
    if any(k in q for k in _NUMBER_MARKERS):
        return "NUMBER"
    if any(k in q for k in _WHO_MARKERS):
        return "WHO"
    return "STATEMENT"


def attribute_groups(question: str) -> list[str]:
    """「二年级、三年级……分别多少」这类题，把属性词分组，每组要各自找到答案。

    返回的属性组按在问题中出现的顺序排列。非多分问题返回空表。
    """
    q = question or ""
    if not any(k in q for k in ("分别", "各", "和", "与", "以及", "、")):
        return []
    body = strip_question_words(q)
    # 按并列连词切开，取每段的尾部核心词（2-8 字）。
    groups: list[str] = []
    for piece in re.split(r"[，,；;、]|分别|以及|或者|或|和|与", body):
        piece = piece.strip()
        if len(piece) < 2:
            continue
        # 每段取最后 2-8 个汉字作为属性锚点。
        han = "".join(ch for ch in piece if "\u4e00" <= ch <= "\u9fff")
        if len(han) >= 2:
            groups.append(han[-8:])
    # 至少两组才值得分开找。
    return groups if len(groups) >= 2 else []


def _loose_contains(sentence: str, term: str) -> bool:
    """有序子序列匹配：「饲养动物」能中「饲养各种动物」。间隔最多 4 字。"""
    if term in sentence:
        return True
    pos = -1
    for ch in term:
        nxt = sentence.find(ch, pos + 1, pos + 6)
        if nxt < 0:
            return False
        pos = nxt
    return True


def split_sentences(text: str) -> list[str]:
    """把证据块切成候选句。

    PDF 换行不是句界：行尾没有终止标点就先拼回上一行；
    「第十七条」这种条款号单独成行时并进后一句。
    """
    joined_lines: list[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or _NOISE_LINE.match(line):
            continue
        # 条款号单独成行（「第十七条」「1.」）：并进下一句。
        if re.fullmatch(r"第[一二三四五六七八九十百0-9]{1,4}条|\d{1,2}[.、]", line):
            joined_lines.append(line)
            continue
        if joined_lines and not re.search(r"[。；;：:]$", joined_lines[-1]):
            joined_lines[-1] += line
        else:
            joined_lines.append(line)
    joined = "\n".join(joined_lines)
    rough = re.split(r"(?<=[。；;])|\n", joined)
    sentences: list[str] = []
    for part in rough:
        s = part.strip(" \t，,。；;：:")
        if not s:
            continue
        if sentences and len(s) < 6 and not NUM_RE.search(s):
            sentences[-1] += "，" + s
        else:
            sentences.append(s)
    return [s for s in sentences if len(s) >= 6]


def _term_hits(sentence: str, terms: list[str]) -> list[str]:
    return [t for t in terms if t in sentence]


def _gram_hits(sentence: str, terms: list[str]) -> list[str]:
    """3 字重叠命中：「三个校区」能中问句里的「哪三个校区」。

    只用于没精确命中的词，避免重复计分。词首 3 字必须命中：
    术语的区分度在词首（硕士/博士），中间碎片会蒙混。
    """
    out: list[str] = []
    for t in terms:
        if len(t) < 4 or t in sentence:
            continue
        if t[:3] in sentence or (len(t) >= 8 and t[:4] in sentence):
            out.append(t)
    return out


def _coverage(hits: list[str], cap: int = 0) -> float:
    """命中的词按长度平方加权；互为子串的词只算最长的那个。

    双向子串判定：「专业学位授」与「士专业学位」重叠但不是子串，
    只留更长的，防止滑窗碎片把短句分数刷过真正命中长术语的句子。
    cap>0 时词长按 cap 封顶：gram 命中的实际证据只是词首 3-4 字，
    不能按完整词长计分（「本科生」前缀命中 23 字长词会白捡 529 分）。
    """
    kept: list[str] = []
    for t in sorted(set(hits), key=lambda t: (-len(t), t)):
        if any(t != u and (t in u or u in t) for u in kept):
            continue
        kept.append(t)
    return sum(min(len(t), cap) ** 2 if cap else len(t) ** 2 for t in kept)


# 链接列表页：一行里多个日期是导航噪声，不是答案。
_DATE_LIST = re.compile(r"20\d{2}[-年.]\d{1,2}[-月.]\d{1,2}")
_NAV_WORDS = ("首页", "概况", "列表", "更多", "上一篇", "下一篇", "点击", "浏览",
              "当前位置", "栏目", "专题", " English", "旧版", "手机版")


_OCR_GARBAGE = re.compile(r"[$#%!&*『』「」<>@~^]")


def _is_noise(sentence: str) -> bool:
    if len(_DATE_LIST.findall(sentence)) >= 2:
        return True
    if sum(1 for w in _NAV_WORDS if w in sentence) >= 2:
        return True
    # OCR 扫出来的句子怪字符密度高，不能当答案。
    if len(sentence) >= 20 and len(_OCR_GARBAGE.findall(sentence)) / len(sentence) > 0.03:
        return True
    # PDF 表格被压成一行碎片：大量孤立数字/符号 token
    # （「百分制 90 100 80 89 - -=绩点系数 45 35 . .」），不是可读答案。
    tokens = sentence.split()
    if len(tokens) >= 6:
        frag = sum(1 for t in tokens if re.fullmatch(r"[\d.\-–—=|]+", t))
        if frag / len(tokens) > 0.4:
            return True
    return False


def _piece_hit(sentence: str, piece: str) -> bool:
    """词族命中：精确，或词首 3 字命中（「浦口区设有」中「浦口区」算该族命中）。"""
    if piece in sentence:
        return True
    return len(piece) >= 4 and piece[:3] in sentence


def score_sentence(sentence: str, terms: list[str], qtype: str, question: str,
                   pieces: list[str] | None = None) -> float:
    """给单句打分：词族宽度为主，覆盖度为辅，题型信号加分，噪声为负。

    问句有多个信息点（词族）时，正确答案往往每个都沾一点；
    只反复命中一个词族的句子多半是噪声。
    """
    if _is_noise(sentence):
        return 0.0
    # 「多久/何时」问的是时间，答案句必须带具体时间量词；
    # 只命中话题词（「补改选」「缓考」）的句子没有回答，直接排除。
    if re.search(r"多久|何时|什么时候", question) and not _TIMEWAY.search(sentence):
        return 0.0
    hits = _term_hits(sentence, terms)
    grams = _gram_hits(sentence, terms)
    if not hits and not grams:
        return 0.0
    # gram 命中的实际证据是词首 3-4 字，覆盖按 4 字封顶计。
    score = _coverage(hits) + 0.5 * _coverage(grams, cap=4)
    if pieces:
        # 词族完整命中 16 分；仅词首 3 字命中（「本科生」撞上
        # 「本科生每学期修读…」）只值 6 分，防止前缀蒙混。
        families = 0.0
        for p in pieces:
            if p in sentence:
                families += 1.0
            elif len(p) >= 4 and p[:3] in sentence:
                families += 0.375
        if families == 0:
            return 0.0
        score += 16 * families
    # 句首（8字内）命中 3 字以上术语：「补改选在开学后2周内进行」
    # 句首就是题眼，这类短句往往就是答案本体。
    head = sentence[:8]
    if any(len(t) >= 3 and t in head for t in hits):
        score += 18
    if _HEADING.match(sentence):
        score *= 0.2
    # 「X是什么/什么是X」：定义句（X是…、X：…、X“…”）是强信号。
    longest = max(hits or grams, key=len)
    if qtype == "STATEMENT" and len(longest) >= 2:
        pos = sentence.find(longest)
        if pos >= 0:
            after = sentence[pos + len(longest) : pos + len(longest) + 4]
            if re.match(r"^[是为：:“\"‘]", after):
                score += len(longest) ** 2
    if qtype == "NUMBER":
        nums = [n for n in NUM_RE.findall(sentence) if not YEAR_RE.fullmatch(n)]
        if not nums:
            return 0.0
        # 表格转成的行（多个 |）：里面的数字是逐行的条目代码，
        # 不是「总数」答案，直接排除，防止「| 10 | 0701 |」蒙混。
        if len(_TABLE_ROW.findall(sentence)) >= 3:
            return 0.0
        # 数字必须紧贴核心属性词。核心词要精确命中：
        # 「硕士专业学位授权点」不能被「博士专业学位授权点」里的碎片蒙混。
        exact = [t for t in hits if len(t) >= 3]
        if not exact:
            return 0.0
        core = max(exact, key=len)
        cpos = sentence.find(core)
        # 问句的疑问焦点：「多少」之后紧贴的属性词（「多少个本科专业」
        # 里的「本科专业」）。数字必须绑在它身上，不能绑在句子里的
        # 其他数字上（「一流本科专业建设点27个」不是本科专业总数）。
        focus = None
        mq = re.search(r"多少\s*个?\s*([一-鿿]{2,12})", question)
        if mq:
            focus = mq.group(1)
        fpos = sentence.find(focus) if focus else -1
        best_dist: int | None = None
        for m in NUM_RE.finditer(sentence):
            if YEAR_RE.fullmatch(m.group()):
                continue
            # 数字与核心词之间隔着「。」就属另一句，距离加倍。
            lo, hi = min(m.start(), cpos), max(m.start(), cpos)
            gap = sentence[lo:hi]
            dist = hi - lo
            if "。" in gap:
                dist *= 2
            if best_dist is None or dist < best_dist:
                best_dist = dist
        if best_dist is None or best_dist > 16:
            return 0.0
        score += 14 / (1 + best_dist / 6)
        # 长核心词（≥5字）是题眼，比短词更能锁定答案句。
        if len(core) >= 5:
            score += 16
        # 疑问焦点命中：数字须紧贴焦点词（12 字内），否则这句在
        # 回答别的问题，直接排除；紧贴则强加分。
        if focus:
            # 时效问题：焦点词必须精确出现，「本科招生专业」不算「本科专业」。
            if focus not in sentence:
                return 0.0
            if fpos < 0:
                return 0.0
            # 焦点词作为更长修饰词的一部分出现时（「本科专业」出现在
            # 「一流本科专业建设点」里），该位置不算数。
            fpos_list: list[int] = []
            start = 0
            while True:
                p = sentence.find(focus, start)
                if p < 0:
                    break
                after = sentence[p + len(focus): p + len(focus) + 4]
                before = sentence[max(0, p - 3): p]
                if not re.match(r"^[建设学点班类]", after) and not re.search(r"[一流二双]$", before):
                    fpos_list.append(p)
                start = p + 1
            if not fpos_list:
                return 0.0
            fdist: int | None = None
            for m in NUM_RE.finditer(sentence):
                if YEAR_RE.fullmatch(m.group()):
                    continue
                for fp in fpos_list:
                    d = min(abs(m.start() - (fp + len(focus))), abs(m.start() - fp))
                    if fdist is None or d < fdist:
                        fdist = d
            if fdist is None or fdist > 12:
                return 0.0
            score += 20
    elif qtype == "LIST":
        # 列举句：冒号、顿号、（1）（2）结构是强信号。
        if "：" in sentence:
            score += 8
        score += 4 * sentence.count("、")
        if re.search(r"[（(]\s*[一二三四五六七八九十\d]\s*[)）]", sentence):
            score += 10
        # 「有哪些X」的答案通常带 现有/设有/分为/包括。
        if re.search(r"现有|设有|分为|包括|涵盖", sentence):
            score += 14
        # 数量词回声：问「哪两种/哪三类」，答句带「两种/三类」。
        echo = re.search(r"哪?([一二三四五六七八九十两\d]+)\s*[种类型]", question)
        if echo and f"{echo.group(1)}种" in sentence.replace("类", "种"):
            score += 18
        # 问「哪N个/哪N方面」且句首（30字内）就有枚举回声「N个校区」，
        # 这句多半就是答案本身；枚举藏在长句中段的只给一半分。
        echo_n = re.search(r"哪?([一二三四五六七八九十两\d]+)\s*个", question)
        if echo_n:
            unit = echo_n.group(1)
            early = sentence[:30]
            if re.search(rf"{re.escape(unit)}\s*个", early):
                score += 18
            elif re.search(rf"{re.escape(unit)}\s*个", sentence):
                score += 8
        # 「包括哪些方面/分为哪几类」的答案常写成「甲、乙、丙、丁、戊"X"」：
        # 多个顿号后接引号词是强枚举信号，比叙事长句更像答案本体。
        if re.search(r"(?:、[^，。；：]{2,10}){2,}[""『「]?[一-鿿]{2,12}[""』」]?", sentence):
            score += 14
    elif qtype == "YESNO":
        if any(k in sentence for k in ("可以", "不得", "严禁", "允许", "应当", "须", "不予", "执行")):
            score += 14
    elif qtype == "WHEN":
        # 时间词要贴着命中词才算数，句子别处带日期不算。
        date = re.search(r"\d{4}\s*年|\d{1,2}\s*月\s*\d{1,2}\s*日|\d{1,2}\s*月", sentence)
        if date:
            near = any(
                abs(date.start() - (sentence.find(t) + len(t))) <= 20 or abs(sentence.find(t) - date.end()) <= 20
                for t in (hits or grams) if sentence.find(t) >= 0
            )
            if near:
                score += 24
            else:
                score *= 0.7
        else:
            score *= 0.5
    elif qtype == "WHO":
        # 「职务+人名」或「人名+是/任」结构。
        if re.search(r"[\u4e00-\u9fff]{2,3}(?:同志)?(?:是|任|担任)", sentence):
            score += 10
    return score


def _trim(sentence: str, limit: int = 200) -> str:
    s = sentence.strip("，,；; ")
    if len(s) <= limit:
        return s
    return s[: limit - 1].rstrip("，,、") + "…"


_CN_DIGIT = {"一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
             "六": "6", "七": "7", "八": "8", "九": "9", "十": "10"}


def _cn_variants(term: str) -> list[str]:
    """「二年级」和「2年级」是同一个属性，生成写法变体做匹配。"""
    out = [term]
    for cn, digit in _CN_DIGIT.items():
        if cn in term:
            out.append(term.replace(cn, digit))
        if digit in term:
            out.append(term.replace(digit, cn))
    return out


# 枚举标记。「第（二）款」「（三）项」是条文交叉引用，不是枚举项：
# 前面有「第」或后面紧跟「款/条/项/章」的要排除。
_ENUM_ITEM = re.compile(
    r"(?<!第)[（(]\s*([一二三四五六七八九十\d]{1,2})\s*[)）](?!\s*[条款项章节])"
)

_ENUM_NO = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
            "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _enum_no(label: str) -> int:
    """枚举标号转数字，用于顺序校验。转不了返回 0。"""
    if label.isdigit():
        return int(label)
    return _ENUM_NO.get(label, 0)


def _anchored_enumeration(hits: list[dict[str, Any]], terms: list[str],
                          limit: int = 300) -> str | None:
    """在原文里找「问题术语 + 连续枚举组」。

    缓考三类、纪律三种这种答案，术语在枚举组前面的小标题里，
    单个枚举项本身不含术语，按句子打分永远找不到。
    锚定词用最长术语，避免被同页无关枚举组抢走。
    """
    strong = [t for t in terms if len(t) >= 3]
    if not strong:
        return None
    # 全局候选竞争：每个 hit 取它最强能锚定的枚举组，跨 hit 比锚定
    # 词长度（证据强度），同长比检索排名。只认「第一个能锚上的 hit」
    # 会被同页弱术语（「学生申请」锚到转学条款）劫持。
    candidates: list[tuple[int, int, str]] = []
    for hi, hit in enumerate(hits):
        text = (hit.get("text") or "").replace("\n", "")
        # 锚定词候选：在该文档里真实出现的术语，长的优先，同长按字串
        # 定序（跨进程确定）。最长术语可能出现在枚举组后面（「违反考场
        # 纪律」在第三项才出现），单锚失败要依次试次长词。
        anchors = [t for t in sorted(strong, key=lambda t: (-len(t), t)) if t in text]
        for anchor in anchors:
            found = _enum_group_at_anchor(text, anchor, strong, limit)
            if found:
                candidates.append((len(anchor), hi, found))
                break
    if not candidates:
        return None
    candidates.sort(key=lambda c: (-c[0], c[1]))
    return candidates[0][2]


def _enum_group_at_anchor(text: str, anchor: str, strong: list[str],
                          limit: int) -> str | None:
    """找一个以 anchor 为前文（80 字内）的连续枚举组，拼成答案。"""
    for m in _ENUM_ITEM.finditer(text):
        if m.group(1) not in ("1", "一"):
            continue
        # 枚举组前文 80 字内要见到锚定词，才认为这组枚举在回答该问题。
        prefix = text[max(0, m.start() - 80) : m.start()]
        if anchor not in prefix:
            continue
        group = [m]
        for nxt in _ENUM_ITEM.finditer(text, m.end()):
            if nxt.group(1) in ("1", "一"):
                break
            # 标号必须连续递增：PDF 多栏抽文本常把（三）（四）（二）（一）
            # 打乱成一行，乱序文本拼出的「枚举组」必然缺项错序。
            if _enum_no(nxt.group(1)) != _enum_no(group[-1].group(1)) + 1:
                break
            if nxt.start() - group[-1].end() > 200:
                break
            group.append(nxt)
            if len(group) >= 8:
                break
        if len(group) < 2:
            continue
        head = prefix.strip("，,。；; ")[-30:]
        # 小标题只取最后一句：80 字窗可能带进上一条款的尾巴
        # （「分及以上）。第三条…」→「第三条…」）。
        if "。" in head:
            head = head.rsplit("。", 1)[1]
        if "；" in head:
            head = head.rsplit("；", 1)[1]
        parts: list[str] = []
        for i, item in enumerate(group):
            start = item.end()
            end = group[i + 1].start() if i + 1 < len(group) else min(len(text), start + 160)
            body = text[start:end].strip("，,。；; ")
            body = re.split(r"(?<=[。])", body)[0] if "。" in body else body
            # 枚举项太短是目录词（「党的建设」「人才培养」），不是答案。
            if len(body) >= 8:
                parts.append(f"（{item.group(1)}）{body}")
            if sum(len(p) for p in parts) > limit:
                break
        if len(parts) >= 2:
            answer = "；".join(parts)
            if head and any(t in head for t in strong):
                answer = f"{head}：{answer}"
            # 佐证：枚举组（含小标题）里还要再见到另一个与锚定词不
            # 重叠的术语。锚定词本身可能只是通用短语（「学生申请」能
            # 锚到转学条款），没有第二术语佐证就是锚错了地方。
            corroborated = any(
                t != anchor and t not in anchor and anchor not in t and t in answer
                for t in strong
            )
            if corroborated:
                return answer
    return None


_STALE_TITLE = ("大事记", "校史", "历史沿革", "光辉历程", "年鉴", "回忆")

# 时效问题里，这些栏目名代表「当前口径」，可以加速。
_CURRENT_TITLE = ("学校简介", "学校章程", "现任领导", "概况", "学科体系")


def _hit_recency_factor(hit: dict[str, Any], question: str) -> float:
    """问句带时效约束时，历史资料和老新闻降权。年份判断优先于栏目加速。

    预决算报告每年一本，不加速（0.9），让最新的学校简介/学生手册（1.4）
    自然压过它们；不带年份的版本说明页（学位授权、质量报告）保持 1.0。
    """
    if not has_recency_constraint(question):
        return 1.0
    title = hit.get("doc_title") or ""
    if any(k in title for k in _STALE_TITLE):
        return 0.25
    years = [int(y) for y in re.findall(r"20(\d{2})", title)]
    if years and max(years) < 23:
        return 0.4
    if any(k in title for k in _CURRENT_TITLE):
        return 1.4
    if any(k in title for k in ("决算", "预算")):
        return 0.9
    # 正文没有时效词的版本说明页（学位授权、质量报告）数据偏旧，
    # 轻微降权让最新的学校简介/学生手册压过它们。
    text = hit.get("text") or ""
    if not re.search(r"截至|目前|现今", text[:3000]):
        return 0.9
    return 1.0


# 内容型关键文档：时效问题里正文通常含「截至…」的现状描述。
_KEY_DOC_TITLES = ("学校简介", "学生手册", "学校章程")


def _year_factor(question: str, hit: dict[str, Any]) -> float:
    """问句限定年份（「2026年招生…」）：文档标题带同年份的优先，
    旧年份的同名章程/简章降权，防止 2020 旧政策压过 2026 新政策。"""
    qy = YEAR_RE.search(question)
    if not qy:
        return 1.0
    title = hit.get("doc_title") or ""
    years = YEAR_RE.findall(title)
    if not years:
        return 1.0
    if qy.group() in years:
        return 1.3
    return 0.6


def _gather(hits: list[dict[str, Any]], terms: list[str], qtype: str,
            question: str, max_sent: int = 3,
            pieces: list[str] | None = None) -> list[tuple[float, str, dict[str, Any]]]:
    """跨所有证据块收集打分过的句子，按分数排序，去重。

    检索没把关键文档（学校简介/学生手册）排上来时，直接到库里捞，
    用内容词做轻量过滤。这类文档是时效问题的「当前口径」来源。
    """
    all_hits = list(hits)
    if has_recency_constraint(question):
        have = {(h.get("doc_title") or "") for h in hits}
        missing = [t for t in _KEY_DOC_TITLES if not any(t in x for x in have)]
        if missing:
            strong = [t for t in terms if len(t) >= 4][:2]
            if strong:
                all_hits += _fetch_key_docs(missing, strong)
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for hit in all_hits:
        text = hit.get("text") or ""
        factor = _hit_recency_factor(hit, question) * _year_factor(question, hit)
        for sentence in split_sentences(text):
            if len(sentence) > 480:
                # 超长句按逗号再切一刀，避免整段当选。
                parts = [p for p in re.split(r"(?<=，)|(?<=、)", sentence) if len(p.strip("，,")) >= 8]
                candidates = parts if parts else [sentence[:480]]
            else:
                candidates = [sentence]
            for cand in candidates:
                s = score_sentence(cand, terms, qtype, question, pieces) * factor
                if s > 0:
                    scored.append((s, cand.strip("，,"), hit))
    # 同分时按文档新旧排：标题/正文带更近年份的排前面，
    # 防止「3.4万（旧）」和「3.6万（新）」同分时旧版碰巧排前。
    def _doc_year(item: tuple[float, str, dict[str, Any]]) -> int:
        title = item[2].get("doc_title") or ""
        years = [int(y) for y in YEAR_RE.findall(title)]
        return max(years) if years else 0
    scored.sort(key=lambda item: (-item[0], -_doc_year(item)))
    # 内容近似去重：前 40 字相同视为同句。不能用更短的 key——
    # 不同年份版本的句子前 20 字往往相同，数字差异在 20-30 字处
    # （「硕士专业学位授权点（类别）12 个」vs「14 个」），短 key 会把
    # 最新版本错杀成重复。
    seen: set[str] = set()
    uniq: list[tuple[float, str, dict[str, Any]]] = []
    for item in scored:
        norm = re.sub(r"[\s，,。；;]", "", item[1])
        key = norm[:40]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(item)
        if len(uniq) >= max_sent * 4:
            break
    # Cross-Encoder 精排：规则分保持主导，CE 只做护栏。
    # CE 是同模型内可比的分对数（logit），但版本号差异（12 个 vs
    # 14 个）它分不出来——所以不能混合重排，只做两件事：
    #  1. 否决：CE 比短名单最优低 4 分以上的候选，规则分减半；
    #  2. 确认：规则分本就不低（≥最优的 50%）且 CE 最高的候选 +15%。
    # 模型缺失时静默跳过，整套规则行为不变。
    if len(uniq) > 1:
        try:
            from . import rerank
            short = uniq[:24]
            ce = rerank.ce_scores(question, [s for _, s, _ in short])
            if ce:
                top_ce = max(ce)
                top_rule = max(s for s, _, _ in short)
                adjusted: list[tuple[float, str, dict[str, Any]]] = []
                for (rs, sent, hit), cs in zip(short, ce):
                    if cs < top_ce - 4.0:
                        adjusted.append((rs * 0.5, sent, hit))
                    elif cs >= top_ce - 0.5 and rs >= top_rule * 0.5:
                        adjusted.append((rs * 1.15, sent, hit))
                    else:
                        adjusted.append((rs, sent, hit))
                adjusted.sort(key=lambda item: (-item[0], -_doc_year(item)))
                uniq = adjusted + uniq[24:]
        except Exception:
            pass
    return uniq


def _fetch_key_docs(titles: list[str], strong_terms: list[str],
                    limit: int = 4) -> list[dict[str, Any]]:
    """从库里捞关键文档（学校简介/学生手册）中含核心词的检索单元。"""
    try:
        from . import db as _db
    except ImportError:
        return []
    out: list[dict[str, Any]] = []
    try:
        conn = _db.connect()
    except Exception:
        return []
    try:
        for title in titles:
            for term in strong_terms:
                rows = conn.execute(
                    "SELECT d.title AS doc_title, r.text, r.locator, r.year "
                    "FROM retrieval_units r JOIN documents d ON d.id = r.document_id "
                    "WHERE d.title LIKE ? AND r.text LIKE ? LIMIT ?",
                    (f"%{title}%", f"%{term}%", limit),
                ).fetchall()
                for row in rows:
                    out.append({"doc_title": row["doc_title"], "text": row["text"],
                                "locator": row["locator"], "year": row["year"]})
    except Exception:
        return out
    return out


def _compose_list(ranked: list[tuple[float, str, dict[str, Any]]],
                  terms: list[str], hits: list[dict[str, Any]],
                  max_sent: int = 4) -> str | None:
    """列举题：先找「术语锚定的枚举组」，否则贪心集覆盖。"""
    anchored = _anchored_enumeration(hits, terms)
    if anchored:
        ans = _trim(anchored, 420)
        return ans if ans.endswith("。") else ans + "。"
    covered: set[str] = set()
    picked: list[str] = []
    # 目标集按包含关系去重（双向子串）：滑窗碎片不算独立目标。
    raw_target = sorted({t for t in terms if len(t) >= 3}, key=len, reverse=True)
    target: set[str] = set()
    for t in raw_target:
        if any(t != u and (t in u or u in t) for u in target):
            continue
        target.add(t)

    def covered_terms(sentence: str) -> set[str]:
        return {t for t in target if t in sentence or t in _gram_hits(sentence, [t])}

    # 贪心集覆盖：只有带来新覆盖的句子才有资格入选。
    greedy: list[tuple[float, str, dict[str, Any]]] = []
    for score, sentence, hit in ranked:
        new = {t for t in covered_terms(sentence) if t not in covered}
        if not new and greedy:
            continue
        covered |= new
        greedy.append((score, sentence, hit))
        if len(greedy) >= max_sent or (target and covered >= target):
            break
    if not greedy:
        return None
    # 覆盖太低的列举不可信。
    if target and len(covered) / len(target) < 0.3:
        return None
    # 首句与后续句分数差太多（>2.2 倍）时，后续句多是搭车噪声，截掉。
    # 「十佳评选办法」这类办法性条款，后续句本身也近满分，会保留，合理。
    top = greedy[0][0]
    picked = [g[1] for g in greedy if g[0] >= top / 2.2] or [greedy[0][1]]
    return "；".join(_trim(s, 160) for s in picked) + "。"


_TABLE_ROW = re.compile(r"\|")


def _compose_statement(ranked: list[tuple[float, str, dict[str, Any]]],
                       max_sent: int = 2) -> str | None:
    """STATEMENT/WHEN/WHO：取最佳句；表格转成的行（多个 |）是噪声，跳过。"""
    clean = [r for r in ranked if len(_TABLE_ROW.findall(r[1])) < 3]
    if not clean:
        clean = ranked
    best = _trim(clean[0][1])
    if len(best) < 60 and len(clean) > 1 and clean[1][0] >= clean[0][0] * 0.6:
        if clean[1][2] is clean[0][2]:
            best += "。" + _trim(clean[1][1], 120)
    return best + "。"


def _compose_number(ranked: list[tuple[float, str, dict[str, Any]]],
                    groups: list[str], question: str,
                    terms: list[str] | None = None) -> str | None:
    """数字题：单问取最佳句；「分别」多问按属性组各找一句。

    最佳句与次佳句同分但内容不同（年份版本不同）时，合并展示，
    不武断选一边。
    """
    if groups and len(groups) >= 2:
        parts: list[str] = []
        for group in groups:
            base = [group] + [group[i:i + size] for size in (4, 3)
                              for i in range(max(0, len(group) - size + 1))]
            gterms = [v for t in base for v in _cn_variants(t)]
            best: tuple[float, str] | None = None
            for score, sentence, _ in ranked:
                if not any(g in sentence or _loose_contains(sentence, g) for g in gterms):
                    continue
                nums = [n for n in NUM_RE.findall(sentence) if not YEAR_RE.fullmatch(n)]
                if not nums:
                    continue
                if best is None or score > best[0]:
                    best = (score, sentence)
            if best:
                parts.append(_trim(best[1], 120))
        if len(parts) >= 2:
            return "；".join(parts) + "。"
        return None
    for score, sentence, _ in ranked:
        if len(_TABLE_ROW.findall(sentence)) >= 3:
            continue
        nums = [n for n in NUM_RE.findall(sentence) if not YEAR_RE.fullmatch(n)]
        if nums:
            # 同分（±10%）的候选句若核心词相同但数字不同：多为年份
            # 版本差异，并列展示，不武断选一边。
            norm = re.sub(r"[\s，,。；;0-9]", "", sentence)
            peers = [s for sc, s, _ in ranked[1:5]
                     if sc >= score * 0.90 and len(_TABLE_ROW.findall(s)) < 3
                     and re.sub(r"[\s，,。；;0-9]", "", s)[:20] == norm[:20]
                     and re.sub(r"[\s，,。；;0-9]", "", s) != norm]
            if peers:
                merged = "；".join(dict.fromkeys([_trim(sentence, 100)] + [_trim(p, 100) for p in peers[:2]]))
                return merged + "。（不同年份口径数据略有差异，以学校最新公布为准）"
            return _trim(sentence) + "。"
    return None


def _compose_yesno(ranked: list[tuple[float, str, dict[str, Any]]],
                   groups: list[str]) -> str | None:
    """是否题：单项取首个有极性的句；「能否A、B或C」每项各找一句。"""
    if groups and len(groups) >= 2:
        parts: list[str] = []
        polarities: set[str] = set()
        for group in groups:
            gterms = [v for t in [group] for v in _cn_variants(t)]
            gterms += [group[i:i + size] for size in (4, 3) for i in range(max(0, len(group) - size + 1))]
            for score, sentence, _ in ranked:
                if not any(g in sentence or _loose_contains(sentence, g) for g in gterms):
                    continue
                if any(k in sentence for k in ("严禁", "不得", "不可以", "不予", "禁止")):
                    polarities.add("no")
                    parts.append(_trim(sentence, 120))
                    break
                if any(k in sentence for k in ("可以", "允许")):
                    polarities.add("yes")
                    parts.append(_trim(sentence, 120))
                    break
        if len(parts) >= 2:
            verdict = "均不可以。" if polarities == {"no"} else ("均可以。" if polarities == {"yes"} else "")
            return verdict + "；".join(parts) + "。"
        return None
    # 单项是否：极性句本身要有足够的词覆盖（分数），不能见「可以」就答。
    # 「执行/实行/按…执行」是肯定极性：问「是否执行加分政策」，
    # 答句「执行各省级招生主管部门公布的照顾加分政策」就是肯定。
    floor = ranked[0][0] * 0.5 if ranked else 0.0
    for score, sentence, _ in ranked:
        if score < floor:
            continue
        if any(k in sentence for k in ("严禁", "不得", "不可以", "不予", "禁止", "不执行", "不再", "不设", "不再受理")):
            return "不可以。" + _trim(sentence) + "。"
        if any(k in sentence for k in ("可以", "允许", "应当", "须", "执行", "实行", "适用")):
            return "可以。" + _trim(sentence) + "。"
    return None


def extract_answer(question: str, hits: list[dict[str, Any]]) -> str | None:
    """主入口：从证据块里抽出可直接当答案的短句。

    返回 None 表示这层没把握，交给上层（LLM 或原文兜底）。
    """
    if not hits:
        return None
    qtype = question_type(question)
    terms = content_terms(question)
    if not terms:
        return None
    pieces = question_pieces(question)
    groups = attribute_groups(question)
    ranked = _gather(hits, terms, qtype, question, pieces=pieces)
    if not ranked:
        return None
    # 置信门：答案句必须命中足够的内容词，否则宁交回上层兜底。
    # 抽取层只抽一句，把完整规则链截成半句反而错——没把握就交回。
    strong_terms = [t for t in terms if len(t) >= 3]
    if strong_terms:
        top = ranked[0][1]
        top_hits = _term_hits(top, strong_terms) + _gram_hits(top, strong_terms)
        coverage = _coverage(top_hits) / max(1.0, _coverage(strong_terms))
        # 命中比例太低，且没有 5 字以上的特征词命中时，不答。
        distinctive = any(len(t) >= 5 and (t in top or t in _gram_hits(top, [t])) for t in strong_terms)
        if coverage < 0.4 and not distinctive:
            return None
    # 多问/规则链问题（「由哪两部分组成」「如何排序」「分别什么规则」），
    # 单句装不下完整答案，抽取硬截反而错，交回上层取完整段落。
    if qtype in ("STATEMENT", "NUMBER") and re.search(
            r"哪[一二两三四五六七八九十\d]+|分别|如何|怎么|规则|流程|手续|构成|组成|计算|依次", question):
        # 只有首句覆盖度明显领先（>1.5 倍次句）且自身够长才抽，否则交回。
        if not (len(ranked) == 1 or ranked[0][0] > ranked[1][0] * 1.5 and len(ranked[0][1]) >= 40):
            return None
    # LIST 里带「依次/分别/排名」的也是要排顺序的规则链，锚定枚举
    # 找不到时硬抽首句必错，交回（缓考三类这类有锚定枚举的不受影响）。
    if qtype == "LIST" and re.search(r"依次|分别|排序|排名|比较", question):
        return None
    # 复合问题：「，」后紧跟第二个疑问点（多少/什么/哪/是否/电话/地址/
    # 路径/入口），或「和/与」连接两个并列名词性疑问（「标识码和招生代码」）。
    # 语气停顿的「，」（「考试中，」「大学，」）不算复合。
    # 复合的关键是「，」前后各有一个独立疑问点。「，哪三类/，哪三个」
    # 是主问本身（前面是背景状语），不是第二问；「，联系电话是什么」
    # 「，周开放时间是多少」才是第二问——特点是逗号后是名词性主语+疑问词。
    compound = bool(re.search(r"，[一-鿿]{2,10}(?:是什么|是多少|是多少|以什么为准|如何|怎么|如何计算)", question))
    compound = compound or bool(re.search(r"(?:是什么|是多少|哪些|何种)[一-鿿]{0,6}(?:和|与)[一-鿿]{2,12}(?:是什么|是多少|以什么|如何|怎么)", question))
    if qtype in ("STATEMENT", "LIST") and compound:
        return None
    # 泛化性原则：抽取层只接它确实赢的高把握场景，其余一律交回上层
    # （LLM 或整段原文兜底）。评测表明 LIST/YESNO/STATEMENT 硬抽会
    # 把本来对的整段答案换错句，净效果是负的。
    #
    # 保留三类：
    #  1. LIST 且能锚定到原文枚举组（缓考三类、纪律处分种类）；
    #  2. NUMBER 单问且首句明显领先（数字+焦点词绑定，校区/授权点/校训数）；
    #  3. STATEMENT 且首句命中定义/美誉信号（「被誉为」「校训是」）且明显领先。
    # 「明显领先」的判定：首句比次句高 1.4 倍，或首句与次句是同源
    # 多版本（前 20 字规范化后相同）——同义句重复不算有竞争对手。
    # NUMBER 题里同源判定须连数字一起比：「3.4万」和「3.6万」文字
    # 相同但数字不同，是版本竞争，不是同义重复。
    def _clear_lead(ignore_numbers: bool = True) -> bool:
        if len(ranked) == 1:
            return True
        if ranked[0][0] > ranked[1][0] * 1.4:
            return True
        pat = r"[\s，,。；;0-9]" if ignore_numbers else r"[\s，,。；;]"
        # 同源多版本常有语气词差异（「总学分一般不得」vs「总学分不得」），
        # 比较前剥掉填充词，否则同义重复被误判成竞争句。
        filler = r"(?:一般|通常|的|了|在|应当|应该)"
        norm0 = re.sub(filler, "", re.sub(pat, "", ranked[0][1]))[:20]
        norm1 = re.sub(filler, "", re.sub(pat, "", ranked[1][1]))[:20]
        return norm0 == norm1
    if qtype == "LIST":
        anchored = _anchored_enumeration(hits, terms)
        if anchored:
            ans = _trim(anchored, 420)
            return ans if ans.endswith("。") else ans + "。"
        # 非枚举列举（校区、专业门类）：首句领先且覆盖足够时取首句。
        if _clear_lead() and len(ranked[0][1]) >= 30:
            return _trim(ranked[0][1]) + "。"
        return None
    if qtype == "NUMBER":
        if not groups and _clear_lead(ignore_numbers=False):
            return _compose_number(ranked, groups, question, terms)
        return None
    if qtype == "STATEMENT":
        top = ranked[0][1]
        # 定义/美誉/标识信号：「被誉为」「校训」「X是…」「X：…」。
        # 这类信号本身足够强（问句就是冲着它来的），不要求首句领先。
        definitional = re.search(r"被誉为|校训|校风|南邮精神|是新中国|是一所", top)
        if definitional:
            return _compose_statement(ranked)
        if _clear_lead() and re.search(r"^[一-鿿]{2,12}[是为：:]", top):
            return _compose_statement(ranked)
        return None
    # WHEN/WHO/YESNO：一律交回上层。
    return None


def answer_coverage(question: str, answer: str, hits: list[dict[str, Any]]) -> float:
    """答案质量校验：内容词被答案覆盖的比例。用于给 LLM 输出打分。"""
    terms = [t for t in content_terms(question) if len(t) >= 3]
    if not terms:
        return 1.0
    flat = re.sub(r"[\s，,。；;]", "", answer or "")
    covered = sum(1 for t in terms if t in flat)
    return covered / len(terms)
