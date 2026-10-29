"""结构化表格问答：直接查 table_rows/tables，绕开 PDF 文本乱序。

PDF 抽文本会把表格打成无序碎片（「百分制90-100 80-89…绩点系数
4.0-5.0…」粘成一行），句子级抽取在这种证据上必错。但采集阶段已经
把表格按 (行, 列) 存进了 table_rows.cells_json，列对齐关系完好。

通用算法，不认识具体题目：
1. 用问句内容词检索候选表（标题/表头/首列命中越多越好）；
2. 在表格里定位「键格」——问句里出现的那个具体值（90-100、优秀、
   文科类）；
3. 定位「属性标签」——问句问的那个维度（绩点系数、学费标准），它
   要么是第一列的行标签（列映射表），要么是表头的列标签（行映射表）；
4. 行列交叉取值。多槽位问题（「文、理、工分别多少」）对每个槽位
   各取一次，全部命中才作答。

没把握一律返回 None（交回句子抽取/整段兜底），宁可不答不可错答。
"""

from __future__ import annotations

import json
import re
from typing import Any

from .db import init_db

_DASHES = re.compile(r"[—–−-]")
_SPACES = re.compile(r"\s+")
# 数字区间键：问句的「90—100」对应表里的「90-100」。
_RANGE_KEY = re.compile(r"\d+(?:\.\d+)?\s*[—–−-]\s*\d+(?:\.\d+)?")


def _norm(text: str) -> str:
    """破折号/空白归一：问句的「90—100」和表里的「90-100」是同一个键。"""
    return _SPACES.sub("", _DASHES.sub("-", text or ""))


def _repair_number(val: str) -> str:
    """修复 PDF 抽取错序的数字格。

    2026 版手册的 PDF 把「4.5」抽成「45 .」（数字先行、标点殿后），
    「4.0-5.0」抽成「40 50 . - .」。模式很固定：尾部游离句点回到
    末位数字前；两段式的连字符也一并归位。修不动就原样返回。
    """
    v = (val or "").strip()
    m = re.fullmatch(r"(\d+(?:\s?\d+)*?)(\d)\s*\.\s*", v)
    if m and " " not in m.group(1):
        return f"{m.group(1)}.{m.group(2)}"
    m = re.fullmatch(r"(\d+)\s*(\d)\s+(\d+)\s*(\d)\s*\.\s*[-–—~]\s*\.\s*", v)
    if m:
        return f"{m.group(1)}.{m.group(2)}-{m.group(3)}.{m.group(4)}"
    return v


def _grid(conn: Any, table_id: int) -> tuple[list[list[str]], int]:
    """取整张表的二维网格 + 真表头行号。

    首行常是通栏标题（「2026年收费公示」占满整行），真正的列标签在
    第二行。判定：前四行里第一行有 ≥2 个互不相同的非空格子。
    """
    rows = conn.execute(
        "SELECT row_index, role, cells_json FROM table_rows WHERE table_id=? ORDER BY row_index",
        (table_id,),
    ).fetchall()
    grid: list[list[str]] = []
    for _, _, cells in rows:
        try:
            grid.append([str(c or "") for c in json.loads(cells or "[]")])
        except Exception:
            grid.append([])
    header_i = -1
    for i, cells in enumerate(grid[:4]):
        nonempty = [c.strip() for c in cells if c.strip()]
        if len({*nonempty}) >= 2 and not all(re.fullmatch(r"\d+", c) for c in nonempty):
            header_i = i
            break
    return grid, header_i


def _find_cells(grid: list[list[str]], key: str) -> list[tuple[int, int]]:
    """键格定位：归一后相等，或格子完整包含键（键≥3 字防误配）。"""
    nk = _norm(key)
    if len(nk) < 2:
        return []
    out: list[tuple[int, int]] = []
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            nc = _norm(cell)
            if nc and (nc == nk or (len(nk) >= 3 and nk in nc)):
                out.append((r, c))
    return out


def _cell(grid: list[list[str]], r: int, c: int) -> str:
    if 0 <= r < len(grid) and 0 <= c < len(grid[r]):
        return grid[r][c].strip()
    return ""


def _is_label_cell(grid: list[list[str]], r: int, c: int, header_i: int) -> bool:
    """首列行标签和表头行都是标签位，不能当答案值。"""
    return r == header_i or c == 0


def _doc_info(conn: Any, table_id: int) -> tuple[str, str, int]:
    row = conn.execute(
        """SELECT d.title, d.url FROM tables t JOIN documents d ON d.id=t.document_id
           WHERE t.id=?""",
        (table_id,),
    ).fetchone()
    if not row:
        return "", "", 0
    years = [int(y) for y in re.findall(r"20(\d{2})", row[0] or "")]
    return row[0] or "", row[1] or "", (max(years) if years else 0)


def _candidate_tables(conn: Any, terms: list[str], limit: int = 24) -> list[int]:
    """按内容词召回候选表：行文本命中加权，标题命中双倍。

    长 n-gram（「绩点系数范围」）几乎不会原文命中，召回要靠中长词
    （「绩点系数」「百分制」），所以长短各取一批，不能被长词挤掉。
    """
    long_terms = [t for t in terms if len(t) >= 5][:4]
    mid_terms = [t for t in terms if 3 <= len(t) <= 4][:10]
    strong = list(dict.fromkeys(long_terms + mid_terms))
    if not strong:
        return []
    scored: dict[int, int] = {}
    for term in strong:
        like = f"%{term}%"
        for (tid,) in conn.execute(
            "SELECT DISTINCT table_id FROM table_rows WHERE text LIKE ? LIMIT 60",
            (like,),
        ):
            scored[tid] = scored.get(tid, 0) + len(term)
        for (tid,) in conn.execute(
            "SELECT id FROM tables WHERE title LIKE ? OR caption LIKE ? LIMIT 20",
            (like, like),
        ):
            scored[tid] = scored.get(tid, 0) + len(term) * 2
    return [t for t, _ in sorted(scored.items(), key=lambda kv: -kv[1])[:limit]]


def _slot_candidates(question: str) -> list[list[str]]:
    """多槽位问题的槽位候选组：「文科、理科、工科、艺术类专业」「仙林
    校区和三牌楼校区」「二年级、三年级」里枚举出的短类目词。

    枚举起点没有标点锚定时（「新生文科、理科…」），首段会吃进前文，
    逐字剥前缀生成多个候选组，交给取值层用「全部槽位都有唯一值」
    来裁决哪组是对的。
    """
    m = re.search(r"((?:[一-鿿]{2,6}、){1,7}[一-鿿]{2,6}?)(?:[和与][一-鿿]{2,6})?[一-鿿]{0,10}(?:，[一-鿿]{0,12})?分别", question)
    if not m:
        m = re.search(r"([一-鿿]{2,6}[和与][一-鿿]{2,6}?)[一-鿿]{0,10}(?:，[一-鿿]{0,12})?分别", question)
    if not m:
        return []
    parts = [p.strip() for p in re.split(r"[、，]|[和与]", m.group(1)) if p.strip()]
    if len(parts) < 2:
        return []
    groups: list[list[str]] = []
    first = parts[0]
    for trim in range(0, min(5, len(first) - 2) + 1):
        head = re.sub(r"^(?:本校|普通|本科|新生|全日制|学生|读者)", "", first[trim:])
        group = [head, *parts[1:]]
        if all(2 <= len(p) <= 8 for p in group) and group not in groups:
            groups.append(group)
    return groups


def _values_in_table(grid: list[list[str]], header_i: int,
                     key: str, attrs: list[str]) -> list[tuple[str, str]]:
    """一张表里「键 × 属性 → 值」的全部取值（值, 可读标签）。

    两种朝向都试：
      A. 属性是第一列行标签 → 值在属性行、键所在列（绩点表这种交叉表）；
      B. 属性是表头列标签 → 值在键所在行、属性列（学费表这种记录表）。
    同一键多行命中时全部返回，交给上层做一致性判断。
    键和属性必须是两个不同的词：同一个词既当键又当属性，命中的
    多半是导航列表页（标题列×日期列），不是映射表。
    """
    keys = _find_cells(grid, key)
    if not keys:
        return []
    out: list[tuple[str, str]] = []
    nk = _norm(key)
    attrs = [a for a in attrs if _norm(a) != nk]
    na_set = {_norm(a) for a in attrs}
    for attr in attrs:
        na = _norm(attr)
        # A：属性行 × 键列。属性行可能有多行（百分制/五级制各自一行
        # 绩点系数），离键行最近的优先，平手取键行下方（子表的值行
        # 在标签行之下）。
        attr_rows = [r for r, row in enumerate(grid)
                     if r != header_i and row and _norm(row[0]) == na]
        for kr, kc in keys:
            if kc == 0:
                continue
            ranked = sorted(attr_rows, key=lambda r: (abs(r - kr), 0 if r > kr else 1))
            for r in ranked[:1]:
                val = _cell(grid, r, kc)
                nv = _norm(val)
                if val and nv not in na_set and nv != nk and not _is_label_cell(grid, r, kc, header_i):
                    col_head = _cell(grid, header_i, kc)
                    label = attr if _norm(col_head) != na else attr
                    out.append((val, label))
        # B：键行 × 属性列
        if header_i >= 0 and header_i < len(grid):
            attr_cols = [c for c, cell in enumerate(grid[header_i]) if na and na in _norm(cell)]
            for kr, kc in keys:
                if kr == header_i:
                    continue
                for c in attr_cols:
                    if c == kc or c == 0:
                        continue
                    val = _cell(grid, kr, c)
                    nv = _norm(val)
                    if val and nv not in na_set and nv != nk and not _is_label_cell(grid, kr, c, header_i):
                        row_head = _cell(grid, kr, 0) or key
                        out.append((val, f"{row_head}·{attr}"))
    return out


# 列表页表头特征：「标题+发布日期」这种表是索引不是映射，
# 行里的日期是发文时间，不是问句要的事件时间。
_LISTING_HEAD = re.compile(r"发布日期|发布时间|更新时间|发文日期|发文时间")
# 文档名后缀：问句点名「按…招生简章/办法/章程」是在指文档，
# 不是映射键；拿它当键只会命中列表页的（标题，日期）行。
_DOCNAME_SUFFIX = ("简章", "办法", "规定", "章程", "简介", "方案",
                   "通知", "细则", "公示", "公告", "计划")


def answer_from_tables(question: str, terms: list[str]) -> tuple[str, str, str] | None:
    """表查询主入口。返回 (答案, 文档标题, 文档URL) 或 None。

    单键问题取第一个解析出唯一值的键；多槽位问题（分别/和/与枚举）
    每个槽位都要解析出唯一值，任一槽位多值或缺失就整体放弃。
    """
    conn = init_db()
    try:
        tids = _candidate_tables(conn, terms)
        if not tids:
            return None
        grids: dict[int, tuple[list[list[str]], int]] = {}
        for tid in tids:
            grid, header_i = _grid(conn, tid)
            if not grid or header_i < 0:
                continue
            head_cells = {_norm(c) for c in grid[header_i] if c.strip()}
            if any(_LISTING_HEAD.search(c) for c in head_cells):
                continue
            grids[tid] = (grid, header_i)
        if not grids:
            return None
        # 属性标签以中长词为主（「绩点系数」），长 n-gram 碎片
        # （「绩点系数范围」）找不到标签但无害，宽进 12 个。
        attrs = [t for t in terms if 3 <= len(t) <= 8][:12]
        if not attrs:
            return None
        slot_groups = _slot_candidates(question)
        keys: list[str] = []
        if not slot_groups:
            keys = [t for t in terms if len(t) >= 2][:10]
            # 短纯数字（「100」）在别的表里太容易撞格，不配当键；
            # 文档名（…简章/办法/章程）是在指文档，也不是映射键。
            keys = [k for k in keys
                    if not re.fullmatch(r"\d{1,3}", k)
                    and not k.endswith(_DOCNAME_SUFFIX)]
            keys += _RANGE_KEY.findall(question)

        def resolve(key: str) -> tuple[str, str, str, str] | None:
            """跨候选表解析一个键：值必须全局唯一；多值时取最新年份
            文档且唯一，否则放弃。返回 (值, 标签, 标题, URL)。"""
            found: list[tuple[str, str, str, str, int]] = []
            for tid, (grid, header_i) in grids.items():
                title, url, year = _doc_info(conn, tid)
                for val, label in _values_in_table(grid, header_i, key, attrs):
                    if len(val) <= 40:
                        found.append((_repair_number(val), label, title, url, year))
            vals = {_norm(v) for v, *_ in found}
            if not vals:
                return None
            if len(vals) > 1:
                best_year = max(y for *_, y in found)
                latest = [f for f in found if f[4] == best_year]
                if best_year == 0 or len({_norm(f[0]) for f in latest}) != 1:
                    return None
                found = latest
            v, label, title, url, _ = found[0]
            # 修完还带「数字夹空格」的值是乱序残渣，答了必错，宁可放弃。
            if re.search(r"\d\s+\d|\d\s+[.,]|[.,]\s+\d", v):
                return None
            # 导航列表页的「值」常是发文日期；问句不问时间时一律否决。
            if re.search(r"20\d{2}\s*[-./年]", v) and not re.search(
                    r"何时|什么时候|哪年|何年|几月|日期|哪天", question):
                return None
            # 值里带多个空白的整串汉字是导航菜单残渣（「培养方案 专业
            # 介绍 …」），不是映射值。
            if len(re.findall(r"\s+", v)) >= 2 and re.search(r"[一-鿿]", v):
                return None
            return v, label, title, url

        for slots in slot_groups:
            per_slot: list[tuple[str, str]] = []
            title = url = ""
            for slot in slots:
                hit = resolve(slot)
                if not hit:
                    break
                val, _, title, url = hit
                per_slot.append((slot, val))
            else:
                return "；".join(f"{k}为{v}" for k, v in per_slot) + "。", title, url
            continue
        if slot_groups:
            return None
        for key in keys:
            hit = resolve(key)
            if hit:
                val, label, title, url = hit
                return f"{key}对应的{label}为{val}。", title, url
        return None
    except Exception:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass
