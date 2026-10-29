"""辅助模型的显式开关。

默认 auto：模型目录和索引都在就启用，和当前运行路径一致。
消融必须设环境变量，不要靠改名目录：

    ASKNUJPT_EMBEDDING=0   关闭向量召回
    ASKNUJPT_RERANK=0      关闭 Cross-Encoder 护栏
    ASKNUJPT_EMBEDDING=1   强制开启（文件缺失则该路为空，不改其他路径）
"""

from __future__ import annotations

import os


def auxiliary(name: str) -> str:
    """返回 off / on / auto。name 用 EMBEDDING 或 RERANK。"""
    raw = os.environ.get(f"ASKNJUPT_{name}", "").strip().lower()
    if raw in {"0", "off", "false", "no"}:
        return "off"
    if raw in {"1", "on", "true", "yes"}:
        return "on"
    return "auto"
