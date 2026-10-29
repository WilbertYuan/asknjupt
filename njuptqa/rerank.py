"""Cross-Encoder 精排：规则打分短名单上的最终裁决。

规则打分（extractive.score_sentence）负责从几百个候选句里圈出
短名单，但对「哪句真的回答了问题」的细微判断有天花板——
2026 年小模型垂域 RAG 的共识是在短名单上跑 Cross-Encoder：
query 和候选句拼接后联合编码，语义交互远深于独立编码。

用法：
    from .rerank import ce_scores
    scores = ce_scores(question, sentences)  # 与输入等长的 float 列表

模型懒加载，任何失败返回 None，调用方静默降级为纯规则排序。
"""

from __future__ import annotations

import os

_MODEL = None
_FAILED = False
_MODEL_DIR = "models/bge-reranker-v2-m3"


def _root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    global _MODEL, _FAILED
    if _MODEL is not None:
        return _MODEL
    if _FAILED:
        return None
    path = os.path.join(_root(), _MODEL_DIR)
    if not os.path.isdir(path):
        _FAILED = True
        return None
    try:
        from sentence_transformers import CrossEncoder
        try:
            _MODEL = CrossEncoder(path, device="cuda:0", max_length=512)
        except Exception:
            _MODEL = CrossEncoder(path, device="cpu", max_length=512)
        return _MODEL
    except Exception:
        _FAILED = True
        return None


def ce_scores(question: str, sentences: list[str]) -> list[float] | None:
    """对 (问题, 候选句) 对打分。分数越高越相关；失败返回 None。

    ASKNUJPT_RERANK=0 时直接返回 None，规则排序保持原样。
    """
    from .switches import auxiliary
    if auxiliary("RERANK") == "off":
        return None
    model = _load()
    if model is None or not sentences:
        return None
    try:
        pairs = [(question[:256], s[:480]) for s in sentences]
        scores = model.predict(pairs, batch_size=32, show_progress_bar=False)
        return [float(s) for s in scores]
    except Exception:
        return None
