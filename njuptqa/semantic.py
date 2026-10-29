"""向量召回 + RRF 融合（混合检索的第二路）。

关键词检索（FTS5/BM25）擅长精确术语，但对「跳级→提前修读」
「教育体系→教育协调发展」这类词汇不匹配的问题原理上无解；
向量召回补上语义这一面。两路各自排名，用 RRF 融合：

    score(d) = Σ 1 / (k + rank_i(d))

不替换原有关键词打分体系，只在召回阶段补充候选块。
模型和索引懒加载，任何一环失败都静默降级为纯关键词检索。
"""

from __future__ import annotations

import os
from typing import Any

_MODEL = None
_INDEX = None
_UNAVAILABLE = False

_MODEL_DIR = "models/bge-small-zh-v1.5"
_INDEX_FILE = "data/db/embeddings_bge_small_zh.npz"
_RRF_K = 60  # RRF 常数，越大越温和


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def available() -> bool:
    """模型和索引都在才启用向量召回。ASKNJUPT_EMBEDDING=0 强制关闭。"""
    global _UNAVAILABLE
    from .switches import auxiliary
    if auxiliary("EMBEDDING") == "off":
        return False
    if _UNAVAILABLE:
        return False
    root = _project_root()
    if not os.path.isdir(os.path.join(root, _MODEL_DIR)):
        _UNAVAILABLE = True
        return False
    if not os.path.isfile(os.path.join(root, _INDEX_FILE)):
        _UNAVAILABLE = True
        return False
    return True


def _load() -> tuple[Any, Any, Any] | None:
    """懒加载模型和 (ids, emb) 索引。加载一次后常驻。"""
    global _MODEL, _INDEX
    if _MODEL is not None and _INDEX is not None:
        return _MODEL, *_INDEX
    if not available():
        return None
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        root = _project_root()
        model = SentenceTransformer(os.path.join(root, _MODEL_DIR), device="cpu")
        data = np.load(os.path.join(root, _INDEX_FILE))
        _MODEL = model
        _INDEX = (data["ids"], data["emb"])
        return _MODEL, *_INDEX
    except Exception:
        return None


def vector_rank(question: str, top: int = 40) -> list[int]:
    """返回按向量相似度排序的 retrieval_unit id 列表（最相关在前）。"""
    loaded = _load()
    if not loaded:
        return []
    model, ids, emb = loaded
    try:
        import numpy as np
        q = model.encode([question[:512]], normalize_embeddings=True,
                         convert_to_numpy=True).astype("float32")[0]
        sims = emb @ q
        if len(sims) <= top:
            order = np.argsort(-sims)
        else:
            part = np.argpartition(-sims, top)[:top]
            order = part[np.argsort(-sims[part])]
        return [int(ids[i]) for i in order]
    except Exception:
        return []


def index_coverage() -> dict[str, Any]:
    """向量索引覆盖了多少检索单元。不重建索引，只报告缺口。"""
    root = _project_root()
    path = os.path.join(root, _INDEX_FILE)
    if not os.path.isfile(path):
        return {"indexed": 0, "units": None, "missing": None, "stale": True}
    import numpy as np
    from .db import init_db
    data = np.load(path)
    indexed = {int(i) for i in data["ids"]}
    conn = init_db()
    try:
        live = {int(i) for (i,) in conn.execute("SELECT id FROM retrieval_units")}
    finally:
        conn.close()
    missing = live - indexed
    extra = indexed - live
    return {
        "indexed": len(indexed),
        "units": len(live),
        "missing": len(missing),
        "extra": len(extra),
        "stale": bool(missing or extra),
    }


def rrf_merge(rank_lists: list[list[int]], limit: int = 60) -> list[int]:
    """多路排名的 RRF 融合。输入各路 id 列表（按相关性排序）。"""
    scores: dict[int, float] = {}
    for ranks in rank_lists:
        for pos, rid in enumerate(ranks):
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (_RRF_K + pos + 1)
    return [rid for rid, _ in sorted(scores.items(), key=lambda kv: -kv[1])][:limit]
