"""NJUPT offline QA system: crawl, parse, retrieve, compute, generate."""

import os

# 向量库在每个请求线程里做矩阵运算时，会按 CPU 核数再开一批线程。
# 连续提问会把这些线程叠上去，进程卡在锁上，页面就一直停在“……”。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__version__ = "0.1.0"
