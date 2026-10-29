"""冻结当前工作区的可复现信息。不提交 git，只写 logs/baseline_freeze.json。"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        if limit is None:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        else:
            h.update(f.read(limit))
    return h.hexdigest()


def main() -> None:
    from njuptqa.semantic import index_coverage
    files = [
        "njuptqa/generate.py",
        "njuptqa/extractive.py",
        "njuptqa/tableqa.py",
        "njuptqa/assemble.py",
        "njuptqa/docresolve.py",
        "njuptqa/evaluate.py",
        "njuptqa/retrieve.py",
        "njuptqa/semantic.py",
        "njuptqa/rerank.py",
        "njuptqa/switches.py",
        "data/db/njupt.sqlite",
        "data/db/embeddings_bge_small_zh.npz",
    ]
    digests = {}
    for rel in files:
        path = ROOT / rel
        digests[rel] = sha256(path) if path.exists() else None
    git = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "sha256": digests,
        "embedding_coverage": index_coverage(),
        "switches": {
            "ASKNJUPT_EMBEDDING": "unset means auto (on when model+index exist)",
            "ASKNJUPT_RERANK": "unset means auto (on when model dir exists)",
            "ASKNJUPT_NO_ASSEMBLE": "1 disables evidence assembly",
            "ASKNJUPT_NO_TABLEQA": "1 disables table lookup",
            "ASKNJUPT_NO_EXTRACT": "1 disables sentence extraction",
        },
        "scoring": {
            "automatic_score": "must_contain / numeric match before any waiver",
            "adjudicated_score": "historical ok, including wording and no_source waivers",
            "no_source": "gold distinctive span absent from retrieval_units; this credits a corpus gap, not a correct answer",
            "known_lifts_before_this_change": {
                "qa1_v21": {"adjudicated": 195, "automatic_strict": 189, "wording": 5, "no_source": 1},
                "qa2_v20": {"adjudicated": 197, "automatic_strict": 126, "wording": 57, "no_source": 14},
            },
        },
        "git_status_short": git.stdout.splitlines()[:200],
    }
    out = ROOT / "logs" / "baseline_freeze.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    cov = payload["embedding_coverage"]
    print(f"wrote {out}")
    print(f"embedding indexed={cov['indexed']} units={cov['units']} missing={cov['missing']} stale={cov['stale']}")


if __name__ == "__main__":
    main()
