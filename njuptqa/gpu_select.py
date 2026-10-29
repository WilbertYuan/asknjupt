"""Pick idle GPUs for one inference, then let the process exit so VRAM is released."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuStat:
    index: int
    free_mib: int
    used_mib: int
    util: int


def parse_nvidia_smi(text: str) -> list[GpuStat]:
    stats: list[GpuStat] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.lower().startswith("index"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        stats.append(
            GpuStat(
                index=int(parts[0]),
                free_mib=_mib(parts[1]),
                used_mib=_mib(parts[2]),
                util=int(float(parts[3])),
            )
        )
    return stats


def select_gpus(
    stats: list[GpuStat],
    *,
    max_cards: int = 4,
    need_mib: int = 3072,
) -> list[int]:
    """Use the idlest card that can hold the model.

    A single free card is enough for Qwen3-0.6B. Extra cards are added only when
    no individual card has ``need_mib`` free, and never more than ``max_cards``.
    """
    if max_cards < 1 or not stats:
        return []
    ranked = sorted(stats, key=lambda gpu: (gpu.util, -gpu.free_mib, gpu.index))
    roomy = [gpu for gpu in ranked if gpu.free_mib >= need_mib]
    if roomy:
        return [roomy[0].index]
    chosen: list[int] = []
    total = 0
    for gpu in ranked:
        if gpu.free_mib < 1024:
            continue
        chosen.append(gpu.index)
        total += gpu.free_mib
        if len(chosen) >= max_cards or total >= need_mib:
            break
    if total >= need_mib:
        return chosen[:max_cards]
    return []


def query_gpus() -> list[GpuStat]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_nvidia_smi(out)


def _mib(value: str) -> int:
    return int(float(value.replace("MiB", "").strip()))
