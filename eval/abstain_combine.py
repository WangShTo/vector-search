#!/usr/bin/env python3
"""把两个拒答信号组合起来，看能不能优于单信号。

## 为什么单独写一个脚本

`abstain_probe.py` 只收数据、只做单信号的阈值扫描。组合规则的搜索会多一层
二维网格，混在一起会让那个脚本变得难读，而且**不需要再调服务** ——
全部基于已经落盘的 `bench/results/abstain-*.json` 离线算。

## 规则

- `and`：两个信号都低才算不相关（保守，误拒少）
- `or` ：任一信号低就算不相关（激进，拒答多）

## 用法

    python eval/abstain_combine.py --input bench/results/abstain-natural.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def quantiles(values: list[float], n: int) -> list[float]:
    """取 n 个分位点做候选阈值（去重后排序）。"""
    ordered = sorted(values)
    if not ordered:
        return []
    picks = set()
    for i in range(n):
        idx = min(len(ordered) - 1, round(i * (len(ordered) - 1) / max(1, n - 1)))
        picks.add(round(ordered[idx], 6))
    return sorted(picks)


def evaluate(pos: list[dict], neg: list[dict], pair: tuple[str, str], mode: str,
             t1: float, t2: float) -> dict:
    a, b = pair

    def low(row: dict) -> bool:
        va, vb = row.get(a), row.get(b)
        if va is None or vb is None:
            return False
        fa, fb = va < t1, vb < t2
        return (fa and fb) if mode == "and" else (fa or fb)

    refused_pos = sum(1 for r in pos if low(r))
    refused_neg = sum(1 for r in neg if low(r))
    return {
        "rule": mode,
        "signalA": a,
        "thresholdA": t1,
        "signalB": b,
        "thresholdB": t2,
        "refusalRate": round(refused_neg / len(neg), 4) if neg else 0.0,
        "falseRefusalRate": round(refused_pos / len(pos), 4) if pos else 0.0,
        "accuracy": round((refused_neg + len(pos) - refused_pos) / (len(pos) + len(neg)), 4),
    }


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="拒答信号组合")
    parser.add_argument("--input", default="bench/results/abstain-natural.json")
    parser.add_argument("--out", default="bench/results/abstain-combined.json")
    parser.add_argument("--pairs", default="vectorTop1:bm25Top1,vectorTop1:ceTop1,bm25Top1:ceTop1")
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()

    data = json.loads((ROOT / args.input).read_text(encoding="utf-8"))
    records = data["records"]
    pos = [r for r in records if r["label"] == "pos"]
    neg = [r for r in records if r["label"] == "neg"]
    print(f"{args.input}：正样本 {len(pos)} / 负样本 {len(neg)}", flush=True)

    results = []
    for spec in args.pairs.split(","):
        a, b = spec.split(":")
        grid_a = quantiles([r[a] for r in pos + neg if r.get(a) is not None], args.steps)
        grid_b = quantiles([r[b] for r in pos + neg if r.get(b) is not None], args.steps)
        for mode in ("and", "or"):
            for t1 in grid_a:
                for t2 in grid_b:
                    results.append(evaluate(pos, neg, (a, b), mode, t1, t2))

    # 两个挑选口径：整体准确率最高、以及在误拒 ≤ 5% 前提下拒答率最高
    best_acc = max(results, key=lambda r: (r["accuracy"], r["refusalRate"]))
    safe = [r for r in results if r["falseRefusalRate"] <= 0.05]
    best_safe = max(safe, key=lambda r: (r["refusalRate"], r["accuracy"])) if safe else None

    print("\n=== 组合规则（按准确率） ===", flush=True)
    print(f"  {best_acc['signalA']} < {best_acc['thresholdA']} {best_acc['rule'].upper()} "
          f"{best_acc['signalB']} < {best_acc['thresholdB']}", flush=True)
    print(f"  拒答率 {best_acc['refusalRate']:.3f}  误拒率 {best_acc['falseRefusalRate']:.3f}  "
          f"准确率 {best_acc['accuracy']:.3f}", flush=True)

    if best_safe:
        print("\n=== 误拒率 ≤ 5% 前提下的最优 ===", flush=True)
        print(f"  {best_safe['signalA']} < {best_safe['thresholdA']} {best_safe['rule'].upper()} "
              f"{best_safe['signalB']} < {best_safe['thresholdB']}", flush=True)
        print(f"  拒答率 {best_safe['refusalRate']:.3f}  误拒率 {best_safe['falseRefusalRate']:.3f}  "
              f"准确率 {best_safe['accuracy']:.3f}", flush=True)

    print("\n=== 单信号对照（来自 abstain_probe） ===", flush=True)
    for key, res in data["analyses"].items():
        best = res.get("best") or {}
        print(f"  {key:<12} 阈值 {best.get('threshold')}  拒答率 {best.get('refusalRate')}  "
              f"误拒率 {best.get('falseRefusalRate')}  AUC {res.get('auc')}", flush=True)

    out = {
        "input": args.input,
        "positiveCount": len(pos),
        "negativeCount": len(neg),
        "bestByAccuracy": best_acc,
        "bestUnderFalseRefusal5": best_safe,
        "singleSignal": data["analyses"],
        "gridSize": len(results),
    }
    out_path = ROOT / args.out
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n结果已写入 {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
