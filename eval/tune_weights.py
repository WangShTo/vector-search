#!/usr/bin/env python3
"""加权 RRF 的权重选择：在调参集上扫网格，在留出集上报分。

## 为什么必须有这一步

等权 RRF 在词面查询上输给纯 BM25（recall@10 0.89 vs 0.96），在语义查询上输给纯向量。
两路可信度并不对称，等权等于让弱的那一路拖累强的那一路。

加权能修正这种不对称 —— 但权重是自由参数：在同一批查询上挑出最好的权重、
再在同一批查询上报分，那个分就不是「样本外表现」，而是在评测集上过拟合的结果。
所以这里把评测集切开，扫权和报分用的是两批不同的查询。

## 做法

- 150 条词面查询按文件顺序切：前 N 条调参（默认 100），其余留出（只报不改）
- 网格只在调参集上扫，按 MRR 选（RRF 是排序融合，MRR 对「对的那条排第几」更敏感）
- 留出集上对比「等权 / 选中权重 / 纯向量 / 纯 BM25」，只有这几组能写进报告
- 16 条语义查询不参与选择（样本太小，容易被调成过拟合），另跑体检

## 用法

    python eval/tune_weights.py --grid 1:1,1:2,1:3,3:1,2:1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVICE = "http://127.0.0.1:8080"
DEV_PS1 = ROOT / "scripts" / "dev.ps1"
PYTHON = sys.executable


def ensure_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass


# ----------------------------------------------------------------------
# 进程与环境
# ----------------------------------------------------------------------

def run(
    cmd: list[str],
    env: dict[str, str] | None = None,
    timeout: float = 900.0,
    capture: bool = True,
) -> str:
    """跑一条命令。

    capture=False 用于 `dev.ps1 start`：它用 Start-Process 拉起守护进程，
    那个子孙进程会一直持有继承来的 stdout 管道，用 capture_output 读会永远等不到 EOF。
    """
    kwargs: dict = {"cwd": str(ROOT), "env": env, "timeout": timeout}
    if capture:
        kwargs.update(capture_output=True, text=True, encoding="utf-8", errors="replace")
    proc = subprocess.run(cmd, **kwargs)
    if proc.returncode != 0:
        detail = ""
        if capture:
            detail = (proc.stdout or "")[-1200:] + (proc.stderr or "")[-1200:]
        raise RuntimeError(f"命令失败（exit={proc.returncode}）：{' '.join(cmd)}\n{detail}")
    return (proc.stdout or "") if capture else ""


def service_ready(timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(f"{SERVICE}/api/stats", timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def restart(weight_vector: float, weight_bm25: float) -> None:
    """改权重只能重启 —— 权重在 Spring 配置里，启动时读一次。"""
    env = dict(os.environ)
    env["VECTOR_SEARCH_VECTOR_WEIGHT"] = str(weight_vector)
    env["VECTOR_SEARCH_BM25_WEIGHT"] = str(weight_bm25)

    try:
        run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(DEV_PS1), "stop"],
            capture=False)
    except RuntimeError:
        pass  # 本来就没在跑时会返回非零，不算错误
    run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(DEV_PS1), "start"],
        env=env, capture=False)
    if not service_ready():
        raise RuntimeError("重启后 /api/stats 不可达")


# ----------------------------------------------------------------------
# 评测
# ----------------------------------------------------------------------

def evaluate(queries: Path, mode: str, label: str, top_k: int) -> dict:
    out_dir = ROOT / "bench" / "results" / "_sweep" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.json"):
        stale.unlink()

    run([
        PYTHON, str(ROOT / "eval" / "evaluate.py"),
        "--queries", str(queries),
        "--service", SERVICE,
        "--top-k", str(top_k),
        "--mode", mode,
        "--label", label,
        "--out", str(out_dir),
    ])

    produced = sorted(out_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not produced:
        raise RuntimeError(f"评测没有产出结果文件：{out_dir}")
    return json.loads(produced[-1].read_text(encoding="utf-8"))


def pick(metrics: dict) -> tuple[float, float, float]:
    return (
        float(metrics.get("mrr", 0.0)),
        float(metrics.get("recall", 0.0)),
        float(metrics.get("ndcg", 0.0)),
    )


# ----------------------------------------------------------------------

def parse_grid(raw: str) -> list[tuple[float, float]]:
    combos: list[tuple[float, float]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        vec, bm = item.split(":")
        combos.append((float(vec), float(bm)))
    if not combos:
        raise ValueError("网格为空")
    return combos


def main() -> int:
    ensure_utf8_stdout()

    parser = argparse.ArgumentParser(description="加权 RRF 权重选择")
    parser.add_argument("--queries", default="data/eval/queries.jsonl")
    parser.add_argument("--grid", default="1:1,1:2,1:3,1:4,2:1,3:1,4:1",
                        help="weight_vector:weight_bm25 列表")
    parser.add_argument("--tune-limit", type=int, default=100, help="调参集大小（按文件顺序取前 N 条）")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--corpus", default="data/eval/corpus.jsonl")
    parser.add_argument("--out", default="bench/results/rrf-weights.json")
    parser.add_argument("--skip-load", action="store_true", help="索引已就绪时跳过重新灌语料")
    args = parser.parse_args()

    queries_path = ROOT / args.queries
    rows = [json.loads(line) for line in queries_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) <= args.tune_limit:
        raise SystemExit(f"查询数 {len(rows)} 不足以切出调参集 {args.tune_limit}")

    tune_rows, test_rows = rows[: args.tune_limit], rows[args.tune_limit:]
    tune_path = ROOT / "data" / "eval" / "_tune.jsonl"
    test_path = ROOT / "data" / "eval" / "_test.jsonl"
    tune_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in tune_rows), encoding="utf-8")
    test_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in test_rows), encoding="utf-8")

    print(f"调参集 {len(tune_rows)} 条 / 留出集 {len(test_rows)} 条（按文件顺序切，不随机）", flush=True)

    if not args.skip_load:
        print("准备索引：重启 → 清空 → 灌评测语料 → 快照", flush=True)
        restart(1.0, 1.0)
        run([PYTHON, str(ROOT / "scripts" / "load_corpus.py"),
             "--service", SERVICE, "--corpus", str(ROOT / args.corpus), "--reset"])
        run(["curl.exe", "-s", "-X", "POST", f"{SERVICE}/api/admin/snapshot"])

    # --- 第一步：只在调参集上扫网格 ---
    grid_results = []
    for weight_vector, weight_bm25 in parse_grid(args.grid):
        label = f"tune-v{weight_vector:g}-b{weight_bm25:g}"
        print(f"  扫参 {weight_vector:g}:{weight_bm25:g} …", end="", flush=True)
        restart(weight_vector, weight_bm25)
        metrics = evaluate(tune_path, "hybrid", label, args.top_k)
        mrr, recall, ndcg = pick(metrics)
        print(f" recall@10 {recall:.4f}  MRR {mrr:.4f}  NDCG {ndcg:.4f}", flush=True)
        grid_results.append({
            "vector": weight_vector,
            "bm25": weight_bm25,
            "recall": recall,
            "mrr": mrr,
            "ndcg": ndcg,
        })

    best = max(grid_results, key=lambda row: (row["mrr"], row["recall"]))
    print(f"调参集上最优：vector={best['vector']:g} bm25={best['bm25']:g}"
          f"（MRR {best['mrr']:.4f}）", flush=True)

    # --- 第二步：留出集上只报不改 ---
    heldout = {}
    plans = [
        ("等权（1:1）", (1.0, 1.0), "hybrid"),
        (f"选中权重（{best['vector']:g}:{best['bm25']:g}）", (best["vector"], best["bm25"]), "hybrid"),
        ("纯向量", (best["vector"], best["bm25"]), "vector"),
        ("纯 BM25", (best["vector"], best["bm25"]), "bm25"),
    ]
    for label, weights, mode in plans:
        print(f"  留出集 {label} …", end="", flush=True)
        restart(*weights)
        metrics = evaluate(test_path, mode, f"test-{mode}-{weights[0]:g}-{weights[1]:g}", args.top_k)
        mrr, recall, ndcg = pick(metrics)
        print(f" recall@10 {recall:.4f}  MRR {mrr:.4f}  NDCG {ndcg:.4f}", flush=True)
        latency = metrics.get("latency", {})
        heldout[label] = {
            "mode": mode,
            "vector_weight": weights[0],
            "bm25_weight": weights[1],
            "recall": recall,
            "mrr": mrr,
            "ndcg": ndcg,
            "p50_total_ms": latency.get("total_p50"),
            "p95_total_ms": latency.get("total_p95"),
        }

    payload = {
        "split": {
            "total_queries": len(rows),
            "tune_queries": len(tune_rows),
            "heldout_queries": len(test_rows),
            "split_rule": "按文件顺序切，前 tune_queries 条调参，其余留出",
        },
        "selection_rule": "调参集上 MRR 最高，MRR 相同比 recall",
        "grid": grid_results,
        "chosen": best,
        "heldout": heldout,
    }

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"结果已写入 {out_path}", flush=True)

    # 扫完把索引恢复成演示语料（否则页面上看到的是评测语料）
    print("恢复演示语料 …", flush=True)
    restart(best["vector"], best["bm25"])
    run(["curl.exe", "-s", "-X", "DELETE", f"{SERVICE}/api/documents"])
    for md in sorted((ROOT / "data" / "corpus-demo").glob("*.md")):
        run(["curl.exe", "-s", "-X", "POST", "-F", f"file=@{md}", f"{SERVICE}/api/documents/upload"])
    run(["curl.exe", "-s", "-X", "POST", f"{SERVICE}/api/admin/snapshot"])
    print("完成。", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
