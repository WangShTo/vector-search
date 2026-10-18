"""把三条检索路线在两套查询上的结果汇总成一份对比，供报告渲染。

## 为什么要汇总

单看任何一套查询都会得出偏颇结论：

- 主评测集（150 条）是**已知项检索**，查询直接取自原文 —— 词面查询，BM25 天然占优
- 语义评测集（16 条）是**人工改写**的提问 —— 语义查询，向量检索占优

必须两套一起看，才能回答「到底该用哪个模式」这个真问题。

## 用法

    python eval/compare_modes.py
    # 产出 bench/results/mode-comparison.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

MODES = ["vector", "bm25", "hybrid"]


def latest_eval_by_mode(results_dir: Path) -> dict:
    """每个模式取最新一次的评测结果（文件名形如 <时间戳>-<mode>.json）。"""
    latest: dict[str, tuple[str, Path]] = {}
    for mode in MODES:
        candidates = sorted(results_dir.glob(f"*-{mode}.json"))
        if candidates:
            newest = candidates[-1]  # 时间戳前缀递增，排序即时间序
            latest[mode] = (newest.stem.rsplit("-", 1)[0], newest)
    return latest


def main() -> int:
    results_dir = Path("bench/results")
    semantic_path = results_dir / "semantic-eval.json"
    if not semantic_path.exists():
        print(f"[错误] 找不到 {semantic_path}，先跑 evaluate_semantic.py", file=sys.stderr)
        return 1

    lexical_raw = latest_eval_by_mode(results_dir)
    if len(lexical_raw) < len(MODES):
        missing = [m for m in MODES if m not in lexical_raw]
        print(f"[错误] 缺少这些模式的主评测结果: {missing}", file=sys.stderr)
        print("       先对每个模式跑一遍 evaluate.py --mode <m> --label <m>", file=sys.stderr)
        return 1

    lexical = {}
    for mode, (_, path) in lexical_raw.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        lexical[mode] = {
            "recall": data["recall"],
            "mrr": data["mrr"],
            "ndcg": data["ndcg"],
            "p50_total_ms": data["latency"]["total_p50"],
            "p50_embedding_ms": data["latency"]["embedding_p50"],
            "query_count": data["query_count"],
        }

    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))

    # 关键结论：平均表现与最差表现
    summary = {}
    for mode in MODES:
        lex = lexical[mode]["recall"]
        sem = semantic[mode]["hit"]
        summary[mode] = {
            "average": (lex + sem) / 2,
            "worst": min(lex, sem),
        }

    comparison = {
        "lexical": {
            "description": "150 条已知项查询（取自文档原文，偏词面匹配）",
            "metrics": lexical,
        },
        "semantic": {
            "description": "16 条人工改写查询（按来源文件判定相关性）",
            "metrics": semantic,
        },
        "summary": summary,
    }

    # ---------- 精排前后对比 ----------
    # 精排的作用是"重新排序"，不是"召回更多"，所以预期是：
    # recall / 命中率基本不变，而 MRR / NDCG 明显提升。
    rerank_lexical_files = sorted(results_dir.glob("*-hybrid-rerank.json"))
    rerank_semantic_path = results_dir / "semantic-eval-rerank.json"
    if rerank_lexical_files and rerank_semantic_path.exists():
        rerank_lexical = json.loads(rerank_lexical_files[-1].read_text(encoding="utf-8"))
        rerank_semantic = json.loads(rerank_semantic_path.read_text(encoding="utf-8"))
        comparison["rerank"] = {
            "lexical": {
                "baseline": {
                    "recall": lexical["hybrid"]["recall"],
                    "mrr": lexical["hybrid"]["mrr"],
                    "ndcg": lexical["hybrid"]["ndcg"],
                    "p50": lexical["hybrid"]["p50_total_ms"],
                },
                "reranked": {
                    "recall": rerank_lexical["recall"],
                    "mrr": rerank_lexical["mrr"],
                    "ndcg": rerank_lexical["ndcg"],
                    "p50": rerank_lexical["latency"]["total_p50"],
                },
            },
            "semantic": {
                "baseline": {
                    "hit": semantic["hybrid"]["hit"],
                    "mrr": semantic["hybrid"]["mrr"],
                    "p50": semantic["hybrid"]["p50"],
                },
                "reranked": {
                    "hit": rerank_semantic["hybrid"]["hit"],
                    "mrr": rerank_semantic["hybrid"]["mrr"],
                    "p50": rerank_semantic["hybrid"]["p50"],
                },
            },
        }

    output = results_dir / "mode-comparison.json"
    output.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    print("模式对比已汇总 -> " + str(output))
    print()
    print(f"  {'模式':<10}{'词面 recall@10':>16}{'语义 命中率@10':>18}{'平均':>10}{'最差':>10}")
    print("  " + "-" * 62)
    for mode in MODES:
        s = summary[mode]
        print(f"  {mode:<10}{lexical[mode]['recall']:>16.4f}{semantic[mode]['hit']:>18.4f}"
              f"{s['average']:>10.4f}{s['worst']:>10.4f}")

    best_avg = max(MODES, key=lambda m: summary[m]["average"])
    best_worst = max(MODES, key=lambda m: summary[m]["worst"])
    print()
    print(f"  平均最好: {best_avg}    最差情况最好: {best_worst}")

    if "rerank" in comparison:
        print()
        print("  精排前后（hybrid）：")
        for name, label in [("lexical", "词面"), ("semantic", "语义")]:
            r = comparison["rerank"][name]
            print(f"    {label}: MRR {r['baseline']['mrr']:.4f} -> {r['reranked']['mrr']:.4f}"
                  f"   P50 {r['baseline']['p50']:.0f}ms -> {r['reranked']['p50']:.0f}ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
