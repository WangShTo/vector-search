"""检索质量与延迟评测。

## 指标为什么是这几个

- **recall@k** —— 相关文档有没有进前 k。近似索引（HNSW）的核心指标，
  衡量"漏没漏"。但只看它会被"全召回但排序很差"骗过去。
- **MRR@k** —— 第一个相关结果排在第几位。衡量"答得够不够靠前"。
- **NDCG@k** —— 位置加权的排序质量，对"正确答案排第一还是排第五"
  敏感。本评测是二值相关性，所以 NDCG 与 MRR 互补。
- **P50/P95/P99 延迟** —— 平均值会掩盖长尾。线上体验由 P99 决定，
  只看平均延迟的优化等于没做。

## 输出

一份 Markdown 报告，可以直接贴进项目 README 或技术博客 ——
<b>这就是简历上那些数字的出处</b>。面试官问"这个数怎么来的"，
把报告链接发给他。

## 用法

    # 前提：Python 模型服务和 Java 检索服务都已启动，且语料已灌入
    python evaluate.py --queries ../data/eval/queries.jsonl --top-k 10

    # 只测 HNSW 检索耗时，跳过 Python 编码（用缓存向量）以隔离变量
    python evaluate.py --queries ../data/eval/queries.jsonl --repeat 3
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


# ----------------------------------------------------------------------
# HTTP（用标准库，不引入额外依赖）
# ----------------------------------------------------------------------

def post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_service(url: str, timeout_seconds: float = 120.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return
        except Exception:  # noqa: BLE001 - 启动期各种失败都算"还没好"
            pass
        time.sleep(2)
    raise SystemExit(f"[错误] 服务 {url} 在 {timeout_seconds:.0f}s 内没有就绪")


# ----------------------------------------------------------------------
# 指标
# ----------------------------------------------------------------------

def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(set(retrieved[:k]) & relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: list[int], relevant: set[int], k: int) -> float:
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """二值相关性下的 NDCG@k。"""
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    # 用最近秩法而不是插值：延迟分布是长尾，插值会低估 P99
    index = min(len(ordered) - 1, max(0, math.ceil(p / 100 * len(ordered)) - 1))
    return ordered[index]


# ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="检索质量与延迟评测")
    parser.add_argument("--queries", required=True, help="queries.jsonl 路径")
    parser.add_argument("--service", default="http://localhost:8080", help="Java 检索服务地址")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=None,
                        help="覆盖 HNSW 的 efSearch；做 recall/延迟 曲线时用")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条（快速冒烟）")
    parser.add_argument("--repeat", type=int, default=1, help="重复轮数")
    parser.add_argument("--label", default="", help="本次运行的标签，写进报告标题")
    parser.add_argument("--out", default="results", help="报告输出目录")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    queries_path = Path(args.queries)
    if not queries_path.exists():
        print(f"[错误] 找不到 {queries_path}，先跑 build_dataset.py", file=sys.stderr)
        return 1

    queries = [json.loads(line) for line in queries_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        queries = queries[: args.limit]
    if not queries:
        print("[错误] 查询集为空", file=sys.stderr)
        return 1

    wait_for_service(f"{args.service}/api/stats", timeout_seconds=args.timeout)

    stats = json.loads(urllib.request.urlopen(f"{args.service}/api/stats", timeout=10).read())
    indexed = stats.get("indexedCount", 0)
    print(f"服务索引条数: {indexed}")
    if indexed == 0:
        print("[警告] 索引为空，先把 corpus.jsonl 灌进去再评测", file=sys.stderr)

    recalls, mrrs, ndcgs = [], [], []
    embedding_ms, search_ms, total_ms = [], [], []
    failures = 0

    for round_index in range(args.repeat):
        for item in queries:
            payload = {"query": item["query"], "topK": args.top_k}
            if args.ef_search is not None:
                payload["efSearch"] = args.ef_search
            try:
                response = post_json(f"{args.service}/api/search", payload, args.timeout)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                failures += 1
                if failures <= 3:
                    print(f"[失败] {item['query'][:30]}... -> {exc}", file=sys.stderr)
                continue

            retrieved = [hit["id"] for hit in response.get("hits", [])]
            relevant = set(item["relevant_ids"])

            recalls.append(recall_at_k(retrieved, relevant, args.top_k))
            mrrs.append(reciprocal_rank(retrieved, relevant, args.top_k))
            ndcgs.append(ndcg_at_k(retrieved, relevant, args.top_k))

            timing = response.get("timing") or {}
            embedding_ms.append(timing.get("embeddingMs", 0))
            search_ms.append(timing.get("searchMs", 0))
            total_ms.append(timing.get("totalMs", 0))

    if not recalls:
        print("[错误] 没有一条成功，检查服务是否正常", file=sys.stderr)
        return 1

    evaluation_count = len(recalls)
    summary = {
        "label": args.label,
        "index_type": stats.get("indexType"),
        "indexed_count": indexed,
        "top_k": args.top_k,
        "ef_search": args.ef_search,
        "query_count": len(queries),
        "repeat": args.repeat,
        "evaluations": evaluation_count,
        "failures": failures,
        "recall": statistics.fmean(recalls),
        "mrr": statistics.fmean(mrrs),
        "ndcg": statistics.fmean(ndcgs),
        "latency": {
            "embedding_p50": percentile(embedding_ms, 50),
            "embedding_p95": percentile(embedding_ms, 95),
            "search_p50": percentile(search_ms, 50),
            "search_p95": percentile(search_ms, 95),
            "search_p99": percentile(search_ms, 99),
            "total_p50": percentile(total_ms, 50),
            "total_p95": percentile(total_ms, 95),
            "total_p99": percentile(total_ms, 99),
        },
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    slug = f"{stamp}-{args.label}" if args.label else stamp
    (out_dir / f"{slug}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = render_report(summary)
    report_path = out_dir / f"{slug}.md"
    report_path.write_text(report, encoding="utf-8")

    print()
    print(report)
    print(f"报告已写入: {report_path}")
    return 0


def render_report(s: dict) -> str:
    lat = s["latency"]
    ef = s["ef_search"] if s["ef_search"] is not None else "配置默认值"
    return f"""# 检索评测报告{' · ' + s['label'] if s['label'] else ''}

## 运行配置

| 项 | 值 |
|---|---|
| 索引实现 | `{s['index_type']}` |
| 索引条数 | {s['indexed_count']} |
| top-k | {s['top_k']} |
| efSearch | {ef} |
| 查询条数 | {s['query_count']} |
| 重复轮数 | {s['repeat']} |
| 成功评测次数 | {s['evaluations']}（失败 {s['failures']}） |

## 检索质量

| 指标 | 数值 |
|---|---|
| recall@{s['top_k']} | **{s['recall']:.4f}** |
| MRR@{s['top_k']} | **{s['mrr']:.4f}** |
| NDCG@{s['top_k']} | **{s['ndcg']:.4f}** |

## 延迟（毫秒）

| 阶段 | P50 | P95 | P99 |
|---|---|---|---|
| 向量编码（含跨进程调用） | {lat['embedding_p50']:.0f} | {lat['embedding_p95']:.0f} | — |
| 索引检索 | {lat['search_p50']:.0f} | {lat['search_p95']:.0f} | {lat['search_p99']:.0f} |
| 端到端 | {lat['total_p50']:.0f} | {lat['total_p95']:.0f} | {lat['total_p99']:.0f} |

> **先看哪一段是瓶颈**：如果「向量编码」远大于「索引检索」，
> 那么去调 HNSW 参数是南辕北辙，应该先优化跨进程调用（批量化、连接复用）。

## 评测方法的局限（必须写清楚）

本评测采用**已知项检索**：查询取自文档自身的一段文字，
因此查询词与文档词高度重合，**指标是上界，不等于线上效果**。

它适合用来做<b>相对比较</b>（改参数前后、两种索引之间），
不适合用来宣称绝对质量。要得到更可信的数字，需要真实用户查询日志，
或人工标注的查询-文档相关性 —— 那是这个项目后续可以做的方向。
"""


if __name__ == "__main__":
    sys.exit(main())
