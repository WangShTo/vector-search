"""分块消融：固定查询集，只改切法，看谁把「答案所在的块」排到前面。

## 为什么这样设计

常见的做法是「每个配置各自生成查询」，但那会让两件事同时变化：切法和查询文本。
数字一变，你分不清是切法带来的，还是换了题带来的。

所以这里**只生成一次查询**：在基准配置（paragraph / 400 / 50）上取 150 条
**连续**文本片段（从句子边界开始、只向后拼，不跨块回绕），随后对每个配置
重新切块，找回「包含该片段原文的那个块」作为正确答案。多次比较里只有
「块怎么切」变了。

## 两个要注意的地方

1. **跨块丢弃**：块切得越小，越容易把一句完整的话拆到两个块里，这条查询
   在新配置里就找不到归属，只能丢弃。丢弃数本身就是切法的一个指标。
2. **公共子集**：为了做真正同题的对比，额外报「在所有配置里都能映射成功」
   的那批查询上的指标 —— 这一列才是横向可比的主指标。

## 每个配置跑三种模式

同一份索引上依次跑 vector / bm25 / hybrid。小块的收益在两路上并不对称
（块越小，BM25 的词频统计越集中；向量这侧则失去上下文），所以分开报。

## 前置条件

服务要先关掉拒答，否则空结果会污染 recall：

    $env:VECTOR_SEARCH_ABSTENTION_ENABLED = 'false'
    .\\scripts\\dev.ps1 start
    python eval/chunk_ablation.py

跑完记得恢复拒答并灌回演示语料（脚本不会自动做，因为它会覆盖索引）。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import STRATEGIES, read_corpus  # noqa: E402

MODES = ["vector", "bm25", "hybrid"]

# (策略, 块大小, 重叠) —— 400/50 是当前默认值，放在中间当锚点
CONFIGS = [
    ("paragraph", 150, 50),
    ("paragraph", 200, 50),
    ("paragraph", 300, 50),
    ("paragraph", 400, 0),
    ("paragraph", 400, 50),
    ("paragraph", 400, 100),
    ("paragraph", 600, 50),
    ("paragraph", 800, 50),
    ("fixed", 400, 50),
]

BASE = ("paragraph", 400, 50)
SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?.\n])")
WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    return WHITESPACE.sub(" ", text).strip()


def post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


# ----------------------------------------------------------------------
# 指标
# ----------------------------------------------------------------------

def recall_at_k(retrieved: list[int], relevant: int, k: int) -> float:
    return 1.0 if relevant in retrieved[:k] else 0.0


def reciprocal_rank(retrieved: list[int], relevant: int, k: int) -> float:
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id == relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[int], relevant: int, k: int) -> float:
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id == relevant:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(p / 100.0 * (len(ordered) - 1)))))
    return ordered[index]


# ----------------------------------------------------------------------
# 切块与查询
# ----------------------------------------------------------------------

def chunk_documents(documents: list[dict], strategy: str, size: int,
                    overlap: int) -> list[dict]:
    chunker = STRATEGIES[strategy]
    chunks: list[dict] = []
    for document in documents:
        for piece in chunker(document["text"], size, overlap):
            chunks.append({
                "id": len(chunks),
                "source": document["source"],
                "text": piece,
            })
    return chunks


def make_contiguous_query(chunk: str, target: int, rng: random.Random) -> str | None:
    """从分块里取一段**连续**文字当查询。

    与 build_dataset.make_query 的区别：那个版本在句子用尽时会绕回块首继续拼，
    拼出来的查询在原文里不是连续片段，无法用于「按原文找回所在块」的映射。
    这里只向后拼，拼不满就接受短查询。
    """
    sentences = [s.strip() for s in SENTENCE_SPLIT.split(chunk) if s.strip()]
    if not sentences:
        return None
    start = rng.randrange(len(sentences))
    query = ""
    for sentence in sentences[start:]:
        query += sentence
        if len(query) >= target:
            break
    query = normalize(query)
    return query if len(query) >= 8 else None


def build_base_queries(documents: list[dict], count: int,
                       seed: int) -> list[dict]:
    strategy, size, overlap = BASE
    chunks = chunk_documents(documents, strategy, size, overlap)
    rng = random.Random(seed)
    picked = rng.sample(chunks, min(count, len(chunks)))
    queries = []
    for chunk in picked:
        text = make_contiguous_query(chunk["text"], 24, rng)
        if text is None:
            continue
        queries.append({"query": text, "source": chunk["source"]})
    return queries


def map_queries(chunks: list[dict], queries: list[dict]) -> tuple[dict, dict]:
    """把固定查询映射到当前配置的块上，返回 (query->块id, 统计)。"""
    normalized = [normalize(c["text"]) for c in chunks]
    mapped: dict[int, int] = {}
    ambiguous = 0
    for index, item in enumerate(queries):
        needle = normalize(item["query"])
        hits = [i for i, text in enumerate(normalized) if needle in text]
        if not hits:
            continue
        if len(hits) > 1:
            ambiguous += 1
        mapped[index] = hits[0]
    stats = {
        "queries_total": len(queries),
        "queries_kept": len(mapped),
        "queries_dropped": len(queries) - len(mapped),
        "queries_ambiguous": ambiguous,
    }
    return mapped, stats


def reset_and_load(service: str, chunks: list[dict], batch_size: int,
                   timeout: float) -> float:
    post_json(f"{service}/api/admin/reset?type=hnsw", {}, timeout)
    started = time.time()
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        payload = {"documents": [
            {"id": c["id"], "text": c["text"], "source": c["source"]}
            for c in batch
        ]}
        post_json(f"{service}/api/documents", payload, timeout)
    return time.time() - started


# ----------------------------------------------------------------------
# 评测
# ----------------------------------------------------------------------

def evaluate_mode(service: str, pairs: list[tuple[str, int]], mode: str,
                  top_k: int, timeout: float) -> dict:
    recalls, mrrs, ndcgs, latencies = [], [], [], []
    failures = 0
    for query_text, relevant in pairs:
        try:
            response = post_json(f"{service}/api/search", {
                "query": query_text, "topK": top_k, "mode": mode,
            }, timeout)
        except (urllib.error.URLError, TimeoutError):
            failures += 1
            continue

        retrieved = [hit["id"] for hit in (response.get("hits") or [])]
        recalls.append(recall_at_k(retrieved, relevant, top_k))
        mrrs.append(reciprocal_rank(retrieved, relevant, top_k))
        ndcgs.append(ndcg_at_k(retrieved, relevant, top_k))
        total_ms = (response.get("timing") or {}).get("totalMs")
        if total_ms is not None:
            latencies.append(float(total_ms))

    if not recalls:
        return {"error": "全部请求失败", "failures": failures}
    return {
        f"recall@{top_k}": round(statistics.fmean(recalls), 4),
        f"mrr@{top_k}": round(statistics.fmean(mrrs), 4),
        f"ndcg@{top_k}": round(statistics.fmean(ndcgs), 4),
        "latency_p50_ms": round(percentile(latencies, 50), 2),
        "latency_p95_ms": round(percentile(latencies, 95), 2),
        "evaluated": len(recalls),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="分块消融实验")
    parser.add_argument("--corpus-dir", default="data/corpus-full",
                        help="源语料目录（评测语料，由 eval/assemble_corpus.py 组装）")
    parser.add_argument("--service", default="http://127.0.0.1:8080")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--query-count", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", default="bench/results/chunk-ablation.json")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--only", default=None, help="只跑某个配置，如 p-400-50")
    args = parser.parse_args()

    documents = read_corpus(Path(args.corpus_dir))
    queries = build_base_queries(documents, args.query_count, args.seed)
    print(f"源文档 {len(documents)} 个，基准查询 {len(queries)} 条"
          f"（generated at {BASE[0]}/{BASE[1]}/{BASE[2]}）")
    print()

    results = []
    kept_sets: list[set[int]] = []
    for strategy, size, overlap in CONFIGS:
        label = f"{strategy[0]}-{size}-{overlap}"
        if args.only and args.only != label:
            continue

        print(f"[{label}] 切块...", flush=True)
        chunks = chunk_documents(documents, strategy, size, overlap)
        mapped, stats = map_queries(chunks, queries)
        kept_sets.append(set(mapped))

        lengths = [len(c["text"]) for c in chunks]
        entry = {
            "label": label, "strategy": strategy,
            "chunk_size": size, "overlap": overlap,
            "chunk_count": len(chunks),
            "avg_chars": round(statistics.fmean(lengths), 1) if lengths else 0.0,
            "p50_chars": percentile(lengths, 50),
            "max_chars": max(lengths) if lengths else 0,
            **stats,
        }
        print(f"  块数 {entry['chunk_count']}  平均 {entry['avg_chars']} 字  "
              f"P50 {entry['p50_chars']}  最大 {entry['max_chars']}")
        print(f"  查询映射 {stats['queries_kept']}/{stats['queries_total']}"
              f"（丢弃 {stats['queries_dropped']}，跨块重复 {stats['queries_ambiguous']}）",
              flush=True)

        print("  重建索引...", flush=True)
        entry["index_seconds"] = round(
            reset_and_load(args.service, chunks, args.batch_size, args.timeout), 1)
        print(f"  建索引 {entry['index_seconds']} 秒", flush=True)

        pairs = [(queries[i]["query"], chunk_id) for i, chunk_id in mapped.items()]
        for mode in MODES:
            metrics = evaluate_mode(args.service, pairs, mode, args.top_k, args.timeout)
            entry[mode] = metrics
            print(f"  {mode:<7} recall@{args.top_k}="
                  f"{metrics.get(f'recall@{args.top_k}', 0):.4f}  "
                  f"MRR={metrics.get(f'mrr@{args.top_k}', 0):.4f}  "
                  f"P95={metrics.get('latency_p95_ms', 0):.1f}ms", flush=True)

        results.append(entry)
        print(flush=True)

    if not results:
        print("[错误] 没有跑任何配置", file=sys.stderr)
        return 1

    # 公共子集：在所有跑过的配置里都能映射成功的那些查询
    common = set.intersection(*kept_sets) if kept_sets else set()
    if common:
        print(f"公共子集：{len(common)} 条查询在所有配置里都能映射，重跑一次做同题对比")
        for entry in results:
            strategy, size, overlap = entry["strategy"], entry["chunk_size"], entry["overlap"]
            chunks = chunk_documents(documents, strategy, size, overlap)
            mapped, _ = map_queries(chunks, queries)
            reset_and_load(args.service, chunks, args.batch_size, args.timeout)
            pairs = [(queries[i]["query"], mapped[i]) for i in sorted(common) if i in mapped]
            entry["common_subset"] = {
                "size": len(pairs),
                "hybrid": evaluate_mode(args.service, pairs, "hybrid", args.top_k, args.timeout),
            }
            hybrid = entry["common_subset"]["hybrid"]
            print(f"  {entry['label']:<10} hybrid recall@{args.top_k}="
                  f"{hybrid.get(f'recall@{args.top_k}', 0):.4f}  "
                  f"MRR={hybrid.get(f'mrr@{args.top_k}', 0):.4f}", flush=True)
        print()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "protocol": {
            "base_config": f"{BASE[0]}/{BASE[1]}/{BASE[2]}",
            "query_count": len(queries),
            "seed": args.seed,
            "document_count": len(documents),
            "common_subset": len(common),
            "abstention": "disabled",
        },
        "configs": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("| 配置 | 块数 | 均长 | 映射成功 | hybrid recall@10 | hybrid MRR | 公共子集 recall | 建索引(s) |")
    print("|---|---|---|---|---|---|---|---|")
    for entry in results:
        hybrid = entry.get("hybrid", {})
        common_hybrid = entry.get("common_subset", {}).get("hybrid", {})
        print(f"| {entry['label']} | {entry['chunk_count']} | {entry['avg_chars']} | "
              f"{entry['queries_kept']}/{entry['queries_total']} | "
              f"{hybrid.get(f'recall@{args.top_k}', 0):.4f} | "
              f"{hybrid.get(f'mrr@{args.top_k}', 0):.4f} | "
              f"{common_hybrid.get(f'recall@{args.top_k}', 0):.4f} | "
              f"{entry['index_seconds']} |")
    print()
    print(f"结果写入 {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
