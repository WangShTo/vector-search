"""按「来源文件」评测语义查询 —— 用来公平对比 vector / bm25 / hybrid。

## 为什么需要另一套评测

主评测集（`build_dataset.py` 产出）用的是**已知项检索**：查询直接从文档里抽一段原文。
这意味着查询词与文档词高度重合，**本质上是词面查询** —— BM25 天然占优，
向量检索被系统性低估。

只用那套数据，会得出「混合检索不如纯 BM25」的结论。那不是混合检索的问题，
是**评测集的构造方式有偏差**。

所以这里补一套**语义查询**：人工改写提问方式，尽量不与原文共用词汇，
并以「答案来自哪个文件」为相关性标准。两套合起来看，才能反映真实场景 ——
真实用户的查询是混合的：有时报个精确的名字，有时描述一个模糊的需求。

## 为什么按文件算相关性，而不是按分块

一个文件会被切成几十个分块（比如 HnswIndex.java 切成 98 块）。
如果把「该文件的所有分块」都算作相关，recall@10 的数学上限只有 10/98，
指标会低得没有意义。

所以这里用**文档级已知项**的评法：只要召回了**任意一个**来自目标文件的分块，
就算命中；MRR 看第一个命中的排名。这是信息检索里标准的 known-item search。

## 用法

    python eval/evaluate_semantic.py --service http://127.0.0.1:8080
    python eval/evaluate_semantic.py --top-k 10 --modes vector,bm25,hybrid
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Windows 控制台默认 GBK，print 非 ASCII 字符会抛 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="按来源文件评测语义查询")
    parser.add_argument("--queries", default="eval/semantic-queries.jsonl")
    parser.add_argument("--corpus", default="data/eval/corpus.jsonl",
                        help="用来把分块 id 映射回来源文件")
    parser.add_argument("--service", default="http://127.0.0.1:8080")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--modes", default="vector,bm25,hybrid")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    queries_path = Path(args.queries)
    corpus_path = Path(args.corpus)
    if not queries_path.exists():
        print(f"[错误] 找不到 {queries_path}", file=sys.stderr)
        return 1
    if not corpus_path.exists():
        print(f"[错误] 找不到 {corpus_path}，先跑 build_dataset.py", file=sys.stderr)
        return 1

    queries = load_jsonl(queries_path)
    # 分块 id -> 来源文件。有了它就能在本地判断相关性，不需要服务返回来源字段
    chunk_source = {c["id"]: c["source"] for c in load_jsonl(corpus_path)}
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    print("=" * 68)
    print("语义查询评测（按来源文件判断相关性）")
    print("=" * 68)
    print(f"  查询条数: {len(queries)}   top-k: {args.top_k}   模式: {', '.join(modes)}")
    print()

    results: dict[str, dict] = {}

    for mode in modes:
        hits_at_k = 0
        reciprocal_ranks: list[float] = []
        latencies: list[float] = []
        failures = 0

        for item in queries:
            targets = set(item["relevant_sources"])
            payload = {"query": item["query"], "topK": args.top_k, "mode": mode}
            try:
                response = post_json(f"{args.service}/api/search", payload, args.timeout)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                failures += 1
                reciprocal_ranks.append(0.0)
                if failures <= 2:
                    print(f"  [失败] {item['query'][:26]}... -> {exc}", file=sys.stderr)
                continue

            latencies.append(response.get("timing", {}).get("totalMs", 0))
            retrieved = [h["id"] for h in response.get("hits", [])]

            first_rank = 0
            for rank, chunk_id in enumerate(retrieved[: args.top_k], start=1):
                if chunk_source.get(chunk_id) in targets:
                    first_rank = rank
                    break

            if first_rank:
                hits_at_k += 1
                reciprocal_ranks.append(1.0 / first_rank)
            else:
                reciprocal_ranks.append(0.0)

        results[mode] = {
            "hit": hits_at_k / len(queries),
            "mrr": statistics.fmean(reciprocal_ranks),
            "p50": statistics.median(latencies) if latencies else 0.0,
            "failures": failures,
        }

    print(f"  {'模式':<10}{'命中率@' + str(args.top_k):>14}{'MRR@' + str(args.top_k):>13}{'P50 延迟':>13}")
    print("  " + "-" * 50)
    for mode in modes:
        r = results[mode]
        print(f"  {mode:<10}{r['hit']:>14.4f}{r['mrr']:>13.4f}{r['p50']:>11.0f} ms")

    print()
    best = max(modes, key=lambda m: (results[m]["hit"], results[m]["mrr"]))
    print(f"  最优: {best}（命中率 {results[best]['hit']:.4f}，MRR {results[best]['mrr']:.4f}）")
    print()
    print("  说明：这是 16 条人工改写的语义查询，样本很小，数字只能看趋势。")
    print("  它的作用是和主评测集（150 条词面查询）交叉验证 ——")
    print("  单看任何一边都会得出偏颇的结论。")

    output = Path("bench/results/semantic-eval.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  结果已写入: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
