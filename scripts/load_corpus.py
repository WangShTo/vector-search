"""把语料灌进检索服务。

读取 build_dataset.py 产出的 corpus.jsonl，分批 POST 给 Java 服务的
/api/documents 接口。

## 为什么强调「分批」

Java 侧的 /api/documents 是批量接口，一次请求带 N 条文本，
Java 再把这 N 条一次性发给 Python 做 embedding。
如果不分批而是一次全发，10 万条文档会产生一个巨大的 HTTP 请求体和
一次巨大的模型 forward —— 直接 OOM。

分批大小是个权衡：
    小批 -> 请求多、每次 forward 的固定开销被放大、慢
    大批 -> 单次 forward 内存占用高、失败要重传的数据多
默认 64 是一个偏保守的值，适合 CPU 推理。

## 用法

    python scripts/load_corpus.py --service http://127.0.0.1:8080
    python scripts/load_corpus.py --corpus data/eval/corpus.jsonl --batch-size 128
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
        url, data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="把语料灌进检索服务")
    parser.add_argument("--corpus", default="data/eval/corpus.jsonl")
    parser.add_argument("--service", default="http://127.0.0.1:8080")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--limit", type=int, default=None, help="只灌前 N 条（调试用）")
    parser.add_argument("--reset", action="store_true",
                        help="先清空索引（切换实现或重新开始时会用到）")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"[错误] 找不到 {corpus_path}", file=sys.stderr)
        print("       先跑: python eval/build_dataset.py --corpus-dir data/corpus --out data/eval",
              file=sys.stderr)
        return 1

    chunks = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        chunks = chunks[: args.limit]
    if not chunks:
        print("[错误] 语料为空", file=sys.stderr)
        return 1

    if args.reset:
        print("清空索引...")
        post_json(f"{args.service}/api/admin/reset?type=hnsw", {}, args.timeout)

    print(f"语料: {corpus_path}（{len(chunks)} 条）")
    print(f"批大小: {args.batch_size}")

    started = time.time()
    indexed = 0
    failed_batches = 0

    for start in range(0, len(chunks), args.batch_size):
        batch = chunks[start : start + args.batch_size]
        payload = {"documents": [{"id": c["id"], "text": c["text"]} for c in batch]}
        try:
            result = post_json(f"{args.service}/api/documents", payload, args.timeout)
            indexed += result["indexed"]
        except urllib.error.URLError as exc:
            failed_batches += 1
            print(f"  [失败] 第 {start // args.batch_size + 1} 批: {exc}", file=sys.stderr)
            if failed_batches >= 3:
                print("[中止] 连续多批失败，检查服务是否正常", file=sys.stderr)
                return 1
            continue

        done = min(start + args.batch_size, len(chunks))
        elapsed = time.time() - started
        rate = done / elapsed if elapsed > 0 else 0
        print(f"\r  已灌入 {done}/{len(chunks)}  索引总数 {result['totalSize']}  "
              f"{rate:.0f} 条/秒", end="", flush=True)

    print()
    elapsed = time.time() - started
    print(f"完成: {indexed} 条，用时 {elapsed:.1f} 秒（{indexed / max(elapsed, 0.001):.0f} 条/秒）")
    if failed_batches:
        print(f"失败批次: {failed_batches}")
        return 1

    print()
    print("灌库速度本身就是一个可报告的指标。如果很慢，先看 /api/search 的 timing 里")
    print("embeddingMs 占比 —— 瓶颈通常在跨进程调用，而不是索引写入。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
