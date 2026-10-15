"""端到端冒烟测试。

在有真实 BGE 模型和真实中文语料的前提下，验证整条链路：
    文本 -> Python 编码 -> Java 索引 -> 检索 -> 回填原文

并且**在同一份真实向量上对比 HNSW 与暴力检索**：
- 检索质量是否一致（HNSW 是近似的，允许少量差异）
- 延迟差多少（这才是引入 HNSW 的理由）

用法（先启动 Python 模型服务和 Java 检索服务）：
    python scripts/smoke_test.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
import urllib.request

# Windows 控制台默认 GBK，print 非 ASCII 字符会抛 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

JAVA = "http://127.0.0.1:8080"
PY = "http://127.0.0.1:8000"

DOCUMENTS = [
    (1, "HNSW 是一种基于分层可导航小世界图的近似最近邻检索算法，通过构建多层图结构加速向量搜索。"),
    (2, "BM25 是经典的稀疏检索算法，基于词频和逆文档频率对文档进行打分排序。"),
    (3, "Java 的垃圾回收器 G1 通过分代收集和区域化管理来降低停顿时间。"),
    (4, "今天食堂的红烧肉特别好吃，配上一碗米饭简直完美。"),
    (5, "向量数据库通过近似最近邻算法在亿级数据中实现毫秒级相似度检索。"),
    (6, "Python 的动态批处理技术可以把多个推理请求合并成一次前向计算，显著提升吞吐量。"),
    (7, "倒排索引是搜索引擎的核心数据结构，它记录每个词出现在哪些文档中。"),
    (8, "堆外内存可以绕过垃圾回收管理，适合存放生命周期长且体积大的数据。"),
    (9, "锁的乐观读策略在读多写少的场景下比互斥锁有更好的吞吐表现。"),
    (10, "编译器把源代码翻译成机器码，中间会经过词法、语法和语义分析等阶段。"),
]

QUERIES = [
    ("怎样加快海量数据的相似度查找", {1, 5}),
    ("如何减少内存回收造成的卡顿", {3, 8}),
    ("让模型一次处理更多请求的办法", {6}),
    ("搜索引擎怎么知道一个词出现在哪里", {7}),
]


def post(path: str, payload: dict, timeout: float = 120.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{JAVA}{path}", data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get(path: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(f"{JAVA}{path}", timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


FAILURES: list[str] = []


def check(ok: bool, message: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {message}")
    if not ok:
        FAILURES.append(message)


def index_documents() -> None:
    payload = {"documents": [{"id": i, "text": t} for i, t in DOCUMENTS]}
    result = post("/api/documents", payload)
    print(f"  写入 {result['indexed']} 条，索引总数 {result['totalSize']}，"
          f"维度 {result['dimension']}")
    print(f"  编码耗时 {result['embeddingMs']}ms，索引耗时 {result['indexMs']}ms")
    check(result["indexed"] == len(DOCUMENTS), "全部文档写入成功")
    check(result["totalSize"] == len(DOCUMENTS), "索引条数与文档数一致")


def run_queries(label: str) -> tuple[list[list[int]], list[float], list[float]]:
    """跑一轮查询，返回 (每条的命中 id 列表, 编码耗时, 检索耗时)。"""
    all_hits: list[list[int]] = []
    embedding_ms: list[float] = []
    search_ms: list[float] = []

    for query, _ in QUERIES:
        response = post("/api/search", {"query": query, "topK": 3})
        hits = [h["id"] for h in response["hits"]]
        all_hits.append(hits)
        embedding_ms.append(response["timing"]["embeddingMs"])
        search_ms.append(response["timing"]["searchMs"])

        top = response["hits"][0]
        text = (top.get("text") or "")[:34]
        print(f"    {query}")
        print(f"      -> #{top['id']} (score={top['score']:.4f}) {text}...")
        print(f"      编码 {response['timing']['embeddingMs']}ms | "
              f"检索 {response['timing']['searchMs']}ms | "
              f"端到端 {response['timing']['totalMs']}ms")

    return all_hits, embedding_ms, search_ms


def main() -> int:
    print("=" * 70)
    print("端到端冒烟测试")
    print("=" * 70)

    # ---- 环境检查 ----
    print("\n[0] 服务健康检查")
    try:
        health = get("/api/stats")
        print(f"  索引实现: {health['indexType']}  维度: {health['dimension']}")
        check(health["modelServiceHealthy"], "Python 模型服务可达")
    except Exception as exc:
        print(f"  无法连接 Java 服务: {exc}")
        return 1

    # ---- HNSW ----
    print("\n[1] HNSW 索引：清空并灌入语料")
    post("/api/admin/reset?type=hnsw", {})
    index_documents()

    print("\n[2] HNSW 检索")
    hnsw_hits, hnsw_emb, hnsw_search = run_queries("hnsw")

    hnsw_stats = get("/api/stats")
    print(f"\n  HNSW 图结构: 最大层={hnsw_stats['hnsw']['maxLevel']} "
          f"总边数={hnsw_stats['hnsw']['totalLinks']} "
          f"层人口={hnsw_stats['hnsw']['levelHistogram']}")

    # ---- 暴力检索 ----
    print("\n[3] 暴力检索：清空并灌入同一份语料（对照组）")
    post("/api/admin/reset?type=brute-force", {})
    index_documents()

    print("\n[4] 暴力检索")
    bf_hits, bf_emb, bf_search = run_queries("brute-force")

    # ---- 对比 ----
    print("\n[5] 结果对比")
    overlaps = []
    for i, (query, _) in enumerate(QUERIES):
        hnsw_top = set(hnsw_hits[i])
        bf_top = set(bf_hits[i])
        overlap = len(hnsw_top & bf_top) / max(1, len(bf_top))
        overlaps.append(overlap)
        print(f"    「{query}」")
        print(f"      HNSW      : {hnsw_hits[i]}")
        print(f"      暴力检索  : {bf_hits[i]}")
        print(f"      重合度    : {overlap:.0%}")

    print(f"\n    平均重合度: {statistics.fmean(overlaps):.0%}")
    check(statistics.fmean(overlaps) >= 0.5,
          f"HNSW 与暴力检索结果平均重合度 {statistics.fmean(overlaps):.0%} >= 50%")

    # ---- 真实数据的相关性验证 ----
    print("\n[6] 语义相关性验证（看 HNSW 的 top-1 是否命中预期文档）")
    for i, (query, expected) in enumerate(QUERIES):
        top1 = hnsw_hits[i][0] if hnsw_hits[i] else -1
        hit = top1 in expected
        print(f"    「{query}」 期望 {sorted(expected)}，实际 top-1 = #{top1} {'✔' if hit else '✘'}")
        if not hit:
            FAILURES.append(f"查询「{query}」的 top-1 未命中预期文档")

    # ---- 延迟 ----
    print("\n[7] 延迟对比（n=10 条，规模太小，此处仅为链路验证）")
    print(f"    HNSW        编码中位数 {statistics.median(hnsw_emb):>5.0f}ms  "
          f"检索中位数 {statistics.median(hnsw_search):>5.0f}ms")
    print(f"    暴力检索    编码中位数 {statistics.median(bf_emb):>5.0f}ms  "
          f"检索中位数 {statistics.median(bf_search):>5.0f}ms")
    print()
    print("    注意：检索耗时在当前规模下两者都接近 0 —— 10 条向量根本体现不出")
    print("    HNSW 的价值。真实差距要等第 2 周用 JMH 在 10 万级数据上跑出来。")
    print("    这也说明：**跨进程编码（几十毫秒）才是当前端到端延迟的瓶颈**，")
    print("    此时去调 HNSW 参数是找错了方向。")

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"结果: {len(FAILURES)} 项失败")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("结果: 端到端链路全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
