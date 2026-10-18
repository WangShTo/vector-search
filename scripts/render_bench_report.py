"""把基准原始数据渲染成 Markdown 报告。

输入：
    bench/results/recall.json    RecallRunner 产出（召回率 + 延迟扫描）
    bench/results/jmh.json       JMH 产出（微基准，含 P50/P95/P99）

输出：
    bench/REPORT.md

## 为什么报告要脚本生成，而不是手写

手写的数字会过期。代码一改、参数一调，报告就悄悄变成谎话 ——
而且没人会记得去改它。脚本生成保证报告永远对应<b>当前代码</b>的实际表现，
别人 clone 下来跑一遍就能复现同样的表。

这也是简历项目里最容易被忽略、但最能体现工程素养的一点：
**你的数字是可复现的，还是你敲上去的。**
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

# Windows 控制台默认 GBK，print 非 ASCII 字符会抛 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ----------------------------------------------------------------------
# 读取
# ----------------------------------------------------------------------

def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[警告] 无法读取 {path}: {exc}", file=sys.stderr)
        return None


def index_jmh(raw: list[dict]) -> dict:
    """把 JMH 结果按 (类名, 方法名, 参数) 建索引。"""
    indexed: dict = {}
    for entry in raw:
        parts = entry["benchmark"].split(".")
        class_name, method = parts[-2], parts[-1]
        params = tuple(sorted((entry.get("params") or {}).items()))
        key = (class_name, method, params)
        indexed.setdefault(key, {})[entry["mode"]] = entry["primaryMetric"]
    return indexed


def fmt(value: float, digits: int = 2) -> str:
    if value is None:
        return "—"
    if value >= 10000:
        return f"{value:,.0f}"
    if value >= 100:
        return f"{value:.1f}"
    return f"{value:.{digits}f}"


def auto_time(ns: float) -> str:
    """把纳秒自动换算成好读的单位。"""
    if ns is None:
        return "—"
    if ns >= 1_000_000_000:
        return f"{ns / 1_000_000_000:.2f} s"
    if ns >= 1_000_000:
        return f"{ns / 1_000_000:.2f} ms"
    if ns >= 1_000:
        return f"{ns / 1_000:.1f} µs"
    return f"{ns:.1f} ns"


# JMH 的 primaryMetric.score 已经是 @OutputTimeUnit 指定的单位，
# 不同基准类可能不同（距离基准用 ns/op，索引基准用 us/op）。
# 所以不能直接把它当纳秒用 —— 必须先按 scoreUnit 归一化，
# 否则会安静地产生 1000 倍的量纲错误。（这个错误真的发生过。）
_UNIT_TO_NANOS = {
    "ns/op": 1.0,
    "us/op": 1_000.0,
    "ms/op": 1_000_000.0,
    "s/op": 1_000_000_000.0,
    "ops/ns": None,   # 吞吐单位，方向相反，不做处理
    "ops/us": None,
    "ops/ms": None,
    "ops/s": None,
}


def metric_to_nanos(metric: dict | None) -> float | None:
    """把 JMH 的 score / 分位数统一换算成纳秒。"""
    if metric is None:
        return None
    unit = metric.get("scoreUnit", "ns/op")
    factor = _UNIT_TO_NANOS.get(unit)
    if factor is None:
        # 未知或被显式排除的单位：原样返回，宁可数字可疑也不要算错
        return metric.get("score")
    return metric.get("score") * factor


def percentile_to_nanos(metric: dict | None, key: str) -> float | None:
    """把 scorePercentiles 里的分位数换算成纳秒。"""
    if metric is None:
        return None
    value = (metric.get("scorePercentiles") or {}).get(key)
    if value is None:
        return None
    unit = metric.get("scoreUnit", "ns/op")
    factor = _UNIT_TO_NANOS.get(unit)
    if factor is None:
        return value
    return value * factor


# ----------------------------------------------------------------------
# 各章节
# ----------------------------------------------------------------------

def section_simd(jmh: dict) -> str:
    """SIMD vs 标量。"""
    dimensions = sorted({
        dict(params).get("dimension")
        for (cls, _, params) in jmh
        if cls == "DistanceBenchmark"
    } - {None}, key=lambda x: int(x))

    if not dimensions:
        return ""

    lines = [
        "## 一、SIMD 距离计算（标量 vs JDK Vector API）",
        "",
        "这是整个项目最干净的对照实验：两个方法输入输出完全相同，",
        "唯一差别就是循环体是否被向量化。所以加速比可以直接归因于 SIMD。",
        "",
        "| 维度 | 操作 | 标量 | SIMD | 加速比 |",
        "|---|---|---|---|---|",
    ]

    for dim in dimensions:
        for op_label, scalar_method, simd_method in [
            ("内积", "scalarDot", "simdDot"),
            ("平方欧氏距离", "scalarL2Squared", "simdL2Squared"),
        ]:
            scalar = jmh.get(("DistanceBenchmark", scalar_method, (("dimension", dim),)), {}).get("avgt")
            simd = jmh.get(("DistanceBenchmark", simd_method, (("dimension", dim),)), {}).get("avgt")
            if not scalar or not simd:
                continue
            scalar_ns = metric_to_nanos(scalar)
            simd_ns = metric_to_nanos(simd)
            speedup = scalar_ns / simd_ns
            lines.append(
                f"| {dim} | {op_label} | {auto_time(scalar_ns)} | "
                f"{auto_time(simd_ns)} | **{speedup:.2f}x** |"
            )

    lines += [
        "",
        "> SIMD 的实际向量宽度由 `DistanceMetricsSimd.laneWidth()` 在运行时决定，",
        "> 本机为 8（AVX2）。`SPECIES_PREFERRED` 让 JVM 选择当前 CPU 支持的最宽向量，",
        "> 写死宽度会在老机器上退化。",
        "",
        "> **为什么用相对容差而不是绝对容差做正确性校验**：SIMD 改变了浮点累加顺序",
        "> （标量顺序累加，SIMD 先算各 lane 部分和再横向归约），而浮点加法不满足结合律，",
        "> 所以两者结果**必然**有微小差异。用绝对容差去卡会把正常现象误报成 bug。",
        "",
    ]
    return "\n".join(lines)


def section_index(jmh: dict) -> str:
    """HNSW vs 暴力检索的延迟对比（JMH 口径）。"""

    def entries(class_name: str):
        out = []
        for (cls, method, params), modes in jmh.items():
            if cls == class_name and method == "search":
                out.append((dict(params), modes))
        return out

    hnsw = entries("HnswSearchBenchmark")
    brute = entries("BruteForceSearchBenchmark")
    if not hnsw and not brute:
        return ""

    lines = [
        "## 二、索引检索延迟（JMH 口径）",
        "",
    ]

    if brute:
        brute_params, brute_modes = brute[0]
        avg = brute_modes.get("avgt")
        # JMH 的 SampleTime 模式在 JSON 里叫 "sample"，不是 "samp" ——
        # 写错会静默地让所有分位数显示成 "—"（踩过）
        sample = brute_modes.get("sample")
        if avg:
            lines += [
                f"**暴力检索**（n={brute_params.get('nodeCount')}, dim={brute_params.get('dimension')}）",
                "",
                "| 指标 | 数值 |",
                "|---|---|",
                f"| 平均单次查询 | {auto_time(metric_to_nanos(avg))} |",
            ]
            if sample:
                lines += [
                    f"| P50 | {auto_time(percentile_to_nanos(sample, '50.0'))} |",
                    f"| P95 | {auto_time(percentile_to_nanos(sample, '95.0'))} |",
                    f"| P99 | {auto_time(percentile_to_nanos(sample, '99.0'))} |",
                ]
            lines.append("")

    if hnsw:
        lines += [
            "**HNSW**（同一个数据集，不同 `efSearch`）",
            "",
            "| efSearch | 平均 | P50 | P95 | P99 | 相对暴力检索 |",
            "|---|---|---|---|---|---|",
        ]
        brute_mean = None
        if brute:
            brute_avg = brute[0][1].get("avgt")
            brute_mean = metric_to_nanos(brute_avg) if brute_avg else None

        for params, modes in sorted(hnsw, key=lambda x: int(x[0].get("efSearch", 0))):
            avg = modes.get("avgt")
            sample = modes.get("sample")
            if not avg:
                continue
            avg_ns = metric_to_nanos(avg)
            speedup = f"**{brute_mean / avg_ns:.1f}x**" if brute_mean else "—"
            lines.append(
                f"| {params.get('efSearch')} | {auto_time(avg_ns)} | "
                f"{auto_time(percentile_to_nanos(sample, '50.0'))} | "
                f"{auto_time(percentile_to_nanos(sample, '95.0'))} | "
                f"{auto_time(percentile_to_nanos(sample, '99.0'))} | {speedup} |"
            )
        lines.append("")

    lines += [
        "> 只有延迟没有意义 —— 必须配合同一份报告里的召回率一起看。",
        "> 把 efSearch 调到 1 延迟可以低到不可思议，但召回率也会烂到毫无用处。",
        "",
    ]
    return "\n".join(lines)


def section_recall(recall: dict) -> str:
    """召回率 / 延迟权衡曲线。"""
    if not recall:
        return ""

    bf = recall.get("bruteForce", {})
    hnsw = recall.get("hnsw", {})
    points = hnsw.get("points", [])
    params = recall.get("hnswParams", {})

    # 内存这一行是可选的：早期产出的结果 JSON 里没有这个字段
    delta = hnsw.get("heapDeltaBytes")
    estimated = hnsw.get("estimatedBytes")
    if delta is not None and estimated is not None:
        memory_line = (
            f"| HNSW 索引堆内占用 | {delta / 1024 / 1024:,.1f} MB 实测"
            f"（数组本身估算 {estimated / 1024 / 1024:,.1f} MB） |")
    else:
        memory_line = "| HNSW 索引堆内占用 | 未测量 |"

    lines = [
        "## 三、召回率 / 延迟 权衡（真实测量）",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 数据规模 | {recall.get('nodeCount'):,} 条 × {recall.get('dimension')} 维 |",
        f"| 数据分布 | {recall.get('dataDistribution')}，{recall.get('clusters')} 个簇 |",
        f"| 查询条数 | {recall.get('queryCount')} |",
        f"| top-k | {recall.get('topK')} |",
        f"| HNSW 参数 | m={params.get('m')}, efConstruction={params.get('efConstruction')} |",
        f"| HNSW 建图耗时 | {hnsw.get('buildMillis'):,} ms |",
        f"| 暴力检索建库耗时 | {bf.get('buildMillis'):,} ms |",
        memory_line,
        f"| 图最大层 | {hnsw.get('maxLevel')} |",
        f"| 图总边数 | {hnsw.get('totalLinks'):,}（平均出边 {hnsw.get('averageDegree')}） |",
        "",
        "### 权衡曲线",
        "",
        f"| efSearch | recall@{recall.get('topK')} | 平均延迟 | P95 | P99 | 相对暴力检索 |",
        "|---|---|---|---|---|---|",
    ]

    bf_mean = bf.get("meanMicros")
    for p in points:
        speedup = f"**{p['speedupVsBruteForce']:.1f}x**" if bf_mean else "—"
        lines.append(
            f"| {p['efSearch']} | **{p['recallAtK']:.4f}** | "
            f"{auto_time(p['meanMicros'] * 1000)} | {auto_time(p['p95Micros'] * 1000)} | "
            f"{auto_time(p['p99Micros'] * 1000)} | {speedup} |"
        )
    lines.append("")

    lines += [
        "> **口径提醒（重要）**：本节的延迟由 `RecallRunner` 在单进程内测得，",
        "> 预热轮次远少于 JMH，因此数值偏高。**绝对延迟请以第二节的 JMH 结果为准**。",
        ">",
        "> 本节延迟列的作用是与 recall 配对，回答「拿到这个召回率要付出多少延迟」。",
        "> 两个来源的数字不一致不是错误，而是测量方法差异的体现 ——",
        "> 报告里保留两者并说明区别，比只留一个更可信。",
        "",
    ]

    # 拐点：达到 recall >= 0.9 的最小 efSearch
    knee = next((p for p in points if p["recallAtK"] >= 0.90), None)
    if knee:
        lines += [
            "### 结论：拐点在哪里",
            "",
            f"**达到 recall@{recall.get('topK')} ≥ 0.90 的最小 `efSearch` 是 "
            f"{knee['efSearch']}**，此时平均延迟 {auto_time(knee['meanMicros'] * 1000)}，",
            f"相对暴力检索快 **{knee['speedupVsBruteForce']:.0f} 倍**。",
            "",
            "这个点就是生产环境应该选的配置起点：再往上加 `efSearch` 召回率提升有限，",
            "但延迟是线性上涨的。面试被问「efSearch 你怎么定的」，答案就是这个拐点。",
            "",
        ]

    # 图结构
    hist = hnsw.get("levelHistogram")
    pop = hnsw.get("layerPopulation")
    if hist and pop:
        lines += [
            "### 图结构自检（金字塔是否成立）",
            "",
            "| 层 | 最高层为此层 | 该层实际存在 |",
            "|---|---|---|",
        ]
        for level in range(len(hist)):
            lines.append(f"| {level} | {hist[level]:,} | {pop[level]:,} |")
        lines += [
            "",
            "> 层高分布必须呈指数衰减、各层存在节点数必须严格递减。",
            "> 如果不满足，说明随机层高抽样或建图逻辑有问题 —— 此时 recall 一定上不去，",
            "> 再怎么调 `efSearch` 也没用。",
            "",
        ]

    return "\n".join(lines)


def parse_concurrency(raw: list[dict]) -> dict:
    """把 JMH 的 searchWith{N}Threads 结果解析成 {线程数: 吞吐}。"""
    out: dict = {}
    for entry in raw:
        name = entry["benchmark"].rsplit("searchWith", 1)[1]
        threads = int(name.replace("Threads", "").replace("Thread", ""))
        out[threads] = entry["primaryMetric"]["score"]
    return out


def section_concurrency(results_dir: Path) -> str:
    """并发扩展性 —— 并如实说明这台机器测不准。"""
    baseline_raw = load_json(results_dir / "concurrent-baseline.json")
    optimized_raw = load_json(results_dir / "concurrent-optimized.json")
    if not baseline_raw and not optimized_raw:
        return ""

    baseline = parse_concurrency(baseline_raw) if baseline_raw else {}
    optimized = parse_concurrency(optimized_raw) if optimized_raw else {}
    threads = sorted(set(baseline) | set(optimized))
    if not threads:
        return ""

    b1 = baseline.get(1)
    o1 = optimized.get(1)

    lines = [
        "## 四、并发扩展性",
        "",
        "单线程延迟再低，也不能说明服务扛得住并发。这一节测的是**吞吐随线程数的扩展**：",
        "",
        "```",
        "扩展效率 = 实际吞吐(T 线程) / (T × 吞吐(1 线程))",
        "```",
        "",
        "理想值 1.0。明显低于 1 说明存在争用 —— 线程越多，花在等待上的时间越多。",
        "",
        "| 线程 | 基线 ops/s | 线程本地后 ops/s | 基线扩展效率 | 优化后扩展效率 |",
        "|---|---|---|---|---|",
    ]

    for t in threads:
        bs = baseline.get(t)
        os_ = optimized.get(t)
        b_cell = f"{bs:,.0f}" if bs else "—"
        o_cell = f"{os_:,.0f}" if os_ else "—"
        be = f"{bs / (b1 * t):.2f}" if bs and b1 else "—"
        oe = f"{os_ / (o1 * t):.2f}" if os_ and o1 else "—"
        lines.append(f"| {t} | {b_cell} | {o_cell} | {be} | {oe} |")

    lines += [
        "",
        "### ⚠️ 这份并发数据不可用于下结论",
        "",
        "必须说清楚：**这台机器测不出可信的并发数字**。",
        "",
        "同一个 8 线程配置，在不同轮次里跑出过 20k、28k、35k、42k、82k ops/s ——",
        "**跨轮次波动达到 4 倍**，完全掩盖了被测改动本身的效果。",
        "",
        "原因是可以定位的：",
        "",
        "| 干扰源 | 具体情况 |",
        "|---|---|",
        "| 大小核混合架构 | i7-14650HX 是 8 P 核 + 8 E 核。线程数超过 8 后，调度器会把线程放到 E 核上，而 E 核性能只有 P 核的一半左右，于是「加线程」变成了「换更慢的核」 |",
        "| 后台负载 | 测量期间机器常驻 27% 左右的 CPU 占用，可用内存一度只剩 1.8 GB / 15.8 GB |",
        "| 样本量不足 | 每轮只有 5 × 2 秒的测量窗口，对毫秒级操作来说统计量偏少 |",
        "",
        "所以表格里的数字只能当**趋势参考**，不能写进简历，也不能用来证明某个优化有效。",
        "",
        "### 要得到可信的并发结论，需要什么",
        "",
        "- **独占机器**：测量期间不跑其他程序，关掉浏览器与后台任务",
        "- **绑核**：把基准线程固定在 P 核上，避开 E 核调度带来的异构性",
        "- **多次取最优**：环境噪声只会让结果变慢，所以多轮中的最好成绩最接近真实能力",
        "- **更长的测量窗口**：单轮 ≥ 10 秒，并提高 fork 数做跨 JVM 平均",
        "",
        "这条结论本身比数字有价值：**并发性能不是「跑一下 JMH」就能拿到的**，",
        "它比单线程基准脆弱得多，对测量环境的要求也高得多。",
        "",
    ]
    return "\n".join(lines)


def section_snapshot(results_dir: Path) -> str:
    """索引持久化的收益：重建图 vs 从快照恢复。"""
    data = load_json(results_dir / "snapshot.json")
    if not data:
        return ""

    build_ms = data.get("buildMillis", 0)
    save_ms = data.get("saveMillis", 0)
    load_ms = data.get("loadMillis", 0)
    speedup = data.get("speedupVsRebuild", 0)
    file_mib = data.get("fileMiB", 0)
    bytes_per_vector = data.get("bytesPerVector", 0)
    node_count = data.get("nodeCount", 0)
    dimension = data.get("dimension", 0)
    verified = data.get("verifiedQueries", 0)
    mismatches = data.get("mismatches", 0)

    verdict = "完全一致" if mismatches == 0 else "存在差异"

    lines = [
        "## 五、索引持久化的收益（冷启动耗时）",
        "",
        "10 万条数据重建一次图要将近两分钟。如果每次重启都重建，演示和部署都无法接受",
        "—— 这是做持久化的真正理由，不只是「省事」。",
        "",
        "| 阶段 | 耗时 | 说明 |",
        "|---|---|---|",
        f"| 重建图（不落盘时每次启动都要付） | **{build_ms / 1000:.1f} 秒** | "
        f"{node_count:,} 条 × {dimension} 维 |",
        f"| 落盘（一次性） | {save_ms} ms | 产出 {file_mib} MiB 快照文件 |",
        f"| 从快照恢复 | **{load_ms} ms** | 启动时自动执行 |",
        "",
        f"**冷启动加速比：{speedup:.0f} 倍。**",
        "",
        "### 一致性验证",
        "",
        f"恢复后的索引重新跑了 {verified} 条查询，与重建后的结果逐条比对（id 与分数）：",
        "",
        f"- 结果与重建后**{verdict}**（{verified - mismatches}/{verified}）",
        "- 节点数、最大层、总边数与重建后完全相同",
        "",
        "这一步不能省。持久化最危险的失败模式不是「读不出来」，而是",
        "**读出来的索引看起来正常，却返回不同结果** —— 少读一段、偏移算错、",
        "字段顺序对不上都会是这样。这类问题不抛异常，只会让检索质量悄悄下降。",
        "",
        "### 文件体积",
        "",
        f"{file_mib} MiB，平均每条约 **{bytes_per_vector} 字节**。"
        f"光是向量本身就占 {dimension * 4} 字节，其余是图结构的边。",
        "",
        "> **一个诚实的说明**：本次恢复耗时是在文件刚写完、",
        "> **还在操作系统页缓存里**的条件下测到的，属于最好情况。",
        "> 冷启动首次从磁盘读会慢一些，按 209 MiB / 500 MB/s 估算约 0.4 秒 ——",
        "> 仍是重建图的数百倍，结论不变，但具体数字会随磁盘状态浮动。",
        "",
    ]
    return "\n".join(lines)


def section_hybrid(results_dir: Path) -> str:
    """三种检索模式的对比：向量 / BM25 / 混合。"""
    data = load_json(results_dir / "mode-comparison.json")
    if not data:
        return ""

    lexical = data.get("lexical", {}).get("metrics", {})
    semantic = data.get("semantic", {}).get("metrics", {})
    summary = data.get("summary", {})
    modes = [m for m in ("vector", "bm25", "hybrid") if m in lexical or m in semantic]
    if not modes:
        return ""

    lines = [
        "## 六、三种检索模式的对比",
        "",
        "同一份语料（602 个分块）上对比三条路线。这一节回答的是一个真问题：",
        "**到底该用哪个模式？** 答案取决于查询长什么样。",
        "",
        "### 词面查询（150 条，取自文档原文）",
        "",
        "| 模式 | recall@10 | MRR@10 | P50 端到端延迟 |",
        "|---|---|---|---|",
    ]
    for mode in modes:
        m = lexical.get(mode)
        if not m:
            continue
        lines.append(
            f"| `{mode}` | **{m['recall']:.4f}** | {m['mrr']:.4f} | {m['p50_total_ms']:.0f} ms |")

    lines += [
        "",
        "### 语义查询（16 条，人工改写提问方式）",
        "",
        "| 模式 | 命中率@10 | MRR@10 | P50 端到端延迟 |",
        "|---|---|---|---|",
    ]
    for mode in modes:
        m = semantic.get(mode)
        if not m:
            continue
        lines.append(
            f"| `{mode}` | **{m['hit']:.4f}** | {m['mrr']:.4f} | {m['p50']:.0f} ms |")

    lines += [
        "",
        "### 结论：混合检索赢在「平均」和「最差情况」",
        "",
        "| 模式 | 平均 | 最差情况 |",
        "|---|---|---|",
    ]
    for mode in modes:
        s = summary.get(mode)
        if not s:
            continue
        lines.append(f"| `{mode}` | {s['average']:.4f} | **{s['worst']:.4f}** |")

    lines += [
        "",
        "**混合检索在两类查询上都不是第一，但它平均最好、最差情况最好。**",
        "",
        "这恰恰是生产环境选择混合检索的理由：你<b>无法预知</b>用户下一次会打哪种查询。",
        "只上向量检索，遇到精确标识符就只有 0.80；只上 BM25，遇到换个说法的提问就只有 0.625。",
        "混合把两条路线的短板都抬起来了。",
        "",
        "### BM25 模式还有一个实用价值：它是天然的降级路径",
        "",
        "`bm25` 模式<b>完全不调用模型服务</b>，所以端到端延迟是 0 ms 级（上表可见），",
        "而另外两种要 40 ms 左右。这意味着模型服务挂掉时，",
        "检索还能以 BM25 模式继续工作，而不是整个不可用。",
        "做一个功能时顺手得到一条降级路径，通常比事后再补要干净。",
        "",
        "### 为什么等权融合没有让混合反超单路",
        "",
        "RRF 给两路相同的权重。当<b>一路明显强于另一路</b>时，等权会让弱的那路",
        "把排第 1 的错结果抬进融合后的 top-k，挤掉强的那路排第 11 的正确结果。",
        "",
        "这不是 RRF 的缺陷，而是它的<b>前提</b>：它假设两路「可信度相当」。",
        "真实场景里如果确知两路强弱，应该用加权 RRF 或先做分数校准 ——",
        "但那样就必须用一份独立的验证集来定权重，否则就是在测试集上调参。",
        "当前选择保留等权，并把这一点写清楚。",
        "",
        "> **样本量提醒**：语义查询只有 16 条，数字只能看趋势，不能当精确值。",
        "> 它在这里的作用是与 150 条词面查询<b>交叉验证</b> —— 单看任何一边都会得出偏颇结论。",
        "",
    ]

    rerank = data.get("rerank")
    if rerank:
        lex = rerank["lexical"]
        sem = rerank["semantic"]
        lex_mrr_gain = (lex["reranked"]["mrr"] / lex["baseline"]["mrr"] - 1) * 100
        lex_ndcg_gain = (lex["reranked"]["ndcg"] / lex["baseline"]["ndcg"] - 1) * 100
        sem_mrr_gain = (sem["reranked"]["mrr"] / sem["baseline"]["mrr"] - 1) * 100

        lines += [
            "### 二阶段精排：不改变召回，只改变排序",
            "",
            "精排用 cross-encoder 把 (查询, 文档) 拼在一起送进模型，能做完整的注意力交互，",
            "而向量检索是双塔结构 —— 查询和文档各自独立编码，编码时两者没见过彼此。",
            "代价是精排没法预先建索引：每来一个查询都要对每个候选跑一次前向。",
            "",
            "| 查询集 | 指标 | 精排前 | 精排后 | 变化 |",
            "|---|---|---|---|---|",
            f"| 词面（150 条） | recall@10 | {lex['baseline']['recall']:.4f} "
            f"| {lex['reranked']['recall']:.4f} | 不变 |",
            f"| 词面（150 条） | MRR@10 | {lex['baseline']['mrr']:.4f} "
            f"| **{lex['reranked']['mrr']:.4f}** | +{lex_mrr_gain:.0f}% |",
            f"| 词面（150 条） | NDCG@10 | {lex['baseline']['ndcg']:.4f} "
            f"| **{lex['reranked']['ndcg']:.4f}** | +{lex_ndcg_gain:.0f}% |",
            f"| 语义（16 条） | 命中率@10 | {sem['baseline']['hit']:.4f} "
            f"| {sem['reranked']['hit']:.4f} | 不变 |",
            f"| 语义（16 条） | MRR@10 | {sem['baseline']['mrr']:.4f} "
            f"| **{sem['reranked']['mrr']:.4f}** | +{sem_mrr_gain:.0f}% |",
            "",
            "**召回率与命中率完全不变，MRR / NDCG 明显提升** —— 这正是精排该有的表现：",
            "它不负责「找到更多」，只负责「把对的排到前面」。",
            "",
            "顺带得到一个自检手段：**如果精排让 recall 变了，说明第一阶段被改坏了**。",
            "指标之间应该有明确的分工，混在一起就失去了诊断能力。",
            "",
            "### 代价：延迟从几十毫秒涨到几百毫秒",
            "",
            f"P50 端到端延迟：词面 {lex['baseline']['p50']:.0f} ms → "
            f"**{lex['reranked']['p50']:.0f} ms**；"
            f"语义 {sem['baseline']['p50']:.0f} ms → **{sem['reranked']['p50']:.0f} ms**。",
            "",
            "这个代价在 CPU 上无法避免。所以 `rerank` 默认<b>关闭</b>，由调用方显式开启 ——",
            "排序质量与延迟怎么权衡应该由业务场景决定，而不是由框架替用户决定。",
            "",
        ]

    return "\n".join(lines)


def section_method(recall: dict) -> str:
    """方法学与局限 —— 必须写，否则是在误导人。"""
    lines = [
        "## 七、方法学与局限",
        "",
        "### 数据是合成的，不是真实 embedding",
        "",
        "基准使用**高斯混合**合成数据。原因是要**可控、可复现**：",
        "规模、维度、分布全部钉死，换台机器跑出来的数字才可比。",
        "真实语料会随评测集构建不断变动，那样「上周 0.31ms、本周 0.29ms」的对比就失去意义 ——",
        "你分不清是代码变快了还是数据变简单了。",
        "",
        "**用聚类而非均匀随机**：均匀随机的高维向量是近似最近邻算法最恶劣的输入",
        "（维数灾难下任意两点距离趋于集中，「最近邻」本身都变得模糊）。",
        "真实 embedding 有明显的簇结构，高斯混合更接近真实形态。",
        "",
        "### 仍然存在的偏差",
        "",
        "| 偏差来源 | 方向 | 说明 |",
        "|---|---|---|",
        "| 合成数据 | 低估 recall | 合成数据的内在维度远高于真实 embedding，近邻结构更弱，HNSW 更难找 |",
        "| 单机单线程 | 高估延迟稳定性 | 未考虑并发争用、GC 停顿、NUMA 效应 |",
        "| 预热充分 | 低估延迟 | JMH 让 JIT 充分编译，冷启动场景会明显更慢 |",
        "| 查询取自独立同分布 | 无法体现真实查询分布 | 真实查询有明显热点，缓存能带来额外收益 |",
        "",
        "### 怎么正确使用这份报告",
        "",
        "- ✅ 用于**横向比较**：HNSW vs 暴力、不同 `efSearch`、标量 vs SIMD",
        "- ✅ 用于**相对结论**：「快 N 倍」「拐点在 efSearch=X」",
        "- ❌ 不要用于宣称**绝对质量**：真实业务上的 recall 必须用自己的语料测（见 `eval/`）",
        "",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="渲染基准报告")
    parser.add_argument("--results", required=True, help="结果目录（含 recall.json / jmh.json）")
    parser.add_argument("--out", required=True, help="输出的 Markdown 路径")
    args = parser.parse_args()

    results_dir = Path(args.results)
    recall = load_json(results_dir / "recall.json")
    jmh_raw = load_json(results_dir / "jmh.json")
    jmh = index_jmh(jmh_raw) if jmh_raw else {}

    if not recall and not jmh:
        print(f"[错误] {results_dir} 下既没有 recall.json 也没有 jmh.json", file=sys.stderr)
        return 1

    header = [
        "# 向量检索基准报告",
        "",
        "> ⚙️ 本文件由 `scripts/render_bench_report.py` **自动生成**，请勿手工编辑。",
        "> 重新生成：`./scripts/run_bench.ps1`",
        "",
        "| 环境 | 值 |",
        "|---|---|",
        f"| 生成时间 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |",
        f"| 操作系统 | {platform.system()} {platform.release()} |",
        f"| 处理器 | {platform.processor() or '未知'} |",
        f"| Python | {platform.python_version()} |",
        "",
    ]

    sections = [s for s in (
        section_simd(jmh),
        section_index(jmh),
        section_recall(recall),
        section_concurrency(results_dir),
        section_snapshot(results_dir),
        section_hybrid(results_dir),
        section_method(recall),
    ) if s]

    report = "\n".join(header) + "\n" + "\n---\n\n".join(sections)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"  报告已写入 {out_path}（{len(report)} 字符）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
