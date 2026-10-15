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


def section_method(recall: dict) -> str:
    """方法学与局限 —— 必须写，否则是在误导人。"""
    lines = [
        "## 四、方法学与局限",
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
