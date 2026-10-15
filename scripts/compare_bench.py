"""对比两次基准结果，生成改造前后的差异表。

## 为什么需要这个脚本

性能优化最忌讳"改完就宣称变快了"。你必须能回答：
改之前是多少、改之后是多少、中间有没有把别的指标弄坏。

这个脚本读两份 `RecallRunner` 产出的 JSON，逐项对照，
并且**主动检查优化是否损害了召回率** —— 这是最容易在优化中被悄悄牺牲的东西。

## 用法

    python scripts/compare_bench.py --before bench/results/recall-old.json \
                                    --after  bench/results/recall.json

也可以用 `--save-baseline` 把当前结果另存一份，方便后续对比：

    python scripts/compare_bench.py --save-baseline bench/results/recall.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Windows 控制台默认用 GBK 编码，直接 print 非 ASCII 字符（µ、✅ 等）
# 会抛 UnicodeEncodeError 并让脚本中途崩溃 —— 而不是打印乱码那么无害。
# 显式切到 UTF-8，无法编码的字符降级处理而不是抛异常。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 召回率下降超过这个幅度就报警。近似索引在不同实现下会有微小差异，
# 所以不要求完全一致，但明显下降必须被看见。
RECALL_REGRESSION_THRESHOLD = 0.005


def load(path: Path) -> dict:
    if not path.exists():
        print(f"[错误] 找不到 {path}", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text(encoding="utf-8"))


def mib(value: float) -> str:
    return f"{value / 1024 / 1024:,.1f}"


def speedup(before: float, after: float) -> str:
    if after <= 0:
        return "—"
    ratio = before / after
    if ratio >= 1.05:
        return f"{ratio:.2f}x 更快"
    if ratio <= 0.95:
        return f"{1 / ratio:.2f}x 更慢"
    return "持平"


def main() -> int:
    parser = argparse.ArgumentParser(description="对比两次基准结果")
    parser.add_argument("--before", help="改造前的 recall.json")
    parser.add_argument("--after", help="改造后的 recall.json")
    parser.add_argument("--save-baseline", metavar="RECALL_JSON",
                        help="把当前结果另存为基线，然后退出")
    parser.add_argument("--baseline-dir", default="bench/results",
                        help="基线存放目录（与 --save-baseline 配合）")
    args = parser.parse_args()

    if args.save_baseline:
        source = Path(args.save_baseline)
        if not source.exists():
            print(f"[错误] 找不到 {source}", file=sys.stderr)
            return 1
        target = Path(args.baseline_dir) / "recall-baseline.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, target)
        print(f"基线已保存: {target}")
        print("之后用 --before 指向它即可对比。")
        return 0

    if not args.before or not args.after:
        parser.error("需要同时提供 --before 与 --after（或使用 --save-baseline）")

    before = load(Path(args.before))
    after = load(Path(args.after))

    b_hnsw, a_hnsw = before["hnsw"], after["hnsw"]
    b_bf, a_bf = before["bruteForce"], after["bruteForce"]

    print("=" * 70)
    print("基准对比")
    print("=" * 70)
    print(f"  改造前: {args.before}")
    print(f"  改造后: {args.after}")

    # 规模必须一致，否则数字不可比 —— 这是最容易被忽略的前提
    for key in ("nodeCount", "dimension", "topK", "queryCount"):
        bv, av = before.get(key), after.get(key)
        if bv != av:
            print(f"\n[警告] {key} 不一致：{bv} vs {av} —— 下面数字不可比！",
                  file=sys.stderr)

    print()
    print("-" * 70)
    print(f"{'结构指标':<24}{'改造前':>16}{'改造后':>16}{'变化':>14}")
    print("-" * 70)

    for label, key, fmt in [
        ("建图耗时 (s)", "buildMillis", lambda v: f"{v / 1000:,.1f}"),
        ("最大层", "maxLevel", lambda v: f"{v:,}"),
        ("总边数", "totalLinks", lambda v: f"{v:,}"),
        ("平均出边", "averageDegree", lambda v: f"{v:,.1f}"),
        ("堆内占用增量 (MB)", "heapDeltaBytes", mib),
        ("估算占用 (MB)", "estimatedBytes", mib),
    ]:
        bv, av = b_hnsw.get(key), a_hnsw.get(key)
        if bv is None or av is None:
            continue
        change = speedup(float(bv), float(av)) if key in ("buildMillis",) else ""
        print(f"{label:<24}{fmt(bv):>16}{fmt(av):>16}{change:>14}")

    print()
    print("-" * 70)
    print(f"{'efSearch':<10}{'延迟(µs)':>22}{'recall@k':>24}")
    print(f"{'':<10}{'改造前':>10}{'改造后':>12}{'改造前':>11}{'改造后':>13}")
    print("-" * 70)

    b_points = {p["efSearch"]: p for p in b_hnsw.get("points", [])}
    a_points = {p["efSearch"]: p for p in a_hnsw.get("points", [])}

    regressions: list[str] = []
    for ef in sorted(set(b_points) & set(a_points)):
        bp, ap = b_points[ef], a_points[ef]
        print(f"{ef:<10}{bp['meanMicros']:>10,.1f}{ap['meanMicros']:>12,.1f}"
              f"{bp['recallAtK']:>11.4f}{ap['recallAtK']:>13.4f}")
        if ap["recallAtK"] < bp["recallAtK"] - RECALL_REGRESSION_THRESHOLD:
            regressions.append(
                f"efSearch={ef}: recall {bp['recallAtK']:.4f} -> {ap['recallAtK']:.4f}")

    print()
    print("-" * 70)
    print(f"{'暴力检索基准':<24}{'改造前':>16}{'改造后':>16}")
    print("-" * 70)
    print(f"{'平均延迟 (µs)':<24}{b_bf['meanMicros']:>16,.1f}{a_bf['meanMicros']:>16,.1f}")
    print(f"{'建库耗时 (ms)':<24}{b_bf['buildMillis']:>16,}{a_bf['buildMillis']:>16,}")

    # 结论
    print()
    print("=" * 70)
    if regressions:
        print("⚠️  召回率出现回退：")
        for line in regressions:
            print(f"    {line}")
        print()
        print("    优化不能以牺牲检索质量为代价。请确认这是可接受的近似误差，")
        print("    还是实现中引入了缺陷（先看穷举对照测试是否仍然通过）。")
    else:
        print("✅ 召回率没有回退 —— 优化没有牺牲检索质量。")

    build_before = b_hnsw["buildMillis"] / 1000
    build_after = a_hnsw["buildMillis"] / 1000
    print(f"\n    建图: {build_before:,.0f}s -> {build_after:,.0f}s "
          f"（{speedup(build_before, build_after)}）")
    if "heapDeltaBytes" in b_hnsw and "heapDeltaBytes" in a_hnsw:
        print(f"    堆占用: {mib(b_hnsw['heapDeltaBytes'])} MB -> "
              f"{mib(a_hnsw['heapDeltaBytes'])} MB "
              f"（{speedup(b_hnsw['heapDeltaBytes'], a_hnsw['heapDeltaBytes'])}）")
    print("=" * 70)

    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
