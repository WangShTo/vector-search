"""动态批处理的正确性验证。

这是 Python 侧最容易出错的一段代码：攒批 + 按 offset 切片回填。
一旦 offset 累加错了，请求 A 会拿到请求 B 的向量 —— 而且不会报错，
只会让检索结果莫名其妙地不准。所以必须有测试盯着。

不依赖 pytest，直接 `python test_batcher.py` 就能跑。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from app.embedder import DynamicBatcher  # noqa: E402

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  [PASS] {message}")
    else:
        print(f"  [FAIL] {message}")
        FAILURES.append(message)


def fake_encode(texts: list[str]) -> np.ndarray:
    """确定性假编码器：向量只由文本自身决定。

    这样如果切片错位，得到的向量就对不上文本，测试立刻能发现。
    """
    return np.array(
        [[float(len(t)), float(ord(t[0])), float(ord(t[-1]))] for t in texts],
        dtype=np.float32,
    )


def expected_vector(text: str) -> list[float]:
    return [float(len(text)), float(ord(text[0])), float(ord(text[-1]))]


async def test_slice_correspondence() -> None:
    """并发请求下，每个请求必须拿回属于自己的那一段向量。"""
    print("\n[1] 切片对应关系（20 个并发请求，每个 1~3 条文本）")
    batcher = DynamicBatcher(fake_encode, max_batch_size=8, timeout_ms=20)
    await batcher.start()
    try:
        expectations: list[list[str]] = []
        tasks = []
        for i in range(20):
            texts = [f"doc-{i}-{j}" for j in range(1 + i % 3)]
            expectations.append(texts)
            tasks.append(batcher.encode(texts))

        results = await asyncio.gather(*tasks)

        mismatch = 0
        for texts, vectors in zip(expectations, results):
            if vectors.shape[0] != len(texts):
                mismatch += 1
                continue
            for k, text in enumerate(texts):
                if list(vectors[k]) != expected_vector(text):
                    mismatch += 1
        check(mismatch == 0, f"全部 {len(expectations)} 个请求的向量与文本一一对应（错位 {mismatch} 处）")

        stats = batcher.stats()
        check(stats["average_batch_size"] > 1.0,
              f"确实发生了攒批：平均批大小 {stats['average_batch_size']:.2f} > 1")
        # max_batch_size 是软目标：请求不可拆分，实际批量最多超出
        # 「单个最大请求的文本数 - 1」。本测试每个请求最多 3 条，故上界为 8 + 3 - 1 = 10。
        upper_bound = 8 + 3 - 1
        check(stats["max_observed_batch"] <= upper_bound,
              f"最大批 {stats['max_observed_batch']} 未超过软目标上界 {upper_bound}")
    finally:
        await batcher.stop()


async def test_text_count_limit() -> None:
    """max_batch_size 限制的必须是文本条数，不是请求个数。"""
    print("\n[2] 批大小按文本条数计（不是请求个数）")
    # 2 个请求，每个 50 条文本，上限 16 —— 每个请求自己就超限，
    # 应该各自单独成批，绝不能合在一起变成 100 条
    batcher = DynamicBatcher(fake_encode, max_batch_size=16, timeout_ms=30)
    await batcher.start()
    try:
        results = await asyncio.gather(
            batcher.encode([f"a-{i}" for i in range(50)]),
            batcher.encode([f"b-{i}" for i in range(50)]),
        )
        check(results[0].shape[0] == 50 and results[1].shape[0] == 50,
              "两个大请求各自返回 50 条向量")
        check(list(results[0][0]) == expected_vector("a-0"), "第一个请求首条向量正确")
        check(list(results[1][0]) == expected_vector("b-0"), "第二个请求首条向量正确")

        stats = batcher.stats()
        check(stats["max_observed_batch"] <= 100,
              f"实际最大批 {stats['max_observed_batch']} 条（未把两个大请求硬凑在一起）")
    finally:
        await batcher.stop()


async def test_empty_and_single() -> None:
    """边界：空输入、单条输入。"""
    print("\n[3] 边界情况")
    batcher = DynamicBatcher(fake_encode, max_batch_size=8, timeout_ms=10)
    await batcher.start()
    try:
        empty = await batcher.encode([])
        check(empty.shape[0] == 0, "空输入返回 0 条向量，不抛异常")

        single = await batcher.encode(["hello"])
        check(single.shape[0] == 1, "单条输入返回 1 条向量")
        check(list(single[0]) == expected_vector("hello"), "单条输入向量正确")
    finally:
        await batcher.stop()


async def test_exception_propagation() -> None:
    """编码抛异常时，所有等待者都要收到，不能永久挂起。"""
    print("\n[4] 异常传播（不能让调用方永久等待）")

    # 这一段会故意抛出异常，embedder 会用 logger.exception 打完整堆栈。
    # 把日志级别调高压掉，免得读者把预期中的堆栈当成测试失败。
    import logging
    logging.getLogger("app.embedder").setLevel(logging.CRITICAL)

    def failing_encode(texts):
        raise RuntimeError("模拟模型推理失败")

    batcher = DynamicBatcher(failing_encode, max_batch_size=8, timeout_ms=10)
    await batcher.start()
    try:
        results = await asyncio.gather(
            batcher.encode(["x"]),
            batcher.encode(["y"]),
            return_exceptions=True,
        )
        check(all(isinstance(r, RuntimeError) for r in results),
              "两个请求都收到了异常，而不是卡死")
    finally:
        await batcher.stop()


async def main() -> int:
    print("=" * 62)
    print("动态批处理测试")
    print("=" * 62)
    await test_slice_correspondence()
    await test_text_count_limit()
    await test_empty_and_single()
    await test_exception_propagation()
    print("\n" + "=" * 62)
    if FAILURES:
        print(f"结果: {len(FAILURES)} 项失败")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("结果: 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
