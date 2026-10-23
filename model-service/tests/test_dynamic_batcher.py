"""DynamicBatcher 的单元测试。

## 为什么这个类能测、而 EmbeddingModel 不能

`DynamicBatcher` 的构造参数里有一个 `encode_fn` —— 推理被注入进来，
所以测批处理逻辑时不需要 torch、不需要下载 1.1 GB 的模型权重。
这正是它当初被写成这个形状（而不是在内部直接调模型）的原因之一。

顺带一提：`EmbeddingModel` 里的 torch 是**方法内延迟 import** 的，
所以 `import app.embedder` 在没装 torch 的机器上也能成功 —— CI 只装 numpy 就够。

## 测什么

批处理最容易错的地方不是「合并」，而是**合并之后怎么切回来**：
每个请求必须拿回严格属于自己那一段，顺序不能错。
所以这里的假编码器把「文本长度」和「首字符码点」编码进向量，
断言时就能逐行验证每一行是不是对应那条文本。
"""

import asyncio

import numpy as np

from app.embedder import DynamicBatcher


def run(coro):
    """在同步测试里跑一段 async 代码，避免引入 pytest-asyncio 依赖。"""
    return asyncio.run(coro)


class RecordingEncoder:
    """假编码器：记录每次被调用时收到的文本，并按文本内容产出可辨识的向量。"""

    def __init__(self, dimension: int = 3) -> None:
        self.calls: list[list[str]] = []
        self.dimension = dimension

    def __call__(self, texts):
        texts = list(texts)
        self.calls.append(texts)
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            # 第 0 列 = 字符数，第 1 列 = 首字符码点：足以判断这一行对应哪条文本
            out[i, 0] = float(len(text))
            out[i, 1] = float(ord(text[0])) if text else 0.0
        return out

    @property
    def batch_count(self) -> int:
        return len(self.calls)

    @property
    def texts_seen(self) -> int:
        return sum(len(c) for c in self.calls)

    @property
    def max_batch(self) -> int:
        return max((len(c) for c in self.calls), default=0)


class ExplodingEncoder:
    """每次都抛异常的编码器，用来验证异常能否传给所有等待者。"""

    def __call__(self, texts):
        raise RuntimeError("模型炸了")


def test_empty_input_returns_empty_array():
    async def main():
        batcher = DynamicBatcher(RecordingEncoder(), max_batch_size=32, timeout_ms=10)
        await batcher.start()
        try:
            out = await batcher.encode([])
            assert out.shape == (0, 0)
            assert out.dtype == np.float32
        finally:
            await batcher.stop()

    run(main())


def test_single_request_keeps_order():
    async def main():
        batcher = DynamicBatcher(RecordingEncoder(), max_batch_size=32, timeout_ms=20)
        await batcher.start()
        try:
            texts = ["短", "长一点点", "中等"]
            out = await batcher.encode(texts)

            assert out.shape == (3, 3)
            # 逐行验证：第 i 行必须对应第 i 条文本
            assert [row[0] for row in out] == [len(t) for t in texts]
        finally:
            await batcher.stop()

    run(main())


def test_concurrent_requests_are_merged_into_fewer_forwards():
    """10 个并发请求应该被合并，而不是每个都单独 forward 一次。"""

    async def main():
        encoder = RecordingEncoder()
        batcher = DynamicBatcher(encoder, max_batch_size=32, timeout_ms=60)
        await batcher.start()
        try:
            results = await asyncio.gather(
                *(batcher.encode([f"req{i}-text"]) for i in range(10))
            )

            # 合并生效：forward 次数必须远少于请求数
            assert encoder.batch_count < 10, f"没有合并：{encoder.batch_count} 次 forward"
            # 但文本一条都不能丢
            assert encoder.texts_seen == 10
            # 软目标 32 条，单个请求只有 1 条 → 任何一批都不该超过 32
            assert encoder.max_batch <= 32

            # 每个请求拿回的仍必须是自己的那一条
            for i, result in enumerate(results):
                assert result.shape[0] == 1
                assert result[0][0] == len(f"req{i}-text")
        finally:
            await batcher.stop()

    run(main())


def test_batch_size_limits_texts_not_requests():
    """max_batch_size 限制的是文本条数，不是请求个数。

    两个各 50 条的请求、软目标 16：任何一个请求自己就超限，所以它们
    **不能**被合并成一次 100 条的 forward —— 那会瞬间打爆内存。
    """

    async def main():
        encoder = RecordingEncoder()
        batcher = DynamicBatcher(encoder, max_batch_size=16, timeout_ms=30)
        await batcher.start()
        try:
            first = [f"a-{i}" for i in range(50)]
            second = [f"b-{i}" for i in range(50)]
            results = await asyncio.gather(batcher.encode(first), batcher.encode(second))

            assert results[0].shape[0] == 50, "第一个请求少了向量"
            assert results[1].shape[0] == 50, "第二个请求少了向量"
            assert results[0][0][0] == len(first[0]), "第一个请求的首条向量对应错了"
            assert results[1][0][0] == len(second[0]), "第二个请求的首条向量对应错了"

            # 关键断言：没有任何一次 forward 同时吞下两个请求（那会是 100 条）
            assert encoder.max_batch <= 50, f"两个大请求被硬凑成 {encoder.max_batch} 条"
        finally:
            await batcher.stop()

    run(main())


def test_each_request_gets_its_own_slice():
    """长度参差不齐的多个请求：切回来的边界必须严格对齐。"""

    async def main():
        encoder = RecordingEncoder()
        batcher = DynamicBatcher(encoder, max_batch_size=64, timeout_ms=50)
        await batcher.start()
        try:
            groups = [["a"], ["bb", "ccc"], ["dddd"], ["e", "ff", "ggg"]]
            outs = await asyncio.gather(*(batcher.encode(g) for g in groups))

            for group, out in zip(groups, outs):
                assert out.shape[0] == len(group), "行数与请求的文本条数不符"
                assert [row[0] for row in out] == [len(t) for t in group], "切片错位了"

            assert encoder.texts_seen == 1 + 2 + 1 + 3
        finally:
            await batcher.stop()

    run(main())


def test_encode_failure_reaches_every_waiter():
    """推理失败时不能让调用方永远挂着 —— 每个等待者都要收到异常。"""

    async def main():
        batcher = DynamicBatcher(ExplodingEncoder(), max_batch_size=32, timeout_ms=30)
        await batcher.start()
        try:
            results = await asyncio.gather(
                batcher.encode(["x"]),
                batcher.encode(["y"]),
                batcher.encode(["z"]),
                return_exceptions=True,
            )
            assert len(results) == 3
            for r in results:
                assert isinstance(r, RuntimeError), f"期望异常，实际 {type(r)}"
        finally:
            await batcher.stop()

    run(main())


def test_request_bigger_than_max_batch_is_still_processed():
    """max_batch_size 是软目标：单个请求本来就超限时也必须处理，不能卡死。"""

    async def main():
        encoder = RecordingEncoder()
        batcher = DynamicBatcher(encoder, max_batch_size=4, timeout_ms=10)
        await batcher.start()
        try:
            texts = [f"t{i:02d}" for i in range(20)]  # 20 条，远超软目标 4
            out = await batcher.encode(texts)

            assert out.shape[0] == 20
            assert [row[0] for row in out] == [len(t) for t in texts]
        finally:
            await batcher.stop()

    run(main())


def test_lone_request_is_flushed_after_timeout():
    """只有一个请求时，靠超时兜底把它送出去 —— 不能等到下一个请求来了才处理。"""

    async def main():
        batcher = DynamicBatcher(RecordingEncoder(), max_batch_size=32, timeout_ms=30)
        await batcher.start()
        try:
            # wait_for 是防挂死的保险：真挂住了这里会超时失败，而不是让 CI 一直等
            out = await asyncio.wait_for(batcher.encode(["孤零零一个请求"]), timeout=2.0)
            assert out.shape[0] == 1
        finally:
            await batcher.stop()

    run(main())


def test_stats_counts_batches_and_texts():
    async def main():
        batcher = DynamicBatcher(RecordingEncoder(), max_batch_size=32, timeout_ms=20)
        await batcher.start()
        try:
            await batcher.encode(["aa", "bbb"])
            stats = batcher.stats()

            assert stats["batches"] >= 1
            assert stats["texts"] == 2
            assert stats["max_observed_batch"] >= 2
            assert stats["average_batch_size"] == stats["texts"] / stats["batches"]
        finally:
            await batcher.stop()

    run(main())


def test_stop_without_start_is_noop():
    """没 start 就 stop 不能抛异常 —— 应用关闭时路径不总是对称的。"""
    batcher = DynamicBatcher(RecordingEncoder(), max_batch_size=8, timeout_ms=10)
    run(batcher.stop())
