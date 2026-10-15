"""动态批处理 + embedding。

## 为什么需要动态批处理

模型推理的成本几乎全部在「一次 forward」的固定开销上，不在文本条数上。
如果每个 HTTP 请求各自调用一次模型：

    100 个并发请求 -> 100 次 forward -> 每次只处理 1 条文本 -> 吞吐极低

动态批处理把时间窗口内到达的请求攒成一批，一次 forward 全部处理：

    100 个并发请求 -> 1 次 forward 处理 100 条文本 -> 吞吐提升一个数量级

代价是每个请求多等最多 batch_timeout_ms 毫秒。这就是**吞吐与延迟的权衡**，
也是这类系统最经典的面试题。

## 实现要点

- worker 是**单线程事件循环 + 线程池执行推理**：推理是阻塞的 CPU/GPU 操作，
  直接放在事件循环里会卡死整个服务
- 用 future 把「攒批」和「等待结果」解耦，调用方看起来仍在 await 自己的请求
- 每个请求拿回的向量切片严格对应它自己的 texts，靠 offset 累积保证顺序不错乱
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class DynamicBatcher:
    """把并发的小请求聚合成大 batch。"""

    def __init__(
        self,
        encode_fn: Callable[[Sequence[str]], np.ndarray],
        max_batch_size: int,
        timeout_ms: int,
    ) -> None:
        self._encode = encode_fn
        self._max_batch_size = max(1, max_batch_size)
        self._timeout_s = max(0.0, timeout_ms / 1000.0)
        self._queue: asyncio.Queue[tuple[list[str], asyncio.Future[np.ndarray]]] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None
        self._batches = 0
        self._texts = 0
        self._max_observed = 0

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._run(), name="dynamic-batcher")
            logger.info(
                "动态批处理已启动: max_batch_size=%d, batch_timeout=%.1fms",
                self._max_batch_size,
                self._timeout_s * 1000,
            )

    async def stop(self) -> None:
        if self._worker_task is None:
            return
        self._worker_task.cancel()
        try:
            await self._worker_task
        except asyncio.CancelledError:
            pass
        self._worker_task = None

    # ------------------------------------------------------------------
    # 调用入口
    # ------------------------------------------------------------------

    async def encode(self, texts: Sequence[str]) -> np.ndarray:
        """编码一批文本。调用方感知不到攒批过程，只是 await 自己的结果。"""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        loop = asyncio.get_running_loop()
        future: asyncio.Future[np.ndarray] = loop.create_future()
        await self._queue.put((list(texts), future))
        return await future

    # ------------------------------------------------------------------
    # worker
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            first = await self._queue.get()
            batch = [first]
            # 按「文本条数」计数，而不是「请求个数」。
            # 一个请求可能带很多条文本，如果按请求数限制，
            # 32 个各含 100 条文本的请求会被塞进同一次 forward（3200 条），
            # 直接撑爆内存 —— max_batch_size 的语义必须是文本条数。
            pending_texts = len(first[0])
            deadline = loop.time() + self._timeout_s

            # 继续攒，直到攒够文本条数或超时。
            #
            # 注意：批量可能略微超过 max_batch_size，最多超出「单个最大请求的文本数 - 1」。
            # 原因是请求不可拆分 —— 从队列取出一个请求后，如果发现加上它会超限，
            # 我们既不能把它拆开，也不该把它塞回队列（会破坏 FIFO 顺序）。
            # 所以 max_batch_size 是「软目标」而非硬上限。
            #
            # 真正防止 OOM 的硬约束在模型侧：EmbeddingModel.encode_sync 会用
            # ENCODE_CHUNK_SIZE 把大批再切成小批，即使这里给了几千条也不会一次性
            # 送进模型。两个参数的关系见 config.py。
            while pending_texts < self._max_batch_size:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                batch.append(item)
                pending_texts += len(item[0])

            await self._dispatch(batch)

    async def _dispatch(
        self,
        batch: list[tuple[list[str], asyncio.Future[np.ndarray]]],
    ) -> None:
        flat: list[str] = [text for texts, _ in batch for text in texts]

        self._batches += 1
        self._texts += len(flat)
        self._max_observed = max(self._max_observed, len(flat))
        logger.debug("执行批次: %d 个请求 / %d 条文本", len(batch), len(flat))

        loop = asyncio.get_running_loop()
        try:
            # 推理是阻塞调用，必须扔到线程池，否则会阻塞事件循环、
            # 导致攒批机制本身失效（新请求进不来，永远攒不满）
            vectors = await loop.run_in_executor(None, self._encode, flat)
        except Exception as exc:  # noqa: BLE001 - 必须把异常传给每个等待者
            logger.exception("批量编码失败，共 %d 条文本", len(flat))
            for _, future in batch:
                if not future.done():
                    future.set_exception(exc)
            return

        # 按 offset 切回各请求自己的那一段，顺序严格对应
        offset = 0
        for texts, future in batch:
            size = len(texts)
            chunk = vectors[offset : offset + size]
            offset += size
            if not future.done():
                future.set_result(chunk)

    # ------------------------------------------------------------------
    # 观测
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, float | int]:
        return {
            "batches": self._batches,
            "texts": self._texts,
            "max_observed_batch": self._max_observed,
            "average_batch_size": (self._texts / self._batches) if self._batches else 0.0,
        }


class EmbeddingModel:
    """BGE 句向量模型封装。"""

    def __init__(self, model_name: str, device: str, chunk_size: int = 64) -> None:
        # 延迟 import：让 --help / 配置校验之类的路径不必加载 torch
        from sentence_transformers import SentenceTransformer

        logger.info("正在加载 embedding 模型: %s (device=%s)", model_name, device)
        self._model = SentenceTransformer(model_name, device=device)
        self._model_name = model_name
        self._chunk_size = max(1, chunk_size)
        self._dimension = int(self._model.get_sentence_embedding_dimension())
        logger.info("模型加载完成，输出维度 %d", self._dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def name(self) -> str:
        return self._model_name

    def encode_sync(self, texts: Sequence[str]) -> np.ndarray:
        """同步编码 —— 由 DynamicBatcher 在线程池里调用。

        normalize_embeddings=True 让输出直接是单位向量，
        这样内积就等于余弦相似度，Java 侧不必再做一次归一化。
        （Java 侧仍会归一化，是幂等的，属于防御性冗余。）
        """
        return self._model.encode(
            list(texts),
            batch_size=self._chunk_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
