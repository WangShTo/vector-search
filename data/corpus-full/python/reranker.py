"""Cross-encoder 精排。

## 为什么需要它

向量检索（bi-encoder）把 query 和文档<b>分别</b>编码成向量再算内积。
这让它可以预先建索引、毫秒级召回百万级文档，但代价是：
query 和文档在编码时没见过彼此，细粒度的语义交互丢失了。

cross-encoder 把 (query, document) <b>拼在一起</b>送进模型，能做完整的
注意力交互，精度显著更高。代价是它没法预先计算 —— 每来一个 query，
都要对每个候选文档跑一次 forward，复杂度 O(候选数)。

所以两者是分工关系，不是替代关系：

    向量召回：百万 -> 100   （快、全、粗）
    rerank  ：100  -> 10    （慢、准、精）

## 为什么懒加载

reranker 模型（几百 MB 到 1GB）在服务启动时加载会让启动多等几十秒，
而第 1~3 周根本用不到它。所以改成首次调用时才加载，
并用双重检查锁保证并发首次调用不会重复加载。
"""

from __future__ import annotations

import logging
import threading
from typing import Sequence

logger = logging.getLogger(__name__)


class LazyReranker:
    """线程安全的懒加载 cross-encoder。"""

    def __init__(self, model_name: str, device: str) -> None:
        self._model_name = model_name
        self._device = device
        self._model = None
        self._load_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_loaded(self):
        # 快路径：已加载则完全不加锁
        if self._model is not None:
            return self._model

        with self._load_lock:
            # 双重检查：可能在等锁期间被别的线程加载完了
            if self._model is None:
                from sentence_transformers import CrossEncoder

                logger.info("首次 rerank 调用，加载模型: %s (device=%s)", self._model_name, self._device)
                self._model = CrossEncoder(self._model_name, device=self._device)
                logger.info("reranker 加载完成")
        return self._model

    def rerank_sync(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int,
    ) -> list[dict[str, float | int]]:
        """同步精排。由调用方放到线程池执行。"""
        model = self._ensure_loaded()
        pairs = [(query, document) for document in documents]
        scores = model.predict(pairs)

        ranked = sorted(
            ((index, float(score)) for index, score in enumerate(scores)),
            key=lambda item: item[1],
            reverse=True,
        )
        if top_n > 0:
            ranked = ranked[:top_n]

        return [{"index": index, "score": score} for index, score in ranked]
