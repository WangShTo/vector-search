"""向量检索模型服务。

职责边界**刻意收得很窄**：这里只做模型推理，不做检索。
索引、BM25、融合排序全部在 Java 侧 —— 那样才能展示后端的存储与并发能力，
而不是写一个"AI 项目"然后所有逻辑都塞进 Python。

接口：
    GET  /health   探活
    POST /embed    批量文本 -> 向量
    POST /rerank   精排候选
    GET  /stats    批处理统计
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config import get_settings
from .embedder import DynamicBatcher, EmbeddingModel
from .reranker import LazyReranker
from .schemas import (
    BatchStats,
    EmbedRequest,
    EmbedResponse,
    HealthResponse,
    RerankItem,
    RerankRequest,
    RerankResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info(
        "启动模型服务: model=%s device=%s max_batch=%d timeout=%dms",
        settings.embedding_model,
        settings.device,
        settings.max_batch_size,
        settings.batch_timeout_ms,
    )

    # embedding 模型在启动时加载：它是每次检索的必经路径，没有懒加载的意义，
    # 早点失败（比如模型名写错、显存不够）比运行到一半才炸好
    embedder = EmbeddingModel(
        settings.embedding_model,
        settings.device,
        settings.encode_chunk_size,
    )
    batcher = DynamicBatcher(
        embedder.encode_sync,
        settings.max_batch_size,
        settings.batch_timeout_ms,
    )
    await batcher.start()

    app.state.settings = settings
    app.state.embedder = embedder
    app.state.batcher = batcher
    # reranker 懒加载，见 reranker.py 的说明
    app.state.reranker = LazyReranker(settings.reranker_model, settings.device)

    logger.info("模型服务就绪，向量维度 %d", embedder.dimension)
    try:
        yield
    finally:
        await batcher.stop()
        logger.info("模型服务已停止")


app = FastAPI(
    title="向量检索模型服务",
    description="BGE embedding + rerank，供 Java 检索服务批量调用",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    embedder = app.state.embedder
    return HealthResponse(
        status="ok",
        embedding_model=embedder.name,
        dimension=embedder.dimension,
        device=app.state.settings.device,
    )


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest) -> EmbedResponse:
    """批量编码。请求会进入动态批处理队列，与同时到达的其他请求合并成一批。"""
    vectors = await app.state.batcher.encode(request.texts)
    embedder = app.state.embedder
    return EmbedResponse(
        # tolist() 把 numpy 数组转成原生 float，否则 JSON 序列化会失败
        vectors=vectors.tolist(),
        dimension=embedder.dimension,
        model=embedder.name,
    )


@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest) -> RerankResponse:
    """精排。首次调用会触发 reranker 模型加载，会明显慢一些。"""
    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(
        None,
        app.state.reranker.rerank_sync,
        request.query,
        request.documents,
        request.top_n,
    )
    return RerankResponse(results=[RerankItem(**item) for item in results])


@app.get("/stats", response_model=BatchStats)
async def stats() -> BatchStats:
    """批处理统计。

    平均批大小是验证动态批处理是否生效的关键指标：
    如果它长期接近 1，说明请求根本没被攒起来，多半是
    uvicorn 的 worker 数 > 1，每个 worker 各有一个队列，请求被打散了。
    """
    return BatchStats(**app.state.batcher.stats())
