"""HTTP 接口的数据契约。

字段名与 Java 侧 client/dto 下的 record 严格对应 ——
改这里的任何一个字段名，都必须同步改 Java 侧，否则会在运行时报错。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="待编码文本")


class EmbedResponse(BaseModel):
    vectors: list[list[float]]
    dimension: int
    model: str


class RerankRequest(BaseModel):
    query: str
    documents: list[str] = Field(..., min_length=1)
    # Java 侧发过来是 top_n（snake_case），保持一致
    top_n: int = 0


class RerankItem(BaseModel):
    index: int
    score: float


class RerankResponse(BaseModel):
    results: list[RerankItem]


class HealthResponse(BaseModel):
    status: str
    embedding_model: str
    dimension: int
    device: str


class BatchStats(BaseModel):
    """批处理统计，用于验证「动态批处理真的把请求攒起来了」。"""

    batches: int
    texts: int
    max_observed_batch: int
    average_batch_size: float
