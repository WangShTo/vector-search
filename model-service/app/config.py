"""模型服务配置。

刻意不引入 pydantic-settings：环境变量一共就这么几项，
一个 dataclass 足够，少一个依赖就少一个版本冲突的可能。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# HuggingFace 官方域名在国内直连不通（实测连接超时），必须在
# huggingface_hub 被导入之前把镜像写进环境变量，否则模型下载会静默挂住，
# 表现为服务启动时卡住十几分钟然后报一个看不懂的连接错误。
#
# 这段代码依赖一个前提：config 是所有 sentence_transformers / transformers
# 导入之前第一个被加载的模块。main.py 的 import 顺序保证了这一点，
# 改动导入顺序时务必保持 config 在最前面。
# 需要切回官方源时：设置环境变量 HF_ENDPOINT=https://huggingface.co
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """服务运行参数，全部可由环境变量覆盖。"""

    # ---- 模型 ----
    # BGE-small-zh-v1.5 输出 768 维，必须与 Java 侧
    # vector-search.index.dimension 保持一致，否则会在写入索引时报错。
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    device: str = os.getenv("EMBEDDING_DEVICE", "cpu")

    # ---- 动态批处理 ----
    # 攒够 max_batch_size 条文本就立刻发车；否则最多等 batch_timeout_ms 毫秒。
    # 这两个值是延迟与吞吐的权衡旋钮：
    #   调大批量 -> 吞吐更高、单条请求延迟更高
    #   调小超时 -> 延迟更低、批量更难攒满、吞吐下降
    # 第 6 周会跑不同组合的压测，画出这条权衡曲线。
    #
    # 注意 max_batch_size 是「软目标」：请求不可拆分，所以实际批量最多会
    # 超出「单个最大请求的文本数 - 1」。见 embedder.py 的说明。
    max_batch_size: int = _env_int("MAX_BATCH_SIZE", 32)
    batch_timeout_ms: int = _env_int("BATCH_TIMEOUT_MS", 10)

    # ---- 模型内部分批（硬约束）----
    # 一次 forward 太大会撑爆内存，这里再切一刀 —— 这才是真正防 OOM 的边界，
    # 无论上游攒了多大的批，送进模型的永远不超过 encode_chunk_size 条。
    encode_chunk_size: int = _env_int("ENCODE_CHUNK_SIZE", 64)

    @property
    def batch_timeout_seconds(self) -> float:
        return self.batch_timeout_ms / 1000.0


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
