package com.vectorsearch.api.dto;

import java.util.List;

/**
 * 检索响应。
 *
 * <p><b>{@code timing} 是刻意暴露的</b>：把「编码」和「检索」两段耗时分开报出来，
 * 能立刻看出瓶颈在哪。如果 embeddingMs 远大于 searchMs，说明优化方向应该是
 * 减少跨进程调用次数，而不是去调 HNSW 参数 —— 这个字段就是让性能问题可见。
 *
 * @param query   原始查询
 * @param hits    检索结果
 * @param indexType 实际使用的索引实现（hnsw / brute-force）
 * @param timing  各阶段耗时
 */
public record SearchResponse(
        String query,
        String mode,
        List<SearchHit> hits,
        String indexType,
        Timing timing
) {

    /**
     * @param embeddingMs 调 Python 服务做向量编码的耗时（含网络往返）。
     *                    {@code bm25} 模式不调模型，此值为 0。
     * @param searchMs    本地检索与融合耗时（不含编码）
     * @param totalMs     端到端耗时
     */
    public record Timing(long embeddingMs, long searchMs, long totalMs) {
    }
}
