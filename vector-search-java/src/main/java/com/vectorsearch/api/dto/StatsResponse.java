package com.vectorsearch.api.dto;

/**
 * 服务与索引状态。
 *
 * <p>把 HNSW 的图结构指标暴露出来，是为了能在浏览器里直接确认
 * 「图建对了没」：层高分布应该呈金字塔状衰减，总边数应该接近
 * {@code 节点数 × m}。如果这两条不对，recall 一定有问题。
 *
 * @param indexType         当前索引实现
 * @param indexedCount      索引中的向量数
 * @param dimension         向量维度
 * @param storedDocuments   文档原文数量，应与 indexedCount 一致
 * @param characterCount    文档总字符数
 * @param hnsw              HNSW 结构指标；暴力检索时为 null
 * @param modelServiceHealthy Python 模型服务是否可达
 */
public record StatsResponse(
        String indexType,
        int indexedCount,
        int dimension,
        int storedDocuments,
        long characterCount,
        HnswStats hnsw,
        Long restoredInMillis,
        boolean modelServiceHealthy
) {

    /**
     * @param m               每层最大出边数
     * @param efConstruction  建图候选池大小
     * @param maxLevel        图的最大层号
     * @param totalLinks      全图总边数
     * @param levelHistogram  各层节点数，index 即层号
     */
    public record HnswStats(
            int m,
            int efConstruction,
            int maxLevel,
            long totalLinks,
            int[] levelHistogram
    ) {
    }
}
