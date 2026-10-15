package com.vectorsearch.core;

import java.util.List;

/**
 * 向量索引的统一抽象。
 *
 * <p>有两个实现，构成一条完整的性能叙事线：
 * <ol>
 *   <li>{@link com.vectorsearch.core.bruteforce.BruteForceIndex} —— 全量遍历，
 *       100% 准确，作为<b>正确性基准</b>和<b>性能对照组</b>。</li>
 *   <li>{@link com.vectorsearch.core.hnsw.HnswIndex} —— 近似最近邻，
 *       用少量 recall 换数量级的延迟下降。</li>
 * </ol>
 *
 * <p>所有向量在入库前会被 L2 归一化，因此返回的 score 即余弦相似度。
 */
public interface VectorIndex {

    /**
     * 写入一个向量。
     *
     * @param id     业务 id，重复写入同一 id 的行为由实现定义
     * @param vector 向量，长度必须等于 {@link #dimension()}
     */
    void add(int id, float[] vector);

    /**
     * 检索 top-k。
     *
     * @param query    查询向量，长度必须等于 {@link #dimension()}
     * @param k        返回条数
     * @param efSearch 候选池大小。仅 HNSW 使用；暴力检索忽略此参数。
     *                 这是 recall / 延迟 的头号旋钮。
     * @return 按相似度降序排列的结果，长度不超过 min(k, size())
     */
    List<SearchResult> search(float[] query, int k, int efSearch);

    /** 已索引的向量条数。 */
    int size();

    /** 向量维度。 */
    int dimension();

    /** 实现名称，用于日志与压测报告。 */
    String name();
}
