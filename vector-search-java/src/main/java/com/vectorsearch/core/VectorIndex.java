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

    /**
     * 遍历所有已索引的节点。
     *
     * <p>为<b>墓碑压实</b>而存在。删除文档用的是墓碑：只把 id 记进
     * {@code DocumentStore} 的删除集合，索引里的向量和邻居边都还在。
     * 时间一长，死节点会占着图里的遍历预算、污染 BM25 的语料统计，
     * 内存也降不下来。压实的做法是「把活着的节点搬进一张新图，扔掉老图」。
     *
     * <p>关键点：<b>压实不需要重新调用模型</b>。向量已经在索引里了，
     * 读出来直接重插即可，省掉的是把整个语料重新编码一遍的几分钟。
     * 但要注意，建图本身的开销<b>并没有省掉</b> —— 10 万点重建一次图约 113 秒，
     * 所以压实是「偶尔做一次的维护操作」，不是每次删除都要跑的东西。
     *
     * <p>实现约束：
     * <ul>
     *   <li>遍历必须持读锁，给出某一时刻的一致快照 —— 压实期间索引仍在对外服务。</li>
     *   <li>回调拿到的向量是<b>只读</b>的，不得修改，也不得在回调返回后继续持有。</li>
     * </ul>
     *
     * @param visitor 接收 (业务 id, 向量)
     */
    void forEach(IndexedVector visitor);

    /** {@link #forEach} 的回调。 */
    @FunctionalInterface
    interface IndexedVector {

        /**
         * @param externalId 业务 id
         * @param vector     L2 归一化后的向量，只读
         */
        void accept(int externalId, float[] vector);
    }
}
