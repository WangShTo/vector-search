package com.vectorsearch.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

/**
 * 索引维护：墓碑压实。
 *
 * <h2>压实修的是什么</h2>
 *
 * <p>删除文档走的是<b>墓碑</b>：{@link DocumentStore} 只把 id 记进删除集合，
 * 向量索引和 BM25 倒排表都原封不动。这样删除是 O(1) 的，
 * 但「已经删掉的文档仍然在索引里」这件事会随时间累积成三个问题：
 *
 * <ol>
 *   <li><b>BM25 的统计量被污染</b>：{@code Bm25Index.averageDocumentLength()} 是
 *       {@code totalTokens / docCount}，IDF 里的文档总数同理 —— 两者都把墓碑算在内。
 *       删得越多，长度归一化和词权重越偏。<b>这类偏差不会报错，只是分数悄悄算错</b>，
 *       是本次压实最主要的目标。</li>
 *   <li><b>图里堆死节点</b>：每次检索的 {@code efSearch} 预算有一部分花在死节点上，
 *       活节点的召回率随之下降。</li>
 *   <li><b>结果可能不足 top-k</b>：候选池大小固定，过滤掉墓碑后可能凑不够数。</li>
 * </ol>
 *
 * <h2>三件事必须一起做</h2>
 *
 * <p>稠密索引、稀疏索引、原文存储三者描述的是同一批文档。只重建其中一个，
 * 它们就会互相矛盾 —— 而且矛盾的方式是「查询结果变差」而不是「报错」，
 * 最难发现。所以压实是一个整体操作：<b>重建图 → 重建倒排表 → 清掉墓碑</b>。
 *
 * <p>整个过程在 {@code synchronized} 里串行执行，避免两次压实交叠
 * （那样两边的 {@code before} 会互相干扰，日志也会对不上）。
 */
@Component
public class IndexMaintenanceService {

    private static final Logger log = LoggerFactory.getLogger(IndexMaintenanceService.class);

    private final VectorIndexManager vectorIndexManager;
    private final SparseIndexManager sparseIndexManager;
    private final DocumentStore documentStore;

    public IndexMaintenanceService(VectorIndexManager vectorIndexManager,
                                   SparseIndexManager sparseIndexManager,
                                   DocumentStore documentStore) {
        this.vectorIndexManager = vectorIndexManager;
        this.sparseIndexManager = sparseIndexManager;
        this.documentStore = documentStore;
    }

    /**
     * 执行一次完整的压实。
     *
     * <p>没有墓碑时直接返回一个「什么都没做」的结果，不做无谓的重建 ——
     * 重建 10 万点的图要一百多秒，不该被一个空操作触发。
     */
    public synchronized CompactionResult compact() {
        int tombstones = documentStore.deletedCount();
        int vectorsBefore = vectorIndexManager.size();

        if (tombstones == 0) {
            log.info("没有墓碑，跳过压实（当前 {} 条向量）", vectorsBefore);
            return new CompactionResult(
                    vectorsBefore, vectorsBefore, 0,
                    sparseIndexManager.termCount(), sparseIndexManager.termCount(),
                    sparseIndexManager.averageDocumentLength(),
                    sparseIndexManager.averageDocumentLength(),
                    0L);
        }

        long start = System.nanoTime();

        // 先记下压缩前的统计量，用来证明「污染确实被修好了」——
        // 平均文档长度和词表大小都会变，这是最直接的证据
        int termsBefore = sparseIndexManager.termCount();
        double averageLengthBefore = sparseIndexManager.averageDocumentLength();

        // 1. 稠密侧：活节点搬进新图。
        //    必须在清墓碑之前做 —— 它要靠 isDeleted 判断哪些该跳过。
        VectorIndexManager.CompactOutcome dense = vectorIndexManager.compact();

        // 2. 稀疏侧：从存活文档整表重建。
        //    BM25 是全局统计量模型（IDF、平均长度），没法局部删词，只能重建。
        sparseIndexManager.rebuildFromDocuments();

        // 3. 墓碑已经不在新图里了，这些 id 留着只剩副作用
        int purged = documentStore.purgeDeleted();

        int termsAfter = sparseIndexManager.termCount();
        double averageLengthAfter = sparseIndexManager.averageDocumentLength();
        long millis = (System.nanoTime() - start) / 1_000_000L;

        log.info("压实完成: 向量 {} -> {}，清掉 {} 个墓碑；BM25 词表 {} -> {}，"
                        + "平均文档长度 {} -> {} token，耗时 {} ms",
                dense.vectorsBefore(), dense.vectorsAfter(), purged,
                termsBefore, termsAfter,
                String.format("%.1f", averageLengthBefore),
                String.format("%.1f", averageLengthAfter),
                millis);

        return new CompactionResult(
                dense.vectorsBefore(), dense.vectorsAfter(), purged,
                termsBefore, termsAfter,
                averageLengthBefore, averageLengthAfter,
                millis);
    }

    /**
     * 压实结果。
     *
     * <p>字段刻意多带了 BM25 的几个统计量：压实的价值有一半是「把被墓碑污染的
     * 语料统计量修正回来」，光说「清掉了 N 个墓碑」看不出这一点。
     *
     * @param vectorsBefore            压实前的向量条数（含墓碑）
     * @param vectorsAfter             压实后的向量条数（只剩活节点）
     * @param tombstonesPurged         从原文存储里真正删掉的条数
     * @param termsBefore              压实前的 BM25 词表大小
     * @param termsAfter               压实后的 BM25 词表大小
     * @param averageDocumentLengthBefore 压实前的平均文档长度（token）
     * @param averageDocumentLengthAfter  压实后的平均文档长度（token）
     * @param elapsedMillis            总耗时
     */
    public record CompactionResult(
            int vectorsBefore,
            int vectorsAfter,
            int tombstonesPurged,
            int termsBefore,
            int termsAfter,
            double averageDocumentLengthBefore,
            double averageDocumentLengthAfter,
            long elapsedMillis
    ) {
    }
}
