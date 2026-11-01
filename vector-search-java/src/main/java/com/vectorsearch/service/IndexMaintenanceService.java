package com.vectorsearch.service;

import com.vectorsearch.config.VectorSearchProperties;
import jakarta.annotation.PreDestroy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

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
 * <p>后两条现在有一层<b>查询期</b>的缓解：{@code SearchService.liveFilter} 把删除判断
 * 推进到遍历里，死节点不再占用候选名额，选择性也按「含墓碑的条目数」算，
 * ef 会自动放大。也就是说<b>删除之后，检索立刻就不受影响了</b>。
 * 压实剩下的价值是把内存和语料统计量收回来 —— 墓碑还占着向量内存，
 * 也还在污染 BM25 的 IDF 与平均长度。两件事都要做，但紧急程度不同。
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
    private final VectorSearchProperties properties;
    private final Executor executor;

    /** 单飞标记：同时只允许一次自动压实排队，避免批量删除放大成 N 次重建。 */
    private final AtomicBoolean compacting = new AtomicBoolean();

    private final AtomicInteger autoCompactions = new AtomicInteger();

    private final java.util.concurrent.ExecutorService worker =
            Executors.newSingleThreadExecutor(runnable -> {
                Thread thread = new Thread(runnable, "index-compaction");
                // 后台线程，不能拦住 JVM 退出
                thread.setDaemon(true);
                return thread;
            });

    @Autowired
    public IndexMaintenanceService(VectorIndexManager vectorIndexManager,
                                   SparseIndexManager sparseIndexManager,
                                   DocumentStore documentStore,
                                   VectorSearchProperties properties) {
        this(vectorIndexManager, sparseIndexManager, documentStore, properties, null);
    }

    /** 测试用：传一个同步执行的 executor，自动压实就不用等线程了。 */
    IndexMaintenanceService(VectorIndexManager vectorIndexManager,
                            SparseIndexManager sparseIndexManager,
                            DocumentStore documentStore,
                            VectorSearchProperties properties,
                            Executor executor) {
        this.vectorIndexManager = vectorIndexManager;
        this.sparseIndexManager = sparseIndexManager;
        this.documentStore = documentStore;
        this.properties = properties;
        this.executor = executor != null ? executor : worker;
    }

    @PreDestroy
    void shutdown() {
        worker.shutdownNow();
    }

    /** 已经自动压实过几次，给运维页看。 */
    public int autoCompactions() {
        return autoCompactions.get();
    }

    public boolean autoCompactEnabled() {
        return properties.maintenance().autoCompact();
    }

    public double autoCompactRatio() {
        return properties.maintenance().autoCompactRatio();
    }

    /** 当前墓碑占比（0~1），没有条目时返回 0。 */
    public double tombstoneRatio() {
        int population = documentStore.totalCount();
        return population <= 0 ? 0 : (double) documentStore.deletedCount() / population;
    }

    /**
     * 删除之后调用：墓碑占比够高就在后台压实一次。
     *
     * <p>为什么是「占比触发」而不是「每次删除都清理」：<b>HNSW 没法删除单点</b>。
     * 一个节点连着几十条边，拿掉它会让图断开、后面所有人的检索都变差。
     * 所以「即时清理」在向量索引这里的正确含义是「立刻让它不参与检索」，
     * 也就是把删除判断推进到遍历里（见 {@code SearchService.liveFilter}），
     * 真正的空间回收只能靠重建整张图 —— 那就必须按比例批量做，不能按条做。
     *
     * <p>BM25 理论上可以精确删掉一篇：记下它的词表，逐个词把 posting 里的那一条拿掉。
     * 但代价是 O(该文档词表的每个词在倒排表里的长度)，长尾高频词能到全库量级；
     * 而且删完 IDF 和平均长度要重算，一样是全局的。所以两条索引统一走压实。
     *
     * @return 是否真的触发了一次自动压实（便于测试与观测）
     */
    public boolean maybeAutoCompact() {
        VectorSearchProperties.Maintenance config = properties.maintenance();
        if (!config.autoCompact()) {
            return false;
        }
        int tombstones = documentStore.deletedCount();
        int population = documentStore.totalCount();
        if (tombstones == 0 || population == 0) {
            return false;
        }
        double ratio = (double) tombstones / population;
        if (ratio < config.autoCompactRatio()) {
            return false;
        }
        if (!compacting.compareAndSet(false, true)) {
            // 已经有一次在排队或正在跑。它压完之后墓碑就没了，
            // 再排一次只是重复重建整张图
            return false;
        }
        log.info("墓碑 {} / {} 条（{}%）达到阈值 {}%，排队自动压实",
                tombstones, population,
                String.format("%.1f", ratio * 100),
                String.format("%.0f", config.autoCompactRatio() * 100));
        executor.execute(() -> {
            try {
                compact();
                autoCompactions.incrementAndGet();
            } catch (RuntimeException e) {
                // 不能把异常吞掉：压失败意味着图里还有死节点，
                // 下一次删除会再试 —— 但日志里必须留下证据
                log.warn("自动压实失败，墓碑仍在索引里，下次删除会重试", e);
            } finally {
                compacting.set(false);
            }
        });
        return true;
    }

    /**
     * 执行一次完整的压实。
     *
     * <p>没有墓碑时直接返回一个「什么都没做」的结果，不做无谓的重建 ——
     * 重建 10 万点的图要一百多秒，不该被一个空操作触发。
     */
    public synchronized CompactionResult compact() {
        // 先把「这一轮重建要排除哪些 id」拍下来。
        // 压实不是原子的：期间还会有删除进来，清理时必须严格限定在这一批，
        // 否则会把后来才删掉、却已经被重建进新图的文档的原文也清掉，
        // 留下一个查不到原文的死节点。
        java.util.Set<Integer> tombstoned = documentStore.deletedIdSnapshot();
        int tombstones = tombstoned.size();
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

        // 3. 墓碑已经不在新图里了，这些 id 留着只剩副作用。
        //    只清这一批 —— 压实期间新来的删除继续当墓碑，下次压实再处理，
        //    它们在检索时本来就被过滤掉，不会泄漏。
        int purged = documentStore.purgeDeleted(tombstoned);

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
