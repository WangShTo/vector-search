package com.vectorsearch.service;

import com.vectorsearch.config.VectorSearchProperties;
import com.vectorsearch.core.SearchResult;
import com.vectorsearch.core.VectorIndex;
import com.vectorsearch.core.bruteforce.BruteForceIndex;
import com.vectorsearch.core.hnsw.HnswIndex;
import com.vectorsearch.core.hnsw.HnswParams;
import com.vectorsearch.core.quant.Quantization;
import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * 索引的生命周期管理：创建、切换、加载、落盘、暴露查询入口。
 *
 * <p><b>为什么要有这一层</b>：暴力检索和 HNSW 是同一份数据上的两种实现，
 * 需要能在不改代码的前提下切换、跑对照实验。压测脚本就是靠
 * {@link #reset(String)} 在两者之间来回切的。
 *
 * <h2>持久化</h2>
 * 启动时如果工作目录下存在快照，就直接读回来，不再重建图 ——
 * 这是「demo」和「服务」的分界：10 万条数据建图要 129 秒，
 * 每次重启都等两分钟是不可接受的，而读快照只要不到 1 秒。
 *
 * <p>快照分两部分：
 * <ul>
 *   <li>{@code vector-index.snapshot} —— 二进制，存向量与整张图</li>
 *   <li>{@code documents.jsonl} —— 文本，存 id 到原文的映射</li>
 * </ul>
 * 分开存是因为两类数据性质不同：定长数值适合二进制，变长文本适合 JSONL。
 */
@Component
public class VectorIndexManager {

    private static final Logger log = LoggerFactory.getLogger(VectorIndexManager.class);

    /** 支持的索引类型。 */
    public static final String TYPE_HNSW = "hnsw";
    /** 与 {@link #TYPE_HNSW} 同一套算法，但向量按 int8 量化存储（内存约 1/4）。 */
    public static final String TYPE_HNSW_SQ8 = "hnsw-sq8";
    public static final String TYPE_BRUTE_FORCE = "brute-force";

    private static final String INDEX_FILE_NAME = "vector-index.snapshot";
    private static final String DOCUMENTS_FILE_NAME = "documents.jsonl";

    private final VectorSearchProperties properties;
    private final DocumentStore documentStore;

    private volatile VectorIndex index;
    private volatile String indexType;
    /** 本次启动是否从快照恢复，以及耗时，供 /api/stats 展示。 */
    private volatile Long restoredInMillis;

    public VectorIndexManager(VectorSearchProperties properties, DocumentStore documentStore) {
        this.properties = properties;
        this.documentStore = documentStore;
        // 初始精度由配置决定；一旦有快照，恢复时会按快照里的精度覆盖
        this.indexType = properties.index().toQuantization() == Quantization.SQ8
                ? TYPE_HNSW_SQ8 : TYPE_HNSW;
        this.index = create(this.indexType);
        log.info("向量索引已初始化: {} (dim={}, 精度 {}, HNSW 参数 {})",
                index.name(), index.dimension(), index.quantization().code(),
                properties.index().toHnswParams());
        // 把解析后的绝对路径打出来：相对路径怎么解析的不该让人猜
        log.info("索引数据目录: {}", dataDir());
    }

    /**
     * 启动时尝试从快照恢复。
     *
     * <p>快照不存在是<b>首次启动的正常情况</b>，只记一条提示；
     * 快照损坏或维度不匹配则记错误并退回空索引 —— 服务要能起来，
     * 不能因为一个坏文件就整个启动失败。
     */
    @PostConstruct
    void restoreIfPresent() {
        Path indexFile = snapshotPath();
        if (!Files.exists(indexFile)) {
            log.info("未找到索引快照，从空索引开始。灌入数据后可调用 "
                    + "POST /api/admin/snapshot 落盘，下次启动即可秒级恢复。");
            return;
        }

        long start = System.nanoTime();
        try {
            HnswIndex restored = HnswIndex.readFrom(indexFile);

            // 维度必须与配置一致，否则检索结果会算错而不是报错 —— 这是最危险的一类不一致
            int configured = properties.index().dimension();
            if (restored.dimension() != configured) {
                log.error("快照维度 {} 与配置 {} 不一致，拒绝加载并退回空索引。"
                                + "请检查 vector-search.index.dimension 与模型配置",
                        restored.dimension(), configured);
                return;
            }

            this.index = restored;
            // 精度跟着快照走，不跟着配置走：快照里存的是什么精度，恢复出来就是什么精度
            this.indexType = restored.quantization() == Quantization.SQ8
                    ? TYPE_HNSW_SQ8 : TYPE_HNSW;

            int documents = documentStore.load(documentsPath());
            long millis = (System.nanoTime() - start) / 1_000_000L;
            this.restoredInMillis = millis;

            log.info("已从快照恢复索引: {} 条向量（{} 精度）, {} 篇文档, 耗时 {} ms（若重建图需数分钟）",
                    restored.size(), restored.quantization().code(), documents, millis);
        } catch (IOException e) {
            log.error("加载索引快照失败，从空索引开始（可删除 {} 后重新灌数据）: {}",
                    indexFile, e.getMessage());
        }
    }

    public VectorIndex index() {
        return index;
    }

    public String indexType() {
        return indexType;
    }

    /** 本次启动从快照恢复的耗时；未恢复时为 null。 */
    public Long restoredInMillis() {
        return restoredInMillis;
    }

    public void add(int id, float[] vector) {
        index.add(id, vector);
    }

    public List<SearchResult> search(float[] query, int k, int efSearch) {
        return index.search(query, k, efSearch);
    }

    /**
     * 带过滤的检索。
     *
     * <p>过滤器和 efSearch 一起交给索引，让图遍历在<b>遍历过程中</b>就知道
     * 哪些节点不参与 —— 而不是搜完再筛。为什么这很重要见
     * {@code bench/REPORT.md} 第八节：搜完再筛在低选择性下会塌。
     */
    public List<SearchResult> search(float[] query, int k, int efSearch,
                                     java.util.function.IntPredicate filter) {
        return index.search(query, k, efSearch, filter);
    }

    public int size() {
        return index.size();
    }

    /**
     * 数据目录的<b>绝对路径</b>。
     *
     * <p>配置里的 {@code data-dir} 通常是相对路径（{@code ./data/index}），
     * 它会相对<b>服务进程的工作目录</b>解析 —— 也就是说从
     * {@code vector-search-java/} 启动和从项目根启动，落盘位置是<b>不一样</b>的。
     *
     * <p>这是个容易踩的坑：快照写成功后按配置里的相对路径去找却找不到。
     * 所以这里统一转成绝对路径，并在启动日志里打出来 —— 文件到底在哪，
     * 看日志就知道，不用猜。
     */
    public Path dataDir() {
        return Path.of(properties.index().dataDir()).toAbsolutePath().normalize();
    }

    public Path snapshotPath() {
        return dataDir().resolve(INDEX_FILE_NAME);
    }

    public Path documentsPath() {
        return dataDir().resolve(DOCUMENTS_FILE_NAME);
    }

    /**
     * 把当前索引与原文一起落盘。
     *
     * @throws IllegalStateException 写盘失败，或当前索引不支持快照
     */
    public synchronized SnapshotResult snapshot() {
        if (!(index instanceof HnswIndex hnsw)) {
            throw new IllegalStateException(
                    "只有 HNSW 索引支持快照，当前是 " + index.name());
        }
        long start = System.nanoTime();
        try {
            hnsw.writeTo(snapshotPath());
            documentStore.save(documentsPath());
            long millis = (System.nanoTime() - start) / 1_000_000L;
            long bytes = Files.size(snapshotPath());

            log.info("索引已落盘: {} 条向量 / {} 篇文档, 快照 {} MB, 耗时 {} ms",
                    index.size(), documentStore.count(),
                    String.format("%.1f", bytes / 1024.0 / 1024.0), millis);
            return new SnapshotResult(
                    snapshotPath().toString(),
                    documentsPath().toString(),
                    bytes,
                    index.size(),
                    documentStore.count(),
                    millis);
        } catch (IOException e) {
            throw new IllegalStateException("写快照失败: " + e.getMessage(), e);
        }
    }

    /**
     * 清空并切换索引实现。
     *
     * <p><b>注意会丢弃已索引的全部向量</b> —— 这是刻意设计：压测需要在
     * 完全干净、同一份数据的前提下重建两种索引，否则测出来的数字不可比。
     *
     * @param type {@link #TYPE_HNSW} 或 {@link #TYPE_BRUTE_FORCE}
     */
    public synchronized void reset(String type) {
        VectorIndex previous = this.index;
        this.index = create(type);
        this.indexType = type;
        this.restoredInMillis = null;
        log.info("索引已切换: {} -> {}（原索引 {} 条向量已丢弃）",
                previous.name(), type, previous.size());
    }

    /**
     * 墓碑压实：把<b>活着的</b>节点搬进一张新的同类型索引，然后原子替换。
     *
     * <h2>为什么需要</h2>
     * 删除文档用的是墓碑 —— {@link DocumentStore} 只记下 id，索引里的向量和邻居边
     * 原封不动地留着。这样删除是 O(1) 的，但代价会随时间累积：
     * <ul>
     *   <li><b>图里堆死节点</b>：每次检索的 {@code efSearch} 预算有一部分花在
     *       已删除的节点上，活节点的召回率随之下降。</li>
     *   <li><b>内存降不下来</b>：向量和边都还在。</li>
     *   <li><b>结果可能不足 top-k</b>：候选池是固定大小，过滤掉墓碑后可能凑不够。</li>
     * </ul>
     *
     * <h2>为什么不需要重新编码</h2>
     * <b>不需要重新调用模型</b> —— 向量已经在索引里了，{@link VectorIndex#forEach}
     * 直接把它们读出来重插即可，省掉的是「把整个语料重新编码一遍」的几分钟。
     *
     * <p>但<b>建图开销省不掉</b>：10 万点重建一次图约 113 秒，所以这是
     * 「偶尔做一次的维护操作」，不是每次删除都要跑的东西。演示规模（百来条）
     * 下是几百毫秒。
     *
     * <h2>为什么要换整张索引而不是原地删</h2>
     * 原地删除要重建邻居边、修入口点、维护空闲槽位，任何一处漏掉都会让图静默变差 ——
     * 这正是「索引热更新」被否决的同一个理由。换新图则没有中间状态：
     * 老索引在替换前一刻都还是完整可用的。
     *
     * <h2>代价</h2>
     * 压实期间两份索引同时存在，<b>内存峰值约 2 倍</b>。
     *
     * @return 压实前后的条数与丢弃的节点数
     */
    public synchronized CompactOutcome compact() {
        long start = System.nanoTime();
        VectorIndex previous = this.index;
        int before = previous.size();

        VectorIndex fresh = create(indexType);
        previous.forEach((externalId, vector) -> {
            if (!documentStore.isDeleted(externalId)) {
                fresh.add(externalId, vector);
            }
        });

        // volatile 写：已经拿到老引用的查询会跑完，新的查询立刻走新图
        this.index = fresh;

        int after = fresh.size();
        int removed = before - after;
        long millis = (System.nanoTime() - start) / 1_000_000L;
        log.info("稠密索引压实完成: {} -> {} 条（丢弃 {} 个墓碑）, 耗时 {} ms",
                before, after, removed, millis);
        return new CompactOutcome(before, after, removed);
    }

    private VectorIndex create(String type) {
        int dimension = properties.index().dimension();
        HnswParams hnswParams = properties.index().toHnswParams();
        return switch (type) {
            case TYPE_HNSW -> new HnswIndex(dimension, hnswParams);
            // 同一套建图/检索代码，只是每条向量从 4 字节压到 1 字节
            case TYPE_HNSW_SQ8 -> new HnswIndex(dimension, hnswParams, 42L, Quantization.SQ8);
            case TYPE_BRUTE_FORCE -> new BruteForceIndex(dimension);
            default -> throw new IllegalArgumentException(
                    "未知索引类型: " + type + "（可选 " + TYPE_HNSW + " / " + TYPE_HNSW_SQ8
                            + " / " + TYPE_BRUTE_FORCE + "）");
        };
    }

    /** 落盘结果。 */
    public record SnapshotResult(
            String indexFile,
            String documentsFile,
            long indexBytes,
            int vectorCount,
            int documentCount,
            long elapsedMillis
    ) {
    }

    /**
     * 稠密索引的压实结果。
     *
     * @param vectorsBefore 压实前的条数（含墓碑）
     * @param vectorsAfter  压实后的条数（只剩活节点）
     * @param removed       丢弃的节点数
     */
    public record CompactOutcome(int vectorsBefore, int vectorsAfter, int removed) {
    }
}
