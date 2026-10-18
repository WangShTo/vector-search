package com.vectorsearch.service;

import com.vectorsearch.config.VectorSearchProperties;
import com.vectorsearch.core.SearchResult;
import com.vectorsearch.core.VectorIndex;
import com.vectorsearch.core.bruteforce.BruteForceIndex;
import com.vectorsearch.core.hnsw.HnswIndex;
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
        this.indexType = TYPE_HNSW;
        this.index = create(TYPE_HNSW);
        log.info("向量索引已初始化: {} (dim={}, HNSW 参数 {})",
                index.name(), index.dimension(), properties.index().toHnswParams());
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
            this.indexType = TYPE_HNSW;

            int documents = documentStore.load(documentsPath());
            long millis = (System.nanoTime() - start) / 1_000_000L;
            this.restoredInMillis = millis;

            log.info("已从快照恢复索引: {} 条向量, {} 篇文档, 耗时 {} ms（若重建图需数分钟）",
                    restored.size(), documents, millis);
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

    private VectorIndex create(String type) {
        int dimension = properties.index().dimension();
        return switch (type) {
            case TYPE_HNSW -> new HnswIndex(dimension, properties.index().toHnswParams());
            case TYPE_BRUTE_FORCE -> new BruteForceIndex(dimension);
            default -> throw new IllegalArgumentException(
                    "未知索引类型: " + type + "（可选 " + TYPE_HNSW + " / " + TYPE_BRUTE_FORCE + "）");
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
}
