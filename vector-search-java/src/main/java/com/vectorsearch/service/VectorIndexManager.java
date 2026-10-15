package com.vectorsearch.service;

import com.vectorsearch.config.VectorSearchProperties;
import com.vectorsearch.core.SearchResult;
import com.vectorsearch.core.VectorIndex;
import com.vectorsearch.core.bruteforce.BruteForceIndex;
import com.vectorsearch.core.hnsw.HnswIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * 索引的生命周期管理：创建、切换、暴露查询入口。
 *
 * <p><b>为什么要有这一层</b>：暴力检索和 HNSW 是同一份数据上的两种实现，
 * 需要能在不改代码的前提下切换、跑对照实验。压测脚本就是靠
 * {@link #reset(String)} 在两者之间来回切的。
 *
 * <p>当前只持有一个索引实例，写路径加了索引自己的锁。第 3 周会在这里
 * 引入索引持久化与热更新（新索引构建完成后原子替换引用，
 * 老索引等正在进行的查询结束再回收）。
 */
@Component
public class VectorIndexManager {

    private static final Logger log = LoggerFactory.getLogger(VectorIndexManager.class);

    /** 支持的索引类型。 */
    public static final String TYPE_HNSW = "hnsw";
    public static final String TYPE_BRUTE_FORCE = "brute-force";

    private final VectorSearchProperties properties;
    private volatile VectorIndex index;
    private volatile String indexType;

    public VectorIndexManager(VectorSearchProperties properties) {
        this.properties = properties;
        this.indexType = TYPE_HNSW;
        this.index = create(TYPE_HNSW);
        log.info("向量索引已初始化: {} (dim={}, HNSW 参数 {})",
                index.name(), index.dimension(), properties.index().toHnswParams());
    }

    public VectorIndex index() {
        return index;
    }

    public String indexType() {
        return indexType;
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
}
