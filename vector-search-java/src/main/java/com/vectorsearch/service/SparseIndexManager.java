package com.vectorsearch.service;

import com.vectorsearch.core.SearchResult;
import com.vectorsearch.core.bm25.Bm25Index;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * 稀疏（BM25）索引的生命周期管理。
 *
 * <h2>为什么 BM25 不需要单独快照</h2>
 * 向量索引必须落盘，是因为<b>建图很贵</b>（10 万点要 113 秒）。
 * 而 BM25 的构建只是「分词 + 往倒排表塞词频」，比建图便宜几个数量级。
 *
 * <p>既然 {@link DocumentStore} 已经把原文持久化了，启动时<b>从原文重建 BM25</b>
 * 就够了 —— 不需要再造一套快照格式、多维护一份文件、还要保证两份数据一致。
 * 少一份需要同步的状态，就少一类 bug。
 *
 * <p>这是「用更简单的方案解决同一个问题」的一个典型例子：
 * 不是所有需要持久化的东西都值得单独序列化，有时候重新算一遍更快也更省心。
 *
 * <h2>为什么等 ApplicationReadyEvent 再重建</h2>
 * BM25 的数据来源是 {@link DocumentStore}，而原文是
 * {@link VectorIndexManager} 在 {@code @PostConstruct} 里加载的。
 * {@code @PostConstruct} 之间的执行顺序没有可靠保证，
 * 所以这里等容器完全就绪之后再重建，避免读到还没加载完的文档。
 */
@Component
public class SparseIndexManager {

    private static final Logger log = LoggerFactory.getLogger(SparseIndexManager.class);

    private final Bm25Index index = new Bm25Index();
    private final DocumentStore documentStore;

    public SparseIndexManager(DocumentStore documentStore) {
        this.documentStore = documentStore;
    }

    /**
     * 容器就绪后从已加载的文档重建 BM25 索引。
     *
     * <p>首次启动或没有快照时文档为空，这里就是一次空操作 ——
     * 之后通过 {@link #add} 增量追加即可。
     */
    @EventListener(ApplicationReadyEvent.class)
    public void rebuildFromDocuments() {
        long start = System.nanoTime();
        index.clear();
        documentStore.forEach((id, text) -> index.add(id, text));
        long millis = (System.nanoTime() - start) / 1_000_000L;

        if (index.size() == 0) {
            log.info("BM25 稀疏索引为空（尚未灌入文档，或快照里没有原文）");
            return;
        }
        log.info("BM25 稀疏索引已重建: {} 篇文档, {} 个词条, 平均文档长度 {} 个 token, 耗时 {} ms",
                index.size(), index.termCount(),
                String.format("%.1f", index.averageDocumentLength()), millis);
    }

    public void add(int id, String text) {
        index.add(id, text);
    }

    public List<SearchResult> search(String query, int k) {
        return index.search(query, k);
    }

    public int size() {
        return index.size();
    }

    public int termCount() {
        return index.termCount();
    }

    public double averageDocumentLength() {
        return index.averageDocumentLength();
    }

    public void clear() {
        index.clear();
    }
}
