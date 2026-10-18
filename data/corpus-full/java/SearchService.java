package com.vectorsearch.service;

import com.vectorsearch.api.dto.IndexDocumentRequest;
import com.vectorsearch.api.dto.IndexResult;
import com.vectorsearch.api.dto.SearchHit;
import com.vectorsearch.api.dto.SearchRequest;
import com.vectorsearch.api.dto.SearchResponse;
import com.vectorsearch.client.ModelServiceClient;
import com.vectorsearch.config.VectorSearchProperties;
import com.vectorsearch.core.SearchResult;
import com.vectorsearch.core.fusion.RrfFusion;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 检索编排：文本 → 向量 / 词项 → 索引查询 → 融合 → 回填原文。
 *
 * <h2>三种检索模式</h2>
 * <table border="1">
 *   <caption>模式对比</caption>
 *   <tr><th>mode</th><th>用什么</th><th>要不要调模型</th><th>擅长</th></tr>
 *   <tr><td>{@code vector}</td><td>HNSW 稠密向量</td><td>要</td><td>语义泛化</td></tr>
 *   <tr><td>{@code bm25}</td><td>倒排 + BM25</td><td><b>不要</b></td><td>精确词匹配</td></tr>
 *   <tr><td>{@code hybrid}</td><td>两者 + RRF 融合</td><td>要</td><td>兼顾</td></tr>
 * </table>
 *
 * <p>暴露三种模式是为了能<b>公平对比</b> —— 只报"混合检索更好"是没说服力的，
 * 必须有单独的向量与 BM25 结果作为基准。
 *
 * <p>另外，{@code bm25} 模式<b>完全不调用模型服务</b>，所以它的端到端延迟
 * 比另外两种低一个数量级。这是一个很实用的降级路径：
 * 模型服务挂了的时候，检索还能以 BM25 模式工作，而不是整个不可用。
 */
@Service
public class SearchService {

    /** 只用稠密向量检索。 */
    public static final String MODE_VECTOR = "vector";
    /** 只用 BM25 稀疏检索。 */
    public static final String MODE_BM25 = "bm25";
    /** 两路融合。 */
    public static final String MODE_HYBRID = "hybrid";

    private final ModelServiceClient modelServiceClient;
    private final VectorIndexManager indexManager;
    private final SparseIndexManager sparseIndexManager;
    private final DocumentStore documentStore;
    private final VectorSearchProperties properties;

    public SearchService(ModelServiceClient modelServiceClient,
                         VectorIndexManager indexManager,
                         SparseIndexManager sparseIndexManager,
                         DocumentStore documentStore,
                         VectorSearchProperties properties) {
        this.modelServiceClient = modelServiceClient;
        this.indexManager = indexManager;
        this.sparseIndexManager = sparseIndexManager;
        this.documentStore = documentStore;
        this.properties = properties;
    }

    public SearchResponse search(SearchRequest request) {
        long start = System.nanoTime();

        String mode = normalizeMode(request.mode());
        int topK = request.topKOrDefault();
        // 融合时两路各自要多召回一些，融合后才有足够的候选可挑
        int pool = Math.max(topK, properties.search().candidatePool());

        int efSearch = request.efSearch() != null
                ? request.efSearch()
                : properties.index().efSearch();

        List<SearchResult> denseHits = null;
        List<SearchResult> sparseHits = null;

        // BM25 模式不调模型 —— 这是它延迟低一个数量级的原因
        boolean needDense = MODE_VECTOR.equals(mode) || MODE_HYBRID.equals(mode);
        boolean needSparse = MODE_BM25.equals(mode) || MODE_HYBRID.equals(mode);

        // 把「编码」与「检索 + 融合」分成两段计时：
        // 编码是跨进程调用，检索是本地计算，两者的优化手段完全不同，
        // 混在一起报就没法定位瓶颈了。
        if (needDense) {
            float[][] embedded = modelServiceClient.embed(List.of(request.query()));
            long afterEmbedding = System.nanoTime();

            long searchStart = System.nanoTime();
            denseHits = indexManager.search(embedded[0], pool, efSearch);
            if (needSparse) {
                sparseHits = sparseIndexManager.search(request.query(), pool);
            }
            List<SearchResult> fused = fuse(mode, denseHits, sparseHits, topK);
            long afterSearch = System.nanoTime();

            return respond(request.query(), mode, fused, start, afterEmbedding, searchStart, afterSearch);
        }

        // 纯 BM25：没有编码阶段
        sparseHits = sparseIndexManager.search(request.query(), pool);
        List<SearchResult> fused = fuse(mode, null, sparseHits, topK);
        long end = System.nanoTime();
        return respond(request.query(), mode, fused, start, start, start, end);
    }

    /** 按模式决定融合方式。 */
    private List<SearchResult> fuse(String mode,
                                    List<SearchResult> denseHits,
                                    List<SearchResult> sparseHits,
                                    int topK) {
        List<SearchResult> fused = switch (mode) {
            case MODE_VECTOR -> denseHits;
            case MODE_BM25 -> sparseHits;
            default -> RrfFusion.fuse(
                    Arrays.asList(denseHits, sparseHits), topK, properties.search().rrfK());
        };
        if (fused == null) {
            return List.of();
        }
        // 单路模式是按 pool 召回的，融合结果也可能超过 topK，这里统一截断
        return fused.size() > topK ? new ArrayList<>(fused.subList(0, topK)) : fused;
    }

    private SearchResponse respond(String query,
                                   String mode,
                                   List<SearchResult> results,
                                   long start,
                                   long afterEmbedding,
                                   long searchStart,
                                   long end) {
        List<SearchHit> hits = new ArrayList<>(results.size());
        for (SearchResult result : results) {
            hits.add(new SearchHit(
                    result.id(),
                    result.score(),
                    documentStore.get(result.id()).orElse(null)));
        }
        return new SearchResponse(
                query,
                mode,
                hits,
                indexManager.indexType(),
                new SearchResponse.Timing(
                        toMillis(afterEmbedding - start),
                        toMillis(end - searchStart),
                        toMillis(end - start)));
    }

    /**
     * 批量写入。
     *
     * <p>整批一次 embedding 调用是关键 —— 逐条调用时，10 万条文档就是
     * 10 万次网络往返，光 TCP/HTTP 开销就能把整个灌库过程拖垮。
     *
     * <p>同时写入两路索引：稠密向量进 HNSW，原文进 BM25 倒排表。
     */
    public IndexResult index(List<IndexDocumentRequest> documents) {
        long start = System.nanoTime();

        List<String> texts = new ArrayList<>(documents.size());
        for (IndexDocumentRequest document : documents) {
            texts.add(document.text());
        }

        float[][] vectors = modelServiceClient.embed(texts);
        long afterEmbedding = System.nanoTime();

        for (int i = 0; i < documents.size(); i++) {
            IndexDocumentRequest document = documents.get(i);
            documentStore.put(document.id(), document.text());
            indexManager.add(document.id(), vectors[i]);
            sparseIndexManager.add(document.id(), document.text());
        }

        long afterIndex = System.nanoTime();

        int dimension = vectors.length > 0 ? vectors[0].length : properties.index().dimension();
        return new IndexResult(
                documents.size(),
                indexManager.size(),
                dimension,
                toMillis(afterEmbedding - start),
                toMillis(afterIndex - afterEmbedding));
    }

    private String normalizeMode(String mode) {
        if (mode == null || mode.isBlank()) {
            return properties.search().defaultMode();
        }
        return mode.trim().toLowerCase();
    }

    private long toMillis(long nanos) {
        return nanos / 1_000_000;
    }
}
