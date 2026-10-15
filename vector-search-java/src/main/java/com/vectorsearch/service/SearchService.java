package com.vectorsearch.service;

import com.vectorsearch.api.dto.IndexDocumentRequest;
import com.vectorsearch.api.dto.IndexResult;
import com.vectorsearch.api.dto.SearchHit;
import com.vectorsearch.api.dto.SearchRequest;
import com.vectorsearch.api.dto.SearchResponse;
import com.vectorsearch.client.ModelServiceClient;
import com.vectorsearch.config.VectorSearchProperties;
import com.vectorsearch.core.SearchResult;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

/**
 * 检索编排：文本 → 向量 → 索引查询 → 回填原文。
 *
 * <p>刻意把每一段耗时分别记录下来返回给调用方。性能优化最忌讳
 * 「凭感觉猜瓶颈」，这些字段让瓶颈自己暴露出来。
 */
@Service
public class SearchService {

    private final ModelServiceClient modelServiceClient;
    private final VectorIndexManager indexManager;
    private final DocumentStore documentStore;
    private final VectorSearchProperties properties;

    public SearchService(ModelServiceClient modelServiceClient,
                         VectorIndexManager indexManager,
                         DocumentStore documentStore,
                         VectorSearchProperties properties) {
        this.modelServiceClient = modelServiceClient;
        this.indexManager = indexManager;
        this.documentStore = documentStore;
        this.properties = properties;
    }

    public SearchResponse search(SearchRequest request) {
        long start = System.nanoTime();

        // 单条查询也走批量接口 —— 保持只有一条代码路径，
        // 避免"批量对了单条错了"这类只在某条路径上出现的 bug
        float[][] embedded = modelServiceClient.embed(List.of(request.query()));
        long afterEmbedding = System.nanoTime();

        int efSearch = request.efSearch() != null
                ? request.efSearch()
                : properties.index().efSearch();

        List<SearchResult> results = indexManager.search(embedded[0], request.topKOrDefault(), efSearch);
        long afterSearch = System.nanoTime();

        List<SearchHit> hits = new ArrayList<>(results.size());
        for (SearchResult result : results) {
            hits.add(new SearchHit(
                    result.id(),
                    result.score(),
                    documentStore.get(result.id()).orElse(null)));
        }

        return new SearchResponse(
                request.query(),
                hits,
                indexManager.indexType(),
                new SearchResponse.Timing(
                        toMillis(afterEmbedding - start),
                        toMillis(afterSearch - afterEmbedding),
                        toMillis(afterSearch - start)));
    }

    /**
     * 批量写入。
     *
     * <p>整批一次 embedding 调用是关键 —— 逐条调用时，10 万条文档就是
     * 10 万次网络往返，光 TCP/HTTP 开销就能把整个灌库过程拖垮。
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

    private long toMillis(long nanos) {
        return nanos / 1_000_000;
    }
}
