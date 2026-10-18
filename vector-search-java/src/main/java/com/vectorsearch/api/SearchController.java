package com.vectorsearch.api;

import com.vectorsearch.api.dto.IndexDocumentsRequest;
import com.vectorsearch.api.dto.IndexResult;
import com.vectorsearch.api.dto.SearchRequest;
import com.vectorsearch.api.dto.SearchResponse;
import com.vectorsearch.api.dto.StatsResponse;
import com.vectorsearch.client.ModelServiceClient;
import com.vectorsearch.core.VectorIndex;
import com.vectorsearch.core.hnsw.HnswIndex;
import com.vectorsearch.service.DocumentStore;
import com.vectorsearch.service.SearchService;
import com.vectorsearch.service.VectorIndexManager;
import jakarta.validation.Valid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")
public class SearchController {

    private static final Logger log = LoggerFactory.getLogger(SearchController.class);

    private final SearchService searchService;
    private final VectorIndexManager indexManager;
    private final DocumentStore documentStore;
    private final ModelServiceClient modelServiceClient;

    public SearchController(SearchService searchService,
                            VectorIndexManager indexManager,
                            DocumentStore documentStore,
                            ModelServiceClient modelServiceClient) {
        this.searchService = searchService;
        this.indexManager = indexManager;
        this.documentStore = documentStore;
        this.modelServiceClient = modelServiceClient;
    }

    /** 语义检索。 */
    @PostMapping("/search")
    public SearchResponse search(@Valid @RequestBody SearchRequest request) {
        return searchService.search(request);
    }

    /** 批量灌入文档（服务端负责切分与向量化）。 */
    @PostMapping("/documents")
    public IndexResult index(@Valid @RequestBody IndexDocumentsRequest request) {
        return searchService.index(request.documents());
    }

    /** 服务与索引状态。 */
    @GetMapping("/stats")
    public StatsResponse stats() {
        VectorIndex index = indexManager.index();

        StatsResponse.HnswStats hnswStats = null;
        if (index instanceof HnswIndex hnsw) {
            hnswStats = new StatsResponse.HnswStats(
                    hnsw.params().m(),
                    hnsw.params().efConstruction(),
                    hnsw.maxLevel(),
                    hnsw.totalLinkCount(),
                    hnsw.levelHistogram());
        }

        return new StatsResponse(
                indexManager.indexType(),
                index.size(),
                index.dimension(),
                documentStore.count(),
                documentStore.totalCharacters(),
                hnswStats,
                indexManager.restoredInMillis(),
                modelServiceClient.isHealthy());
    }

    /**
     * 把当前索引与原文一起落盘。
     *
     * <p>落盘之后重启服务会直接从快照恢复 —— 10 万条数据建图要两分钟，
     * 而读快照不到一秒。这是一个「灌一次、长期用」的操作。
     *
     * <p>需要重新灌数据时，删掉 {@code data/index/} 下的两个文件即可。
     */
    @PostMapping("/admin/snapshot")
    public VectorIndexManager.SnapshotResult snapshot() {
        log.info("收到索引落盘请求");
        return indexManager.snapshot();
    }

    /**
     * 清空并切换索引实现 —— 压测对照用。
     *
     * <p>{@code POST /api/admin/reset?type=brute-force} 之后重新灌一遍数据，
     * 就能量出同一份数据在两种实现下的延迟差距。
     */
    @PostMapping("/admin/reset")
    public StatsResponse reset(@RequestParam(defaultValue = VectorIndexManager.TYPE_HNSW) String type) {
        log.warn("收到索引重置请求，切换到: {}（当前数据将被清空）", type);
        indexManager.reset(type);
        documentStore.clear();
        return stats();
    }
}
