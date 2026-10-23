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
import com.vectorsearch.service.IndexMaintenanceService;
import com.vectorsearch.service.SearchService;
import com.vectorsearch.service.SparseIndexManager;
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
    private final SparseIndexManager sparseIndexManager;
    private final DocumentStore documentStore;
    private final ModelServiceClient modelServiceClient;
    private final IndexMaintenanceService maintenanceService;

    public SearchController(SearchService searchService,
                            VectorIndexManager indexManager,
                            SparseIndexManager sparseIndexManager,
                            DocumentStore documentStore,
                            ModelServiceClient modelServiceClient,
                            IndexMaintenanceService maintenanceService) {
        this.searchService = searchService;
        this.indexManager = indexManager;
        this.sparseIndexManager = sparseIndexManager;
        this.documentStore = documentStore;
        this.modelServiceClient = modelServiceClient;
        this.maintenanceService = maintenanceService;
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
                new StatsResponse.SparseStats(
                        sparseIndexManager.size(),
                        sparseIndexManager.termCount(),
                        Math.round(sparseIndexManager.averageDocumentLength() * 10) / 10.0),
                indexManager.restoredInMillis(),
                modelServiceClient.isHealthy(),
                documentStore.deletedCount());
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
     * 墓碑压实：把活着的文档搬进新索引，清掉已删除的残留。
     *
     * <p>删除用的是墓碑，被删的向量和邻居边都还留在索引里。删得多了会有三个后果：
     * BM25 的语料统计（IDF、平均文档长度）被墓碑污染、图里死节点占用遍历预算、
     * 结果可能凑不够 top-k。压实把这三件事一起修好。
     *
     * <p><b>不需要重新调用模型</b>：向量已经在索引里，读出来重插即可，
     * 省掉的是重编码整个语料的几分钟。但建图开销省不掉 ——
     * 10 万点重建一次图约 113 秒，演示规模（百来条）是几百毫秒。
     *
     * <p>没有墓碑时是空操作，直接返回。
     */
    @PostMapping("/admin/compact")
    public IndexMaintenanceService.CompactionResult compact() {
        log.info("收到索引压实请求");
        return maintenanceService.compact();
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
        // 两路索引必须一起清，否则会出现"向量索引空了但 BM25 还留着旧数据"的错位
        sparseIndexManager.clear();
        documentStore.clear();
        return stats();
    }
}
