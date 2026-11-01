package com.vectorsearch.service;

import com.vectorsearch.api.dto.IndexDocumentRequest;
import com.vectorsearch.api.dto.IndexResult;
import com.vectorsearch.api.dto.SearchHit;
import com.vectorsearch.api.dto.SearchRequest;
import com.vectorsearch.api.dto.SearchResponse;
import com.vectorsearch.client.ModelServiceClient;
import com.vectorsearch.client.ModelServiceException;
import com.vectorsearch.client.dto.RerankResponse;
import com.vectorsearch.config.VectorSearchProperties;
import com.vectorsearch.core.SearchResult;
import com.vectorsearch.core.fusion.RrfFusion;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
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

    private static final Logger log = LoggerFactory.getLogger(SearchService.class);

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

        boolean rerankRequested = request.rerank() != null
                ? request.rerank()
                : properties.search().rerankDefault();

        // 来源过滤：把「只在这个文件里找」变成索引能理解的谓词
        java.util.function.IntPredicate filter = sourceFilter(request.source());
        double selectivity = selectivity(request.source());
        // 过滤会把候选池里的名额筛掉且不补回来，所以 ef 要按选择性放大
        int efEffective = effectiveEf(efSearch, pool, selectivity);

        List<SearchResult> denseHits = null;
        List<SearchResult> sparseHits = null;

        // BM25 模式不调模型 —— 这是它延迟低一个数量级的原因
        boolean needDense = MODE_VECTOR.equals(mode) || MODE_HYBRID.equals(mode);
        boolean needSparse = MODE_BM25.equals(mode) || MODE_HYBRID.equals(mode);

        // 把「编码」与「检索 + 融合 + 精排」分成两段计时：
        // 编码是跨进程调用，后面是本地计算与另一类模型调用，优化手段完全不同，
        // 混在一起报就没法定位瓶颈了。
        long embeddingEndsAt = start;
        if (needDense) {
            float[][] embedded = modelServiceClient.embed(List.of(request.query()));
            embeddingEndsAt = System.nanoTime();
            // 谓词按 externalId 判，索引不知道原文里哪些 id 已被逻辑删除；
            // 这两件事是正交的，删除过滤仍在下面统一做
            denseHits = indexManager.search(embedded[0], pool, efEffective, filter);
        }
        if (needSparse) {
            // BM25 走的是倒排表，没有「图」可按选择性放大 ef，只能多取再筛。
            // 它的单次开销本来就在亚毫秒级，多取一些是划算的。
            int sparsePool = filter == null ? pool : sparsePool(pool, selectivity);
            List<SearchResult> raw = sparseIndexManager.search(request.query(), sparsePool);
            if (filter == null) {
                sparseHits = raw;
            } else {
                sparseHits = new ArrayList<>(raw.size());
                for (SearchResult hit : raw) {
                    if (filter.test(hit.id())) {
                        sparseHits.add(hit);
                    }
                }
            }
        }
        long searchStart = needDense ? embeddingEndsAt : start;

        // 第一阶段：召回 + 融合
        //
        // 这里故意按 pool（而不是 topK）来截断：被逻辑删除的文档仍然留在索引里，
        // 检索时要把它们过滤掉。如果一开始就只留 topK 条，删掉几篇之后
        // 结果就会不足 topK —— 先多留一些、过滤完再截断才对。
        List<SearchResult> candidates = fuse(mode, denseHits, sparseHits, Math.max(topK, pool));
        List<SearchResult> results = new ArrayList<>(candidates.size());
        for (SearchResult candidate : candidates) {
            if (documentStore.isDeleted(candidate.id())) {
                continue;
            }
            results.add(candidate);
            if (results.size() >= topK) {
                break;
            }
        }

        // 第二阶段：精排
        boolean reranked = false;
        if (rerankRequested && results.size() > 1) {
            try {
                results = applyRerank(request.query(), results, topK);
                reranked = true;
            } catch (ModelServiceException e) {
                // 精排是增强而不是必需：模型服务挂了也应该把召回结果返回去，
                // 而不是整个查询失败。降级会体现在响应里的 reranked=false。
                log.warn("精排失败，返回未精排的结果: {}", e.getMessage());
            }
        }
        long end = System.nanoTime();

        return respond(request.query(), mode, reranked, results, start, embeddingEndsAt, searchStart, end);
    }

    /**
     * 二阶段精排：用 cross-encoder 对候选重新打分。
     *
     * <p><b>为什么值得多花这一百多毫秒</b>：向量检索是双塔结构，
     * 查询和文档<b>各自独立</b>编码成向量再算内积 —— 编码时两者没见过彼此，
     * 细粒度的语义交互全部丢失。cross-encoder 把 (查询, 文档) 拼在一起送进模型，
     * 能做完整的注意力交互，精度明显更高。
     *
     * <p>代价是它没法预先建索引：每来一个查询，都要对每个候选跑一次前向。
     * 所以必须先由召回阶段把候选压到几十条，这就是两阶段检索的分工。
     *
     * @return 精排后的结果；候选不足时原样返回
     */
    private List<SearchResult> applyRerank(String query, List<SearchResult> candidates, int topK) {
        int pool = Math.min(candidates.size(), properties.search().rerankCandidates());
        List<SearchResult> head = candidates.subList(0, pool);

        List<String> texts = new ArrayList<>(pool);
        for (SearchResult candidate : head) {
            // 原文缺失时用空串：reranker 会给它很低的分数，自然排到后面去，
            // 比直接跳过更可预测
            texts.add(documentStore.get(candidate.id()).orElse(""));
        }

        // 让 reranker 把所有候选都排一遍（而不是只要 topK），
        // 这样候选不足 topK 时也不会丢结果
        List<RerankResponse.Item> ranked = modelServiceClient.rerank(query, texts, pool);
        if (ranked.isEmpty()) {
            return candidates;
        }

        List<SearchResult> reranked = new ArrayList<>(ranked.size());
        for (RerankResponse.Item item : ranked) {
            SearchResult source = head.get(item.index());
            // 分数换成 cross-encoder 的相关性分数 —— 与余弦相似度、RRF 分数
            // 都不是一个量纲，所以响应里用 reranked 标记区分
            reranked.add(new SearchResult(source.id(), item.score()));
        }

        // 被精排截断掉的尾巴接在后面：宁可多给调用方几条，也不要凭空丢结果
        if (candidates.size() > pool) {
            reranked.addAll(candidates.subList(pool, candidates.size()));
        }
        return reranked.size() > topK ? new ArrayList<>(reranked.subList(0, topK)) : reranked;
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

    /** 过滤检索允许的最大 ef。再大就接近全图扫描了，不如换策略。 */
    private static final int MAX_EFFECTIVE_EF = 4096;

    /**
     * 把「来源」变成按 id 判的谓词。
     *
     * <p>为什么来源不塞进索引：索引只管向量和 id，来源是业务属性。
     * 让索引通过谓词来问「这个 id 合不合格」，职责就分清了 ——
     * 明天要按时间、按标签过滤，索引一行都不用改。
     */
    private java.util.function.IntPredicate sourceFilter(String source) {
        if (source == null || source.isBlank()) {
            return null;
        }
        String wanted = source.trim();
        return id -> wanted.equals(documentStore.sourceOf(id));
    }

    /**
     * 来源的选择性 = 匹配文档数 / 存活文档数。
     *
     * <p>不过滤时返回 1。来源不存在时返回一个极小的正数 ——
     * 反正匹配不到东西，多遍历一些也无所谓，但别让它除零。
     */
    private double selectivity(String source) {
        if (source == null || source.isBlank()) {
            return 1.0;
        }
        int total = documentStore.count();
        if (total <= 0) {
            return 1.0;
        }
        Integer matching = documentStore.sourceCounts().get(source.trim());
        if (matching == null || matching <= 0) {
            return 1.0 / total;
        }
        return Math.min(1.0, (double) matching / total);
    }

    /**
     * 带过滤的检索该用多大 ef。
     *
     * <p>结论来自实测（{@code bench/REPORT.md} 第八节）：过滤会把候选池里的名额
     * 筛掉而<b>不补回来</b>。1% 选择性下 ef=50 平均只回 0.7 条、recall 0.07；
     * 把 ef 放大到 k/选择性 —— 即「保证池子里大约有 k 个合格候选」——
     * recall 回到 0.94，而且比「不过滤、只放大返回池」更快。
     */
    private int effectiveEf(int baseEf, int k, double selectivity) {
        if (selectivity >= 1.0) {
            return baseEf;
        }
        double needed = Math.ceil(k / Math.max(selectivity, 1e-6));
        return (int) Math.min(Math.max(baseEf, needed), MAX_EFFECTIVE_EF);
    }

    /**
     * BM25 没有 ef 可放大 —— 倒排表是按词直接查的，不存在「遍历深度」这个旋钮，
     * 只能多取一些再筛。它的单次开销在亚毫秒级，多取是划算的。
     */
    private int sparsePool(int pool, double selectivity) {
        double needed = Math.ceil(pool / Math.max(selectivity, 1e-6));
        int capped = (int) Math.min(needed, 2000);
        return Math.max(pool, capped);
    }

    private SearchResponse respond(String query,
                                   String mode,
                                   boolean reranked,
                                   List<SearchResult> results,
                                   long start,
                                   long embeddingEndsAt,
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
                reranked,
                hits,
                indexManager.indexType(),
                new SearchResponse.Timing(
                        toMillis(embeddingEndsAt - start),
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
        return index(documents, DocumentStore.DocMeta.unknown());
    }

    /**
     * 批量写入，并给这一批文档打上同样的元数据。
     *
     * <p>一批文档来自同一次上传，所以共用一份来源信息 —— 切块会让一篇文件
     * 变成很多条记录，但它们的来源是同一个文件名。
     */
    public IndexResult index(List<IndexDocumentRequest> documents, DocumentStore.DocMeta meta) {
        long start = System.nanoTime();

        List<String> texts = new ArrayList<>(documents.size());
        for (IndexDocumentRequest document : documents) {
            texts.add(document.text());
        }

        float[][] vectors = modelServiceClient.embed(texts);
        long afterEmbedding = System.nanoTime();

        for (int i = 0; i < documents.size(); i++) {
            IndexDocumentRequest document = documents.get(i);
            documentStore.put(document.id(), document.text(), meta);
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
