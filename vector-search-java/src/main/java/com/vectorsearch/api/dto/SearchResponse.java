package com.vectorsearch.api.dto;

import java.util.List;

/**
 * 检索响应。
 *
 * <p><b>{@code timing} 是刻意暴露的</b>：把「编码」和「检索」两段耗时分开报出来，
 * 能立刻看出瓶颈在哪。如果 embeddingMs 远大于 searchMs，说明优化方向应该是
 * 减少跨进程调用次数，而不是去调 HNSW 参数 —— 这个字段就是让性能问题可见。
 *
 * @param query      原始查询
 * @param hits       检索结果。**拒答时为空列表**，不是「返回低分结果」
 * @param indexType  实际使用的索引实现（hnsw / brute-force）
 * @param timing     各阶段耗时
 * @param abstention 拒答信息。关掉拒答或未触发时也可能为 {@code null}
 */
public record SearchResponse(
        String query,
        String mode,
        Boolean reranked,
        List<SearchHit> hits,
        String indexType,
        Timing timing,
        Abstention abstention
) {

    /**
     * @param embeddingMs 调 Python 服务做向量编码的耗时（含网络往返）。
     *                    {@code bm25} 模式不调模型，此值为 0。
     * @param searchMs    本地检索与融合耗时（不含编码）
     * @param totalMs     端到端耗时
     */
    public record Timing(long embeddingMs, long searchMs, long totalMs) {
    }

    /**
     * 拒答信息。
     *
     * <p>把判据一起返回，是为了让「为什么没结果」可排查 —— 用户看到的是一次空结果，
     * 但页面和日志需要知道是哪一路、哪个分数没过线。少了这两个数字，
     * 线上出现「明明有答案却拒答」时只能靠猜。
     *
     * @param abstained  是否拒答
     * @param rule       实际生效的规则（vector / or / and）
     * @param reason     可读理由，例如 {@code cosine 0.412 < 0.550}
     * @param vectorTop1 向量余弦最高分（该模式不走向量时为空）
     * @param bm25Top1   BM25 最高分（该模式不走 BM25 时为空）
     */
    public record Abstention(boolean abstained,
                             String rule,
                             String reason,
                             Float vectorTop1,
                             Float bm25Top1) {
    }
}
