package com.vectorsearch.api.dto;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;

/**
 * 检索请求。
 *
 * @param query    查询文本
 * @param topK     返回条数，默认 10
 * @param efSearch HNSW 查询候选池大小，默认取配置值。
 *                 暴露成请求参数是为了能在线演示 recall/延迟 权衡 ——
 *                 同一个 query 用 ef=10 和 ef=500 各跑一次，结果和耗时的差别很直观
 */
public record SearchRequest(
        @NotBlank(message = "query 不能为空")
        String query,

        @Min(value = 1, message = "topK 至少为 1")
        @Max(value = 100, message = "topK 最多为 100")
        Integer topK,

        @Min(value = 1, message = "efSearch 至少为 1")
        @Max(value = 2000, message = "efSearch 最多为 2000")
        Integer efSearch,

        /**
         * 检索模式：{@code vector} / {@code bm25} / {@code hybrid}。
         * 不传则用配置里的默认值（默认 hybrid）。
         *
         * <p>暴露成请求参数是为了能<b>公平对比</b>三条路线 ——
         * 只报"混合更好"没有说服力，必须有单路结果作为基准。
         */
        @Pattern(regexp = "vector|bm25|hybrid",
                message = "mode 只能是 vector / bm25 / hybrid")
        String mode,

        /**
         * 是否做二阶段精排。不传则用配置里的默认值（默认关）。
         *
         * <p>精排用 cross-encoder 把 (查询, 文档) 拼在一起送进模型，
         * 精度明显高于向量内积，但要对每个候选跑一次前向计算 ——
         * 是数量级的延迟代价，所以默认不开启。
         */
        Boolean rerank,

        /**
         * 只在这个来源里检索（上传时的文件名）。不传表示全库。
         *
         * <p>这是<b>带过滤的检索</b>，不是「搜完再筛」：过滤条件会下推到图遍历里。
         * 两者的差别在低选择性下是数量级的 —— 只占 1% 的来源，搜完再筛 recall 接近 0，
         * 下推之后仍然能到 0.94。实测见 {@code bench/REPORT.md} 第八节。
         */
        String source
) {

    public int topKOrDefault() {
        return topK == null ? 10 : topK;
    }
}
