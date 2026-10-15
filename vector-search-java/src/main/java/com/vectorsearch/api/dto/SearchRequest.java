package com.vectorsearch.api.dto;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;

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
        Integer efSearch
) {

    public int topKOrDefault() {
        return topK == null ? 10 : topK;
    }
}
