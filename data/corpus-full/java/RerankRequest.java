package com.vectorsearch.client.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

/**
 * 请求 Python 服务做精排（rerank）。
 *
 * <p>两阶段检索的第二阶段：向量召回负责"快而全"（从百万里捞出 100 条），
 * rerank 负责"准"（用 cross-encoder 对这 100 条精细打分）。
 * cross-encoder 精度远高于向量内积，但无法预先建索引，只能对少量候选做。
 *
 * <p>第 4 周接入，当前先把契约定下来。
 *
 * @param query     用户原始查询
 * @param documents 待精排的候选文档
 * @param topN      返回前 N 条
 */
public record RerankRequest(
        String query,
        List<String> documents,
        @JsonProperty("top_n") int topN
) {
}
