package com.vectorsearch.api.dto;

/**
 * 索引写入结果。
 *
 * @param indexed       本次成功写入的条数
 * @param totalSize     写入后索引中的总向量数
 * @param dimension     向量维度
 * @param embeddingMs   本批向量化耗时
 * @param indexMs       本批写入索引耗时
 */
public record IndexResult(
        int indexed,
        int totalSize,
        int dimension,
        long embeddingMs,
        long indexMs
) {
}
