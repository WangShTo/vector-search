package com.vectorsearch.client.dto;

import java.util.List;

/**
 * Python 服务的 embedding 响应。
 *
 * @param vectors   与请求 texts 顺序严格对应、长度相同的向量列表
 * @param dimension 向量维度，Java 侧会校验它是否与索引配置一致 ——
 *                  维度不一致会导致索引写入时才发现，越早失败越好
 * @param model     实际使用的模型名，便于排查"换了模型但没重建索引"
 */
public record EmbedResponse(List<float[]> vectors, int dimension, String model) {
}
