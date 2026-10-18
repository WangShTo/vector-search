package com.vectorsearch.client.dto;

import java.util.List;

/**
 * Python 服务的 rerank 响应。
 *
 * @param results 按分数降序排列
 */
public record RerankResponse(List<Item> results) {

    /**
     * @param index 对应请求 documents 中的下标 —— 用下标而非原文回传，
     *              避免大文本在网络上来回搬运
     * @param score 相关性分数，越大越相关
     */
    public record Item(int index, float score) {
    }
}
