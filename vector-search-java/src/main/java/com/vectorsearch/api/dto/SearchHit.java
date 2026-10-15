package com.vectorsearch.api.dto;

/**
 * 一条检索结果。
 *
 * @param id    文档 id
 * @param score 余弦相似度，越大越相关
 * @param text  文档原文，便于调用方直接展示
 */
public record SearchHit(int id, float score, String text) {
}
