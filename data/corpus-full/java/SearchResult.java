package com.vectorsearch.core;

/**
 * 一条检索结果。
 *
 * @param id    文档 id（对外暴露的业务 id）
 * @param score 余弦相似度，范围 [-1, 1]，<b>越大越相似</b>。
 *              对外统一用相似度而不是距离，避免调用方混淆方向。
 */
public record SearchResult(int id, float score) implements Comparable<SearchResult> {

    /** 相似度降序；分数相同时按 id 升序，保证结果稳定可复现。 */
    @Override
    public int compareTo(SearchResult other) {
        int byScore = Float.compare(other.score, this.score);
        return byScore != 0 ? byScore : Integer.compare(this.id, other.id);
    }

    @Override
    public String toString() {
        return "SearchResult{id=" + id + ", score=" + String.format("%.4f", score) + '}';
    }
}
