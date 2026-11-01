package com.vectorsearch.core;

import com.vectorsearch.core.hnsw.HnswIndex;
import com.vectorsearch.core.hnsw.HnswParams;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.function.IntPredicate;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 带过滤的检索 —— 验证「过滤下推到图遍历」确实比「搜完再筛」强。
 *
 * <p>这个测试守的是一个很容易被改回去的设计：把 filter 写成
 * {@code search(...).stream().filter(...)} 看起来更简单、更「函数式」，
 * 但在低选择性下 recall 会塌（1% 选择性实测 0.07 vs 0.94）。
 * 没有这个测试，将来重构时它会被当成等价改写。
 */
class FilteredRetrievalTest {

    private static final int DIMENSION = 32;
    private static final int SIZE = 1_200;
    /** 每 100 个 id 里只有 2 个合格 —— 2% 选择性，够低，能暴露问题。 */
    private static final int MATCH_EVERY = 100;
    private static final int MATCH_COUNT = 2;

    private static final IntPredicate TWO_PERCENT =
            id -> Math.floorMod(id, MATCH_EVERY) < MATCH_COUNT;

    private static HnswIndex buildIndex(float[][] vectors) {
        HnswIndex index = new HnswIndex(DIMENSION, HnswParams.of(16, 200), 7L);
        for (int i = 0; i < vectors.length; i++) {
            index.add(i, vectors[i]);
        }
        return index;
    }

    private static float[][] randomVectors(long seed) {
        Random random = new Random(seed);
        float[][] vectors = new float[FilteredRetrievalTest.SIZE][DIMENSION];
        for (float[] vector : vectors) {
            for (int d = 0; d < DIMENSION; d++) {
                vector[d] = (float) random.nextGaussian();
            }
            normalize(vector);
        }
        return vectors;
    }

    private static void normalize(float[] vector) {
        double norm = 0;
        for (float value : vector) {
            norm += value * value;
        }
        norm = Math.sqrt(norm);
        for (int i = 0; i < vector.length; i++) {
            vector[i] /= (float) norm;
        }
    }

    private static double dot(float[] a, float[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += (double) a[i] * b[i];
        }
        return sum;
    }

    /** 暴力检索，但只在本地的合格集合里找 —— 这就是这块的 ground truth。 */
    private static List<Integer> bruteForce(float[][] vectors, float[] query, int k, IntPredicate filter) {
        record Hit(int id, double score) {
        }
        List<Hit> hits = new ArrayList<>();
        for (int id = 0; id < vectors.length; id++) {
            if (filter.test(id)) {
                hits.add(new Hit(id, dot(query, vectors[id])));
            }
        }
        hits.sort((a, b) -> Double.compare(b.score(), a.score()));
        List<Integer> ids = new ArrayList<>(k);
        for (int i = 0; i < Math.min(k, hits.size()); i++) {
            ids.add(hits.get(i).id());
        }
        return ids;
    }

    private static double recall(List<Integer> expected, List<Integer> actual) {
        if (expected.isEmpty()) {
            return 1.0;
        }
        Set<Integer> actualSet = new HashSet<>(actual);
        int hit = 0;
        for (int id : expected) {
            if (actualSet.contains(id)) {
                hit++;
            }
        }
        return (double) hit / expected.size();
    }

    @Test
    @DisplayName("过滤检索只返回满足条件的 id")
    void returnsOnlyMatchingIds() {
        HnswIndex index = buildIndex(randomVectors(11));
        float[] query = randomVectors(12)[0];

        List<com.vectorsearch.core.SearchResult> hits = index.search(query, 5, 200, TWO_PERCENT);

        assertTrue(!hits.isEmpty(), "2% 选择性下也应该能返回结果");
        for (com.vectorsearch.core.SearchResult hit : hits) {
            assertTrue(TWO_PERCENT.test(hit.id()),
                    "返回了不合格的 id " + hit.id() + "，说明过滤没有下推到遍历里");
        }
    }

    @Test
    @DisplayName("过滤下推的召回明显高于搜完再筛")
    void filterAwareBeatsPostFiltering() {
        float[][] vectors = randomVectors(21);
        HnswIndex index = buildIndex(vectors);
        float[] query = randomVectors(22)[0];
        int k = 5;

        List<Integer> expected = bruteForce(vectors, query, k, TWO_PERCENT);

        // 下推：索引在遍历时就只把合格节点放进候选池
        List<Integer> aware = index.search(query, k, 200, TWO_PERCENT)
                .stream().map(com.vectorsearch.core.SearchResult::id).toList();

        // 搜完再筛：先不加条件取 40 条，再自己过滤。这是「看起来更简单」的写法。
        List<Integer> post = index.search(query, 40, 50).stream()
                .map(com.vectorsearch.core.SearchResult::id)
                .filter(id -> TWO_PERCENT.test(id))
                .limit(k)
                .toList();

        double awareRecall = recall(expected, aware);
        double postRecall = recall(expected, post);

        assertTrue(awareRecall >= 0.6,
                "过滤下推的 recall 应该接近完整，实际 " + awareRecall);
        assertTrue(awareRecall > postRecall,
                "下推 recall(" + awareRecall + ") 应该高于搜完再筛(" + postRecall + ")");
    }

    @Test
    @DisplayName("没有节点满足条件时返回空，而不是硬塞几条不合格的")
    void emptyResultWhenNothingMatches() {
        HnswIndex index = buildIndex(randomVectors(31));
        float[] query = randomVectors(32)[0];

        List<com.vectorsearch.core.SearchResult> hits =
                index.search(query, 5, 100, id -> false);

        assertTrue(hits.isEmpty(), "过滤器全 false 时不应该返回任何结果，实际 " + hits.size());
    }

    @Test
    @DisplayName("传 null 过滤器等价于不带过滤")
    void nullFilterEqualsUnfiltered() {
        HnswIndex index = buildIndex(randomVectors(41));
        float[] query = randomVectors(42)[0];

        List<Integer> plain = index.search(query, 10, 100).stream()
                .map(com.vectorsearch.core.SearchResult::id).toList();
        List<Integer> withNull = index.search(query, 10, 100, null).stream()
                .map(com.vectorsearch.core.SearchResult::id).toList();

        assertEquals(plain, withNull, "null 过滤器必须与三参数版本逐条一致");
    }
}
