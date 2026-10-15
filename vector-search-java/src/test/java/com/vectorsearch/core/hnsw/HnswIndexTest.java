package com.vectorsearch.core.hnsw;

import com.vectorsearch.core.SearchResult;
import com.vectorsearch.core.VectorIndex;
import com.vectorsearch.core.bruteforce.BruteForceIndex;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * HNSW 正确性验证。
 *
 * <p>核心方法：<b>用暴力检索的结果当 ground truth，算 recall@k</b>。
 * 这是近似索引唯一有意义的正确性指标 —— 逐条比对"是否完全一致"是错的，
 * 因为 HNSW 本来就是近似的，它永远不可能 100% 一致（除非 ef 开到 n）。
 *
 * <p>本测试同时扮演一个角色：<b>性能对比的对照组生成器</b>。
 * 第 2 周起，同样的数据规模会喂给 JMH，量出两者的延迟差距。
 */
class HnswIndexTest {

    private static final int DIMENSION = 32;
    private static final int NODE_COUNT = 2_000;
    private static final int QUERY_COUNT = 50;
    private static final int K = 10;
    private static final long SEED = 42L;

    @Test
    @DisplayName("recall@10 应不低于 0.90（m=16, efConstruction=200, efSearch=100）")
    void recallShouldBeAcceptable() {
        float[][] vectors = randomVectors(NODE_COUNT, DIMENSION, SEED);
        float[][] queries = randomVectors(QUERY_COUNT, DIMENSION, SEED + 1);

        VectorIndex bruteForce = new BruteForceIndex(DIMENSION);
        HnswIndex hnsw = new HnswIndex(DIMENSION, HnswParams.of(16, 200), SEED);

        for (int i = 0; i < NODE_COUNT; i++) {
            bruteForce.add(i, vectors[i]);
            hnsw.add(i, vectors[i]);
        }

        assertEquals(NODE_COUNT, hnsw.size());
        assertEquals(NODE_COUNT, bruteForce.size());

        double recall = averageRecall(bruteForce, hnsw, queries);

        // 把实测值打出来 —— 调参时靠它，而不是靠"看起来还行"
        System.out.printf("recall@%d = %.4f  (m=%d, efConstruction=%d, efSearch=%d, n=%d, dim=%d)%n",
                K, recall, hnsw.params().m(), hnsw.params().efConstruction(),
                100, NODE_COUNT, DIMENSION);

        assertTrue(recall >= 0.90,
                "recall@" + K + " 过低: " + recall + "，检查建图逻辑或邻居选择启发式");
    }

    @Test
    @DisplayName("efSearch 调大，recall 应单调不降（这是 recall/延迟 权衡的验证）")
    void largerEfSearchShouldNotHurtRecall() {
        float[][] vectors = randomVectors(NODE_COUNT, DIMENSION, SEED);
        float[][] queries = randomVectors(QUERY_COUNT, DIMENSION, SEED + 1);

        VectorIndex bruteForce = new BruteForceIndex(DIMENSION);
        HnswIndex hnsw = new HnswIndex(DIMENSION, HnswParams.of(16, 200), SEED);
        for (int i = 0; i < NODE_COUNT; i++) {
            bruteForce.add(i, vectors[i]);
            hnsw.add(i, vectors[i]);
        }

        double recallAt10 = recallAt(bruteForce, hnsw, queries, 10);
        double recallAt200 = recallAt(bruteForce, hnsw, queries, 200);

        System.out.printf("efSearch=10  -> recall=%.4f%nefSearch=200 -> recall=%.4f%n",
                recallAt10, recallAt200);

        // 允许极小浮动（近似算法的结果不保证严格单调），但不允许明显倒退
        assertTrue(recallAt200 >= recallAt10 - 1e-9,
                "efSearch 调大后 recall 反而下降: " + recallAt10 + " -> " + recallAt200);
    }

    @Test
    @DisplayName("efSearch >= 节点数时，结果应与暴力检索完全一致")
    void exhaustiveEfShouldMatchGroundTruth() {
        int smallCount = 200;
        float[][] vectors = randomVectors(smallCount, DIMENSION, SEED);
        float[][] queries = randomVectors(10, DIMENSION, SEED + 1);

        VectorIndex bruteForce = new BruteForceIndex(DIMENSION);
        HnswIndex hnsw = new HnswIndex(DIMENSION, HnswParams.of(16, 200), SEED);
        for (int i = 0; i < smallCount; i++) {
            bruteForce.add(i, vectors[i]);
            hnsw.add(i, vectors[i]);
        }

        // ef 开到比全量还大，等价于在图上做穷举 —— 此时近似误差应当消失
        for (float[] query : queries) {
            List<SearchResult> expected = bruteForce.search(query, K, smallCount);
            List<SearchResult> actual = hnsw.search(query, K, smallCount);
            assertEquals(ids(expected), ids(actual),
                    "ef 开到全量时结果必须与暴力检索一致");
        }
    }

    @Test
    @DisplayName("层高分布呈指数衰减，且各层存在节点数严格递减（金字塔结构）")
    void levelHistogramShouldDecay() {
        float[][] vectors = randomVectors(NODE_COUNT, DIMENSION, SEED);
        HnswIndex hnsw = new HnswIndex(DIMENSION, HnswParams.of(16, 200), SEED);
        for (int i = 0; i < NODE_COUNT; i++) {
            hnsw.add(i, vectors[i]);
        }

        // 这两个量很容易混淆，测试里必须分开验证：
        //   histogram[l]  = 最高层恰好是 l 的节点数（层高分布）
        //   population[l] = 第 l 层实际存在的节点数（累计，因为节点存在于 0..自身层高 的所有层）
        int[] histogram = hnsw.levelHistogram();
        int[] population = hnsw.layerPopulation();

        int total = 0;
        double averageLevel = 0;
        for (int level = 0; level < histogram.length; level++) {
            total += histogram[level];
            averageLevel += (double) level * histogram[level] / NODE_COUNT;
            System.out.printf("level %2d: 层高为此层 %5d 个，该层实际存在 %5d 个%n",
                    level, histogram[level], population[level]);
        }
        System.out.printf("平均层高 = %.3f，最大层 = %d%n", averageLevel, hnsw.maxLevel());

        // 每个节点都必须被计入某一层，且每个节点都存在于第 0 层
        assertEquals(NODE_COUNT, total, "层高直方图总和应等于节点总数");
        assertEquals(NODE_COUNT, population[0], "第 0 层必须包含全部节点");

        for (int level = 1; level < histogram.length; level++) {
            assertTrue(histogram[level] <= histogram[level - 1],
                    "level " + level + " 的层高计数不该多于上一层");
            assertTrue(population[level] < population[level - 1],
                    "level " + level + " 的存在节点数应严格少于上一层");
        }

        // 几何分布下绝大多数节点层高为 0 —— 这是金字塔结构的直接体现
        assertTrue(histogram[0] > NODE_COUNT * 0.8,
                "层高为 0 的节点应占绝大多数，实际 " + histogram[0] + "/" + NODE_COUNT);
    }

    @Test
    @DisplayName("边界情况：空索引、k 大于节点数、维度不匹配")
    void edgeCases() {
        HnswIndex hnsw = new HnswIndex(DIMENSION, HnswParams.of(16, 200), SEED);

        assertTrue(hnsw.search(randomVectors(1, DIMENSION, 7)[0], K, 100).isEmpty(),
                "空索引应返回空结果");

        hnsw.add(1, randomVectors(1, DIMENSION, 7)[0]);
        assertEquals(1, hnsw.search(randomVectors(1, DIMENSION, 8)[0], K, 100).size(),
                "只有一个节点时应返回它自己");

        // k 超过节点数应返回全部，而不是抛异常
        for (int i = 0; i < 5; i++) {
            hnsw.add(100 + i, randomVectors(1, DIMENSION, 20 + i)[0]);
        }
        assertEquals(6, hnsw.search(randomVectors(1, DIMENSION, 9)[0], 100, 100).size());

        // k <= 0 返回空
        assertTrue(hnsw.search(randomVectors(1, DIMENSION, 9)[0], 0, 100).isEmpty());

        // 维度不匹配必须尽早失败，而不是算出垃圾结果
        assertThrows(IllegalArgumentException.class,
                () -> hnsw.add(2, new float[DIMENSION + 1]));
        assertThrows(IllegalArgumentException.class,
                () -> hnsw.search(new float[DIMENSION + 1], K, 100));
    }

    // ------------------------------------------------------------------
    // 辅助
    // ------------------------------------------------------------------

    private double averageRecall(VectorIndex groundTruth, VectorIndex candidate, float[][] queries) {
        return recallAt(groundTruth, candidate, queries, 100);
    }

    private double recallAt(VectorIndex groundTruth, VectorIndex candidate,
                            float[][] queries, int efSearch) {
        double total = 0;
        for (float[] query : queries) {
            Set<Integer> expected = ids(groundTruth.search(query, K, 0));
            Set<Integer> actual = ids(candidate.search(query, K, efSearch));
            expected.retainAll(actual);
            total += (double) expected.size() / K;
        }
        return total / queries.length;
    }

    private Set<Integer> ids(List<SearchResult> results) {
        Set<Integer> ids = new HashSet<>();
        for (SearchResult r : results) {
            ids.add(r.id());
        }
        return ids;
    }

    /** 生成随机向量。不做归一化 —— 索引内部会归一化，正好顺带验证这条路径。 */
    private float[][] randomVectors(int count, int dimension, long seed) {
        Random random = new Random(seed);
        float[][] vectors = new float[count][dimension];
        for (int i = 0; i < count; i++) {
            for (int j = 0; j < dimension; j++) {
                vectors[i][j] = random.nextFloat() * 2f - 1f;
            }
        }
        return vectors;
    }
}
