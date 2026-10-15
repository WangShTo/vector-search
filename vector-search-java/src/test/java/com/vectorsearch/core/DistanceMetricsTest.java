package com.vectorsearch.core;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 距离函数验证，包含标量版与 SIMD 版的一致性对照。
 *
 * <p><b>SIMD 版必须与标量版数值一致</b> —— 这是 SIMD 优化的经典陷阱：
 * 浮点累加顺序改变会带来微小误差，如果两者差异过大，说明尾部处理
 * 或 lane 归约写错了。这个测试就是拿来抓这种 bug 的。
 */
class DistanceMetricsTest {

    private static final float EPSILON = 1e-4f;

    @Test
    @DisplayName("内积 / 欧氏距离的基本性质")
    void basicProperties() {
        float[] a = {1f, 2f, 3f};
        float[] b = {4f, 5f, 6f};

        assertEquals(32f, DistanceMetrics.dot(a, b), EPSILON);       // 4+10+18
        assertEquals(27f, DistanceMetrics.l2Squared(a, b), EPSILON);  // 9+9+9

        // 与自己的距离为 0，相似度为 1
        assertEquals(1f, DistanceMetrics.cosineSimilarity(a, a), EPSILON);
        assertEquals(0f, DistanceMetrics.l2Squared(a, a), EPSILON);
    }

    @Test
    @DisplayName("归一化后内积等于余弦相似度")
    void normalizedDotEqualsCosine() {
        float[] a = {1f, 2f, 3f, 4f};
        float[] b = {4f, 3f, 2f, 1f};

        float cosine = DistanceMetrics.cosineSimilarity(a, b);
        float dotOfNormalized = DistanceMetrics.dot(
                DistanceMetrics.normalizedCopy(a),
                DistanceMetrics.normalizedCopy(b));

        assertEquals(cosine, dotOfNormalized, EPSILON);
    }

    @Test
    @DisplayName("零向量归一化不应除零")
    void zeroVectorNormalization() {
        float[] zero = {0f, 0f, 0f};
        DistanceMetrics.normalizeInPlace(zero);
        for (float v : zero) {
            assertEquals(0f, v);
        }
    }

    @Test
    @DisplayName("SIMD 版与标量版结果一致（含非 lane 整数倍的维度）")
    void simdMatchesScalar() {
        System.out.println("SIMD lane 宽度 = " + DistanceMetricsSimd.laneWidth());

        // 特意选 768（BGE 维度，是 8 的倍数）和 769/100（不是整数倍）
        // 两种都要测，尾部处理是最容易写错的地方
        for (int dimension : new int[]{1, 7, 8, 100, 384, 768, 769}) {
            float[] a = randomVector(dimension, 1L);
            float[] b = randomVector(dimension, 2L);

            assertRelativelyClose(DistanceMetrics.dot(a, b), DistanceMetricsSimd.dot(a, b),
                    "维度 " + dimension + " 的 dot 结果不一致");
            assertRelativelyClose(DistanceMetrics.l2Squared(a, b), DistanceMetricsSimd.l2Squared(a, b),
                    "维度 " + dimension + " 的 l2Squared 结果不一致");
        }
    }

    @Test
    @DisplayName("偏移版必须与独立数组的结果一致（这是扁平存储正确性的前提）")
    void offsetVariantsMatchStandalone() {
        // 特意混入非 lane 整数倍的维度：尾部标量收尾 + 偏移量叠加，
        // 这两个逻辑叠在一起是最容易出现 off-by-one 的地方
        for (int dimension : new int[]{1, 7, 8, 9, 100, 512, 769}) {
            int count = 4;
            float[][] standalone = new float[count][];
            float[] flat = new float[count * dimension];
            java.util.Random random = new java.util.Random(1000 + dimension);

            for (int i = 0; i < count; i++) {
                standalone[i] = new float[dimension];
                for (int d = 0; d < dimension; d++) {
                    standalone[i][d] = random.nextFloat() * 2f - 1f;
                }
                // 模拟扁平存储：第 i 条向量落在 i*dimension 处
                System.arraycopy(standalone[i], 0, flat, i * dimension, dimension);
            }

            for (int i = 0; i < count; i++) {
                for (int j = 0; j < count; j++) {
                    int offsetI = i * dimension;
                    int offsetJ = j * dimension;

                    assertEquals(
                            DistanceMetrics.dot(standalone[i], standalone[j]),
                            DistanceMetrics.dot(flat, offsetI, flat, offsetJ, dimension),
                            "标量偏移版 dot 不一致 (dim=" + dimension + ", " + i + "," + j + ")");

                    assertRelativelyClose(
                            DistanceMetricsSimd.dot(standalone[i], standalone[j]),
                            DistanceMetricsSimd.dot(flat, offsetI, flat, offsetJ, dimension),
                            "SIMD 偏移版 dot 不一致 (dim=" + dimension + ", " + i + "," + j + ")");

                    assertRelativelyClose(
                            DistanceMetricsSimd.l2Squared(standalone[i], standalone[j]),
                            DistanceMetricsSimd.l2Squared(flat, offsetI, flat, offsetJ, dimension),
                            "SIMD 偏移版 l2Squared 不一致 (dim=" + dimension + ", " + i + "," + j + ")");
                }
            }

            // 偏移版归一化不应污染相邻的向量
            float[] neighborBefore = java.util.Arrays.copyOfRange(flat, 0, dimension);
            DistanceMetrics.normalizeInPlace(flat, dimension, dimension);
            assertEquals(1f, DistanceMetrics.l2Squared(flat, dimension, new float[dimension], 0, dimension), 1e-3f,
                    "偏移版归一化后模长应为 1");
            for (int d = 0; d < dimension; d++) {
                assertEquals(neighborBefore[d], flat[d],
                        "归一化第 1 条向量时污染了第 0 条（dim=" + dimension + "）");
            }
        }
    }

    /**
     * 按<b>相对</b>误差比较两个 float。
     *
     * <p>这里必须用相对容差，不能用绝对容差 —— 原因是 SIMD 与标量的
     * <b>累加顺序不同</b>：标量是顺序累加，SIMD 是先把各 lane 部分和算出来
     * 再做横向归约。浮点加法不满足结合律，所以两者结果必然有微小差异。
     *
     * <p>实测：768 维、量级约 518 时，绝对差约 3e-4。而 float32 在这个量级上的
     * ULP 本身就约 6e-5，累加 768 项的误差累积到 1e-4 量级完全正常。
     * 用绝对容差 1e-4 去卡，会把这个<b>正常现象</b>误报成 bug。
     *
     * <p>相对容差取 1e-5：float32 的相对精度约 1e-7，累加 768 项后放宽两个
     * 数量级足够安全，同时仍然能抓到真正的实现错误（例如尾部元素没处理，
     * 那会产生百分之几的偏差）。
     */
    private void assertRelativelyClose(float expected, float actual, String message) {
        float tolerance = Math.max(1e-6f, Math.abs(expected) * 1e-5f);
        assertEquals(expected, actual, tolerance, message);
    }

    @Test
    @DisplayName("维度不一致必须抛异常，而不是静默算错")
    void dimensionMismatchThrows() {
        float[] a = {1f, 2f, 3f};
        float[] b = {1f, 2f};
        assertThrows(IllegalArgumentException.class, () -> DistanceMetrics.dot(a, b));
        assertThrows(IllegalArgumentException.class, () -> DistanceMetrics.l2Squared(a, b));
        assertThrows(IllegalArgumentException.class, () -> DistanceMetricsSimd.dot(a, b));
    }

    @Test
    @DisplayName("归一化后向量模长为 1")
    void normalizedVectorHasUnitNorm() {
        float[] v = randomVector(128, 99L);
        DistanceMetrics.normalizeInPlace(v);
        assertEquals(1f, DistanceMetrics.l2Squared(v, new float[128]), 1e-3f);
        assertTrue(DistanceMetrics.l2Squared(v, new float[128]) > 0f);
    }

    private float[] randomVector(int dimension, long seed) {
        java.util.Random random = new java.util.Random(seed);
        float[] v = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            v[i] = random.nextFloat() * 2f - 1f;
        }
        return v;
    }
}
