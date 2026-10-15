package com.vectorsearch.core;

/**
 * 标量（scalar）距离与相似度计算。
 *
 * <p>这是<b>对照组基线</b>。第 5 周会用 JDK Vector API 写一份 SIMD 版本
 * （{@link DistanceMetricsSimd}），然后用 JMH 量出真实加速比。
 * 没有这个基线，"提升 3.2x" 就是无源之水 —— 面试官必问「跟什么比」。
 *
 * <p>向量在入库前统一做 L2 归一化，因此内积等价于余弦相似度。
 * 这是文本 embedding 的标准做法（BGE 官方推荐 normalized + inner product）。
 */
public final class DistanceMetrics {

    private DistanceMetrics() {
    }

    /** 内积。向量已归一化时，等价于余弦相似度。 */
    public static float dot(float[] a, float[] b) {
        checkSameLength(a, b);
        float sum = 0f;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    /** 平方欧氏距离。注意不是开方后的值 —— 排序只需要单调量，省一次 sqrt。 */
    public static float l2Squared(float[] a, float[] b) {
        checkSameLength(a, b);
        float sum = 0f;
        for (int i = 0; i < a.length; i++) {
            float d = a[i] - b[i];
            sum += d * d;
        }
        return sum;
    }

    public static float l2(float[] a, float[] b) {
        return (float) Math.sqrt(l2Squared(a, b));
    }

    /** 余弦相似度，范围 [-1, 1]，越大越相似。 */
    public static float cosineSimilarity(float[] a, float[] b) {
        checkSameLength(a, b);
        float dotProduct = 0f;
        float normA = 0f;
        float normB = 0f;
        for (int i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        float denom = (float) (Math.sqrt(normA) * Math.sqrt(normB));
        return denom == 0f ? 0f : dotProduct / denom;
    }

    /** 原地 L2 归一化。零向量保持原样，不做除零。 */
    public static void normalizeInPlace(float[] v) {
        float sumOfSquares = 0f;
        for (float x : v) {
            sumOfSquares += x * x;
        }
        float norm = (float) Math.sqrt(sumOfSquares);
        if (norm == 0f) {
            return;
        }
        float inv = 1f / norm;
        for (int i = 0; i < v.length; i++) {
            v[i] *= inv;
        }
    }

    /** 返回归一化后的副本，不改动入参。 */
    public static float[] normalizedCopy(float[] v) {
        float[] copy = v.clone();
        normalizeInPlace(copy);
        return copy;
    }

    /**
     * HNSW 内部按"距离越小越近"排序，而归一化后的内积越大越相似。
     * 这个转换把相似度统一成距离语义，避免两套比较逻辑混用出错。
     */
    public static float cosineDistanceFromDot(float dot) {
        return 1f - dot;
    }

    private static void checkSameLength(float[] a, float[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException(
                    "向量维度不一致: " + a.length + " vs " + b.length);
        }
    }
}
