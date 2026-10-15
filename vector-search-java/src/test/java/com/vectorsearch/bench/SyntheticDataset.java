package com.vectorsearch.bench;

import java.util.Random;

/**
 * 基准测试用的合成数据集。
 *
 * <h2>为什么基准测试用合成数据，而不用真实 embedding</h2>
 * 基准要的是<b>可控、可复现</b>：规模、维度、分布全部钉死，换台机器跑出来的
 * 数字才可比。真实语料会随着评测集构建不断变动，那么"上周 0.31ms、本周 0.29ms"
 * 这种对比就失去了意义 —— 你分不清是代码变快了还是数据变简单了。
 *
 * <h2>为什么用「聚类」而不是均匀随机 —— 这一点很关键</h2>
 * 均匀随机的高维向量是近似最近邻算法<b>最恶劣</b>的输入：
 * 维度一高，任意两点之间的距离会趋于集中（维数灾难），
 * "最近邻"这个概念本身都变得模糊，HNSW 的图结构也就失去了意义。
 *
 * 真实 embedding 不是这样的 —— 它们分布在一个低维流形附近，
 * 有明显的簇结构。所以本类默认生成<b>高斯混合</b>数据：
 * 先撒若干个簇心，再在簇心周围加噪声。这才是 embedding 的真实形态。
 *
 * <p><b>即便如此，仍然要清醒</b>：合成数据上的 recall 仍然只是参考值，
 * 不能替代真实语料上的评测（那由 {@code eval/} 下的脚手架负责）。
 * 这里的数据主要用于横向比较（HNSW vs 暴力、不同 efSearch、标量 vs SIMD），
 * 而不是宣称绝对质量。
 */
public final class SyntheticDataset {

    /** 数据集的向量与查询。查询从同样的分布中独立抽样，不与任何向量重合。 */
    public record Dataset(float[][] vectors, float[][] queries, int clusters) {

        public int size() {
            return vectors.length;
        }

        public int dimension() {
            return vectors.length == 0 ? 0 : vectors[0].length;
        }
    }

    private SyntheticDataset() {
    }

    /**
     * 高斯混合数据集。
     *
     * <h2>噪声标度必须与维度无关 —— 这里踩过一个坑</h2>
     * 最初写的是「每个维度加 {@code N(0, 0.25)} 的噪声」。这在低维下没问题，
     * 但高维下完全错了：噪声向量的<b>模长</b>是 {@code 0.25 * sqrt(d)}，
     * 512 维时约等于 5.66，而簇心是单位向量、模长只有 1.0。
     * 结果是<b>噪声彻底淹没了簇结构</b>，生成出来的数据近似均匀分布 ——
     * 正是维数灾难下近似最近邻算法最恶劣的输入。
     *
     * <p>症状很有迷惑性：n=5000 时 recall 还有 0.94，n=20000 时崩到 0.075。
     * 看起来像"图建得不好"或"参数没调好"，实际是<b>数据本身没有结构</b>，
     * 任何算法都做不好。
     *
     * <p>正确做法是让噪声模长与维度无关：每维噪声取
     * {@code noiseLevel / sqrt(dimension)}，这样总噪声模长就约等于
     * {@code noiseLevel}，与维度解耦。
     *
     * @param count      向量条数
     * @param dimension  向量维度
     * @param queryCount 查询条数
     * @param clusters   簇的数量。经验上取 {@code sqrt(count)} 量级比较接近真实语料
     * @param noiseLevel 簇内噪声的<b>模长</b>（不是每维标准差）。
     *                   簇心模长为 1，所以 0.35 意味着信噪比约 3:1 —— 簇结构清晰，
     *                   但簇内仍有明确的近邻次序，这才是真实 embedding 的形态。
     * @param seed       固定种子 —— 基准数据必须每次完全一致
     */
    public static Dataset clustered(int count, int dimension, int queryCount,
                                    int clusters, double noiseLevel, long seed) {
        Random random = new Random(seed);

        // 关键：把噪声模长换算成每维标准差，使总噪声模长 ≈ noiseLevel
        float noisePerDimension = (float) (noiseLevel / Math.sqrt(dimension));

        // 簇心：决定数据的宏观结构
        float[][] centers = new float[clusters][dimension];
        for (float[] center : centers) {
            for (int d = 0; d < dimension; d++) {
                center[d] = (float) random.nextGaussian();
            }
            // 簇心归一化到单位模长，于是噪声水平有明确的参照物
            normalize(center);
        }

        float[][] vectors = new float[count][dimension];
        for (int i = 0; i < count; i++) {
            // 按簇均匀分配，保证每簇样本量相近
            float[] center = centers[i % clusters];
            for (int d = 0; d < dimension; d++) {
                vectors[i][d] = center[d] + (float) random.nextGaussian() * noisePerDimension;
            }
        }

        float[][] queries = new float[queryCount][dimension];
        for (int i = 0; i < queryCount; i++) {
            float[] center = centers[random.nextInt(clusters)];
            for (int d = 0; d < dimension; d++) {
                queries[i][d] = center[d] + (float) random.nextGaussian() * noisePerDimension;
            }
        }

        return new Dataset(vectors, queries, clusters);
    }

    /** 默认噪声水平（模长口径）。 */
    public static final double DEFAULT_NOISE_LEVEL = 0.35;

    /** 使用默认噪声水平的便捷重载。 */
    public static Dataset clustered(int count, int dimension, int queryCount,
                                    int clusters, long seed) {
        return clustered(count, dimension, queryCount, clusters, DEFAULT_NOISE_LEVEL, seed);
    }

    /** 均匀随机数据 —— 作为"最恶劣输入"的对照，用于说明聚类假设有多重要。 */
    public static Dataset uniform(int count, int dimension, int queryCount, long seed) {
        Random random = new Random(seed);
        float[][] vectors = new float[count][dimension];
        for (float[] vector : vectors) {
            for (int d = 0; d < dimension; d++) {
                vector[d] = random.nextFloat() * 2f - 1f;
            }
        }
        float[][] queries = new float[queryCount][dimension];
        for (float[] query : queries) {
            for (int d = 0; d < dimension; d++) {
                query[d] = random.nextFloat() * 2f - 1f;
            }
        }
        return new Dataset(vectors, queries, 0);
    }

    /** 按向量条数推荐簇数，近似 {@code sqrt(n)}。 */
    public static int recommendedClusters(int count) {
        return Math.max(2, (int) Math.sqrt(count));
    }

    private static void normalize(float[] v) {
        double sum = 0;
        for (float x : v) {
            sum += (double) x * x;
        }
        double norm = Math.sqrt(sum);
        if (norm == 0) {
            return;
        }
        for (int i = 0; i < v.length; i++) {
            v[i] /= (float) norm;
        }
    }
}
