package com.vectorsearch.core.hnsw;

import com.vectorsearch.core.SearchResult;
import com.vectorsearch.core.bruteforce.BruteForceIndex;
import com.vectorsearch.core.quant.Quantization;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * SQ8 量化索引的行为验证。
 *
 * <p>量化是「用精度换内存」，所以测试的重点不是「跑得通」，而是三件事：
 * <ul>
 *   <li>召回确实没有明显下降（用暴力检索当 ground truth）</li>
 *   <li>内存确实降到了 1/4 左右</li>
 *   <li>落盘再读回来，量化精度没有丢</li>
 * </ul>
 */
class QuantizedHnswTest {

    private static final int DIMENSION = 64;
    private static final int CLUSTERS = 20;
    private static final int COUNT = 800;
    private static final int K = 10;
    private static final int EF_SEARCH = 64;

    @Test
    void 量化后召回不应明显下降() {
        Random random = new Random(7L);
        float[][] vectors = clusteredVectors(random);

        BruteForceIndex truth = new BruteForceIndex(DIMENSION);
        HnswIndex exact = new HnswIndex(DIMENSION, HnswParams.of(16, 100), 42L, Quantization.NONE);
        HnswIndex quantized = new HnswIndex(DIMENSION, HnswParams.of(16, 100), 42L, Quantization.SQ8);
        for (int i = 0; i < COUNT; i++) {
            truth.add(i, vectors[i]);
            exact.add(i, vectors[i]);
            quantized.add(i, vectors[i]);
        }

        double exactRecall = recall(truth, exact, random);
        double quantizedRecall = recall(truth, quantized, random);

        // 64 维、800 点的规模下，量化噪声应远小于簇内距离。
        // 允许相对回落，但绝对值必须仍然可用 —— 否则这个交换就不值得做
        assertTrue(quantizedRecall > 0.9, "SQ8 recall 过低: " + quantizedRecall);
        assertTrue(quantizedRecall >= exactRecall - 0.05,
                "SQ8 相对 f32 掉得太多: " + exactRecall + " -> " + quantizedRecall);
    }

    @Test
    void 量化后内存约为四分之一() {
        Random random = new Random(11L);
        float[][] vectors = clusteredVectors(random);

        HnswIndex exact = new HnswIndex(DIMENSION, HnswParams.of(16, 100), 42L, Quantization.NONE);
        HnswIndex quantized = new HnswIndex(DIMENSION, HnswParams.of(16, 100), 42L, Quantization.SQ8);
        for (int i = 0; i < COUNT; i++) {
            exact.add(i, vectors[i]);
            quantized.add(i, vectors[i]);
        }

        long exactBytes = exact.estimatedBytes();
        long quantizedBytes = quantized.estimatedBytes();

        // 只有向量段变小，图结构（邻接表、层号）不跟着缩，
        // 所以整体不会正好是 1/4，但必须明显更小
        assertTrue(quantizedBytes < exactBytes * 0.6,
                "SQ8 内存应明显小于 f32: " + quantizedBytes + " vs " + exactBytes);
        assertTrue(quantizedBytes > 0);
    }

    @Test
    void 快照往返保留量化精度(@TempDir Path dir) throws IOException {
        Random random = new Random(13L);
        float[][] vectors = clusteredVectors(random);

        HnswIndex original = new HnswIndex(DIMENSION, HnswParams.of(16, 100), 42L, Quantization.SQ8);
        for (int i = 0; i < COUNT; i++) {
            original.add(i, vectors[i]);
        }

        Path file = dir.resolve("quantized.snapshot");
        original.writeTo(file);

        HnswIndex restored = HnswIndex.readFrom(file);

        assertEquals(Quantization.SQ8, restored.quantization(), "读回来的仍是 SQ8");
        assertEquals("hnsw-sq8", restored.name());
        assertEquals(original.size(), restored.size());

        // 同样的查询必须给出同样的结果 —— 若量化参数没存进文件，这里会立刻不一致
        float[] query = vectors[0];
        List<SearchResult> before = original.search(query, K, EF_SEARCH);
        List<SearchResult> after = restored.search(query, K, EF_SEARCH);
        assertEquals(before.size(), after.size());
        for (int i = 0; i < before.size(); i++) {
            assertEquals(before.get(i).id(), after.get(i).id(), "第 " + i + " 位不一致");
            assertEquals(before.get(i).score(), after.get(i).score(), 1e-4f);
        }
    }

    @Test
    void 压实读出的向量可以原样写回() {
        Random random = new Random(17L);
        float[][] vectors = clusteredVectors(random);

        HnswIndex index = new HnswIndex(DIMENSION, HnswParams.of(16, 100), 42L, Quantization.SQ8);
        for (int i = 0; i < COUNT; i++) {
            index.add(i, vectors[i]);
        }

        // 压实 = 「读出 float → 重新 add」。若量化不幂等，这里会累积误差
        HnswIndex rebuilt = new HnswIndex(DIMENSION, HnswParams.of(16, 100), 42L, Quantization.SQ8);
        index.forEach((externalId, vector) -> rebuilt.add(externalId, vector));

        assertEquals(index.size(), rebuilt.size());
        float[] query = vectors[COUNT / 2];
        List<SearchResult> a = index.search(query, K, EF_SEARCH);
        List<SearchResult> b = rebuilt.search(query, K, EF_SEARCH);
        assertEquals(a.size(), b.size());
        for (int i = 0; i < a.size(); i++) {
            assertEquals(a.get(i).id(), b.get(i).id());
        }
    }

    // ------------------------------------------------------------------

    /** 有簇结构的向量：全随机构成的点彼此近似正交，recall 本身就没意义。 */
    private static float[][] clusteredVectors(Random random) {
        float[][] centers = new float[CLUSTERS][];
        for (int c = 0; c < CLUSTERS; c++) {
            centers[c] = GaussianUnitVector.next(random, DIMENSION);
        }

        float[][] vectors = new float[COUNT][];
        for (int i = 0; i < COUNT; i++) {
            float[] center = centers[i % CLUSTERS];
            float[] v = new float[DIMENSION];
            double norm = 0;
            for (int d = 0; d < DIMENSION; d++) {
                v[d] = center[d] + (float) (random.nextGaussian() * 0.3);
                norm += v[d] * v[d];
            }
            float scale = (float) (1.0 / Math.sqrt(norm));
            for (int d = 0; d < DIMENSION; d++) {
                v[d] *= scale;
            }
            vectors[i] = v;
        }
        return vectors;
    }

    private static double recall(BruteForceIndex truth, HnswIndex candidate, Random random) {
        int queries = 40;
        int hits = 0;
        int total = 0;
        for (int q = 0; q < queries; q++) {
            float[] query = GaussianUnitVector.next(random, DIMENSION);
            Set<Integer> expected = ids(truth.search(query, K, EF_SEARCH));
            Set<Integer> actual = ids(candidate.search(query, K, EF_SEARCH));
            for (int id : expected) {
                if (actual.contains(id)) {
                    hits++;
                }
            }
            total += expected.size();
        }
        return total == 0 ? 0 : (double) hits / total;
    }

    private static Set<Integer> ids(List<SearchResult> results) {
        Set<Integer> ids = new HashSet<>();
        for (SearchResult r : results) {
            ids.add(r.id());
        }
        return ids;
    }

    /** 生成 L2 归一化的高斯随机向量。 */
    private static final class GaussianUnitVector {
        static float[] next(Random random, int dimension) {
            float[] v = new float[dimension];
            double norm = 0;
            for (int i = 0; i < dimension; i++) {
                v[i] = (float) random.nextGaussian();
                norm += v[i] * v[i];
            }
            float scale = (float) (1.0 / Math.sqrt(norm));
            for (int i = 0; i < dimension; i++) {
                v[i] *= scale;
            }
            return v;
        }
    }
}
