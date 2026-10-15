package com.vectorsearch.bench;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.vectorsearch.core.SearchResult;
import com.vectorsearch.core.bruteforce.BruteForceIndex;
import com.vectorsearch.core.hnsw.HnswIndex;
import com.vectorsearch.core.hnsw.HnswParams;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * 召回率测量器 —— 与 JMH 基准配对使用。
 *
 * <h2>为什么这个不能交给 JMH</h2>
 * JMH 只测<b>时间</b>，不测<b>正确性</b>。而 HNSW 是近似算法：
 * 把 efSearch 调到 1，延迟可以低到不可思议，但召回率也会烂到毫无用处。
 *
 * <p>所以「快」必须和「准」一起报。这个类负责算准的那一半：
 * 以<b>暴力检索的结果为 ground truth</b>，测 HNSW 在不同 efSearch 下的 recall@k。
 *
 * <h2>输出的 JSON 长这样</h2>
 * <pre>
 * {
 *   "nodeCount": 100000, "dimension": 512, "topK": 10,
 *   "bruteForce": { "meanMicros": 48000, "p95Micros": 52000 },
 *   "hnsw": {
 *     "buildMillis": 68000, "maxLevel": 5, "totalLinks": 3200000,
 *     "points": [
 *       { "efSearch": 10, "recallAtK": 0.81, "meanMicros": 180, "speedup": 266.7 },
 *       ...
 *     ]
 *   }
 * }
 * </pre>
 *
 * <p>运行：
 * <pre>
 *   java -cp ... com.vectorsearch.bench.RecallRunner
 *   java -Dbench.n=10000 -cp ... com.vectorsearch.bench.RecallRunner   # 快速验证
 * </pre>
 */
public final class RecallRunner {

    private static final ObjectMapper MAPPER = new ObjectMapper()
            .enable(SerializationFeature.INDENT_OUTPUT);

    private RecallRunner() {
    }

    public static void main(String[] args) throws Exception {
        int nodeCount = Integer.getInteger("bench.n", 100_000);
        int dimension = Integer.getInteger("bench.dim", 512);
        int topK = Integer.getInteger("bench.topK", 10);
        int m = Integer.getInteger("bench.m", 16);
        int efConstruction = Integer.getInteger("bench.efConstruction", 200);
        int queryCount = Integer.getInteger("bench.queries", 200);
        String outPath = System.getProperty("bench.out", "bench/results/recall.json");
        long seed = Long.getLong("bench.seed", 42L);

        int[] efSearchValues = parseEfSearch(System.getProperty("bench.efSearch", "10,20,50,100,200,400"));

        System.out.println("=".repeat(74));
        System.out.printf("召回率测量  n=%d  dim=%d  topK=%d  m=%d  efConstruction=%d%n",
                nodeCount, dimension, topK, m, efConstruction);
        System.out.println("=".repeat(74));

        // ---------- 数据集 ----------
        int clusters = SyntheticDataset.recommendedClusters(nodeCount);
        SyntheticDataset.Dataset dataset =
                SyntheticDataset.clustered(nodeCount, dimension, queryCount, clusters, seed);
        System.out.printf("数据集: %d 条 %d 维向量，%d 个簇，%d 条查询%n",
                dataset.size(), dataset.dimension(), clusters, dataset.queries().length);

        // ---------- ground truth：暴力检索 ----------
        System.out.println("\n[1/3] 构建暴力检索索引（ground truth）");
        BruteForceIndex bruteForce = new BruteForceIndex(dimension);
        long bfBuildStart = System.nanoTime();
        for (int i = 0; i < dataset.size(); i++) {
            bruteForce.add(i, dataset.vectors()[i]);
        }
        long bfBuildMillis = (System.nanoTime() - bfBuildStart) / 1_000_000L;
        System.out.printf("      建库耗时 %d ms%n", bfBuildMillis);

        System.out.println("      计算 ground truth 并计时...");
        List<Set<Integer>> groundTruth = new ArrayList<>(dataset.queries().length);
        long[] bfLatencies = new long[dataset.queries().length];
        for (int q = 0; q < dataset.queries().length; q++) {
            long start = System.nanoTime();
            List<SearchResult> hits = bruteForce.search(dataset.queries()[q], topK, 0);
            bfLatencies[q] = System.nanoTime() - start;
            groundTruth.add(idsOf(hits));
        }
        Map<String, Object> bruteForceStats = new LinkedHashMap<>();
        bruteForceStats.put("buildMillis", bfBuildMillis);
        bruteForceStats.put("meanMicros", round(meanMicros(bfLatencies)));
        bruteForceStats.put("p50Micros", round(percentile(bfLatencies, 50)));
        bruteForceStats.put("p95Micros", round(percentile(bfLatencies, 95)));
        bruteForceStats.put("p99Micros", round(percentile(bfLatencies, 99)));
        System.out.printf("      暴力检索单次: 均值 %.1f ms, P95 %.1f ms%n",
                meanMicros(bfLatencies) / 1000.0, percentile(bfLatencies, 95) / 1000.0);

        // ---------- HNSW ----------
        System.out.printf("%n[2/3] 构建 HNSW 索引（m=%d, efConstruction=%d）%n", m, efConstruction);
        HnswIndex hnsw = new HnswIndex(dimension, HnswParams.of(m, efConstruction), seed);
        long hnswBuildStart = System.nanoTime();
        for (int i = 0; i < dataset.size(); i++) {
            hnsw.add(i, dataset.vectors()[i]);
        }
        long hnswBuildMillis = (System.nanoTime() - hnswBuildStart) / 1_000_000L;
        double speedupOfBuild = (double) hnswBuildMillis / Math.max(1, bfBuildMillis);
        System.out.printf("      建图耗时 %d ms（是暴力建库的 %.2f 倍）%n",
                hnswBuildMillis, speedupOfBuild);
        System.out.printf("      最大层=%d  总边数=%d  单点平均出边=%.1f%n",
                hnsw.maxLevel(), hnsw.totalLinkCount(),
                (double) hnsw.totalLinkCount() / Math.max(1, nodeCount));

        // ---------- 扫描 efSearch ----------
        System.out.printf("%n[3/3] 扫描 efSearch，测 recall@%d 与延迟%n%n", topK);
        System.out.printf("  %-10s %-12s %-14s %-14s %-12s%n",
                "efSearch", "recall@10", "均值(us)", "P95(us)", "相对暴力");
        System.out.println("  " + "-".repeat(66));

        double bruteForceMean = meanMicros(bfLatencies);
        List<Map<String, Object>> points = new ArrayList<>();

        for (int efSearch : efSearchValues) {
            // 预热：让 JIT 把查询路径编译好，否则前几十次会明显偏慢
            for (int q = 0; q < Math.min(50, dataset.queries().length); q++) {
                hnsw.search(dataset.queries()[q], topK, efSearch);
            }

            double recallSum = 0;
            long[] latencies = new long[dataset.queries().length];
            for (int q = 0; q < dataset.queries().length; q++) {
                long start = System.nanoTime();
                List<SearchResult> hits = hnsw.search(dataset.queries()[q], topK, efSearch);
                latencies[q] = System.nanoTime() - start;

                Set<Integer> expected = new HashSet<>(groundTruth.get(q));
                expected.retainAll(idsOf(hits));
                recallSum += (double) expected.size() / Math.max(1, groundTruth.get(q).size());
            }

            double recall = recallSum / dataset.queries().length;
            double meanUs = meanMicros(latencies);
            double p95Us = percentile(latencies, 95);

            Map<String, Object> point = new LinkedHashMap<>();
            point.put("efSearch", efSearch);
            point.put("recallAtK", round(recall, 4));
            point.put("meanMicros", round(meanUs));
            point.put("p50Micros", round(percentile(latencies, 50)));
            point.put("p95Micros", round(p95Us));
            point.put("p99Micros", round(percentile(latencies, 99)));
            point.put("speedupVsBruteForce", round(bruteForceMean / Math.max(meanUs, 0.001), 1));
            points.add(point);

            System.out.printf("  %-10d %-12.4f %-14.1f %-14.1f %-12s%n",
                    efSearch, recall, meanUs, p95Us,
                    String.format("%.0fx", bruteForceMean / Math.max(meanUs, 0.001)));
        }

        // ---------- 输出 ----------
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("nodeCount", nodeCount);
        result.put("dimension", dimension);
        result.put("topK", topK);
        result.put("clusters", clusters);
        result.put("queryCount", dataset.queries().length);
        result.put("seed", seed);
        result.put("dataDistribution", "gaussian-mixture");

        Map<String, Object> params = new LinkedHashMap<>();
        params.put("m", m);
        params.put("efConstruction", efConstruction);
        result.put("hnswParams", params);

        result.put("bruteForce", bruteForceStats);

        Map<String, Object> hnswStats = new LinkedHashMap<>();
        hnswStats.put("buildMillis", hnswBuildMillis);
        hnswStats.put("maxLevel", hnsw.maxLevel());
        hnswStats.put("totalLinks", hnsw.totalLinkCount());
        hnswStats.put("averageDegree", round((double) hnsw.totalLinkCount() / Math.max(1, nodeCount), 1));
        hnswStats.put("levelHistogram", hnsw.levelHistogram());
        hnswStats.put("layerPopulation", hnsw.layerPopulation());
        hnswStats.put("points", points);
        result.put("hnsw", hnswStats);

        Path outputPath = Path.of(outPath);
        if (outputPath.getParent() != null) {
            Files.createDirectories(outputPath.getParent());
        }
        MAPPER.writeValue(outputPath.toFile(), result);
        System.out.printf("%n结果已写入: %s%n", outputPath.toAbsolutePath());

        // 顺便给出一句可以直接抄进报告的结论
        Map<String, Object> knee = points.stream()
                .filter(p -> ((Number) p.get("recallAtK")).doubleValue() >= 0.90)
                .findFirst().orElse(null);
        if (knee != null) {
            System.out.printf("达到 recall>=0.90 的最小 efSearch = %s（延迟 %.1f us，相对暴力 %.0fx）%n",
                    knee.get("efSearch"), knee.get("meanMicros"), knee.get("speedupVsBruteForce"));
        } else {
            System.out.println("警告: 没有配置能达到 recall>=0.90，需要调大 efConstruction 或 m");
        }
    }

    private static int[] parseEfSearch(String raw) {
        return Arrays.stream(raw.split(","))
                .map(String::trim)
                .filter(s -> !s.isEmpty())
                .mapToInt(Integer::parseInt)
                .toArray();
    }

    private static Set<Integer> idsOf(List<SearchResult> results) {
        Set<Integer> ids = new HashSet<>(results.size() * 2);
        for (SearchResult result : results) {
            ids.add(result.id());
        }
        return ids;
    }

    private static double meanMicros(long[] nanos) {
        if (nanos.length == 0) {
            return 0;
        }
        long sum = 0;
        for (long value : nanos) {
            sum += value;
        }
        return sum / (double) nanos.length / 1000.0;
    }

    private static double percentile(long[] nanos, double p) {
        if (nanos.length == 0) {
            return 0;
        }
        long[] sorted = nanos.clone();
        Arrays.sort(sorted);
        int index = (int) Math.ceil(p / 100.0 * sorted.length) - 1;
        index = Math.max(0, Math.min(sorted.length - 1, index));
        return sorted[index] / 1000.0;
    }

    private static double round(double value) {
        return round(value, 1);
    }

    private static double round(double value, int digits) {
        double factor = Math.pow(10, digits);
        return Math.round(value * factor) / factor;
    }
}
