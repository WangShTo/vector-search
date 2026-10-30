package com.vectorsearch.bench;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.vectorsearch.core.SearchResult;
import com.vectorsearch.core.bruteforce.BruteForceIndex;
import com.vectorsearch.core.hnsw.HnswIndex;
import com.vectorsearch.core.hnsw.HnswParams;
import com.vectorsearch.core.quant.Quantization;

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
 * 量化（SQ8）的三角权衡：内存 / 延迟 / 召回。
 *
 * <p>一个优化如果只报告"内存降了 4 倍"是没有说服力的——降内存的常见代价是
 * 检索变慢（要解码）或者变不准（丢精度）。所以这里必须把三者放在同一张表里：
 * 同一份数据集、同一组查询、同一套 HNSW 参数，只换向量存储精度。
 *
 * <p>召回用暴力检索作 ground truth —— 用 f32 索引当基准会把两个近似算法的
 * 误差混在一起，量不出量化的真实代价。
 *
 * <p>运行：
 * <pre>
 *   java -Dbench.n=100000 -cp ... com.vectorsearch.bench.QuantizationBench
 * </pre>
 */
public final class QuantizationBench {

    private static final ObjectMapper MAPPER = new ObjectMapper()
            .enable(SerializationFeature.INDENT_OUTPUT);

    private QuantizationBench() {
    }

    public static void main(String[] args) throws Exception {
        int nodeCount = Integer.getInteger("bench.n", 100_000);
        int dimension = Integer.getInteger("bench.dim", 512);
        int m = Integer.getInteger("bench.m", 16);
        int efConstruction = Integer.getInteger("bench.efConstruction", 200);
        int topK = Integer.getInteger("bench.topK", 10);
        int efSearch = Integer.getInteger("bench.efSearch", 50);
        int recallQueries = Integer.getInteger("bench.queries", 60);
        int latencyRounds = Integer.getInteger("bench.latencyRounds", 500);
        String outPath = System.getProperty("bench.out", "bench/results/quantization.json");

        System.out.println("=".repeat(74));
        System.out.printf("量化权衡测量  n=%d  dim=%d  m=%d  efConstruction=%d  efSearch=%d%n",
                nodeCount, dimension, m, efConstruction, efSearch);
        System.out.println("=".repeat(74));

        SyntheticDataset.Dataset dataset = SyntheticDataset.clustered(
                nodeCount, dimension, recallQueries,
                SyntheticDataset.recommendedClusters(nodeCount), 42L);

        // ---------- 暴力检索：ground truth ----------
        System.out.println("构建暴力检索基准（ground truth）...");
        BruteForceIndex truth = new BruteForceIndex(dimension);
        for (int i = 0; i < dataset.size(); i++) {
            truth.add(i, dataset.vectors()[i]);
        }
        List<Set<Integer>> expected = new ArrayList<>(recallQueries);
        for (int q = 0; q < recallQueries; q++) {
            expected.add(ids(truth.search(dataset.queries()[q], topK, efSearch)));
        }
        System.out.println("  完成");

        List<Map<String, Object>> variants = new ArrayList<>();
        variants.add(measure(Quantization.NONE, dataset, expected, recallQueries, latencyRounds,
                m, efConstruction, topK, efSearch, dimension));
        variants.add(measure(Quantization.SQ8, dataset, expected, recallQueries, latencyRounds,
                m, efConstruction, topK, efSearch, dimension));

        // ---------- 对比 ----------
        Map<String, Object> f32 = variants.get(0);
        Map<String, Object> sq8 = variants.get(1);

        double memoryRatio = (double) (long) f32.get("estimatedBytes")
                / Math.max(1, (long) sq8.get("estimatedBytes"));
        double latencyRatio = (double) (double) sq8.get("p50Micros")
                / Math.max(1e-9, (double) f32.get("p50Micros"));
        double recallDelta = (double) sq8.get("recall") - (double) f32.get("recall");
        double fileRatio = (double) (long) f32.get("snapshotBytes")
                / Math.max(1, (long) sq8.get("snapshotBytes"));

        System.out.println();
        System.out.printf("%-8s %12s %14s %10s %10s %10s %10s%n",
                "精度", "建图(秒)", "内存(MB)", "召回", "P50(µs)", "P95(µs)", "快照(MB)");
        for (Map<String, Object> v : variants) {
            System.out.printf("%-8s %12.1f %14.1f %10.4f %10.1f %10.1f %10.1f%n",
                    v.get("mode"),
                    (long) v.get("buildMillis") / 1000.0,
                    (long) v.get("estimatedBytes") / 1024.0 / 1024.0,
                    v.get("recall"),
                    v.get("p50Micros"),
                    v.get("p95Micros"),
                    (long) v.get("snapshotBytes") / 1024.0 / 1024.0);
        }
        System.out.println();
        System.out.printf("★ 内存 %.2f 倍（快照 %.2f 倍）  延迟 %.2f 倍  召回差 %+.4f%n",
                memoryRatio, fileRatio, latencyRatio, recallDelta);

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("nodeCount", nodeCount);
        result.put("dimension", dimension);
        result.put("m", m);
        result.put("efConstruction", efConstruction);
        result.put("efSearch", efSearch);
        result.put("topK", topK);
        result.put("recallQueries", recallQueries);
        result.put("variants", variants);
        Map<String, Object> ratios = new LinkedHashMap<>();
        ratios.put("memory", round(memoryRatio));
        ratios.put("snapshot", round(fileRatio));
        ratios.put("latencyP50", round(latencyRatio));
        ratios.put("recallDelta", round(recallDelta));
        result.put("ratios", ratios);

        Path outputPath = Path.of(outPath);
        if (outputPath.getParent() != null) {
            Files.createDirectories(outputPath.getParent());
        }
        MAPPER.writeValue(outputPath.toFile(), result);
        System.out.printf("%n结果已写入: %s%n", outputPath.toAbsolutePath());
    }

    private static Map<String, Object> measure(
            Quantization quantization, SyntheticDataset.Dataset dataset,
            List<Set<Integer>> expected, int recallQueries, int latencyRounds,
            int m, int efConstruction, int topK, int efSearch, int dimension) throws Exception {

        System.out.printf("%n---------- %s ----------%n", quantization.code());

        HnswIndex index = new HnswIndex(dimension, HnswParams.of(m, efConstruction), 42L, quantization);
        long buildStart = System.nanoTime();
        for (int i = 0; i < dataset.size(); i++) {
            index.add(i, dataset.vectors()[i]);
        }
        long buildMillis = (System.nanoTime() - buildStart) / 1_000_000L;
        System.out.printf("建图: %d ms（%.1f 秒）%n", buildMillis, buildMillis / 1000.0);
        System.out.printf("估算内存: %.1f MB%n", index.estimatedBytes() / 1024.0 / 1024.0);

        // 召回
        int hits = 0;
        int total = 0;
        for (int q = 0; q < recallQueries; q++) {
            Set<Integer> actual = ids(index.search(dataset.queries()[q], topK, efSearch));
            Set<Integer> want = expected.get(q);
            for (int id : want) {
                if (actual.contains(id)) {
                    hits++;
                }
            }
            total += want.size();
        }
        double recall = total == 0 ? 0 : (double) hits / total;
        System.out.printf("recall@%d = %.4f%n", topK, recall);

        // 延迟（单线程，预热后取分位）
        long[] samples = new long[latencyRounds];
        float[] query = dataset.queries()[0];
        for (int i = 0; i < 200; i++) {
            index.search(query, topK, efSearch);
        }
        for (int i = 0; i < latencyRounds; i++) {
            long start = System.nanoTime();
            index.search(query, topK, efSearch);
            samples[i] = System.nanoTime() - start;
        }
        Arrays.sort(samples);
        double p50 = samples[samples.length / 2] / 1000.0;
        double p95 = samples[(int) (samples.length * 0.95)] / 1000.0;
        double p99 = samples[(int) (samples.length * 0.99)] / 1000.0;
        System.out.printf("延迟: P50 %.1f µs  P95 %.1f µs  P99 %.1f µs%n", p50, p95, p99);

        // 快照体积（落盘后的大小最能反映真实内存占用）
        Path file = Files.createTempFile("quant-bench", ".snapshot");
        file.toFile().deleteOnExit();
        index.writeTo(file);
        long snapshotBytes = Files.size(file);
        System.out.printf("快照: %.1f MB%n", snapshotBytes / 1024.0 / 1024.0);
        HnswIndex verify = HnswIndex.readFrom(file);
        if (verify.size() != index.size() || verify.quantization() != quantization) {
            throw new IllegalStateException("快照往返失败: " + verify.size() + " / " + verify.quantization());
        }
        Files.deleteIfExists(file);

        Map<String, Object> row = new LinkedHashMap<>();
        row.put("mode", quantization.code());
        row.put("buildMillis", buildMillis);
        row.put("estimatedBytes", index.estimatedBytes());
        row.put("snapshotBytes", snapshotBytes);
        row.put("recall", round(recall));
        row.put("p50Micros", round(p50));
        row.put("p95Micros", round(p95));
        row.put("p99Micros", round(p99));
        return row;
    }

    private static Set<Integer> ids(List<SearchResult> results) {
        Set<Integer> ids = new HashSet<>();
        for (SearchResult r : results) {
            ids.add(r.id());
        }
        return ids;
    }

    private static double round(double value) {
        return Math.round(value * 10_000.0) / 10_000.0;
    }
}
