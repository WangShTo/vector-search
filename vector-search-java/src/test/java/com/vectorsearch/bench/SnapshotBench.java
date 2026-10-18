package com.vectorsearch.bench;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.vectorsearch.core.SearchResult;
import com.vectorsearch.core.hnsw.HnswIndex;
import com.vectorsearch.core.hnsw.HnswParams;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 测量索引快照的收益：重建图 vs 从快照恢复。
 *
 * <p>这是持久化功能的<b>存在理由</b>——如果恢复不比重建快，这个功能就没有意义。
 * 所以不能只说"实现了持久化"，必须量出两者的耗时差。
 *
 * <p>同时验证一件事：恢复后的索引<b>检索结果必须与重建后完全一致</b>。
 * 快照坏了却还能"正常"返回结果，是这类功能最危险的失败模式。
 *
 * <p>运行：
 * <pre>
 *   java -Dbench.n=100000 -cp ... com.vectorsearch.bench.SnapshotBench
 * </pre>
 */
public final class SnapshotBench {

    private static final ObjectMapper MAPPER = new ObjectMapper()
            .enable(SerializationFeature.INDENT_OUTPUT);

    private SnapshotBench() {
    }

    public static void main(String[] args) throws Exception {
        int nodeCount = Integer.getInteger("bench.n", 100_000);
        int dimension = Integer.getInteger("bench.dim", 512);
        int m = Integer.getInteger("bench.m", 16);
        int efConstruction = Integer.getInteger("bench.efConstruction", 200);
        int topK = Integer.getInteger("bench.topK", 10);
        int efSearch = Integer.getInteger("bench.efSearch", 50);
        int verifyQueries = Integer.getInteger("bench.queries", 30);
        String outPath = System.getProperty("bench.out", "bench/results/snapshot.json");

        System.out.println("=".repeat(70));
        System.out.printf("索引快照测量  n=%d  dim=%d  m=%d  efConstruction=%d%n",
                nodeCount, dimension, m, efConstruction);
        System.out.println("=".repeat(70));

        // ---------- 1. 建图（这就是每次冷启动都要付的成本）----------
        SyntheticDataset.Dataset dataset = SyntheticDataset.clustered(
                nodeCount, dimension, verifyQueries + 1,
                SyntheticDataset.recommendedClusters(nodeCount), 42L);

        HnswIndex index = new HnswIndex(dimension, HnswParams.of(m, efConstruction), 42L);
        long buildStart = System.nanoTime();
        for (int i = 0; i < dataset.size(); i++) {
            index.add(i, dataset.vectors()[i]);
        }
        long buildMillis = (System.nanoTime() - buildStart) / 1_000_000L;
        System.out.printf("建图: %d ms（%.1f 秒）%n", buildMillis, buildMillis / 1000.0);

        // 记下检索结果用于恢复后比对
        float[][] queries = new float[verifyQueries][];
        List<List<SearchResult>> expected = new ArrayList<>(verifyQueries);
        for (int q = 0; q < verifyQueries; q++) {
            queries[q] = dataset.queries()[q];
            expected.add(index.search(queries[q], topK, efSearch));
        }

        // ---------- 2. 落盘 ----------
        Path file = Files.createTempFile("vector-index-bench", ".snapshot");
        file.toFile().deleteOnExit();

        long saveStart = System.nanoTime();
        index.writeTo(file);
        long saveMillis = (System.nanoTime() - saveStart) / 1_000_000L;
        long fileBytes = Files.size(file);
        System.out.printf("落盘: %d ms，文件 %.1f MB%n", saveMillis, fileBytes / 1024.0 / 1024.0);

        // ---------- 3. 恢复 ----------
        long loadStart = System.nanoTime();
        HnswIndex restored = HnswIndex.readFrom(file);
        long loadMillis = (System.nanoTime() - loadStart) / 1_000_000L;
        System.out.printf("恢复: %d ms（%.1f 秒）%n", loadMillis, loadMillis / 1000.0);

        // ---------- 4. 验证一致性 ----------
        int mismatches = 0;
        for (int q = 0; q < verifyQueries; q++) {
            List<SearchResult> actual = restored.search(queries[q], topK, efSearch);
            List<SearchResult> want = expected.get(q);
            if (actual.size() != want.size()) {
                mismatches++;
                continue;
            }
            for (int i = 0; i < want.size(); i++) {
                if (want.get(i).id() != actual.get(i).id()
                        || Math.abs(want.get(i).score() - actual.get(i).score()) > 1e-6f) {
                    mismatches++;
                    break;
                }
            }
        }
        System.out.printf("一致性: %d/%d 条查询的结果与重建后完全一致%n",
                verifyQueries - mismatches, verifyQueries);
        System.out.printf("结构: 节点 %d -> %d，最大层 %d -> %d，总边数 %d -> %d%n",
                index.size(), restored.size(),
                index.maxLevel(), restored.maxLevel(),
                index.totalLinkCount(), restored.totalLinkCount());

        double speedup = (double) buildMillis / Math.max(1, loadMillis);
        System.out.println();
        System.out.printf("★ 冷启动耗时: 重建 %d ms -> 恢复 %d ms（%.0f 倍）%n",
                buildMillis, loadMillis, speedup);

        // ---------- 5. 输出 ----------
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("nodeCount", nodeCount);
        result.put("dimension", dimension);
        result.put("m", m);
        result.put("efConstruction", efConstruction);
        result.put("topK", topK);
        result.put("efSearch", efSearch);
        result.put("buildMillis", buildMillis);
        result.put("saveMillis", saveMillis);
        result.put("loadMillis", loadMillis);
        result.put("speedupVsRebuild", Math.round(speedup * 10) / 10.0);
        result.put("fileBytes", fileBytes);
        result.put("fileMiB", Math.round(fileBytes / 1024.0 / 1024.0 * 10) / 10.0);
        result.put("bytesPerVector", Math.round((double) fileBytes / nodeCount));
        result.put("verifiedQueries", verifyQueries);
        result.put("mismatches", mismatches);

        Path outputPath = Path.of(outPath);
        if (outputPath.getParent() != null) {
            Files.createDirectories(outputPath.getParent());
        }
        MAPPER.writeValue(outputPath.toFile(), result);
        System.out.printf("%n结果已写入: %s%n", outputPath.toAbsolutePath());

        if (mismatches > 0) {
            System.out.println("异常: 恢复后的检索结果与重建后不一致，快照有问题！");
            System.exit(1);
        }
    }
}
