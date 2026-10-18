package com.vectorsearch.core.hnsw;

import com.vectorsearch.core.SearchResult;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 索引快照的正确性验证。
 *
 * <p>持久化最容易出的错不是"读写文件失败"，而是<b>读回来的索引看起来正常、
 * 却能检索出不同结果</b> —— 比如少读了一段、偏移算错、字段顺序对不上。
 * 这种问题不会抛异常，只会让检索质量悄悄下降。
 *
 * <p>所以这里最核心的一条测试是：<b>存盘前与读回后，同一批查询的结果必须逐条一致</b>
 * （包括 id 与分数）。只比对 size 或 recall 是不够的，那样会漏掉细微的错位。
 */
class HnswSnapshotTest {

    private static final int DIMENSION = 64;
    private static final int NODE_COUNT = 3_000;
    private static final int QUERY_COUNT = 40;
    private static final int TOP_K = 10;
    private static final int EF_SEARCH = 100;
    private static final long SEED = 2024L;

    @Test
    @DisplayName("存盘再读回，检索结果必须逐条一致（含 id 与分数）")
    void roundTripPreservesSearchResults(@TempDir Path tempDir) throws IOException {
        HnswIndex original = buildIndex();
        float[][] queries = randomVectors(QUERY_COUNT, DIMENSION, SEED + 7);

        // 存盘前的检索结果作为基准
        List<List<SearchResult>> before = new java.util.ArrayList<>();
        for (float[] query : queries) {
            before.add(original.search(query, TOP_K, EF_SEARCH));
        }

        Path snapshot = tempDir.resolve("index.snapshot");
        original.writeTo(snapshot);
        assertTrue(Files.exists(snapshot), "快照文件应该被创建");

        HnswIndex restored = HnswIndex.readFrom(snapshot);

        assertEquals(original.size(), restored.size(), "节点数应一致");
        assertEquals(original.dimension(), restored.dimension(), "维度应一致");
        assertEquals(original.maxLevel(), restored.maxLevel(), "最大层应一致");

        for (int q = 0; q < queries.length; q++) {
            List<SearchResult> expected = before.get(q);
            List<SearchResult> actual = restored.search(queries[q], TOP_K, EF_SEARCH);

            assertEquals(expected.size(), actual.size(), "第 " + q + " 条查询命中数应一致");
            for (int i = 0; i < expected.size(); i++) {
                assertEquals(expected.get(i).id(), actual.get(i).id(),
                        "第 " + q + " 条查询第 " + i + " 位的 id 应一致");
                // 分数也必须一致：说明向量是按原样读回来的，没有错位或精度损失
                assertEquals(expected.get(i).score(), actual.get(i).score(), 1e-6f,
                        "第 " + q + " 条查询第 " + i + " 位的分数应一致");
            }
        }
    }

    @Test
    @DisplayName("图结构与各层人口在存盘前后必须一致")
    void roundTripPreservesStructure(@TempDir Path tempDir) throws IOException {
        HnswIndex original = buildIndex();
        Path snapshot = tempDir.resolve("index.snapshot");
        original.writeTo(snapshot);
        HnswIndex restored = HnswIndex.readFrom(snapshot);

        assertArrayEquals(original.levelHistogram(), restored.levelHistogram(),
                "层高分布应一致");
        assertArrayEquals(original.layerPopulation(), restored.layerPopulation(),
                "各层人口应一致");
        assertEquals(original.totalLinkCount(), restored.totalLinkCount(),
                "总边数应一致");
        assertEquals(original.params().m(), restored.params().m(), "m 应一致");
        assertEquals(original.params().efConstruction(), restored.params().efConstruction(),
                "efConstruction 应一致");
    }

    @Test
    @DisplayName("空索引也能正常存盘与读回")
    void emptyIndexRoundTrip(@TempDir Path tempDir) throws IOException {
        HnswIndex empty = new HnswIndex(DIMENSION, HnswParams.of(16, 200), SEED);
        Path snapshot = tempDir.resolve("empty.snapshot");
        empty.writeTo(snapshot);

        HnswIndex restored = HnswIndex.readFrom(snapshot);
        assertEquals(0, restored.size(), "空索引读回后仍应为空");
        assertTrue(restored.search(randomVectors(1, DIMENSION, 1L)[0], TOP_K, EF_SEARCH).isEmpty(),
                "空索引检索应返回空结果，而不是抛异常");
    }

    @Test
    @DisplayName("文件被篡改时必须报错，绝不能静默返回错误结果")
    void corruptedFileIsRejected(@TempDir Path tempDir) throws IOException {
        HnswIndex original = buildIndex();
        Path snapshot = tempDir.resolve("index.snapshot");
        original.writeTo(snapshot);

        // 翻转数据段里的一个字节，模拟磁盘损坏
        byte[] bytes = Files.readAllBytes(snapshot);
        int target = bytes.length - 100;
        bytes[target] = (byte) (bytes[target] ^ 0xFF);
        Files.write(snapshot, bytes);

        IOException error = assertThrows(IOException.class, () -> HnswIndex.readFrom(snapshot));
        assertTrue(error.getMessage().contains("CRC"),
                "应当报 CRC 校验失败，实际是: " + error.getMessage());
    }

    @Test
    @DisplayName("不是索引快照的文件必须被识别出来")
    void wrongMagicIsRejected(@TempDir Path tempDir) throws IOException {
        Path fake = tempDir.resolve("not-an-index.bin");
        Files.write(fake, new byte[1024]);

        IOException error = assertThrows(IOException.class, () -> HnswIndex.readFrom(fake));
        assertTrue(error.getMessage().contains("魔数"),
                "应当报魔数不匹配，实际是: " + error.getMessage());
    }

    @Test
    @DisplayName("快照文件大小应与数据量吻合（说明没有把容量里的空白也写出去）")
    void snapshotSizeMatchesData(@TempDir Path tempDir) throws IOException {
        HnswIndex original = buildIndex();
        Path snapshot = tempDir.resolve("index.snapshot");
        original.writeTo(snapshot);

        long actual = Files.size(snapshot);

        // 头部 64 + 段表 7×28（kind + offset + length + crc），再加上各段的理论字节数
        // （不含上层邻接表，那部分取决于层高分布）
        long headerBytes = 64 + 7 * 28;
        long perNode = (long) DIMENSION * Float.BYTES
                + Integer.BYTES      // externalIdStore
                + 1                  // levelStore
                + (long) original.params().mMax0() * Integer.BYTES  // layer0Links
                + Short.BYTES        // layer0Count
                + Integer.BYTES;     // upperOffset
        long lowerBound = headerBytes + NODE_COUNT * perNode;

        assertTrue(actual >= lowerBound,
                "文件大小 " + actual + " 应不小于下界 " + lowerBound);
        // 上界放宽到位图数据的 1.5 倍：上层邻接表很小，但留足余量避免偶发失败
        assertTrue(actual < lowerBound * 3 / 2,
                "文件大小 " + actual + " 明显超出预期下界 " + lowerBound
                        + "，可能把容量里的空白也写进去了");

        System.out.printf("快照文件: %.2f MB（%d 节点，%d 维）%n",
                actual / 1024.0 / 1024.0, NODE_COUNT, DIMENSION);
    }

    // ------------------------------------------------------------------

    private HnswIndex buildIndex() {
        HnswIndex index = new HnswIndex(DIMENSION, HnswParams.of(16, 200), SEED);
        float[][] vectors = randomVectors(NODE_COUNT, DIMENSION, SEED);
        for (int i = 0; i < vectors.length; i++) {
            index.add(i, vectors[i]);
        }
        return index;
    }

    private float[][] randomVectors(int count, int dimension, long seed) {
        Random random = new Random(seed);
        float[][] vectors = new float[count][dimension];
        for (int i = 0; i < count; i++) {
            for (int d = 0; d < dimension; d++) {
                vectors[i][d] = random.nextFloat() * 2f - 1f;
            }
        }
        return vectors;
    }
}
