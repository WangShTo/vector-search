package com.vectorsearch.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.vectorsearch.config.VectorSearchProperties;
import com.vectorsearch.core.SearchResult;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.TreeSet;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 墓碑压实的测试。
 *
 * <p>这组测试的重点不是「方法不抛异常」，而是要证明三件<b>可观测</b>的事：
 * <ol>
 *   <li>被删除的向量真的离开了索引；</li>
 *   <li>BM25 的语料统计量（平均文档长度）在压实后<b>只由存活文档决定</b> ——
 *       这是压实最主要的目标，因为被墓碑污染时它不会报错，只会让分数悄悄偏；</li>
 *   <li>存活文档的向量原样保留，检索还能命中。</li>
 * </ol>
 */
class IndexMaintenanceServiceTest {

    private static final int DIM = 512;

    /** 把四个组件装在一起，省掉每个测试重复的样板。 */
    private record Fixture(
            DocumentStore documentStore,
            VectorIndexManager vectorIndexManager,
            SparseIndexManager sparseIndexManager,
            IndexMaintenanceService maintenance
    ) {
    }

    private Fixture fixture() {
        return fixture(new VectorSearchProperties.Maintenance(true, 0.1));
    }

    private Fixture fixture(VectorSearchProperties.Maintenance maintenance) {
        // 配置用 record 的兜底默认值（维度 512、HNSW m=16 efConstruction=200）
        VectorSearchProperties properties =
                new VectorSearchProperties(null, 0, null, null, null, maintenance);
        DocumentStore store = new DocumentStore(new ObjectMapper());
        VectorIndexManager vector = new VectorIndexManager(properties, store);
        SparseIndexManager sparse = new SparseIndexManager(store);
        // 传同步 executor：自动压实当场跑完，测试才能断言压实之后的状态
        return new Fixture(store, vector, sparse,
                new IndexMaintenanceService(vector, sparse, store, properties, Runnable::run));
    }

    /** 可复现的随机向量，不归一化 —— add 会自己归一化。 */
    private static float[] vector(int seed) {
        Random random = new Random(seed);
        float[] v = new float[DIM];
        for (int i = 0; i < DIM; i++) {
            v[i] = (float) random.nextGaussian();
        }
        return v;
    }

    @Test
    void compact_dropsTombstonedVectors() {
        Fixture f = fixture();
        for (int id = 1; id <= 10; id++) {
            f.documentStore().put(id, "文档 " + id + " 的正文内容");
            f.vectorIndexManager().add(id, vector(id));
        }
        for (int id = 1; id <= 4; id++) {
            f.documentStore().remove(id);
        }

        var result = f.maintenance().compact();

        assertEquals(10, result.vectorsBefore(), "压实前图里应该有 10 条（含 4 个墓碑）");
        assertEquals(6, result.vectorsAfter(), "压实后只剩 6 条活节点");
        assertEquals(4, result.tombstonesPurged(), "从原文存储里清掉 4 条");
        assertEquals(6, f.vectorIndexManager().size(), "索引里的条数必须同步降下来");
        assertEquals(6, f.documentStore().count(), "存活文档数不变");
        assertEquals(0, f.documentStore().deletedCount(), "墓碑集合应该被清空");
    }

    @Test
    void compact_repairsPollutedBm25Statistics() {
        Fixture f = fixture();
        String longText = "向量检索把文本编码成稠密向量，再用近似最近邻算法找出最相似的若干条结果。"
                + "倒排索引相反，它记录每个词出现在哪些文档里，用词频和逆文档频率打分。";
        String shortText = "分层图是一种近似最近邻索引结构。";

        f.documentStore().put(1, longText);
        f.vectorIndexManager().add(1, vector(1));
        // 走增量路径 add() 而不是 rebuildFromDocuments()：线上的导入流程就是
        // 一篇一篇 add 进来的，这才是墓碑能积累起来的真实场景。
        // （rebuildFromDocuments() 按 DocumentStore 的存活文档重建，天然不含墓碑）
        f.sparseIndexManager().add(1, longText);
        f.documentStore().put(2, shortText);
        f.vectorIndexManager().add(2, vector(2));
        f.sparseIndexManager().add(2, shortText);

        // 此时的平均文档长度把两篇都算进去了
        double polluted = f.sparseIndexManager().averageDocumentLength();

        // 删除只动 DocumentStore，<b>不会通知 BM25</b> —— 统计量仍是两篇的平均。
        // 这就是不压实时的真实偏差：不报错，只是长度归一化基准偏了，
        // 要等到下次重启触发 rebuildFromDocuments 才会变干净。
        f.documentStore().remove(1);
        assertEquals(polluted, f.sparseIndexManager().averageDocumentLength(), 1e-9,
                "前提：只删文档不压实，BM25 的统计量不会自己变干净");

        f.maintenance().compact();

        // 干净参考：只有短文档时的统计量
        DocumentStore cleanStore = new DocumentStore(new ObjectMapper());
        cleanStore.put(2, shortText);
        SparseIndexManager clean = new SparseIndexManager(cleanStore);
        clean.rebuildFromDocuments();

        assertEquals(clean.averageDocumentLength(), f.sparseIndexManager().averageDocumentLength(), 1e-9,
                "压实后 BM25 的长度归一化基准应由存活文档唯一决定");
        assertNotEquals(polluted, f.sparseIndexManager().averageDocumentLength(),
                "被污染的值和修好的值必须真的不同，否则这个测试什么都证明不了");
    }

    @Test
    void compact_keepsLiveVectorsSearchable() {
        Fixture f = fixture();
        float[] query = vector(1);
        f.documentStore().put(1, "向量检索的原理");
        f.vectorIndexManager().add(1, query);
        f.documentStore().put(2, "倒排索引与 BM25");
        f.vectorIndexManager().add(2, vector(2));
        f.documentStore().put(3, "HNSW 分层图");
        f.vectorIndexManager().add(3, vector(3));

        f.documentStore().remove(2);
        f.maintenance().compact();

        List<SearchResult> hits = f.vectorIndexManager().search(query, 5, 50);

        assertEquals(2, hits.size(), "只剩两条活节点");
        assertEquals(1, hits.get(0).id(), "查询向量就是文档 1 自己的向量，必须排第一");
        assertTrue(hits.get(0).score() > 0.999,
                "压实只是搬家，向量本身不该被改动，分数应该还接近 1");
        assertTrue(hits.stream().noneMatch(h -> h.id() == 2),
                "被删除的文档不该重新出现在结果里");
    }

    @Test
    void compact_isNoOpWhenThereAreNoTombstones() {
        Fixture f = fixture();
        f.documentStore().put(1, "只有一篇文档");
        f.vectorIndexManager().add(1, vector(1));

        var result = f.maintenance().compact();

        assertEquals(0, result.tombstonesPurged());
        assertEquals(0L, result.elapsedMillis(), "没有墓碑就该直接返回，不做无谓的重建");
        assertEquals(1, result.vectorsBefore());
        assertEquals(1, result.vectorsAfter());
        assertEquals(1, f.vectorIndexManager().size());
    }

    @Test
    void forEach_exposesEveryIndexedNode() {
        Fixture f = fixture();
        f.vectorIndexManager().add(7, vector(7));
        f.vectorIndexManager().add(9, vector(9));

        TreeSet<Integer> ids = new TreeSet<>();
        f.vectorIndexManager().index().forEach((id, v) -> ids.add(id));

        assertEquals(List.of(7, 9), new ArrayList<>(ids));
    }

    @Test
    void forEach_worksForBruteForceToo() {
        Fixture f = fixture();
        f.vectorIndexManager().reset(VectorIndexManager.TYPE_BRUTE_FORCE);
        f.vectorIndexManager().add(3, vector(3));
        f.vectorIndexManager().add(5, vector(5));

        List<Integer> ids = new ArrayList<>();
        f.vectorIndexManager().index().forEach((id, v) -> ids.add(id));
        ids.sort(null);

        assertEquals(List.of(3, 5), ids);
    }

    // ---------- 自动压实 ----------

    @Test
    void autoCompact_triggersWhenTombstoneRatioReachesThreshold() {
        Fixture f = fixture(new VectorSearchProperties.Maintenance(true, 0.1));
        for (int id = 1; id <= 10; id++) {
            f.documentStore().put(id, "第 " + id + " 篇文档的正文");
            f.vectorIndexManager().add(id, vector(id));
        }
        for (int id = 1; id <= 2; id++) {
            f.documentStore().remove(id);
        }
        assertEquals(0.2, f.maintenance().tombstoneRatio(), 1e-9,
                "前提：10 篇删 2 篇，墓碑占比 20%");

        assertTrue(f.maintenance().maybeAutoCompact(), "20% ≥ 10%，应该触发");

        assertEquals(0, f.documentStore().deletedCount(), "同步 executor 下当场就该压完");
        assertEquals(8, f.vectorIndexManager().size());
        assertEquals(1, f.maintenance().autoCompactions());
    }

    @Test
    void autoCompact_skipsBelowThreshold() {
        Fixture f = fixture(new VectorSearchProperties.Maintenance(true, 0.1));
        for (int id = 1; id <= 20; id++) {
            f.documentStore().put(id, "第 " + id + " 篇文档的正文");
            f.vectorIndexManager().add(id, vector(id));
        }
        f.documentStore().remove(1);

        assertFalse(f.maintenance().maybeAutoCompact(), "5% < 10%，不该触发");
        assertEquals(1, f.documentStore().deletedCount(), "墓碑应该还留着");
        assertEquals(20, f.vectorIndexManager().size(), "没压实，图里不该少东西");
        assertEquals(0, f.maintenance().autoCompactions());
    }

    @Test
    void autoCompact_canBeDisabled() {
        Fixture f = fixture(new VectorSearchProperties.Maintenance(false, 0.1));
        for (int id = 1; id <= 10; id++) {
            f.documentStore().put(id, "第 " + id + " 篇文档的正文");
            f.vectorIndexManager().add(id, vector(id));
        }
        for (int id = 1; id <= 9; id++) {
            f.documentStore().remove(id);
        }
        assertEquals(0.9, f.maintenance().tombstoneRatio(), 1e-9);

        assertFalse(f.maintenance().maybeAutoCompact(), "关掉之后占比再高也不触发");
        assertEquals(9, f.documentStore().deletedCount(), "墓碑原样留着，等人工压实");
    }

    @Test
    void autoCompact_isIdempotentWhenNothingToCompact() {
        Fixture f = fixture(new VectorSearchProperties.Maintenance(true, 0.1));
        f.documentStore().put(1, "唯一的一篇文档");
        f.vectorIndexManager().add(1, vector(1));

        // 没有墓碑时占比是 0，不该为了「空操作」把整张图重建一遍
        assertFalse(f.maintenance().maybeAutoCompact());
        assertEquals(0, f.maintenance().autoCompactions());
    }

    @Test
    void autoCompactRatio_rejectsNonsenseValues() {
        // 0 会让每次删除都触发重建，1 会让它永远不触发 —— 两个都很难从现象倒推
        org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
                () -> new VectorSearchProperties.Maintenance(true, 0));
        org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
                () -> new VectorSearchProperties.Maintenance(true, 1));
        org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
                () -> new VectorSearchProperties.Maintenance(true, 1.5));
    }

    @Test
    void purgeDeleted_onlyRemovesTheSnapshotTakenBeforeRebuild() {
        Fixture f = fixture();
        f.documentStore().put(1, "第一篇文档");
        f.documentStore().put(2, "第二篇文档");
        f.documentStore().remove(1);

        // 压实开始：记下「这次重建要排除谁」
        java.util.Set<Integer> snapshot = f.documentStore().deletedIdSnapshot();

        // 重建期间又删了一篇 —— 它<b>已经</b>被重建进新图了，
        // 原文一旦被清掉就会留下一个查不到内容的死节点
        f.documentStore().remove(2);

        assertEquals(1, f.documentStore().purgeDeleted(snapshot));

        assertEquals(1, f.documentStore().totalCount(), "只该清掉快照里的那一篇");
        assertEquals(1, f.documentStore().deletedCount(), "后来删的必须继续当墓碑");
    }
}
