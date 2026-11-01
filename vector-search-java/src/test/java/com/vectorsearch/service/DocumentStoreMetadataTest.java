package com.vectorsearch.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 文档元数据（来源）的存取与计数。
 *
 * <p>来源计数不是「顺便存一下」的展示字段：带过滤的检索靠它算选择性，
 * 选择性决定 ef 要放大多少倍。所以这里的每一条都要准。
 */
class DocumentStoreMetadataTest {

    private static DocumentStore newStore() {
        return new DocumentStore(new ObjectMapper());
    }

    @Test
    @DisplayName("来源被记录，并按来源分组计数")
    void recordsSourceAndCounts() {
        DocumentStore store = newStore();
        store.put(0, "第一篇", DocumentStore.DocMeta.of("a.md", 1L));
        store.put(1, "第二篇", DocumentStore.DocMeta.of("a.md", 1L));
        store.put(2, "第三篇", DocumentStore.DocMeta.of("b.md", 1L));

        assertEquals("a.md", store.sourceOf(0));
        assertEquals(Map.of("a.md", 2, "b.md", 1), store.sourceCounts());
    }

    @Test
    @DisplayName("来源为空时记成占位值，而不是 null 或空串")
    void blankSourceFallsBackToPlaceholder() {
        DocumentStore store = newStore();
        store.put(0, "没有来源的文档");

        assertEquals(DocumentStore.DocMeta.UNKNOWN_SOURCE, store.sourceOf(0));
        assertTrue(store.sourceCounts().containsKey(DocumentStore.DocMeta.UNKNOWN_SOURCE));
    }

    @Test
    @DisplayName("逻辑删除会让来源计数减一")
    void removeDecrementsSourceCount() {
        DocumentStore store = newStore();
        store.put(0, "甲", DocumentStore.DocMeta.of("a.md", 1L));
        store.put(1, "乙", DocumentStore.DocMeta.of("a.md", 1L));

        store.remove(0);

        assertEquals(Map.of("a.md", 1), store.sourceCounts());
    }

    @Test
    @DisplayName("同一个 id 改了来源，旧来源要扣掉")
    void reimportWithDifferentSourceMovesCount() {
        DocumentStore store = newStore();
        store.put(0, "旧内容", DocumentStore.DocMeta.of("a.md", 1L));
        store.put(0, "新内容", DocumentStore.DocMeta.of("b.md", 2L));

        assertEquals(Map.of("b.md", 1), store.sourceCounts());
        assertEquals("b.md", store.sourceOf(0));
    }

    @Test
    @DisplayName("压实会清掉墓碑的元数据，但不动存活文档的计数")
    void purgeDropsMetadataOnlyForDeleted() {
        DocumentStore store = newStore();
        store.put(0, "甲", DocumentStore.DocMeta.of("a.md", 1L));
        store.put(1, "乙", DocumentStore.DocMeta.of("a.md", 1L));
        store.remove(0);

        assertEquals(1, store.purgeDeleted());

        assertEquals(Map.of("a.md", 1), store.sourceCounts());
        assertEquals("a.md", store.sourceOf(1));
        assertFalse(store.get(0).isPresent(), "墓碑压实后原文应该没了");
    }

    @Test
    @DisplayName("落盘再读回，来源不丢")
    void saveAndLoadPreserveSource(@TempDir Path dir) throws IOException {
        Path file = dir.resolve("documents.jsonl");
        DocumentStore store = newStore();
        store.put(0, "甲", DocumentStore.DocMeta.of("a.md", 111L));
        store.put(1, "乙", DocumentStore.DocMeta.of("b.md", 222L));
        store.save(file);

        DocumentStore reloaded = newStore();
        assertEquals(2, reloaded.load(file));

        assertEquals("a.md", reloaded.sourceOf(0));
        assertEquals("b.md", reloaded.sourceOf(1));
        assertEquals(111L, reloaded.metadataOf(0).importedAt());
        assertEquals(Map.of("a.md", 1, "b.md", 1), reloaded.sourceCounts());
    }

    @Test
    @DisplayName("老格式 JSONL（没有 source 字段）也能读，来源落到占位值")
    void loadsLegacyLinesWithoutSource(@TempDir Path dir) throws IOException {
        Path file = dir.resolve("legacy.jsonl");
        java.nio.file.Files.writeString(file,
                "{\"id\":5,\"text\":\"老文件里的一行\",\"deleted\":false}\n");

        DocumentStore store = newStore();
        assertEquals(1, store.load(file));

        assertEquals(DocumentStore.DocMeta.UNKNOWN_SOURCE, store.sourceOf(5));
        assertEquals(Map.of(DocumentStore.DocMeta.UNKNOWN_SOURCE, 1), store.sourceCounts());
    }
}
