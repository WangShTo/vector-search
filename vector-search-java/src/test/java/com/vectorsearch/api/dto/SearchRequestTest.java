package com.vectorsearch.api.dto;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 来源集合的归一化规则。
 *
 * <p>多选来源时最容易出错的是「并集怎么算」和「空白怎么处理」：
 * 前端会传 {@code ["", null]} 这类脏数据，而选择性直接决定 ef 放大多少 ——
 * 多算一个不存在的来源只会多查一遍，少算则会让 ef 给小、召回塌。
 */
class SearchRequestTest {

    private static SearchRequest of(String source, List<String> sources) {
        return new SearchRequest("q", null, null, null, null, source, sources);
    }

    @Test
    @DisplayName("只传单值 source：集合里就一个")
    void singleSource() {
        assertEquals(Set.of("a.md"), of("a.md", null).sourceSet());
    }

    @Test
    @DisplayName("只传 sources：全部保留，去重")
    void multipleSources() {
        assertEquals(Set.of("a.md", "b.md"),
                of(null, List.of("a.md", "b.md", "a.md")).sourceSet());
    }

    @Test
    @DisplayName("两个都传：取并集")
    void unionOfBothFields() {
        assertEquals(Set.of("a.md", "b.md"),
                of("b.md", List.of("a.md")).sourceSet());
    }

    @Test
    @DisplayName("空白与 null 被剔除，不会变成一个「来源叫空串」的过滤器")
    void blanksDropped() {
        SearchRequest request = of("  ", List.of("", "   ", "c.md"));
        assertEquals(Set.of("c.md"), request.sourceSet());
    }

    @Test
    @DisplayName("前后空白被裁掉，与文档里记的来源能对上")
    void trimmed() {
        assertEquals(Set.of("手册.md"), of("  手册.md  ", null).sourceSet());
    }

    @Test
    @DisplayName("都不传：空集，等价于不过滤")
    void emptyMeansNoFilter() {
        assertTrue(of(null, null).sourceSet().isEmpty());
        assertTrue(of(null, List.of()).sourceSet().isEmpty());
    }
}
