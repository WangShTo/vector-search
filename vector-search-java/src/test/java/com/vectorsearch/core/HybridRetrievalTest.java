package com.vectorsearch.core;

import com.vectorsearch.core.bm25.Bm25Index;
import com.vectorsearch.core.bm25.Bm25Tokenizer;
import com.vectorsearch.core.fusion.RrfFusion;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 混合检索三件套的单元测试：分词 / BM25 打分 / RRF 融合。
 *
 * <p>这三块里最容易错的是<b>分词</b> —— 中文没有分隔符，
 * 切错了后面全都白搭，而且不会报错，只会让检索质量悄悄变差。
 * 所以分词单独用精确断言，BM25 与 RRF 断言相对顺序。
 */
class HybridRetrievalTest {

    // =====================================================================
    // 分词
    // =====================================================================

    @Test
    @DisplayName("中文按二元组切分，不需要词典")
    void chineseBigrams() {
        assertEquals(List.of("向量", "量检", "检索"),
                Bm25Tokenizer.tokenize("向量检索"));
    }

    @Test
    @DisplayName("英文标识符整体成词并转小写")
    void asciiIdentifiers() {
        assertEquals(List.of("hnsw"), Bm25Tokenizer.tokenize("HNSW"));
        assertEquals(List.of("efsearch"), Bm25Tokenizer.tokenize("efSearch"));
        // 太短的单词（1 个字符）当噪声丢掉
        assertEquals(List.of(), Bm25Tokenizer.tokenize("a"));
    }

    @Test
    @DisplayName("中英混排各走各的路径，不互相污染")
    void mixedScripts() {
        // "efSearch 参数" -> 英文词 + 中文二元组
        assertEquals(List.of("efsearch", "参数"),
                Bm25Tokenizer.tokenize("efSearch 参数"));

        // "用HNSW建图" -> 孤立的"用" + 英文词 + 二元组"建图"（末尾不再多吐一个单字）
        assertEquals(List.of("用", "hnsw", "建图"),
                Bm25Tokenizer.tokenize("用HNSW建图"));
    }

    @Test
    @DisplayName("标点与空白切断累积，不产生跨标点的伪词")
    void punctuationBreaksTokens() {
        // 逗号不该把两边的英文粘成一个词
        assertEquals(List.of("alpha", "beta"), Bm25Tokenizer.tokenize("alpha,beta"));
        // 破折号同理
        assertEquals(List.of("bge", "small", "zh"), Bm25Tokenizer.tokenize("bge-small-zh"));
    }

    // =====================================================================
    // BM25 打分
    // =====================================================================

    @Test
    @DisplayName("命中精确词的文档排第一")
    void exactTermWins() {
        Bm25Index index = new Bm25Index();
        index.add(1, "今天食堂的红烧肉特别好吃");
        index.add(2, "HNSW 是一种近似最近邻检索算法，通过分层图加速查找");
        index.add(3, "倒排索引记录每个词出现在哪些文档里");

        List<SearchResult> hits = index.search("近似最近邻", 2);
        assertEquals(2, hits.get(0).id(), "应当命中讲 HNSW 的那篇");
    }

    @Test
    @DisplayName("罕见词的权重高于到处都有的词（IDF 生效）")
    void rareTermsOutweighCommonOnes() {
        // 三篇文档等长（都是 2 个 token），只让词的分布不同，
        // 这样分数差异只能归因于 IDF，排除了长度归一化的干扰
        Bm25Index index = new Bm25Index();
        index.add(1, "检索 苹果");
        index.add(2, "检索 香蕉");
        index.add(3, "检索 efsearch");

        // "检索" 三篇都有 -> IDF 极低；"efsearch" 只有第 3 篇 -> IDF 很高
        List<SearchResult> hits = index.search("检索 efsearch", 3);
        assertEquals(3, hits.get(0).id(),
                "同时命中高频词与罕见词的文档应当排第一");
    }

    @Test
    @DisplayName("相同词频下，短文档得分高于长文档（长度归一化生效）")
    void shorterDocumentWins() {
        Bm25Index index = new Bm25Index();
        index.add(1, "向量检索");
        index.add(2, "向量检索"
                + "这是一段为了凑长度而写的无关内容，反复出现各种没有信息量的词，"
                + "把文档撑得很长很长很长很长很长很长很长很长很长很长很长很长");

        List<SearchResult> hits = index.search("向量检索", 2);
        assertEquals(1, hits.get(0).id(), "短文档应当得分更高");
    }

    @Test
    @DisplayName("空索引与空查询都返回空结果，而不是抛异常")
    void edgeCases() {
        Bm25Index index = new Bm25Index();
        assertTrue(index.search("任何查询", 10).isEmpty(), "空索引应返回空");

        index.add(1, "一些文本");
        assertTrue(index.search("   ", 10).isEmpty(), "空查询应返回空");
        assertTrue(index.search("完全不存在的词汇组合", 10).isEmpty(),
                "没有任何词命中时应返回空");
        assertTrue(index.search("文本", 0).isEmpty(), "k<=0 应返回空");
    }

    @Test
    @DisplayName("词表与平均文档长度可观测")
    void observability() {
        Bm25Index index = new Bm25Index();
        index.add(1, "向量检索");
        index.add(2, "向量检索");

        assertEquals(2, index.size());
        assertTrue(index.termCount() >= 3, "词表应包含至少 3 个二元组");
        assertTrue(index.averageDocumentLength() > 0, "平均文档长度应为正");
    }

    // =====================================================================
    // RRF 融合
    // =====================================================================

    @Test
    @DisplayName("两路都靠前的文档应当胜出（融合的核心价值）")
    void consistentRankingWins() {
        // A: 第 1 路第 1，第 2 路第 3
        // B: 第 1 路第 2，第 2 路第 1   <- 两路都靠前
        // C: 第 1 路第 3，第 2 路第 2
        List<SearchResult> first = List.of(hit(1, 0.9f), hit(2, 0.8f), hit(3, 0.7f));
        List<SearchResult> second = List.of(hit(2, 30f), hit(3, 20f), hit(1, 10f));

        List<SearchResult> fused = RrfFusion.fuse(List.of(first, second), 3, RrfFusion.DEFAULT_K);

        assertEquals(3, fused.size());
        assertEquals(2, fused.get(0).id(), "两路都靠前的 B 应当第一");
        assertEquals(1, fused.get(1).id(), "A 与 C 次之，A 因为有一个第 1 名略胜");
        assertEquals(3, fused.get(2).id());
    }

    @Test
    @DisplayName("分数尺度差异不影响融合结果（只依赖排名）")
    void scaleInvariant() {
        List<SearchResult> small = List.of(hit(1, 0.9f), hit(2, 0.8f));
        // 同样的排名，但分数放大一万倍
        List<SearchResult> huge = List.of(hit(1, 9000f), hit(2, 8000f));

        List<SearchResult> a = RrfFusion.fuse(List.of(small), 2, RrfFusion.DEFAULT_K);
        List<SearchResult> b = RrfFusion.fuse(List.of(huge), 2, RrfFusion.DEFAULT_K);

        assertEquals(a.get(0).id(), b.get(0).id(), "分数放大不该改变融合后的顺序");
        assertEquals(a.get(0).score(), b.get(0).score(), 1e-9f, "融合分数只取决于排名");
    }

    @Test
    @DisplayName("只在一路出现的文档也能进结果")
    void singleSourceStillCounts() {
        List<SearchResult> first = List.of(hit(1, 0.9f));
        List<SearchResult> second = List.of(hit(9, 50f));

        List<SearchResult> fused = RrfFusion.fuse(List.of(first, second), 5, RrfFusion.DEFAULT_K);
        assertEquals(2, fused.size(), "两路各自的独有文档都应保留");
    }

    @Test
    @DisplayName("融合结果稳定可复现（同分时按 id 排序）")
    void deterministicTieBreak() {
        // 两篇文档在两路里的排名完全对称，融合分数会精确相等
        List<SearchResult> first = List.of(hit(2, 1f), hit(1, 1f));
        List<SearchResult> second = List.of(hit(1, 1f), hit(2, 1f));

        List<SearchResult> a = RrfFusion.fuse(List.of(first, second), 2, RrfFusion.DEFAULT_K);
        List<SearchResult> b = RrfFusion.fuse(List.of(first, second), 2, RrfFusion.DEFAULT_K);

        assertEquals(a.get(0).id(), b.get(0).id(), "两次融合的顺序必须一致");
        assertEquals(1, a.get(0).id(), "同分时按 id 升序");
    }

    @Test
    @DisplayName("空输入返回空结果")
    void fusionEdgeCases() {
        assertTrue(RrfFusion.fuse(List.of(), 5, 60).isEmpty());
        assertTrue(RrfFusion.fuse(List.of(List.of()), 5, 60).isEmpty());
        assertTrue(RrfFusion.fuse(List.of(List.of(hit(1, 1f))), 0, 60).isEmpty());
    }

    private static SearchResult hit(int id, float score) {
        return new SearchResult(id, score);
    }
}
