package com.vectorsearch.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.vectorsearch.api.dto.IndexDocumentRequest;
import com.vectorsearch.client.ModelServiceClient;
import com.vectorsearch.config.AbstentionProperties;
import com.vectorsearch.config.VectorSearchProperties;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * 批量灌入时，来源必须按<b>每条文档</b>各自落库。
 *
 * <p>为什么专门测这个：切好块的语料一行一条、每行的来源不同，而
 * {@code scripts/load_corpus.py} 走的是批量接口。如果批量接口只认请求级来源，
 * 灌完之后整库都是「未标注」，页面上的来源筛选直接失去意义 —— 而且不会报错，
 * 只是安静地少了一个功能。
 */
class IndexDocumentSourceTest {

    private static final int DIMENSION = 8;

    /** 不打网络的假模型客户端：按顺序给单位向量，够索引建图用。 */
    private static final class StubModelServiceClient extends ModelServiceClient {

        StubModelServiceClient(VectorSearchProperties properties) {
            super(new ObjectMapper(), properties);
        }

        @Override
        public float[][] embed(List<String> texts) {
            float[][] vectors = new float[texts.size()][DIMENSION];
            for (int i = 0; i < texts.size(); i++) {
                vectors[i][i % DIMENSION] = 1.0f;
            }
            return vectors;
        }
    }

    private static VectorSearchProperties properties() {
        return new VectorSearchProperties(
                null,
                0,
                new VectorSearchProperties.Index(
                        DIMENSION, 16, 200, 100, "./target/test-index", 300, 50, "f32"),
                null,
                null,
                null);
    }

    @Test
    @DisplayName("同一批里每条的 source 各自生效，没带的记「未标注」")
    void keepsPerDocumentSource() {
        VectorSearchProperties properties = properties();
        DocumentStore store = new DocumentStore(new ObjectMapper());
        VectorIndexManager indexManager = new VectorIndexManager(properties, store);
        SparseIndexManager sparseIndexManager = new SparseIndexManager(store);
        SearchService service = new SearchService(
                new StubModelServiceClient(properties),
                indexManager,
                sparseIndexManager,
                store,
                properties,
                new AbstentionProperties(false, null, null, null));

        service.index(List.of(
                new IndexDocumentRequest(0, "第一篇正文", "a.md"),
                new IndexDocumentRequest(1, "第二篇正文", "b.md"),
                new IndexDocumentRequest(2, "第三篇正文")));

        assertEquals("a.md", store.sourceOf(0));
        assertEquals("b.md", store.sourceOf(1));
        assertEquals(DocumentStore.DocMeta.UNKNOWN_SOURCE, store.sourceOf(2));
        assertEquals(1, store.sourceCounts().get("a.md"));
        assertEquals(1, store.sourceCounts().get("b.md"));
    }
}
