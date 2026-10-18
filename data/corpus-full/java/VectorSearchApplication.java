package com.vectorsearch;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;

/**
 * 向量检索服务入口。
 *
 * <p>Java 侧负责「存储 + 检索」：HNSW 索引、BM25 倒排、混合召回与融合排序。
 * Python 侧只负责模型推理（embedding / rerank），通过 HTTP 批量调用。
 *
 * <p>{@code @ConfigurationPropertiesScan} 让 {@code VectorSearchProperties}
 * 这个 record 能被绑定上 —— record 没法用 {@code @Component} 构造器注入，
 * 靠扫描才是标准做法。
 */
@SpringBootApplication
@ConfigurationPropertiesScan
public class VectorSearchApplication {

    public static void main(String[] args) {
        SpringApplication.run(VectorSearchApplication.class, args);
    }
}
