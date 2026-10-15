package com.vectorsearch.config;

import com.vectorsearch.core.hnsw.HnswParams;
import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * 服务配置，绑定 application.yml 的 {@code vector-search.*}。
 *
 * <p>用 record 做配置类：天然不可变，且 Spring Boot 3 支持构造器绑定。
 * 需要 {@code @ConfigurationPropertiesScan} 才会被扫到（已加在主类上）。
 *
 * <p>HNSW 参数放在配置里而不是硬编码，是第 5 周消融实验的前提 ——
 * 改 yml 就能换一组参数跑评测，不用改代码重新打包。
 */
@ConfigurationProperties(prefix = "vector-search")
public record VectorSearchProperties(
        String modelServiceUrl,
        int modelTimeoutMs,
        Index index
) {

    /** 兜底默认值，避免 yml 缺项时启动直接 NPE。 */
    public VectorSearchProperties {
        if (modelServiceUrl == null || modelServiceUrl.isBlank()) {
            modelServiceUrl = "http://localhost:8000";
        }
        if (modelTimeoutMs <= 0) {
            modelTimeoutMs = 5000;
        }
        if (index == null) {
            index = new Index(768, 16, 200, 100, "./data/index");
        }
    }

    /**
     * @param dimension        向量维度，必须与 Python 侧模型输出一致（BGE-small-zh 为 768）
     * @param m                HNSW 每层最大出边数
     * @param efConstruction   HNSW 建图候选池大小
     * @param efSearch         查询候选池大小，recall / 延迟 的主旋钮
     * @param dataDir          索引落盘目录
     */
    public record Index(
            int dimension,
            int m,
            int efConstruction,
            int efSearch,
            String dataDir
    ) {
        public Index {
            if (dimension <= 0) {
                dimension = 768;
            }
            if (m < 2) {
                m = 16;
            }
            if (efConstruction < m) {
                efConstruction = Math.max(200, m);
            }
            if (efSearch <= 0) {
                efSearch = 100;
            }
            if (dataDir == null || dataDir.isBlank()) {
                dataDir = "./data/index";
            }
        }

        public HnswParams toHnswParams() {
            return HnswParams.of(m, efConstruction);
        }
    }
}
