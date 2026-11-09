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
        Index index,
        Search search,
        Security security,
        Maintenance maintenance
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
            index = new Index(512, 16, 200, 100, "./data/index", 300, 50, "f32");
        }
        if (search == null) {
            search = new Search("hybrid", 50, 60, false, 20, 1.0, 1.0);
        }
        if (security == null) {
            security = new Security("");
        }
        if (maintenance == null) {
            maintenance = new Maintenance(true, 0.1);
        }
    }

    /**
     * 索引维护配置。
     *
     * @param autoCompact        删除后是否在后台自动压实。关掉就退回「手动调
     *                           {@code POST /api/admin/compact}」的老行为。
     * @param autoCompactRatio   墓碑占索引条目的比例达到多少就触发。
     *                           0.1 的意思是「删掉了十分之一就该整理了」——
     *                           墓碑不清理有两个后果：候选名额被死节点吃掉，
     *                           以及检索要为了补偿而放大 ef。但它也不能太频繁：
     *                           压实要把整张图重建一遍，10 万点要一百多秒。
     */
    public record Maintenance(boolean autoCompact, double autoCompactRatio) {
        public Maintenance {
            if (autoCompactRatio <= 0 || autoCompactRatio >= 1) {
                // 写错就报错，不要静默接受 —— 0 会让每次删除都触发重建，
                // 1 会让它永远不触发，两个都很难从「服务变慢了」倒推出来
                throw new IllegalArgumentException(
                        "vector-search.maintenance.auto-compact-ratio 必须在 (0,1) 之间，实际是 "
                                + autoCompactRatio);
            }
        }
    }

    /**
     * 鉴权配置。
     *
     * @param apiKey 管理密钥。**留空表示不启用鉴权**，这是本地开发的默认值 ——
     *               本地跑着玩还要配个密钥是纯粹的摩擦。
     *               一旦部署到公网就必须设置，否则任何人都能删光知识库。
     *               用 {@link Security#enabled()} 判断是否启用，不要直接判空字符串。
     */
    public record Security(String apiKey) {
        public Security {
            if (apiKey == null) {
                apiKey = "";
            }
            apiKey = apiKey.trim();
        }

        public boolean enabled() {
            return !apiKey.isEmpty();
        }
    }

    /**
     * 检索模式、融合与精排参数。
     *
     * @param defaultMode      请求不指定 mode 时用哪个：{@code vector} / {@code bm25} / {@code hybrid}
     * @param candidatePool    融合前每一路召回多少条。太小会让融合没得挑，
     *                         太大会把噪声也带进来。50 是常见起点。
     * @param rrfK             RRF 平滑常数。见 {@link com.vectorsearch.core.fusion.RrfFusion}，
     *                         60 是原论文经验值，对结果不敏感。
     * @param rerankDefault    请求不指定 rerank 时是否默认做精排。
     *                         <b>默认关</b> —— 精排要在 CPU 上对每个候选跑一次
     *                         cross-encoder 前向，是数量级的延迟代价，应该显式开启。
     * @param rerankCandidates 精排多少条候选。这是延迟与精度的主旋钮：
     *                         20 条时 CPU 上大约几百毫秒，50 条就要一秒以上。
     * @param vectorWeight     融合时稠密那一路的权重，默认 1.0（等权）。
     *                         <b>0 表示这一路不参与</b>，此时 hybrid 退化成单路检索。
     *                         <p>权重是可选旋钮，不是必须调的参数：等权没有超参数，
     *                         也就没有过拟合的余地；一旦加权，选值必须用<b>独立于报告
     *                         数据</b>的验证集，否则报告上的数字就不再是样本外表现。
     * @param bm25Weight       融合时 BM25 那一路的权重，默认 1.0。
     *                         <p>用 {@code Double} 而不是 {@code double} 是有原因的：
     *                         yml 里不写这项时前者绑定成 {@code null}、后者绑定成
     *                         {@code 0.0}。用基本类型会把「没配」误当成「权重为零」，
     *                         然后 hybrid 模式静悄悄地少掉一路。
     */
    public record Search(String defaultMode, int candidatePool, int rrfK,
                         boolean rerankDefault, int rerankCandidates,
                         Double vectorWeight, Double bm25Weight) {
        public Search {
            if (defaultMode == null || defaultMode.isBlank()) {
                defaultMode = "hybrid";
            }
            if (candidatePool <= 0) {
                candidatePool = 50;
            }
            if (rrfK <= 0) {
                rrfK = 60;
            }
            if (rerankCandidates <= 0) {
                rerankCandidates = 20;
            }
            if (vectorWeight == null) {
                vectorWeight = 1.0;
            }
            if (bm25Weight == null) {
                bm25Weight = 1.0;
            }
            if (vectorWeight < 0 || bm25Weight < 0
                    || Double.isNaN(vectorWeight) || Double.isNaN(bm25Weight)) {
                // 负权重等于把排序反过来（排得越靠后加分越多），不是「少用一点」
                throw new IllegalArgumentException(
                        "vector-search.search.vector-weight / bm25-weight 必须 >= 0，实际是 "
                                + vectorWeight + " / " + bm25Weight);
            }
            if (vectorWeight + bm25Weight <= 0) {
                // 两路都关掉就没有 hybrid 这个模式了，与其给空结果不如启动就报错
                throw new IllegalArgumentException(
                        "vector-search.search.vector-weight 与 bm25-weight 不能同时为 0");
            }
            if (!defaultMode.equals("vector")
                    && !defaultMode.equals("bm25")
                    && !defaultMode.equals("hybrid")) {
                throw new IllegalArgumentException(
                        "vector-search.search.default-mode 只能是 vector / bm25 / hybrid，实际是 "
                                + defaultMode);
            }
        }

        /** 融合用的权重数组，顺序是 [稠密, 稀疏]。 */
        public double[] fusionWeights() {
            return new double[]{vectorWeight, bm25Weight};
        }
    }

    /**
     * @param dimension        向量维度，必须与 Python 侧模型输出一致。
     *                         bge-small-zh-v1.5 为 512，bge-base-zh-v1.5 为 768。
     * @param m                HNSW 每层最大出边数
     * @param efConstruction   HNSW 建图候选池大小
     * @param efSearch         查询候选池大小，recall / 延迟 的主旋钮
     * @param dataDir          索引落盘目录
     * @param chunkSize        上传文档切块的目标字符数
     * @param chunkOverlap     超长段落硬切时的重叠字符数
     * @param quantization     向量存储精度：{@code f32}（默认）或 {@code sq8}
     */
    public record Index(
            int dimension,
            int m,
            int efConstruction,
            int efSearch,
            String dataDir,
            int chunkSize,
            int chunkOverlap,
            String quantization
    ) {
        public Index {
            if (dimension <= 0) {
                dimension = 512;
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
            if (chunkSize < 64) {
                chunkSize = 300;
            }
            if (chunkOverlap < 0 || chunkOverlap >= chunkSize) {
                chunkOverlap = Math.min(50, chunkSize / 4);
            }
            if (quantization == null || quantization.isBlank()) {
                quantization = "f32";
            }
            quantization = quantization.trim().toLowerCase(java.util.Locale.ROOT);
            if (!quantization.equals("f32") && !quantization.equals("sq8")) {
                // 和 default-mode 一样：配置写错就启动失败，不要静默退化成默认值。
                // 量化写错会让人以为「开了 SQ8」，实际还在跑 f32，内存账全错。
                throw new IllegalArgumentException(
                        "vector-search.index.quantization 只能是 f32 / sq8，实际是 " + quantization);
            }
        }

        /** 配置值到枚举的映射。 */
        public com.vectorsearch.core.quant.Quantization toQuantization() {
            return quantization.equals("sq8")
                    ? com.vectorsearch.core.quant.Quantization.SQ8
                    : com.vectorsearch.core.quant.Quantization.NONE;
        }

        public HnswParams toHnswParams() {
            return HnswParams.of(m, efConstruction);
        }
    }
}
