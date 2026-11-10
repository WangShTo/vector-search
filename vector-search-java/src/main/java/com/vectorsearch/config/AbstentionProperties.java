package com.vectorsearch.config;

import com.vectorsearch.core.retrieval.AbstentionPolicy;
import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * 拒答（abstention）配置，绑定 {@code vector-search.abstention.*}。
 *
 * <p>单独一个配置类而不是塞进 {@link VectorSearchProperties}：后者是 record，
 * 加一个字段就要改掉所有构造点（测试里全是显式 new 的）。一个可选特性不值得
 * 让所有调用方跟着改签名。
 *
 * @param enabled         是否启用。**不配就是启用** —— 拒答是产品该有的默认行为，
 *                        「问什么都能答」才是需要显式打开的那个开关。
 * @param rule            判定规则：{@code vector} 只看向量余弦（最保守）/
 *                        {@code or} 任一低就拒（默认）/ {@code and} 都低才拒。
 * @param vectorThreshold 余弦阈值。默认 0.55 是在 13 条自然问句 + 50 条负样本上
 *                        标定的（{@code eval/abstain_probe.py}），实测拒答 84%、误拒 0/13。
 *                        语义上就是「最像的那条都只有这么点相似度，那就是没有」。
 * @param bm25Threshold   BM25 阈值。**&lt;= 0 表示这一路不参与投票**（默认 8.0，
 *                        同为那次标定的结果）。注意 BM25 的绝对分跟语料走，
 *                        换一批文档就会漂 —— 换语料必须重新标定，
 *                        不想担这个责任就把这里设成 0，退回纯余弦规则。
 */
@ConfigurationProperties(prefix = "vector-search.abstention")
public record AbstentionProperties(
        Boolean enabled,
        String rule,
        Double vectorThreshold,
        Double bm25Threshold
) {

    public AbstentionProperties {
        if (enabled == null) {
            enabled = Boolean.TRUE;
        }
        if (rule == null || rule.isBlank()) {
            rule = AbstentionPolicy.RULE_OR;
        }
        rule = rule.trim().toLowerCase(java.util.Locale.ROOT);
        if (!rule.equals(AbstentionPolicy.RULE_VECTOR)
                && !rule.equals(AbstentionPolicy.RULE_OR)
                && !rule.equals(AbstentionPolicy.RULE_AND)) {
            // 和 default-mode / quantization 一样：写错就启动失败。
            // 静默退化成默认值会让人以为「已经开了拒答」，其实规则不是他想的那条。
            throw new IllegalArgumentException(
                    "vector-search.abstention.rule 只能是 vector / or / and，实际是 " + rule);
        }
        if (vectorThreshold == null) {
            vectorThreshold = 0.55;
        }
        if (vectorThreshold <= 0 || vectorThreshold >= 1 || Double.isNaN(vectorThreshold)) {
            // 归一化之后的余弦落在 [-1,1]，阈值取到 1 以上等于全部拒答
            throw new IllegalArgumentException(
                    "vector-search.abstention.vector-threshold 必须在 (0,1) 之间，实际是 "
                            + vectorThreshold);
        }
        if (bm25Threshold == null) {
            bm25Threshold = 8.0;
        }
        if (bm25Threshold < 0 || Double.isNaN(bm25Threshold)) {
            throw new IllegalArgumentException(
                    "vector-search.abstention.bm25-threshold 必须 >= 0（0 表示不参与），实际是 "
                            + bm25Threshold);
        }
    }

    public boolean on() {
        return Boolean.TRUE.equals(enabled);
    }
}
