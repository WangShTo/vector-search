package com.vectorsearch.core.retrieval;

/**
 * 拒答判定：这条路要不要承认「库里没有相关内容」。
 *
 * <h2>为什么需要它</h2>
 *
 * 一个检索服务如果永远返回 top-k，它永远不会说「不知道」。用户问「明天上海下雨吗」，
 * 它照样从知识库里挑十条最像的，分数看着还挺高 —— 这比报错更糟，因为**错误是静默的**。
 *
 * <h2>为什么不用融合分做判据</h2>
 *
 * 直觉上「融合分低于某个值就是不相关」最省事，但这是错的，而且错得有结构性原因：
 * RRF 的 k=60，单路 rank1 贡献 1/61 ≈ 0.0164，融合分被压在 0.016~0.033 这条窄带里，
 * 它编码的是**排名**，不是**相关程度**。实测（{@code bench/results/abstain-scores.json}）：
 * 120 条相关查询与 50 条不相关查询的融合分区间**完全重合**（0.0164~0.0328），
 * AUC 只有 0.87，是所有候选信号里最差的一个。
 *
 * <h2>用什么</h2>
 *
 * 用两路各自的**原生分**，它们都还没被排名归一化掉：
 *
 * <table border="1">
 *   <caption>实测判别力（AUC，越高越能分开相关与不相关）</caption>
 *   <tr><th>信号</th><th>自然问句</th><th>已知项检索</th><th>额外开销</th></tr>
 *   <tr><td>向量余弦 top1</td><td>0.915</td><td>0.999</td><td>0（本来就算）</td></tr>
 *   <tr><td>BM25 top1</td><td>0.901</td><td>0.973</td><td>0（本来就算）</td></tr>
 *   <tr><td>cross-encoder top1</td><td>0.891</td><td>0.981</td><td>每个候选一次前向</td></tr>
 *   <tr><td>RRF 融合分</td><td>0.775</td><td>0.873</td><td>0</td></tr>
 * </table>
 *
 * <p>结论：<b>免费的两个信号比要花几百毫秒的 cross-encoder 还好</b>。所以判据用余弦与 BM25，
 * 不启用精排也能拒答。cross-encoder 的分只在已经开了精排时才有参考价值。
 *
 * <h2>阈值从哪来</h2>
 *
 * 不能拍脑袋，要在带负样本的评测集上扫（{@code eval/abstain_probe.py}）：
 * 13 条自然问句 + 50 条负样本，按「误拒率最低时拒答率最高」选点。
 * 默认的 0.55 / 8.0 就是这么来的，实测拒答 84%、误拒 0/13。
 *
 * <p><b>但这两个数是有边界的</b>，用之前必须知道：
 * <ul>
 *   <li>样本只有 13 条正样本。观测到 0 次误拒，95% 置信上界仍有约 21% ——
 *       「误拒率 0%」这句话是不成立的，只能说「13 条里没观察到」。</li>
 *   <li>两个阈值都紧贴在正样本最小值下方（余弦 0.5739、BM25 8.05），
 *       是那块数据的边界，不是安全余量。</li>
 *   <li><b>BM25 的绝对分是跟语料走的</b>：同样的查询换一批文档，分就变了
 *       （IDF 与平均文档长度都会影响）。换语料必须重新标定。
 *       余弦没有这个问题，归一化之后大致稳定 —— 这是默认规则里两项不对称的原因。</li>
 * </ul>
 */
public final class AbstentionPolicy {

    /** 规则：只看余弦。 */
    public static final String RULE_VECTOR = "vector";
    /** 规则：任一信号低就拒答（默认）。 */
    public static final String RULE_OR = "or";
    /** 规则：两个信号都低才拒答。 */
    public static final String RULE_AND = "and";

    private AbstentionPolicy() {
    }

    /**
     * 两路各自的最佳分数。缺失用 {@code null} 表示 ——
     * 例如 {@code mode=bm25} 时根本没有向量分，这和「向量分很低」是两回事。
     *
     * @param vectorTop1 余弦相似度最高的那条的分数
     * @param bm25Top1   BM25 分数最高的那条的分数
     */
    public record Signals(Float vectorTop1, Float bm25Top1) {
    }

    /**
     * 判定结果。
     *
     * @param abstained 是否拒答
     * @param reason    拒答理由，可读字符串（会写进响应，方便页面与排查）
     */
    public record Decision(boolean abstained, String reason) {

        public static Decision answer() {
            return new Decision(false, null);
        }
    }

    /**
     * 判定要不要拒答。
     *
     * @param signals        两路原生分
     * @param rule           {@link #RULE_VECTOR} / {@link #RULE_OR} / {@link #RULE_AND}
     * @param vectorThreshold 余弦阈值，必须 &gt; 0
     * @param bm25Threshold   BM25 阈值，**&lt;= 0 表示这一路不参与投票**
     *                        （BM25 量纲跟语料强相关，没标定过就不该让它投票）
     */
    public static Decision decide(Signals signals,
                                  String rule,
                                  double vectorThreshold,
                                  double bm25Threshold) {
        boolean hasVector = signals.vectorTop1() != null;
        boolean hasBm25 = signals.bm25Top1() != null && bm25Threshold > 0;
        if (!hasVector && !hasBm25) {
            // 两路都拿不到分：要么索引是空的，要么模式既不查向量也不查 BM25。
            // 这种情况「没有结果」就是正确答案，但理由要说清楚，别让人以为是阈值卡掉了。
            return new Decision(true, "no_candidates");
        }

        boolean vectorLow = hasVector && signals.vectorTop1() < vectorThreshold;
        boolean bm25Low = hasBm25 && signals.bm25Top1() < bm25Threshold;

        String effective = rule == null ? RULE_OR : rule.trim().toLowerCase(java.util.Locale.ROOT);
        boolean abstain = switch (effective) {
            case RULE_VECTOR -> vectorLow;
            // AND 只在两路都能投票时才「都低」成立；缺一路时退化成那一路的判断，
            // 否则关掉一路权重就会让 AND 永远不成立 —— 那是静默失效。
            case RULE_AND -> hasVector && hasBm25 ? (vectorLow && bm25Low) : (vectorLow || bm25Low);
            default -> vectorLow || bm25Low;
        };

        if (!abstain) {
            return Decision.answer();
        }
        return new Decision(true, describe(vectorLow, bm25Low, signals, vectorThreshold, bm25Threshold));
    }

    private static String describe(boolean vectorLow,
                                   boolean bm25Low,
                                   Signals signals,
                                   double vectorThreshold,
                                   double bm25Threshold) {
        if (vectorLow && bm25Low) {
            return String.format("both_low: cosine %.3f < %.3f, bm25 %.2f < %.2f",
                    signals.vectorTop1(), vectorThreshold, signals.bm25Top1(), bm25Threshold);
        }
        if (vectorLow) {
            return String.format("cosine %.3f < %.3f", signals.vectorTop1(), vectorThreshold);
        }
        return String.format("bm25 %.2f < %.2f", signals.bm25Top1(), bm25Threshold);
    }
}
