package com.vectorsearch.core.retrieval;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 拒答判据。
 *
 * <p>为什么这些用例值得写：拒答是<b>唯一会主动减少结果</b>的功能，
 * 判错的代价是不对称的 —— 该拒没拒只是多给几条噪声，不该拒却拒了会让
 * 明明存在的答案凭空消失，而用户无从判断是自己记错了还是系统有问题。
 * 所以每条规则的边界都要钉死。
 *
 * <p>阈值来自实测（{@code eval/abstain_probe.py}，自然问句场景）：
 * 余弦 0.55 / BM25 8.0 是「拒答 84%、13 条正样本误拒 0」的点。
 */
class AbstentionPolicyTest {

    private static final double VECTOR_THRESHOLD = 0.55;
    private static final double BM25_THRESHOLD = 8.0;

    private static AbstentionPolicy.Decision decide(Float cosine, Float bm25, String rule) {
        return AbstentionPolicy.decide(
                new AbstentionPolicy.Signals(cosine, bm25), rule, VECTOR_THRESHOLD, BM25_THRESHOLD);
    }

    @Test
    @DisplayName("两路都没有候选时拒答，理由是 no_candidates")
    void noCandidatesAtAll() {
        AbstentionPolicy.Decision decision = decide(null, null, AbstentionPolicy.RULE_OR);

        assertTrue(decision.abstained());
        assertEquals("no_candidates", decision.reason());
    }

    @Test
    @DisplayName("or 规则：余弦低就拒，哪怕 BM25 很高")
    void orRuleRejectsOnLowCosineAlone() {
        AbstentionPolicy.Decision decision = decide(0.32f, 25.0f, AbstentionPolicy.RULE_OR);

        assertTrue(decision.abstained());
    }

    @Test
    @DisplayName("or 规则：BM25 低也拒，哪怕余弦很高")
    void orRuleRejectsOnLowBm25Alone() {
        AbstentionPolicy.Decision decision = decide(0.88f, 3.1f, AbstentionPolicy.RULE_OR);

        assertTrue(decision.abstained());
    }

    @Test
    @DisplayName("or 规则：两路都不错就放行")
    void orRulePassesWhenBothLookGood() {
        AbstentionPolicy.Decision decision = decide(0.72f, 14.5f, AbstentionPolicy.RULE_OR);

        assertFalse(decision.abstained());
    }

    @Test
    @DisplayName("and 规则更保守：只有一路低不拒，两路都低才拒")
    void andRuleNeedsBothLow() {
        assertFalse(decide(0.40f, 20.0f, AbstentionPolicy.RULE_AND).abstained(),
                "BM25 很高，不该拒");
        assertFalse(decide(0.80f, 2.0f, AbstentionPolicy.RULE_AND).abstained(),
                "余弦很高，不该拒");
        assertTrue(decide(0.40f, 2.0f, AbstentionPolicy.RULE_AND).abstained(),
                "两路都低才拒");
    }

    @Test
    @DisplayName("vector 规则只看余弦，BM25 不参与")
    void vectorRuleIgnoresBm25() {
        assertFalse(decide(0.70f, 0.5f, AbstentionPolicy.RULE_VECTOR).abstained());
        assertTrue(decide(0.50f, 99.0f, AbstentionPolicy.RULE_VECTOR).abstained());
    }

    @Test
    @DisplayName("bm25 阈值设为 0 表示这一路不投票")
    void zeroBm25ThresholdDisablesThatVote() {
        // 余弦很高、BM25 低 —— 但 BM25 不投票，所以放行
        AbstentionPolicy.Decision decision = AbstentionPolicy.decide(
                new AbstentionPolicy.Signals(0.90f, 1.0f),
                AbstentionPolicy.RULE_OR, VECTOR_THRESHOLD, 0.0);

        assertFalse(decision.abstained());
    }

    @Test
    @DisplayName("只走了一路时，规则退化为那一路的判断")
    void singleRouteDegradesGracefully() {
        // bm25 模式：只有 BM25 分
        assertTrue(decide(null, 2.0f, AbstentionPolicy.RULE_OR).abstained());
        assertFalse(decide(null, 20.0f, AbstentionPolicy.RULE_AND).abstained());

        // vector 模式：只有余弦
        assertTrue(decide(0.30f, null, AbstentionPolicy.RULE_AND).abstained());
        assertFalse(decide(0.80f, null, AbstentionPolicy.RULE_OR).abstained());
    }

    @Test
    @DisplayName("阈值是「低于」不是「低于等于」")
    void thresholdIsStrictlyBelow() {
        assertFalse(decide((float) VECTOR_THRESHOLD, 99.0f, AbstentionPolicy.RULE_VECTOR).abstained(),
                "正好等于阈值不算低");
    }
}
