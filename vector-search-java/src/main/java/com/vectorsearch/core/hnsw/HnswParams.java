package com.vectorsearch.core.hnsw;

/**
 * HNSW 构建与查询参数。
 *
 * <p>这几个数是 HNSW 的全部旋钮，第 5 周的消融实验就是逐个改它们并量指标：
 * <ul>
 *   <li>{@code m} —— 每层每个节点的最大出边数。调大：图更连通、recall 更高、
 *       内存和构建时间线性上升。经验值 12~48，16 是常见起点。</li>
 *   <li>{@code efConstruction} —— 建图时每层的候选池大小。调大：图质量更好、
 *       构建更慢。对查询延迟<b>无影响</b>，只影响一次性的构建成本。</li>
 *   <li>{@code efSearch} —— 查询时候选池大小，<b>不属于构建参数</b>。
 *       它是 runtime 可调的 recall/延迟 权衡旋钮（见 {@code VectorIndex#search}）。</li>
 * </ul>
 *
 * <p><b>面试必问</b>：「m 和 efConstruction 你怎么定的？」
 * 答案不能是"抄的默认值"，要能说出：m 影响的是图的连通度和内存，
 * efConstruction 只影响构建质量不影响查询速度，所以线上可以放心调大
 * efConstruction、而 m 一旦定下要改就得重建索引。
 *
 * @param m              每层最大出边数（layer 0 会放宽到 {@code 2*m}）
 * @param efConstruction 建图候选池大小
 * @param levelMultiplier 层级衰减系数，标准取 {@code 1/ln(m)}
 */
public record HnswParams(int m, int efConstruction, double levelMultiplier) {

    /** 层级上限。理论上服从几何分布，实际拦一下避免病态分配。 */
    public static final int MAX_LEVEL_CAP = 32;

    public HnswParams {
        if (m < 2) {
            throw new IllegalArgumentException("m 必须 >= 2，实际 " + m);
        }
        if (efConstruction < m) {
            throw new IllegalArgumentException(
                    "efConstruction(" + efConstruction + ") 不能小于 m(" + m + ")");
        }
        if (levelMultiplier <= 0 || Double.isNaN(levelMultiplier)) {
            throw new IllegalArgumentException("levelMultiplier 必须为正数");
        }
    }

    /**
     * 按标准论文取值构造：{@code mL = 1 / ln(M)}。
     *
     * <p>这个系数决定随机层高的分布，进而决定上层图的稀疏程度。
     */
    public static HnswParams of(int m, int efConstruction) {
        return new HnswParams(m, efConstruction, 1.0 / Math.log(m));
    }

    /**
     * layer 0 的最大出边数。
     *
     * <p>论文给 layer 0 放宽到 {@code 2*m}：底层承载全部节点，需要更高连通度
     * 才能保证搜索不走进死胡同；而上层节点少，用 m 就够，也更省内存。
     */
    public int mMax0() {
        return m * 2;
    }

    /** 指定层的最大出边数。 */
    public int maxLinks(int layer) {
        return layer == 0 ? mMax0() : m;
    }

    @Override
    public String toString() {
        return String.format("HnswParams{m=%d, efConstruction=%d, mMax0=%d, mL=%.4f}",
                m, efConstruction, mMax0(), levelMultiplier);
    }
}
