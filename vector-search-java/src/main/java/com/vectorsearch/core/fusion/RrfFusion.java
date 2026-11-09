package com.vectorsearch.core.fusion;

import com.vectorsearch.core.SearchResult;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Reciprocal Rank Fusion（倒数排名融合）—— 把多路检索结果合成一个排名。
 *
 * <h2>为什么不能直接把分数加权相加</h2>
 * 这是混合检索里最容易做错的地方。我们有两路检索：
 *
 * <table border="1">
 *   <caption>分数不可比</caption>
 *   <tr><th></th><th>取值范围</th><th>含义</th></tr>
 *   <tr><td>向量检索（余弦）</td><td>[-1, 1]</td><td>语义相似度</td></tr>
 *   <tr><td>BM25</td><td>[0, +∞)</td><td>词频加权的匹配强度，无上界</td></tr>
 * </table>
 *
 * <p>直接把两者相加，等于让 BM25 单方面决定结果 —— 它的分数动辄二三十，
 * 而余弦最多 1.0。想归一化也不行：两种分数的分布形状完全不同，
 * min-max 归一化会被个别极端值带偏。
 *
 * <h2>RRF 的做法：只看排名，不看分数</h2>
 * <pre>
 *   score(d) = Σ_i  1 / (k + rank_i(d))
 * </pre>
 *
 * <p>每路检索里排第 1 的文档得 {@code 1/(k+1)}，排第 5 的得 {@code 1/(k+5)}。
 * 如果一篇文档在两路里都排得靠前，它的分数会自然叠加。
 *
 * <p>这样做的好处：
 * <ul>
 *   <li><b>绕开了分数不可比问题</b>，只依赖"谁排在前面"这个排名信息</li>
 *   <li>对异常分数免疫：某一路给出一个离谱的高分，只要排名没变，结果就不受影响</li>
 *   <li>平滑常数 {@code k} 基本不用调 —— 60 是论文的常用值，对结果影响很小。
 *       真正需要调的是<b>各路权重</b>，而那是可选项：不调就是等权，
 *       调了就要用独立验证集选，见 {@link #fuse(List, double[], int, int)}</li>
 * </ul>
 *
 * <p>代价是丢掉了分数里的"置信度"信息：排第 1 但分数 0.99 与
 * 排第 1 但分数 0.31，在 RRF 里完全等价。想利用这个信息就得做分数校准，
 * 那是另一个话题。
 */
public final class RrfFusion {

    /**
     * 平滑常数。作用是压低"排名靠前"的绝对优势比例：
     * k 越小，第 1 名与第 2 名的差距越大；k 越大，各排名越接近。
     * 60 是原论文的经验值，实践中对结果不敏感。
     */
    public static final int DEFAULT_K = 60;

    private RrfFusion() {
    }

    /**
     * 融合多路检索结果，各路等权。
     *
     * @param rankings 各路结果，每路已按相关度降序排列
     * @param topK     返回条数
     * @param k        平滑常数，一般用 {@link #DEFAULT_K}
     */
    public static List<SearchResult> fuse(List<List<SearchResult>> rankings, int topK, int k) {
        return fuse(rankings, null, topK, k);
    }

    /**
     * 融合多路检索结果，各路可带权重。
     *
     * <p><b>为什么要权重</b>：等权 RRF 隐含一个假设 —— 每一路的可信度相同。
     * 实测不成立：在词面查询上 BM25 recall 0.96、向量 0.80，
     * 等权融合出来的 0.89 比 BM25 单路还差。原因是向量那一路的第 1 名
     * 与 BM25 那一路的第 1 名在公式里分量相同，可前者有一半是错的，
     * 它把对的文档从第一名挤了下去。给 BM25 加权就能压住这种「平票时的噪声」。
     *
     * <p>公式变成：
     * <pre>
     *   score(d) = Σ_i  w_i / (k + rank_i(d))
     * </pre>
     *
     * <p>权重为 0 表示这一路完全不参与（等价于退化成单路检索）。
     * 负数和 NaN 也按 0 处理 —— 排名分数非负，负权重会让「排得越靠后越加分」，
     * 那不是加权而是把排序反过来，不该被一个配置项静默触发。
     *
     * <p><b>代价</b>：多了一个要调的超参数，而调参必须用<b>独立于报告数据</b>的
     * 验证集选，否则报告里那个数字就不再是「没见过的数据上的表现」。
     * 这就是等权 RRF 好看的地方 —— 它没有旋钮，也就没有过拟合的余地。
     *
     * @param weights 各路权重，下标与 {@code rankings} 对齐；长度不足时缺的按 1.0 算；
     *                传 {@code null} 等价于全 1.0
     */
    public static List<SearchResult> fuse(List<List<SearchResult>> rankings,
                                          double[] weights,
                                          int topK,
                                          int k) {
        if (topK <= 0 || rankings == null || rankings.isEmpty()) {
            return List.of();
        }
        int smoothing = k <= 0 ? DEFAULT_K : k;

        // 用 double 累加：多路融合时浮点误差会被放大，而最终排名对精度敏感
        Map<Integer, Double> fusedScores = new HashMap<>();

        for (int i = 0; i < rankings.size(); i++) {
            List<SearchResult> ranking = rankings.get(i);
            if (ranking == null) {
                continue;
            }
            double weight = (weights != null && i < weights.length) ? weights[i] : 1.0;
            if (!(weight > 0)) {
                // 0 / 负数 / NaN：这一路不计入
                continue;
            }
            for (int rank = 0; rank < ranking.size(); rank++) {
                int id = ranking.get(rank).id();
                // rank 从 0 开始，公式里从 1 开始
                fusedScores.merge(id, weight / (smoothing + rank + 1), Double::sum);
            }
        }

        if (fusedScores.isEmpty()) {
            return List.of();
        }

        List<SearchResult> all = new ArrayList<>(fusedScores.size());
        for (Map.Entry<Integer, Double> entry : fusedScores.entrySet()) {
            all.add(new SearchResult(entry.getKey(), entry.getValue().floatValue()));
        }

        // 分数相同时按 id 升序 —— 融合后的分数容易出现精确相等，
        // 没有稳定的次级排序的话，同一份数据两次查询可能给出不同顺序
        all.sort(Comparator.comparingDouble(SearchResult::score).reversed()
                .thenComparingInt(SearchResult::id));

        return all.size() <= topK ? all : new ArrayList<>(all.subList(0, topK));
    }
}
