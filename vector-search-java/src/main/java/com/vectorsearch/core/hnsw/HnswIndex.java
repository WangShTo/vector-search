package com.vectorsearch.core.hnsw;

import com.vectorsearch.core.DistanceMetrics;
import com.vectorsearch.core.SearchResult;
import com.vectorsearch.core.VectorIndex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * HNSW（Hierarchical Navigable Small World）近似最近邻索引 —— 从零实现。
 *
 * <h2>核心思想</h2>
 * 把向量组织成一张<b>分层</b>的邻近图：上层节点稀疏、边长跨得大，用于快速跳到目标区域；
 * 下层节点稠密，用于精确收敛。查询时从顶层贪心下降到第 0 层，每层只走"比当前更近"的邻居。
 *
 * <h2>与暴力检索的关系</h2>
 * 这是<b>近似</b>算法：用少量 recall 损失换数量级的延迟下降。
 * 因此必须用 {@link com.vectorsearch.core.bruteforce.BruteForceIndex} 的结果当
 * ground truth 来量化 recall —— 否则"很快"是没有意义的。
 *
 * <h2>本版本是刻意的 v1（正确优先）</h2>
 * 为了给后续每一步优化留下可量化的对照组，这里<b>故意</b>用了朴素实现：
 * <ul>
 *   <li>向量存成 {@code List<float[]>} —— 对象头 + 指针跳转，第 3 周改扁平数组 + 堆外内存</li>
 *   <li>距离用标量 {@link DistanceMetrics} —— 第 5 周换 Vector API SIMD</li>
 *   <li>{@code visited} 用 {@code HashSet<Integer>} —— 每层查询都在装箱，第 3 周换 int 数组 + 版本号戳</li>
 *   <li>锁用 {@link ReentrantReadWriteLock} —— 第 6 周换 StampedLock 乐观读</li>
 * </ul>
 * 每一处替换都能单独跑出加速比，这些数字就是简历和面试里的弹药。
 *
 * <h2>并发模型</h2>
 * 读（{@link #search}）加读锁，可并发；写（{@link #add}）加写锁，独占。
 * 典型用法是离线构建、在线查询，因此读多写少。
 */
public class HnswIndex implements VectorIndex {

    private final int dimension;
    private final HnswParams params;
    private final Random random;
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();

    // ---------- 节点存储 ----------
    /** 内部节点的归一化向量，下标即内部 id。 */
    private final List<float[]> vectors = new ArrayList<>();
    /** 内部 id → 业务 id。 */
    private final List<Integer> externalIds = new ArrayList<>();
    /** links[node][layer] = 该节点在第 layer 层的邻居（内部 id 数组）。 */
    private final List<int[][]> links = new ArrayList<>();
    /** 每个节点的最高层号。 */
    private final List<Integer> levels = new ArrayList<>();

    // ---------- 图入口 ----------
    private int entryPoint = -1;
    private int maxLevel = -1;

    public HnswIndex(int dimension, HnswParams params) {
        this(dimension, params, 42L);
    }

    /**
     * @param seed 固定随机种子，让每次构建出的图完全一致 ——
     *             压测报告要可复现，随机图会让两次跑的 recall 对不上
     */
    public HnswIndex(int dimension, HnswParams params, long seed) {
        if (dimension <= 0) {
            throw new IllegalArgumentException("dimension 必须为正数: " + dimension);
        }
        this.dimension = dimension;
        this.params = params;
        this.random = new Random(seed);
    }

    // =====================================================================
    // 写入
    // =====================================================================

    @Override
    public void add(int id, float[] vector) {
        validate(vector);
        lock.writeLock().lock();
        try {
            insertLocked(id, vector);
        } finally {
            lock.writeLock().unlock();
        }
    }

    private void insertLocked(int externalId, float[] vector) {
        float[] v = DistanceMetrics.normalizedCopy(vector);
        int nodeId = vectors.size();
        int level = randomLevel();

        vectors.add(v);
        externalIds.add(externalId);
        levels.add(level);

        int[][] nodeLinks = new int[level + 1][];
        for (int l = 0; l <= level; l++) {
            nodeLinks[l] = new int[0];
        }
        links.add(nodeLinks);

        // 第一个节点：直接成为入口，没有图可走
        if (entryPoint == -1) {
            entryPoint = nodeId;
            maxLevel = level;
            return;
        }

        int current = entryPoint;
        float currentSim = similarity(v, current);

        // ---- 阶段 1：从顶层贪心下降到 level+1 层 ----
        // 上层图稀疏，用 ef=1 的贪心就够，目的是快速逼近目标区域
        for (int l = maxLevel; l > level; l--) {
            current = greedyDescend(v, current, currentSim, l);
            currentSim = similarity(v, current);
        }

        // ---- 阶段 2：从 min(level, maxLevel) 层逐层下降，边搜边连 ----
        for (int l = Math.min(level, maxLevel); l >= 0; l--) {
            List<SearchResult> candidates = searchLayer(v, current, params.efConstruction(), l);
            int maxLinks = params.maxLinks(l);

            // 用启发式而非"最近的 m 个"：优先保留方向分散的邻居，
            // 否则邻居全挤在同一侧，查询时容易走进局部最优出不来
            int[] neighbors = selectNeighborsHeuristic(v, candidates, maxLinks);
            links.get(nodeId)[l] = neighbors;

            // 建立反向边。注意反向边可能因为超出上限而被剪枝，
            // 这是 HNSW 的正常行为：边的存在不保证双向对称
            for (int neighbor : neighbors) {
                addLink(neighbor, l, nodeId);
            }

            if (!candidates.isEmpty()) {
                current = candidates.get(0).id();
                currentSim = similarity(v, current);
            }
        }

        // 新节点层高超过当前最高层时，它成为新的入口点
        if (level > maxLevel) {
            maxLevel = level;
            entryPoint = nodeId;
        }
    }

    /** 在指定层贪心走到局部最优。用于上层快速逼近。 */
    private int greedyDescend(float[] query, int start, float startSim, int layer) {
        int current = start;
        float currentSim = startSim;
        boolean improved = true;
        while (improved) {
            improved = false;
            for (int neighbor : links.get(current)[layer]) {
                float sim = similarity(query, neighbor);
                if (sim > currentSim) {
                    currentSim = sim;
                    current = neighbor;
                    improved = true;
                }
            }
        }
        return current;
    }

    /**
     * 加一条边；超出上限时用启发式剪枝。
     *
     * <p>剪枝不是"删掉最远的那个"那么简单 —— 要在「当前邻居 ∪ 新邻居」上重跑
     * 启发式，保持方向多样性。这是 HNSW 质量的关键细节，也是常见实现错误。
     */
    private void addLink(int node, int layer, int target) {
        int[] current = links.get(node)[layer];
        for (int existing : current) {
            if (existing == target) {
                return; // 边已存在
            }
        }

        int maxLinks = params.maxLinks(layer);
        if (current.length < maxLinks) {
            int[] next = Arrays.copyOf(current, current.length + 1);
            next[current.length] = target;
            links.get(node)[layer] = next;
            return;
        }

        float[] nodeVec = vectors.get(node);
        List<SearchResult> candidates = new ArrayList<>(current.length + 1);
        for (int existing : current) {
            candidates.add(new SearchResult(existing, similarity(nodeVec, existing)));
        }
        candidates.add(new SearchResult(target, similarity(nodeVec, target)));
        links.get(node)[layer] = selectNeighborsHeuristic(nodeVec, candidates, maxLinks);
    }

    // =====================================================================
    // 查询
    // =====================================================================

    @Override
    public List<SearchResult> search(float[] query, int k, int efSearch) {
        validate(query);
        if (k <= 0) {
            return List.of();
        }
        float[] q = DistanceMetrics.normalizedCopy(query);

        lock.readLock().lock();
        try {
            if (entryPoint == -1) {
                return List.of();
            }

            int current = entryPoint;
            float currentSim = similarity(q, current);

            // 上层：ef=1 的贪心下降，只找方向
            for (int l = maxLevel; l > 0; l--) {
                current = greedyDescend(q, current, currentSim, l);
                currentSim = similarity(q, current);
            }

            // 第 0 层：用 efSearch 的候选池做真正的收敛
            // ef 至少要 >= k，否则候选池比要的条数还少，结果必然缺
            int ef = Math.max(efSearch, k);
            List<SearchResult> candidates = searchLayer(q, current, ef, 0);

            int take = Math.min(k, candidates.size());
            List<SearchResult> result = new ArrayList<>(take);
            for (int i = 0; i < take; i++) {
                SearchResult hit = candidates.get(i);
                result.add(new SearchResult(externalIds.get(hit.id()), hit.score()));
            }
            return result;
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * 在指定层搜索，返回按相似度<b>降序</b>（最近在前）排列的候选。
     *
     * <p>这是 HNSW 论文的 SEARCH-LAYER，两个堆的分工是关键：
     * <ul>
     *   <li>{@code candidates} —— 大顶堆（按相似度），每次弹出<b>最近</b>的待扩展节点</li>
     *   <li>{@code result} —— 小顶堆（按相似度），堆顶是<b>已保留结果里最差</b>的，
     *       用于 O(1) 判断新候选值不值得收、以及淘汰谁</li>
     * </ul>
     *
     * <p>终止条件：弹出来的候选比"已保留结果里最差的"还差，
     * 说明前沿已经没有希望，可以停。
     *
     * @param entry 起始节点（必须是第 layer 层存在的节点）
     */
    private List<SearchResult> searchLayer(float[] query, int entry, int ef, int layer) {
        // TODO(第3周): HashSet<Integer> 每次查询都装箱 + 分配，是热点路径上的主要开销之一。
        //  改为 int[] visitedStamp + 递增版本号，可省掉全部装箱与清空成本。
        Set<Integer> visited = new HashSet<>();

        // 大顶堆：堆顶 = 最近的候选
        PriorityQueue<SearchResult> candidates =
                new PriorityQueue<>(Comparator.comparingDouble(SearchResult::score).reversed());
        // 小顶堆：堆顶 = 已保留结果里最差的
        PriorityQueue<SearchResult> result =
                new PriorityQueue<>(Comparator.comparingDouble(SearchResult::score));

        SearchResult start = new SearchResult(entry, similarity(query, entry));
        visited.add(entry);
        candidates.offer(start);
        result.offer(start);

        while (!candidates.isEmpty()) {
            SearchResult nearest = candidates.poll();
            SearchResult worstKept = result.peek();

            // nearest 已经比已保留的最差还差 —— 前沿无望，收工
            if (worstKept != null && nearest.score() < worstKept.score()) {
                break;
            }

            for (int neighbor : links.get(nearest.id())[layer]) {
                if (!visited.add(neighbor)) {
                    continue;
                }
                float sim = similarity(query, neighbor);
                SearchResult candidate = new SearchResult(neighbor, sim);
                SearchResult currentWorst = result.peek();

                boolean worthKeeping = result.size() < ef
                        || currentWorst == null
                        || sim > currentWorst.score();

                if (worthKeeping) {
                    candidates.offer(candidate);
                    result.offer(candidate);
                    if (result.size() > ef) {
                        result.poll(); // 淘汰最差的
                    }
                }
            }
        }

        List<SearchResult> sorted = new ArrayList<>(result);
        sorted.sort(Comparator.comparingDouble(SearchResult::score).reversed());
        return sorted;
    }

    // =====================================================================
    // 邻居选择启发式（论文 Algorithm 4）
    // =====================================================================

    /**
     * 从候选集中挑出最多 {@code m} 个邻居。
     *
     * <p><b>为什么不能直接取最近的 m 个</b>：那样邻居会聚在同一个方向，
     * 查询时容易一头扎进局部最优出不来。这里加了多样性约束 ——
     * 只有当候选 {@code e} 距离查询点<b>比距离任一已选邻居都更近</b>时才收它。
     * 直觉上就是"这个新邻居提供了别人提供不了的方向"。
     *
     * <p>不足 {@code m} 时用被丢弃的候选中最近的补齐（论文的
     * {@code keepPrunedConnections}），保证图上连通度不至于太差。
     *
     * @param query      查询（或建图时的新节点）向量
     * @param candidates 候选，可乱序
     * @param m          最多选几个
     */
    private int[] selectNeighborsHeuristic(float[] query, List<SearchResult> candidates, int m) {
        if (candidates.isEmpty() || m <= 0) {
            return new int[0];
        }

        List<SearchResult> sorted = new ArrayList<>(candidates);
        sorted.sort(Comparator.comparingDouble(SearchResult::score).reversed()); // 最近在前

        List<Integer> selected = new ArrayList<>(m);
        List<Integer> discarded = new ArrayList<>();

        for (SearchResult candidate : sorted) {
            if (selected.size() >= m) {
                discarded.add(candidate.id());
                continue;
            }

            boolean addsNewDirection = true;
            for (int chosen : selected) {
                // candidate 与已选邻居的相似度，若高于它与 query 的相似度，
                // 说明它已经被已选邻居"覆盖"了，收进来只是冗余
                float simToChosen = similarity(vectors.get(candidate.id()), chosen);
                if (simToChosen > candidate.score()) {
                    addsNewDirection = false;
                    break;
                }
            }

            if (addsNewDirection) {
                selected.add(candidate.id());
            } else {
                discarded.add(candidate.id());
            }
        }

        // keepPrunedConnections：连通度兜底
        for (int id : discarded) {
            if (selected.size() >= m) {
                break;
            }
            selected.add(id);
        }

        int[] out = new int[selected.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = selected.get(i);
        }
        return out;
    }

    // =====================================================================
    // 辅助
    // =====================================================================

    /**
     * 抽样层高。服从几何分布，期望值随 {@code 1/ln(m)} 衰减 ——
     * 这保证了上层节点数指数级减少，形成"金字塔"结构。
     */
    private int randomLevel() {
        double r = random.nextDouble();
        if (r == 0.0) {
            r = Double.MIN_VALUE; // 避免 log(0) = -Infinity
        }
        int level = (int) Math.floor(-Math.log(r) * params.levelMultiplier());
        return Math.min(level, HnswParams.MAX_LEVEL_CAP);
    }

    /** 向量已归一化，所以内积就是余弦相似度。 */
    private float similarity(float[] query, int nodeId) {
        return DistanceMetrics.dot(query, vectors.get(nodeId));
    }

    private void validate(float[] vector) {
        if (vector == null) {
            throw new IllegalArgumentException("向量不能为 null");
        }
        if (vector.length != dimension) {
            throw new IllegalArgumentException(
                    "向量维度不匹配: 期望 " + dimension + "，实际 " + vector.length);
        }
    }

    // =====================================================================
    // 观测（供压测报告与测试使用）
    // =====================================================================

    @Override
    public int size() {
        return vectors.size();
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public String name() {
        return "hnsw";
    }

    public HnswParams params() {
        return params;
    }

    /** 图的最大层号，即入口点所在层。 */
    public int maxLevel() {
        return maxLevel;
    }

    /** 指定节点的最高层号，用于验证层高分布是否符合几何分布。 */
    public int levelOf(int nodeId) {
        lock.readLock().lock();
        try {
            return levels.get(nodeId);
        } finally {
            lock.readLock().unlock();
        }
    }

    /** 全图总边数（含各层，反向边单独计数），用于估算内存占用。 */
    public long totalLinkCount() {
        lock.readLock().lock();
        try {
            long total = 0;
            for (int[][] nodeLinks : links) {
                for (int[] layerLinks : nodeLinks) {
                    total += layerLinks.length;
                }
            }
            return total;
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * 各层「最高层恰好是本层」的节点数。index 即层号。
     *
     * <p>这是<b>层高分布</b>，用来验证随机层高的抽样是否正确（应服从几何分布）。
     *
     * <p>注意它与 {@link #layerPopulation()} 的区别：本方法数的是
     * 「最高层正好是 l」，而 layerPopulation 数的是「至少达到 l 层」。
     * 描述金字塔结构时要用后者。这两个量混淆是很容易犯的错。
     */
    public int[] levelHistogram() {
        lock.readLock().lock();
        try {
            int[] histogram = new int[maxLevel + 1];
            for (int level : levels) {
                histogram[level]++;
            }
            return histogram;
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * 各层实际「存在」的节点数（累计口径）。index 即层号。
     *
     * <p>HNSW 中每个节点都存在于从 0 到它自身层高的<b>所有</b>层，
     * 所以 population[0] 恒等于总节点数，且逐层递减。
     * 这才是"金字塔"的真实形状。
     */
    public int[] layerPopulation() {
        lock.readLock().lock();
        try {
            int[] population = new int[maxLevel + 1];
            for (int level : levels) {
                for (int l = 0; l <= level; l++) {
                    population[l]++;
                }
            }
            return population;
        } finally {
            lock.readLock().unlock();
        }
    }
}
