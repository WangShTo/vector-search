package com.vectorsearch.core.hnsw;

import com.vectorsearch.core.DistanceMetrics;
import com.vectorsearch.core.DistanceMetricsSimd;
import com.vectorsearch.core.SearchResult;
import com.vectorsearch.core.VectorIndex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Random;
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
 * <h2>存储布局：全部扁平化</h2>
 *
 * 最初每个节点用独立对象表示（{@code List<float[]>} + {@code List<int[][]>}）。
 * 10 万节点时这会产生约 <b>30 万个对象</b>：每个 {@code float[]}、每个 {@code int[][]}、
 * 每层一个 {@code int[]}，再加上 {@code List<Integer>} 的装箱。后果有两个：
 * 一是遍历邻居时不停追指针，缓存命中率差；二是垃圾回收要管理海量小对象。
 *
 * <p>现在整个索引只有 <b>7 个数组</b>：
 *
 * <pre>
 *   vectorStore    float[]  节点向量连续存放，第 i 条在 i*dimension 处
 *   externalIdStore int[]   业务 id
 *   levelStore     byte[]   层高（最大 32，一字节足够）
 *   layer0Links    int[]    第 0 层邻接表，步长固定为 mMax0
 *   layer0Count    short[]  第 0 层每个节点的邻居数
 *   upperStore     int[]    第 1 层以上，按 (节点, 层) 紧凑打包
 *   upperOffset    int[]    每个节点的上层槽位起点，-1 表示没有上层
 * </pre>
 *
 * <p>访问标记<b>不在其中</b>：它是线程本地的，每个线程各持一份 int[capacity]。
 * 原因是读路径不能写共享状态 —— 见 {@link #visitedScratch}。
 *
 * <p><b>关键的兼容性问题</b>：扁平存储要求距离计算支持"数组 + 偏移量"而不是
 * 两个独立数组，否则每算一次距离都得先把向量复制出来，扁平化就白做了。
 * 这正是 {@link DistanceMetricsSimd#dot(float[], int, float[], int, int)} 存在的理由 ——
 * 数据连续之后，SIMD 从整块大数组按偏移加载，比原来追引用跳转更友好。
 *
 * <h2>为什么不做堆外内存</h2>
 * Java 21 的 {@code java.lang.foreign} 仍是<b>预览特性</b>，用它会要求
 * {@code --enable-preview} 并把项目锁死在特定 JDK 版本上。
 * 而绕过它的方案（直接 {@code ByteBuffer}）无法配合 Vector API 的
 * {@code fromArray}，只能退化回标量距离 —— 那会丢掉 5.3 倍的 SIMD 收益，
 * 得不偿失。所以当前选择是「扁平 + SIMD」，堆外留待 JDK 22+ 之后再做。
 * 这个取舍本身比硬塞一个半成品堆外实现更值得说。
 *
 * <h2>并发模型</h2>
 * 读（{@link #search}）加读锁，可并发；写（{@link #add}）加写锁，独占。
 */
public class HnswIndex implements VectorIndex {

    /** 初始容量，之后按需翻倍。 */
    private static final int INITIAL_CAPACITY = 1024;

    /** 上层邻接表的初始容量（int 个数）。 */
    private static final int INITIAL_UPPER_STORE = 4096;

    private final int dimension;
    private final HnswParams params;
    private final Random random;
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();

    // ---------- 扁平存储 ----------
    private float[] vectorStore;
    private int[] externalIdStore;
    private byte[] levelStore;
    private int[] layer0Links;
    private short[] layer0Count;
    private int[] upperStore;
    private int[] upperOffset;
    private int upperStoreSize;

    private int capacity;
    private int nodeCount;

    // ---------- 图入口 ----------
    private int entryPoint = -1;
    private int maxLevel = -1;

    /**
     * 每个线程自己的访问标记 —— <b>读路径上唯一的可写状态</b>。
     *
     * <p>最初 {@code visitedStamp} 是索引里的一个共享 {@code int[]}，
     * 所有线程共用一个数组和一个令牌。单线程下这没问题，多线程下却是灾难：
     * 每做一次搜索，每个线程都要往同一片内存写标记，缓存行在核心之间
     * 来回失效（<b>伪共享</b>），线程越多争得越凶。
     *
     * <p>改成线程本地之后，读路径不再写任何共享状态。这不只是省掉争用 ——
     * 它还是「读操作可以无锁并发」这个结论的<b>前提</b>：只要读路径不写共享状态，
     * 理论上就不需要读锁。
     *
     * <p>代价是每个线程各持有一份 {@code int[capacity]}。2 万节点约 80 KB/线程，
     * 几百个线程才会到几十 MB —— 对检索服务来说是划算的交换。
     */
    private final ThreadLocal<VisitedScratch> visitedScratch =
            ThreadLocal.withInitial(VisitedScratch::new);

    /** 线程本地的搜索暂存区。容量按索引容量对齐，索引扩容后由搜索路径重建。 */
    private static final class VisitedScratch {
        int[] visited = new int[0];
        int token;
    }

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
        if (params.mMax0() > Short.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "m 过大，第 0 层邻居数 " + params.mMax0() + " 超出 short 表示范围");
        }
        this.dimension = dimension;
        this.params = params;
        this.random = new Random(seed);
        this.capacity = INITIAL_CAPACITY;
        allocate(capacity);
    }

    // =====================================================================
    // 容量管理
    // =====================================================================

    private void allocate(int newCapacity) {
        this.capacity = newCapacity;
        this.vectorStore = new float[newCapacity * dimension];
        this.externalIdStore = new int[newCapacity];
        this.levelStore = new byte[newCapacity];
        this.layer0Links = new int[newCapacity * params.mMax0()];
        this.layer0Count = new short[newCapacity];
        this.upperOffset = new int[newCapacity];
        Arrays.fill(this.upperOffset, -1);
        this.upperStore = new int[INITIAL_UPPER_STORE];
        this.upperStoreSize = 0;
    }

    private void ensureCapacity(int required) {
        if (required <= capacity) {
            return;
        }
        int newCapacity = Math.max(required, capacity * 2);

        // 步长固定，整块拷贝即可
        float[] newVectors = new float[newCapacity * dimension];
        System.arraycopy(vectorStore, 0, newVectors, 0, nodeCount * dimension);
        vectorStore = newVectors;

        int[] newLayer0 = new int[newCapacity * params.mMax0()];
        System.arraycopy(layer0Links, 0, newLayer0, 0, nodeCount * params.mMax0());
        layer0Links = newLayer0;

        externalIdStore = Arrays.copyOf(externalIdStore, newCapacity);
        levelStore = Arrays.copyOf(levelStore, newCapacity);
        layer0Count = Arrays.copyOf(layer0Count, newCapacity);
        // visitedStamp 已改为线程本地，容器扩容时不必跟着拷贝；
        // 各线程在下次搜索时按 capacity 自行重建。

        int[] newUpperOffset = Arrays.copyOf(upperOffset, newCapacity);
        Arrays.fill(newUpperOffset, capacity, newCapacity, -1);
        upperOffset = newUpperOffset;

        capacity = newCapacity;
    }

    private void ensureUpperStore(int required) {
        if (required <= upperStore.length) {
            return;
        }
        upperStore = Arrays.copyOf(upperStore, Math.max(required, upperStore.length * 2));
    }

    // =====================================================================
    // 邻接表读写（把两种布局的差异收敛在这四个方法里）
    // =====================================================================

    private int neighborCount(int node, int layer) {
        if (layer == 0) {
            return layer0Count[node];
        }
        int offset = upperOffset[node];
        if (offset < 0) {
            return 0;
        }
        return upperStore[offset + (layer - 1) * (params.m() + 1)];
    }

    private int neighborAt(int node, int layer, int index) {
        if (layer == 0) {
            return layer0Links[node * params.mMax0() + index];
        }
        return upperStore[upperOffset[node] + (layer - 1) * (params.m() + 1) + 1 + index];
    }

    private void appendNeighbor(int node, int layer, int count, int target) {
        if (layer == 0) {
            layer0Links[node * params.mMax0() + count] = target;
            layer0Count[node] = (short) (count + 1);
        } else {
            int slot = upperOffset[node] + (layer - 1) * (params.m() + 1);
            upperStore[slot] = count + 1;
            upperStore[slot + 1 + count] = target;
        }
    }

    private void setNeighbors(int node, int layer, int[] neighbors) {
        if (layer == 0) {
            System.arraycopy(neighbors, 0, layer0Links, node * params.mMax0(), neighbors.length);
            layer0Count[node] = (short) neighbors.length;
            return;
        }
        int slot = upperOffset[node] + (layer - 1) * (params.m() + 1);
        upperStore[slot] = neighbors.length;
        System.arraycopy(neighbors, 0, upperStore, slot + 1, neighbors.length);
    }

    // =====================================================================
    // 距离
    // =====================================================================

    /** query 是一段独立缓冲（从偏移量 queryOffset 开始），与扁平存储中的节点比。 */
    private float similarity(float[] query, int queryOffset, int nodeId) {
        // 用 SIMD：距离计算是建图与查询的绝对热点。
        // 512 维下实测 61.3ns vs 标量 326.6ns（见 bench/REPORT.md）。
        return DistanceMetricsSimd.dot(
                query, queryOffset, vectorStore, nodeId * dimension, dimension);
    }

    /** 两个节点之间比。 */
    private float similarityNodes(int nodeA, int nodeB) {
        return DistanceMetricsSimd.dot(
                vectorStore, nodeA * dimension, vectorStore, nodeB * dimension, dimension);
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
        ensureCapacity(nodeCount + 1);

        int nodeId = nodeCount;
        int level = randomLevel();
        int vectorOffset = nodeId * dimension;

        // 向量直接写进扁平存储，就地归一化 —— 不产生任何临时数组
        System.arraycopy(vector, 0, vectorStore, vectorOffset, dimension);
        DistanceMetrics.normalizeInPlace(vectorStore, vectorOffset, dimension);

        externalIdStore[nodeId] = externalId;
        levelStore[nodeId] = (byte) level;
        layer0Count[nodeId] = 0;
        upperOffset[nodeId] = -1;

        // 上层槽位一次性预留：每个 (节点, 层) 占 m+1 个 int（第 0 个存邻居数）
        if (level >= 1) {
            int slotSize = level * (params.m() + 1);
            ensureUpperStore(upperStoreSize + slotSize);
            Arrays.fill(upperStore, upperStoreSize, upperStoreSize + slotSize, 0);
            upperOffset[nodeId] = upperStoreSize;
            upperStoreSize += slotSize;
        }

        nodeCount++;

        // 第一个节点：直接成为入口，没有图可走
        if (entryPoint == -1) {
            entryPoint = nodeId;
            maxLevel = level;
            return;
        }

        // 查询就是刚写进存储的这个节点本身，用偏移量指过去
        float[] query = vectorStore;
        int queryOffset = vectorOffset;

        int current = entryPoint;
        float currentSim = similarity(query, queryOffset, current);

        // ---- 阶段 1：从顶层贪心下降到 level+1 层 ----
        for (int l = maxLevel; l > level; l--) {
            current = greedyDescend(query, queryOffset, current, currentSim, l);
            currentSim = similarity(query, queryOffset, current);
        }

        // ---- 阶段 2：从 min(level, maxLevel) 层逐层下降，边搜边连 ----
        for (int l = Math.min(level, maxLevel); l >= 0; l--) {
            List<SearchResult> candidates =
                    searchLayer(query, queryOffset, current, params.efConstruction(), l);
            int maxLinks = params.maxLinks(l);

            int[] neighbors = selectNeighborsHeuristic(query, queryOffset, candidates, maxLinks);
            setNeighbors(nodeId, l, neighbors);

            // 建立反向边。反向边可能因超出上限被剪枝，这是 HNSW 的正常行为
            for (int neighbor : neighbors) {
                addLink(neighbor, l, nodeId);
            }

            if (!candidates.isEmpty()) {
                current = candidates.get(0).id();
                currentSim = similarity(query, queryOffset, current);
            }
        }

        if (level > maxLevel) {
            maxLevel = level;
            entryPoint = nodeId;
        }
    }

    /** 在指定层贪心走到局部最优。用于上层快速逼近。 */
    private int greedyDescend(float[] query, int queryOffset, int start, float startSim, int layer) {
        int current = start;
        float currentSim = startSim;
        boolean improved = true;
        while (improved) {
            improved = false;
            int count = neighborCount(current, layer);
            for (int i = 0; i < count; i++) {
                int neighbor = neighborAt(current, layer, i);
                float sim = similarity(query, queryOffset, neighbor);
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
        int count = neighborCount(node, layer);
        for (int i = 0; i < count; i++) {
            if (neighborAt(node, layer, i) == target) {
                return; // 边已存在
            }
        }

        int maxLinks = params.maxLinks(layer);
        if (count < maxLinks) {
            appendNeighbor(node, layer, count, target);
            return;
        }

        List<SearchResult> candidates = new ArrayList<>(count + 1);
        for (int i = 0; i < count; i++) {
            int existing = neighborAt(node, layer, i);
            candidates.add(new SearchResult(existing, similarityNodes(node, existing)));
        }
        candidates.add(new SearchResult(target, similarityNodes(node, target)));

        // 剪枝的"查询点"就是这个节点自己
        int[] pruned = selectNeighborsHeuristic(
                vectorStore, node * dimension, candidates, maxLinks);
        setNeighbors(node, layer, pruned);
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

        lock.readLock().lock();
        try {
            if (entryPoint == -1) {
                return List.of();
            }

            // 归一化到临时缓冲，不改动调用方的数组。
            // 这是一次 2KB 的分配，相对微秒级的检索开销可以忽略
            float[] normalized = DistanceMetrics.normalizedCopy(query);

            int current = entryPoint;
            float currentSim = similarity(normalized, 0, current);

            // 上层：ef=1 的贪心下降，只找方向
            for (int l = maxLevel; l > 0; l--) {
                current = greedyDescend(normalized, 0, current, currentSim, l);
                currentSim = similarity(normalized, 0, current);
            }

            // 第 0 层：用 efSearch 的候选池做真正的收敛
            int ef = Math.max(efSearch, k);
            List<SearchResult> candidates = searchLayer(normalized, 0, current, ef, 0);

            int take = Math.min(k, candidates.size());
            List<SearchResult> result = new ArrayList<>(take);
            for (int i = 0; i < take; i++) {
                SearchResult hit = candidates.get(i);
                result.add(new SearchResult(externalIdStore[hit.id()], hit.score()));
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
     * <p>终止条件：弹出来的候选比"已保留结果里最差的"还差，说明前沿已经没有希望。
     *
     * @param entry 起始节点（必须是第 layer 层存在的节点）
     */
    private List<SearchResult> searchLayer(float[] query, int queryOffset,
                                           int entry, int ef, int layer) {
        // 线程本地的访问标记：容量跟着索引容量走，索引扩容后重建一次。
        // 把数组引用存进局部变量，避免循环里反复走 ThreadLocal。
        VisitedScratch scratch = visitedScratch.get();
        if (scratch.visited.length < capacity) {
            scratch.visited = new int[capacity];
            scratch.token = 0;
        }
        int token = nextVisitToken(scratch);
        int[] visited = scratch.visited;

        PriorityQueue<SearchResult> candidates =
                new PriorityQueue<>(Comparator.comparingDouble(SearchResult::score).reversed());
        PriorityQueue<SearchResult> result =
                new PriorityQueue<>(Comparator.comparingDouble(SearchResult::score));

        SearchResult start = new SearchResult(entry, similarity(query, queryOffset, entry));
        visited[entry] = token;
        candidates.offer(start);
        result.offer(start);

        while (!candidates.isEmpty()) {
            SearchResult nearest = candidates.poll();
            SearchResult worstKept = result.peek();

            if (worstKept != null && nearest.score() < worstKept.score()) {
                break;
            }

            int count = neighborCount(nearest.id(), layer);
            for (int i = 0; i < count; i++) {
                int neighbor = neighborAt(nearest.id(), layer, i);
                if (visited[neighbor] == token) {
                    continue;
                }
                visited[neighbor] = token;

                float sim = similarity(query, queryOffset, neighbor);
                SearchResult candidate = new SearchResult(neighbor, sim);
                SearchResult currentWorst = result.peek();

                if (result.size() < ef || currentWorst == null || sim > currentWorst.score()) {
                    candidates.offer(candidate);
                    result.offer(candidate);
                    if (result.size() > ef) {
                        result.poll();
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
     *
     * <p>不足 {@code m} 时用被丢弃候选中最近的补齐（论文的
     * {@code keepPrunedConnections}），保证图上连通度不至于太差。
     *
     * @param query       查询向量所在的数组
     * @param queryOffset 查询向量在该数组中的起始下标
     * @param candidates  候选（id 是内部节点号，score 是与 query 的相似度）
     * @param m           最多选几个
     */
    private int[] selectNeighborsHeuristic(float[] query, int queryOffset,
                                           List<SearchResult> candidates, int m) {
        if (candidates.isEmpty() || m <= 0) {
            return new int[0];
        }

        List<SearchResult> sorted = new ArrayList<>(candidates);
        sorted.sort(Comparator.comparingDouble(SearchResult::score).reversed());

        List<Integer> selected = new ArrayList<>(m);
        List<Integer> discarded = new ArrayList<>();

        for (SearchResult candidate : sorted) {
            if (selected.size() >= m) {
                discarded.add(candidate.id());
                continue;
            }

            boolean addsNewDirection = true;
            for (int chosen : selected) {
                // candidate 与已选邻居的相似度若高于它与 query 的相似度，
                // 说明它已经被已选邻居"覆盖"了，收进来只是冗余
                float simToChosen = similarityNodes(candidate.id(), chosen);
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
     * 取下一个访问令牌（线程本地）。
     *
     * <p>版本号戳的关键在于"不用清空数组"：只要换一个令牌，
     * 上一轮的标记自然就全部失效了。代价是 int 会耗尽 ——
     * 真到 21 亿次搜索时把数组清零并从头开始，实际不可能触发，
     * 但必须处理，否则会变成静默的漏访问。
     */
    private int nextVisitToken(VisitedScratch scratch) {
        if (++scratch.token == Integer.MAX_VALUE) {
            Arrays.fill(scratch.visited, 0);
            scratch.token = 1;
        }
        return scratch.token;
    }

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
        return nodeCount;
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

    /** 指定节点的最高层号。 */
    public int levelOf(int nodeId) {
        lock.readLock().lock();
        try {
            return levelStore[nodeId];
        } finally {
            lock.readLock().unlock();
        }
    }

    /** 全图总边数（含各层，反向边单独计数）。 */
    public long totalLinkCount() {
        lock.readLock().lock();
        try {
            long total = 0;
            for (int node = 0; node < nodeCount; node++) {
                total += layer0Count[node];
            }
            int stride = params.m() + 1;
            for (int node = 0; node < nodeCount; node++) {
                int offset = upperOffset[node];
                if (offset < 0) {
                    continue;
                }
                int levels = levelStore[node];
                for (int l = 1; l <= levels; l++) {
                    total += upperStore[offset + (l - 1) * stride];
                }
            }
            return total;
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * 各层「最高层恰好是本层」的节点数（层高分布）。index 即层号。
     *
     * <p>注意它与 {@link #layerPopulation()} 的区别：本方法数的是
     * 「最高层正好是 l」，而 layerPopulation 数的是「至少达到 l 层」。
     */
    public int[] levelHistogram() {
        lock.readLock().lock();
        try {
            int[] histogram = new int[maxLevel + 1];
            for (int node = 0; node < nodeCount; node++) {
                histogram[levelStore[node]]++;
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
     * 所以 population[0] 恒等于总节点数，且逐层递减。这是金字塔的真实形状。
     */
    public int[] layerPopulation() {
        lock.readLock().lock();
        try {
            int[] population = new int[maxLevel + 1];
            for (int node = 0; node < nodeCount; node++) {
                for (int l = 0; l <= levelStore[node]; l++) {
                    population[l]++;
                }
            }
            return population;
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * 存储占用估算（字节）。
     *
     * <p>这是<b>估算</b>，只统计索引自身的 7 个数组，不含对象头与对齐填充，
     * 也<b>不含线程本地的访问标记</b>（那部分按线程数计，不属于索引本身）。
     *
     * <p>它的价值在于和改造前的对象化布局做对比：那时 10 万节点会产生
     * 约 30 万个对象，现在的对象数是常数级。
     */
    public long estimatedBytes() {
        lock.readLock().lock();
        try {
            return (long) capacity * dimension * Float.BYTES
                    + (long) capacity * Integer.BYTES      // externalIdStore
                    + (long) capacity                       // levelStore
                    + (long) capacity * params.mMax0() * Integer.BYTES
                    + (long) capacity * Short.BYTES         // layer0Count
                    + (long) capacity * Integer.BYTES       // upperOffset
                    + (long) upperStoreSize * Integer.BYTES;
        } finally {
            lock.readLock().unlock();
        }
    }

    /** 当前已分配的节点槽位数（容量），用于观察扩容行为。 */
    public int capacity() {
        return capacity;
    }
}
