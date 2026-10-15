package com.vectorsearch.bench;

import com.vectorsearch.core.SearchResult;
import com.vectorsearch.core.hnsw.HnswIndex;
import com.vectorsearch.core.hnsw.HnswParams;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;

/**
 * HNSW 检索延迟基准。
 *
 * <h2>JMH 的一个关键行为：每个参数组合都跑在全新的 JVM 里</h2>
 *
 * 这是写 JMH 基准时最容易误解的一点：<b>{@code @Fork(1)} 不是"整个基准共用
 * 一个 JVM"，而是"每个参数组合各 fork 一个 JVM"</b>。JMH 这么做是为了
 * 防止不同参数组合之间通过 JIT 状态、堆布局互相污染。
 *
 * <p>后果是：<b>静态字段无法跨参数组合共享</b>。本类的 {@link #CACHE}
 * 只在单个 JVM 内有效，所以 {@code efSearch} 有几个取值，HNSW 图就会被
 * 重建几次。这是实测出来的 —— 一开始以为静态缓存能省掉重复建图，
 * 结果日志里 {@code [HNSW 建图]} 按参数个数原样打印了 4 次。
 *
 * <p>应对办法有两个，本类都用了：
 * <ol>
 *   <li>把 {@code efSearch} 的取值压到 3 个，控制重复建图的次数</li>
 *   <li>完整的 recall/延迟 曲线交给 {@code RecallRunner} ——
 *       它只建一次图、然后扫全部 efSearch，成本低得多。
 *       JMH 只负责在几个点上给出精确的延迟分布</li>
 * </ol>
 *
 * <h2>为什么用 SampleTime 而不是只测平均</h2>
 * 平均值会掩盖长尾，而线上体验由 P99 决定。JMH 的 SampleTime 模式会直接
 * 输出 P50/P90/P99/P99.9，这正是要写进报告的数字。
 *
 * <p>运行（先用小规模验证链路，再跑全量）：
 * <pre>
 *   -p nodeCount=10000    快速验证
 *   -p nodeCount=100000   出正式数据
 * </pre>
 */
@BenchmarkMode({Mode.AverageTime, Mode.SampleTime})
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 2, timeUnit = TimeUnit.SECONDS)
@Fork(value = 1, jvmArgsAppend = {"--add-modules", "jdk.incubator.vector", "-Xmx4g"})
@State(Scope.Benchmark)
public class HnswSearchBenchmark {

    /**
     * 单次 JVM 内的索引缓存，key 为建图参数。
     *
     * <p><b>注意作用范围</b>：JMH 为每个参数组合 fork 一个全新 JVM，
     * 所以这个缓存只在<b>同一个 JVM 内</b>有效，不会跨参数组合省下建图成本。
     * 它的实际作用是在同一组合的多轮预热/测量之间复用索引。
     */
    private static final Map<String, SharedIndex> CACHE = new ConcurrentHashMap<>();

    private record SharedIndex(HnswIndex index, float[][] queries, long buildMillis) {
    }

    @Param({"100000"})
    public int nodeCount;

    @Param({"512"})
    public int dimension;

    @Param({"16"})
    public int m;

    @Param({"200"})
    public int efConstruction;

    /**
     * 查询时的候选池大小 —— 这就是 recall / 延迟 的那根权衡旋钮。
     *
     * <p>刻意只取 3 个值：每多一个取值就多一个 JVM fork、多建一次图
     * （见类注释）。完整的曲线由 RecallRunner 扫描，这里只需要几个
     * 有代表性的点，用来给出精确的延迟分布。
     */
    @Param({"10", "50", "200"})
    public int efSearch;

    /** 每次检索返回的条数。 */
    @Param({"10"})
    public int topK;

    private SharedIndex shared;

    @Setup(Level.Trial)
    public void setup() {
        String key = nodeCount + ":" + dimension + ":" + m + ":" + efConstruction;
        shared = CACHE.computeIfAbsent(key, k -> build());
    }

    private SharedIndex build() {
        int queryCount = 200;
        SyntheticDataset.Dataset dataset = SyntheticDataset.clustered(
                nodeCount, dimension, queryCount,
                SyntheticDataset.recommendedClusters(nodeCount), 42L);

        HnswIndex index = new HnswIndex(dimension, HnswParams.of(m, efConstruction), 42L);
        long start = System.nanoTime();
        for (int i = 0; i < dataset.size(); i++) {
            index.add(i, dataset.vectors()[i]);
        }
        long buildMillis = (System.nanoTime() - start) / 1_000_000L;

        System.out.printf(
                "[HNSW 建图] n=%d dim=%d m=%d efConstruction=%d 簇数=%d -> %d ms（%.1f 万点/秒）%n",
                nodeCount, dimension, m, efConstruction, dataset.clusters(),
                buildMillis, nodeCount / 10_000.0 / (buildMillis / 1000.0));

        return new SharedIndex(index, dataset.queries(), buildMillis);
    }

    @Benchmark
    public List<SearchResult> search() {
        // 轮换查询向量：固定用同一条会被 CPU 缓存和分支预测给出过于乐观的数字
        float[] query = shared.queries()[ThreadLocalRandom.current().nextInt(shared.queries().length)];
        return shared.index().search(query, topK, efSearch);
    }

    public static long buildMillisOf(int nodeCount) {
        for (Map.Entry<String, SharedIndex> entry : CACHE.entrySet()) {
            if (entry.getKey().startsWith(nodeCount + ":")) {
                return entry.getValue().buildMillis();
            }
        }
        return -1;
    }
}
