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
import org.openjdk.jmh.annotations.Threads;
import org.openjdk.jmh.annotations.Warmup;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;

/**
 * 并发检索的吞吐基准 —— 用来回答「你的索引在多线程下还行不行」。
 *
 * <h2>为什么要有这个基准</h2>
 * 前面所有数字都是<b>单线程</b>的。而检索服务的真实负载是「离线建好索引、
 * 在线多线程查询」，也就是<b>读多写零</b>。这时候限制吞吐的通常不是算法，
 * 而是读路径上的共享状态与锁。
 *
 * <p>基准测的是<b>吞吐（ops/s）而不是延迟</b>：单线程延迟再低，
 * 如果加线程之后总吞吐不涨，说明瓶颈在锁争用上。所以这里用
 * {@link Mode#Throughput}，并给出 1 / 4 / 8 / 16 线程的扩展曲线。
 *
 * <h2>怎么看结果</h2>
 * 关键指标不是吞吐绝对值，而是<b>扩展效率</b>：
 *
 * <pre>
 *   扩展效率 = 实际吞吐(T 线程) / (T × 吞吐(1 线程))
 * </pre>
 *
 * 理想值 1.0。明显低于 1 就说明有争用：线程越多，花在等锁上的时间越多，
 * 加机器反而更亏。这条曲线比任何单点数字都更能说明并发质量。
 *
 * <h2>为什么用 2 万条而不是 10 万条</h2>
 * 关注的是<b>扩展性</b>而不是绝对性能，规模小不影响结论；
 * 而 JMH 为每个 {@code @Threads} 取值 fork 一个独立 JVM，也就是每个取值
 * 都要重新建一次图。10 万条建图约 2 分钟，四个取值就是 8 分钟纯等待；
 * 2 万条约 25 秒，四个取值 100 秒，性价比高得多。
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 2, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 3, time = 2, timeUnit = TimeUnit.SECONDS)
@Fork(value = 1, jvmArgsAppend = {"--add-modules", "jdk.incubator.vector", "-Xmx4g"})
@State(Scope.Benchmark)
public class ConcurrentSearchBenchmark {

    private static final Map<String, SharedIndex> CACHE = new ConcurrentHashMap<>();

    private record SharedIndex(HnswIndex index, float[][] queries) {
    }

    @Param({"20000"})
    public int nodeCount;

    @Param({"512"})
    public int dimension;

    @Param({"16"})
    public int m;

    @Param({"200"})
    public int efConstruction;

    @Param({"50"})
    public int efSearch;

    @Param({"10"})
    public int topK;

    private SharedIndex shared;

    @Setup(Level.Trial)
    public void setup() {
        String key = nodeCount + ":" + dimension + ":" + m + ":" + efConstruction;
        shared = CACHE.computeIfAbsent(key, k -> build());
    }

    private SharedIndex build() {
        SyntheticDataset.Dataset dataset = SyntheticDataset.clustered(
                nodeCount, dimension, 512,
                SyntheticDataset.recommendedClusters(nodeCount), 42L);

        HnswIndex index = new HnswIndex(dimension, HnswParams.of(m, efConstruction), 42L);
        long start = System.nanoTime();
        for (int i = 0; i < dataset.size(); i++) {
            index.add(i, dataset.vectors()[i]);
        }
        long buildMillis = (System.nanoTime() - start) / 1_000_000L;
        System.out.printf("[并发基准建图] n=%d dim=%d -> %d ms%n",
                nodeCount, dimension, buildMillis);
        return new SharedIndex(index, dataset.queries());
    }

    /** 每次检索随机取一条查询向量，避免所有线程打在同一条上。 */
    private List<SearchResult> searchOnce() {
        float[] query = shared.queries()[ThreadLocalRandom.current().nextInt(shared.queries().length)];
        return shared.index().search(query, topK, efSearch);
    }

    @Benchmark
    @Threads(1)
    public List<SearchResult> searchWith1Thread() {
        return searchOnce();
    }

    @Benchmark
    @Threads(4)
    public List<SearchResult> searchWith4Threads() {
        return searchOnce();
    }

    @Benchmark
    @Threads(8)
    public List<SearchResult> searchWith8Threads() {
        return searchOnce();
    }

    @Benchmark
    @Threads(16)
    public List<SearchResult> searchWith16Threads() {
        return searchOnce();
    }
}
