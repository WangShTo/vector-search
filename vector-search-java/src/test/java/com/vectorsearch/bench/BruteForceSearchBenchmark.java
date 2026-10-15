package com.vectorsearch.bench;

import com.vectorsearch.core.SearchResult;
import com.vectorsearch.core.bruteforce.BruteForceIndex;
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
 * 暴力检索延迟基准 —— <b>对照组</b>。
 *
 * <p>这个类存在的全部意义，是给 HNSW 的"快"提供一个可比的基准。
 * 没有它，"HNSW 把延迟降到 0.3ms" 这句话是空的 —— 降到 0.3ms 是从多少降下来的？
 *
 * <h2>为什么测量轮次比 HNSW 少</h2>
 * 暴力检索是 O(n·d)：10 万条 512 维向量，一次查询就是 5000 万次乘加，
 * 单次耗时在几十毫秒量级。JMH 每轮固定时间，单次操作越慢、采样数越少。
 * 用 SampleTime 求 P99 需要足够多的样本，所以这里刻意压短测量时间，
 * 避免为了统计意义而把整个基准拖到十几分钟。
 *
 * <p><b>这也正是引入 HNSW 的理由本身</b>：O(n·d) 的复杂度在真实规模上不可用。
 */
@BenchmarkMode({Mode.AverageTime, Mode.SampleTime})
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 2, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 3, time = 2, timeUnit = TimeUnit.SECONDS)
@Fork(value = 1, jvmArgsAppend = {"--add-modules", "jdk.incubator.vector", "-Xmx4g"})
@State(Scope.Benchmark)
public class BruteForceSearchBenchmark {

    private static final Map<String, SharedIndex> CACHE = new ConcurrentHashMap<>();

    private record SharedIndex(BruteForceIndex index, float[][] queries, long buildMillis) {
    }

    /**
     * 与 HNSW 基准保持同样的默认规模，否则两者的数字不可比。
     * 小规模可以用 {@code -p nodeCount=10000} 快速验证链路。
     */
    @Param({"100000"})
    public int nodeCount;

    @Param({"512"})
    public int dimension;

    @Param({"10"})
    public int topK;

    private SharedIndex shared;

    @Setup(Level.Trial)
    public void setup() {
        String key = nodeCount + ":" + dimension;
        shared = CACHE.computeIfAbsent(key, k -> build());
    }

    private SharedIndex build() {
        SyntheticDataset.Dataset dataset = SyntheticDataset.clustered(
                nodeCount, dimension, 200,
                SyntheticDataset.recommendedClusters(nodeCount), 42L);

        BruteForceIndex index = new BruteForceIndex(dimension);
        long start = System.nanoTime();
        for (int i = 0; i < dataset.size(); i++) {
            index.add(i, dataset.vectors()[i]);
        }
        long buildMillis = (System.nanoTime() - start) / 1_000_000L;

        System.out.printf("[暴力检索建库] n=%d dim=%d -> %d ms%n",
                nodeCount, dimension, buildMillis);

        return new SharedIndex(index, dataset.queries(), buildMillis);
    }

    @Benchmark
    public List<SearchResult> search() {
        float[] query = shared.queries()[ThreadLocalRandom.current().nextInt(shared.queries().length)];
        // efSearch 参数对暴力检索无意义 —— 它本来就遍历全部，这正是它的定义
        return shared.index().search(query, topK, 0);
    }
}
