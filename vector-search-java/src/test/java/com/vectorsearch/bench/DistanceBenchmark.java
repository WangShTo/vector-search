package com.vectorsearch.bench;

import com.vectorsearch.core.DistanceMetrics;
import com.vectorsearch.core.DistanceMetricsSimd;
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

import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * 距离计算的微基准：标量 vs SIMD。
 *
 * <p>这是整个项目里<b>最干净</b>的一个对照实验 —— 两个方法输入输出完全相同，
 * 唯一差别就是循环体的向量化。所以测出来的加速比可以直接归因于 SIMD。
 *
 * <p><b>为什么必须用 JMH</b>：
 * <ul>
 *   <li>JIT 会把短循环优化到面目全非，预热轮次不可省</li>
 *   <li>死代码消除（DCE）会把"算了但没用"的结果直接删掉 ——
 *       所以方法必须<b>返回</b>结果，JMH 会用 Blackhole 消费它</li>
 *   <li>循环里重复算同一对向量，分支预测和缓存都会给出乐观到失真的数字，
 *       JMH 的多次迭代 + fork 新 JVM 能压掉一部分这类偏差</li>
 * </ul>
 *
 * <p>运行：
 * <pre>
 *   mvn -q test-compile
 *   java --add-modules jdk.incubator.vector -cp ... org.openjdk.jmh.Main "DistanceBenchmark"
 * </pre>
 */
@BenchmarkMode({Mode.AverageTime, Mode.SampleTime})
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 3, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Fork(value = 2, jvmArgsAppend = {"--add-modules", "jdk.incubator.vector"})
@State(Scope.Thread)
public class DistanceBenchmark {

    /** 512 是 BGE-small-zh 的维度（本项目实际使用），768 与 1024 是更大模型的维度。 */
    @Param({"128", "512", "768"})
    public int dimension;

    private float[] a;
    private float[] b;

    @Setup(Level.Trial)
    public void setup() {
        Random random = new Random(42);
        a = randomVector(dimension, random);
        b = randomVector(dimension, random);
        // 归一化后内积即余弦相似度，与实际使用场景一致
        DistanceMetrics.normalizeInPlace(a);
        DistanceMetrics.normalizeInPlace(b);
    }

    @Benchmark
    public float scalarDot() {
        return DistanceMetrics.dot(a, b);
    }

    @Benchmark
    public float simdDot() {
        return DistanceMetricsSimd.dot(a, b);
    }

    @Benchmark
    public float scalarL2Squared() {
        return DistanceMetrics.l2Squared(a, b);
    }

    @Benchmark
    public float simdL2Squared() {
        return DistanceMetricsSimd.l2Squared(a, b);
    }

    private static float[] randomVector(int dimension, Random random) {
        float[] v = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            v[i] = random.nextFloat() * 2f - 1f;
        }
        return v;
    }
}
