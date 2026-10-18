package com.vectorsearch.core;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * 基于 JDK Vector API 的 SIMD 距离计算。
 *
 * <p>标量版本一次处理 1 个 float，这里一次处理 {@code SPECIES.length()} 个
 * （AVX2 上是 8，AVX-512 上是 16）。配合 {@code fma}（乘加融合）能把
 * 距离计算的循环体压成一条指令。
 *
 * <p><b>为什么这是加分项</b>：这不是背八股能背出来的东西，需要理解
 * CPU 的 SIMD 执行模型、内存对齐、以及尾部元素的处理。JMH 跑出来的
 * 加速比是可直接写进简历的真实数字。
 *
 * <p><b>编译与运行要求</b>：Java 17 中该模块仍是 incubator，必须加
 * {@code --add-modules jdk.incubator.vector}。pom.xml 里已配置好
 * （编译、测试、spring-boot:run 三处）。
 */
public final class DistanceMetricsSimd {

    /**
     * SPECIES_PREFERRED 让 JVM 在运行时选当前 CPU 支持的最宽向量宽度。
     * 写死 SPECIES_256 会在只有 SSE 的老机器上退化，不推荐。
     */
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private DistanceMetricsSimd() {
    }

    /** 当前运行平台实际使用的向量 lane 数（8 = AVX2，16 = AVX-512）。 */
    public static int laneWidth() {
        return SPECIES.length();
    }

    /** 内积。两个数组长度必须相等。 */
    public static float dot(float[] a, float[] b) {
        checkSameLength(a, b);
        return dot(a, 0, b, 0, a.length);
    }

    /**
     * 带偏移与长度的内积 —— <b>扁平存储的关键</b>。
     *
     * <p>把所有向量拼成一个大数组后，第 i 条向量的数据位于
     * {@code a[i * dimension]} 开始处。有了这个方法，就不必再把每条向量
     * 复制成独立的 {@code float[]}：直接传入整个大数组和偏移量即可。
     *
     * <p>这正是扁平化布局能与 SIMD 共存的原因 —— 数据连续，向量化加载
     * 反而比原来追着 {@code List<float[]>} 的引用跳转更友好。
     *
     * @param a       数组 A
     * @param aOffset A 的起始下标
     * @param b       数组 B
     * @param bOffset B 的起始下标
     * @param length  参与计算的元素个数
     */
    public static float dot(float[] a, int aOffset, float[] b, int bOffset, int length) {
        int i = 0;
        int bound = SPECIES.loopBound(length);
        FloatVector acc = FloatVector.zero(SPECIES);

        for (; i < bound; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, aOffset + i);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, bOffset + i);
            acc = va.fma(vb, acc);
        }

        float sum = acc.reduceLanes(VectorOperators.ADD);
        // 尾部（维度不是 lane 整数倍时才会进入）
        for (; i < length; i++) {
            sum += a[aOffset + i] * b[bOffset + i];
        }
        return sum;
    }

    /** 平方欧氏距离。两个数组长度必须相等。 */
    public static float l2Squared(float[] a, float[] b) {
        checkSameLength(a, b);
        return l2Squared(a, 0, b, 0, a.length);
    }

    /** 带偏移与长度的平方欧氏距离。语义同 {@link #dot(float[], int, float[], int, int)}。 */
    public static float l2Squared(float[] a, int aOffset, float[] b, int bOffset, int length) {
        int i = 0;
        int bound = SPECIES.loopBound(length);
        FloatVector acc = FloatVector.zero(SPECIES);

        for (; i < bound; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, aOffset + i);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, bOffset + i);
            FloatVector diff = va.sub(vb);
            acc = diff.fma(diff, acc);
        }

        float sum = acc.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            float d = a[aOffset + i] - b[bOffset + i];
            sum += d * d;
        }
        return sum;
    }

    private static void checkSameLength(float[] a, float[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException(
                    "向量维度不一致: " + a.length + " vs " + b.length);
        }
    }
}
