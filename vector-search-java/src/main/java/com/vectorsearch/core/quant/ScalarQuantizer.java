package com.vectorsearch.core.quant;

/**
 * 8 位对称标量量化（SQ8）。
 *
 * <h2>为什么不需要每条向量存 min/max</h2>
 * 常见做法是每条向量存自己的 min/max 再仿射映射到 [0, 255]，代价是每条向量
 * 多 8 字节元数据，解码时还要先读它。本项目的向量入库前都做了 L2 归一化，
 * 所以<b>每个分量的上界是已知的</b>，一个全局常数就够了。
 *
 * <h2>满量程取多少：这里踩过一次真实的坑</h2>
 * 第一版取的是 {@code scale = 127}，理由是「归一化后每个分量必然落在 [-1, 1]」。
 * 这句话本身没错，但它<b>不是有效上界</b>：512 维单位向量的分量服从 σ ≈ 1/√512 ≈ 0.044
 * 的分布，实际只占满量程的 4% 左右。<b>量化格白白浪费了 96%</b>，
 * 相对误差从 0.4% 恶化到相当于「每个分量只剩 2~3 个有效比特」。
 *
 * <p>实测后果很直观（10 万点、512 维、efSearch=50）：recall@10 从 1.0000 掉到 0.6567，
 * 建图时间还因为图退化变长 2.3 倍。改成按维度取满量程之后 recall 回到 0.99 以上。
 *
 * <p>所以这里的满量程不是一个常数，而是随维度算的：归一化向量的分量 RMS 是
 * {@code 1/√d}，取 {@code 4/√d} 作为设计满量程（约 4σ，正常数据几乎不会被钳）。
 *
 * <h2>编码 / 解码</h2>
 * <pre>
 *   编码：q = clamp(round(x * scale), -127, 127)
 *   解码：x ≈ q / scale
 * </pre>
 * 用 127 而不是 128 是为了保持正负对称：若用 128，{@code -1.0} 能精确表示而
 * {@code 1.0} 会溢出到 128 再被钳成 127，两侧精度不一致。
 *
 * <p>这里刻意做成纯函数、无状态：量化逻辑能独立测试，也不需要跟着索引的生命周期走。
 * 唯一的可变部分是 {@code scale}，它跟着索引走、并且写进快照 —— 否则读回来的
 * 索引会用错误的刻度解码，结果安静地全错。
 */
public final class ScalarQuantizer {

    /** 最大量化值。127 而非 128，保持正负对称。 */
    public static final int MAX_CODE = 127;

    /** v2 及更早的快照用的固定刻度（分量按 [-1,1] 理解），只为兼容读取。 */
    public static final float LEGACY_SCALE = 127f;

    private ScalarQuantizer() {
    }

    /**
     * 按维度推荐满量程对应的刻度。
     *
     * <p>归一化向量各分量 RMS 为 {@code 1/√d}，取 4 倍 RMS 作为设计满量程；
     * 维度很低时 {@code 4/√d} 会超过 1，此时退回 1（分量上界）。
     */
    public static float recommendedScale(int dimension) {
        double bound = Math.min(1.0, 4.0 / Math.sqrt(Math.max(1, dimension)));
        return (float) (MAX_CODE / bound);
    }

    /** 编码单个分量。 */
    public static byte encode(float x, float scale) {
        int q = Math.round(x * scale);
        if (q > MAX_CODE) {
            q = MAX_CODE;
        } else if (q < -MAX_CODE) {
            q = -MAX_CODE;
        }
        return (byte) q;
    }

    /** 解码单个分量。 */
    public static float decode(byte q, float scale) {
        return q / scale;
    }

    /** 整段编码：从 {@code src} 的 {@code srcOffset} 开始，写到 {@code dst} 的 {@code dstOffset}。 */
    public static void encode(float[] src, int srcOffset, byte[] dst, int dstOffset, int length, float scale) {
        for (int i = 0; i < length; i++) {
            dst[dstOffset + i] = encode(src[srcOffset + i], scale);
        }
    }

    /** 整段解码。 */
    public static void decode(byte[] src, int srcOffset, float[] dst, int dstOffset, int length, float scale) {
        for (int i = 0; i < length; i++) {
            dst[dstOffset + i] = src[srcOffset + i] / scale;
        }
    }

    /**
     * 单个分量的理论最大舍入误差：半个量化格。
     *
     * <p>用来在报告里给出一个「不该超过」的上界 —— 实测若远超它，说明问题
     * 不在量化本身，而在别处（例如归一化没做、或者距离函数写错）。
     */
    public static double maxAbsoluteError(float scale) {
        return 0.5 / scale;
    }
}
