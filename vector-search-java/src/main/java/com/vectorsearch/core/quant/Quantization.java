package com.vectorsearch.core.quant;

/**
 * 向量存储精度。
 *
 * <p>这是本项目里<b>唯一一个会影响「内存 / 延迟 / 召回」三角</b>的开关，
 * 所以它被做成一等公民而不是一个布尔标志 —— 调用方看到这个名字，
 * 就知道自己在权衡什么。
 */
public enum Quantization {

    /** 每条向量原样存 float32：最快、最准、最占内存。 */
    NONE("f32"),

    /**
     * 每条向量压到 8 位整数（SQ8）：内存降到 1/4，代价是量化噪声。
     *
     * <p>向量入库前已做 L2 归一化，每个分量落在 [-1, 1]，所以不需要为
     * 每条向量单独存 min/max —— 一个固定比例因子就够了。
     *
     * @see ScalarQuantizer
     */
    SQ8("sq8");

    private final String code;

    Quantization(String code) {
        this.code = code;
    }

    /** 用于指标与快照的短代号。 */
    public String code() {
        return code;
    }

    /** 每条向量的字节数。 */
    public int bytesPerVector(int dimension) {
        return this == SQ8 ? dimension : dimension * Float.BYTES;
    }

    public static Quantization fromCode(String code) {
        if (code == null || code.isBlank()) {
            return NONE;
        }
        for (Quantization q : values()) {
            if (q.code.equalsIgnoreCase(code.trim())) {
                return q;
            }
        }
        throw new IllegalArgumentException("未知的量化模式: " + code + "（支持 f32 / sq8）");
    }
}
