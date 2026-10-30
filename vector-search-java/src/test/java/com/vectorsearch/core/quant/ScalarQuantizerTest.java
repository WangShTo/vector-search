package com.vectorsearch.core.quant;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ScalarQuantizerTest {

    /** 归一化 512 维向量的典型刻度：满量程 = 4/√512 ≈ 0.177，刻度 ≈ 718。 */
    private static final float SCALE_512 = ScalarQuantizer.recommendedScale(512);

    @Test
    void 刻度随维度变化而不是固定常数() {
        // 维度越高，单位向量单个分量越小，刻度必须跟着放大，否则量化格被浪费
        float scale512 = ScalarQuantizer.recommendedScale(512);
        float scale64 = ScalarQuantizer.recommendedScale(64);
        float scale2 = ScalarQuantizer.recommendedScale(2);

        assertTrue(scale512 > scale64, "512 维的刻度应大于 64 维: " + scale512 + " vs " + scale64);
        assertTrue(scale64 > 127, "64 维也应用比固定 127 更细的刻度: " + scale64);
        // 低维时 4/√d 会超过 1，满量程退回 1，刻度就是 127
        assertEquals(127f, scale2, 1e-6f);
    }

    @Test
    void 编码解码往返误差不超过半个量化格() {
        Random random = new Random(7L);
        double worst = 0;

        for (int i = 0; i < 10_000; i++) {
            // 单位向量在 512 维下的分量幅值约 4/√512
            float x = (float) ((random.nextDouble() * 2 - 1) * 4.0 / Math.sqrt(512));
            float roundTrip = ScalarQuantizer.decode(ScalarQuantizer.encode(x, SCALE_512), SCALE_512);
            worst = Math.max(worst, Math.abs(x - roundTrip));
        }

        assertTrue(worst <= ScalarQuantizer.maxAbsoluteError(SCALE_512) + 1e-9,
                "最大往返误差 " + worst + " 超过理论上界 " + ScalarQuantizer.maxAbsoluteError(SCALE_512));
    }

    @Test
    void 超量程被钳住而不是溢出() {
        // 用 127 而不是 128：正负两侧对称，+满量程与 -满量程都能表示
        assertEquals(127, ScalarQuantizer.encode(1.0f, SCALE_512));
        assertEquals(-127, ScalarQuantizer.encode(-1.0f, SCALE_512));
        assertEquals(127, ScalarQuantizer.encode(10f, SCALE_512), "远超满量程应钳到 127");
        assertEquals(-127, ScalarQuantizer.encode(-10f, SCALE_512), "远低于负满量程应钳到 -127");
        assertEquals(0, ScalarQuantizer.encode(0f, SCALE_512));
    }

    @Test
    void 极小值被量化为零而不是被放大() {
        assertEquals(0, ScalarQuantizer.encode(0.0001f, SCALE_512));
        assertEquals(0f, ScalarQuantizer.decode((byte) 0, SCALE_512), 0f);
    }

    @Test
    void 整段编解码保持一致() {
        Random random = new Random(11L);
        int length = 512;
        float[] src = new float[length];
        for (int i = 0; i < length; i++) {
            src[i] = (float) ((random.nextDouble() * 2 - 1) * 4.0 / Math.sqrt(512));
        }

        byte[] packed = new byte[length];
        ScalarQuantizer.encode(src, 0, packed, 0, length, SCALE_512);

        float[] restored = new float[length];
        ScalarQuantizer.decode(packed, 0, restored, 0, length, SCALE_512);

        for (int i = 0; i < length; i++) {
            assertEquals(src[i], restored[i],
                    (float) ScalarQuantizer.maxAbsoluteError(SCALE_512) + 1e-9f);
        }
    }

    @Test
    void 重复量化是幂等的() {
        // 压实会「读出 float → 重新写入」。若量化不幂等，压几次就掉一次精度
        Random random = new Random(13L);
        for (int i = 0; i < 1_000; i++) {
            float x = (float) ((random.nextDouble() * 2 - 1) * 4.0 / Math.sqrt(512));
            byte once = ScalarQuantizer.encode(x, SCALE_512);
            byte twice = ScalarQuantizer.encode(ScalarQuantizer.decode(once, SCALE_512), SCALE_512);
            assertEquals(once, twice, "解码后再编码必须得到同一个值");
        }
    }

    @Test
    void 代号解析与每向量字节数() {
        assertEquals(Quantization.NONE, Quantization.fromCode("f32"));
        assertEquals(Quantization.SQ8, Quantization.fromCode("SQ8"));
        assertEquals(Quantization.NONE, Quantization.fromCode(""));
        assertEquals(Quantization.NONE, Quantization.fromCode(null));
        assertEquals(512, Quantization.SQ8.bytesPerVector(512));
        assertEquals(2048, Quantization.NONE.bytesPerVector(512));
    }
}
