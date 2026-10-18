package com.vectorsearch.core;

import java.util.ArrayList;
import java.util.List;

/**
 * 文本切分 —— 把上传的长文档切成适合检索的小块。
 *
 * <h2>为什么必须先切分再灌库</h2>
 * 一篇 5000 字的文档如果整篇编码成一个向量，向量的「语义重心」会被摊平：
 * 它既不像开头，也不像中间，更不像结尾。用户问其中某一个细节时，
 * 这个向量的相似度对任何问题都不高 —— 检索效果会明显变差。
 *
 * <p>而且整篇文档命中后，用户还得自己在 5000 字里找答案。
 * 切成 300 字左右的块，就能直接返回相关的那个片段。
 *
 * <h2>为什么先按段落切、再按长度切</h2>
 * 直接按固定长度硬切会把句子拦腰截断，产生「所以这两种方法」这种半句话。
 * 先按空行分段、把相邻段落攒到接近目标长度，边界基本能落在语义完整的地方。
 * 只有单段本身就超长时，才退化成固定窗口 + 重叠。
 *
 * <h2>overlap 的作用</h2>
 * 相邻块之间重叠一小段，避免「答案刚好跨在两个块的边界上」时两边都不完整。
 * 代价是索引体积变大（重叠越多越冗余）。
 *
 * <p>这套逻辑与 Python 侧 {@code eval/build_dataset.py} 的分块保持同一思路，
 * 免得"灌库时怎么切"和"评测时怎么切"不一致 —— 那样评测数字就不可信了。
 */
public final class TextSplitter {

    private TextSplitter() {
    }

    /**
     * @param text      原文
     * @param chunkSize 每块的目标字符数
     * @param overlap   超长段落硬切时的重叠字符数
     */
    public static List<String> split(String text, int chunkSize, int overlap) {
        List<String> chunks = new ArrayList<>();
        if (text == null || text.isBlank()) {
            return chunks;
        }
        int size = Math.max(64, chunkSize);
        int overlapChars = Math.max(0, Math.min(overlap, size / 2));

        String normalized = text.replace("\r\n", "\n").replace('\r', '\n').trim();
        StringBuilder buffer = new StringBuilder();

        for (String paragraph : normalized.split("\n\\s*\n")) {
            String current = paragraph.trim();
            if (current.isEmpty()) {
                continue;
            }

            if (current.length() > size) {
                // 单段就超长：先把已攒的发出去，再对这一段做窗口切分
                flush(buffer, chunks);
                chunks.addAll(sliceLongParagraph(current, size, overlapChars));
                continue;
            }

            if (buffer.length() + current.length() + 1 > size) {
                flush(buffer, chunks);
            }
            if (buffer.length() > 0) {
                buffer.append('\n');
            }
            buffer.append(current);
        }

        flush(buffer, chunks);
        return chunks;
    }

    private static void flush(StringBuilder buffer, List<String> chunks) {
        if (buffer.length() > 0) {
            String chunk = buffer.toString().trim();
            if (!chunk.isEmpty()) {
                chunks.add(chunk);
            }
            buffer.setLength(0);
        }
    }

    private static List<String> sliceLongParagraph(String paragraph, int size, int overlapChars) {
        List<String> pieces = new ArrayList<>();
        int step = Math.max(1, size - overlapChars);
        int start = 0;
        while (start < paragraph.length()) {
            int end = Math.min(start + size, paragraph.length());
            String piece = paragraph.substring(start, end).trim();
            if (!piece.isEmpty()) {
                pieces.add(piece);
            }
            if (end >= paragraph.length()) {
                break;
            }
            start += step;
        }
        return pieces;
    }
}
