package com.vectorsearch.core.bm25;

import java.util.ArrayList;
import java.util.List;

/**
 * BM25 用的分词器 —— 中英混合，<b>不依赖任何词典或外部分词库</b>。
 *
 * <h2>为什么中文不能按空格切</h2>
 * 英文天然有空格分隔，切词几乎是免费的。中文没有分隔符，
 * 一句话「向量检索的原理」按字切开是「向」「量」「检」「索」……
 * 单字信息量太低，噪声极大。
 *
 * <h2>为什么选二元组（bigram）</h2>
 * 正经做法是用分词词典（jieba、HanLP），但那要引入词典文件、还要处理
 * 未登录词与领域词。这里用一个更简单且效果不错的方案：<b>相邻两字成一组</b>。
 *
 * <pre>
 *   向量检索  ->  向量 / 量检 / 检索
 * </pre>
 *
 * <p>二元组的两个好处：
 * <ol>
 *   <li>不需要任何词典，任何领域文本都能处理</li>
 *   <li>「检索」这个词对应的二元组就是它自己，所以词级匹配能力还在</li>
 * </ol>
 *
 * <p>代价是索引体积变大（每个汉字参与两个二元组），以及会切出一些
 * 跨词边界的无效二元组（比如「的检」）。实测影响不大 ——
 * 无效二元组在 IDF 计算里自然会被压权重，因为它在很多文档里都出现。
 *
 * <h2>中英混合怎么处理</h2>
 * 逐字符扫描，按字符类型分流：
 * <ul>
 *   <li>ASCII 字母/数字 → 累积成一个单词（{@code efSearch}、{@code HNSW}），转小写</li>
 *   <li>CJK 字符 → 与上一个 CJK 字符组成二元组</li>
 *   <li>其它（标点、空白）→ 切断当前单词与 CJK 链</li>
 * </ul>
 *
 * <h2>这里踩过一个很隐蔽的坑</h2>
 * 最初在 ASCII 分支里<b>每读一个字符就调一次 flush</b>，想表达"遇到非 ASCII 时收尾"。
 * 结果 flush 会把缓冲区清空，单词永远长不到 2 个字符 ——
 * <b>所有英文标识符都被丢掉了</b>，而中文部分完全正常。
 *
 * <p>症状很迷惑：中文检索照常工作，只有 {@code efSearch}、{@code HNSW}
 * 这类词查不到，而且不抛任何异常。这正是 BM25 那一路最该擅长的场景。
 * 正确的写法是：ASCII 分支只负责累积，收尾交给"遇到别的字符时"和"字符串结束"。
 */
public final class Bm25Tokenizer {

    /** 短于这个长度的 ASCII 单词直接丢弃（单字母噪声太大）。 */
    private static final int MIN_WORD_LENGTH = 2;

    private Bm25Tokenizer() {
    }

    /**
     * 把一段文本切成 token 列表。
     *
     * <p>返回的列表可能包含重复项 —— 调用方需要自己统计词频，
     * 因为 BM25 的 tf 是"出现了几次"而不是"有没有出现"。
     */
    public static List<String> tokenize(String text) {
        List<String> tokens = new ArrayList<>();
        if (text == null || text.isEmpty()) {
            return tokens;
        }

        StringBuilder word = new StringBuilder();
        int previousCjk = -1;
        int cjkRunLength = 0;

        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);

            if (isAsciiLetterOrDigit(c)) {
                // 只累积，不在这里 flush —— 见类注释里那个坑
                word.append(Character.toLowerCase(c));
                // 英文出现意味着前面的中文链结束
                flushCjkRun(tokens, previousCjk, cjkRunLength);
                previousCjk = -1;
                cjkRunLength = 0;
            } else if (isCjk(c)) {
                flushWord(word, tokens);
                if (previousCjk >= 0) {
                    // 与上一个汉字组成二元组
                    tokens.add(new String(new char[]{(char) previousCjk, c}));
                    cjkRunLength++;
                } else {
                    cjkRunLength = 1;
                }
                previousCjk = c;
            } else {
                // 标点、空白、全角符号：两边都要收尾
                flushWord(word, tokens);
                flushCjkRun(tokens, previousCjk, cjkRunLength);
                previousCjk = -1;
                cjkRunLength = 0;
            }
        }

        flushWord(word, tokens);
        flushCjkRun(tokens, previousCjk, cjkRunLength);
        return tokens;
    }

    /**
     * 收尾中文链。
     *
     * <p>只有<b>孤立单个汉字</b>才单独成词。多字词已经由二元组覆盖了，
     * 再把最后一个字单独吐出来只是噪声 —— 而且会造成"同一个词在不同位置
     * 切法不一致"的怪现象（词中间的汉字不出单字，词尾的却出）。
     */
    private static void flushCjkRun(List<String> tokens, int previousCjk, int runLength) {
        if (previousCjk >= 0 && runLength == 1) {
            tokens.add(String.valueOf((char) previousCjk));
        }
    }

    private static void flushWord(StringBuilder word, List<String> tokens) {
        if (word.length() >= MIN_WORD_LENGTH) {
            tokens.add(word.toString());
        }
        word.setLength(0);
    }

    private static boolean isAsciiLetterOrDigit(char c) {
        return (c >= 'a' && c <= 'z')
                || (c >= 'A' && c <= 'Z')
                || (c >= '0' && c <= '9');
    }

    /**
     * 是否是中日韩统一表意文字。
     *
     * <p>覆盖基本区与扩展 A 区。生僻字（扩展 B 及以上）在这些语料里几乎不会出现，
     * 为了不让字符判断变成一大串区间，这里只到扩展 A。
     */
    private static boolean isCjk(char c) {
        return (c >= 0x4E00 && c <= 0x9FFF)   // CJK 统一表意文字（基本区）
                || (c >= 0x3400 && c <= 0x4DBF); // 扩展 A
    }
}
