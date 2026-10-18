package com.vectorsearch.core.bm25;

import com.vectorsearch.core.SearchResult;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * BM25 稀疏检索 —— 从零实现，倒排索引 + 词频饱和 + 文档长度归一化。
 *
 * <h2>为什么向量检索之外还要 BM25</h2>
 * 两者擅长的是<b>不同</b>的东西，这正是混合检索存在的理由：
 *
 * <table border="1">
 *   <caption>互补性</caption>
 *   <tr><th></th><th>向量检索（稠密）</th><th>BM25（稀疏）</th></tr>
 *   <tr><td>擅长</td><td>语义泛化："内存回收" ≈ "垃圾回收"</td>
 *       <td>精确匹配：标识符、型号、错误码、人名</td></tr>
 *   <tr><td>不擅长</td><td>没见过的新词、专有名词；字面精确匹配</td>
 *       <td>同义替换、换一种说法</td></tr>
 *   <tr><td>失败模式</td><td>查询里的罕见词被"平均"掉</td>
 *       <td>用户换了说法就完全找不到</td></tr>
 * </table>
 *
 * <p>所以生产上的搜索引擎几乎都是两者混用 —— Elasticsearch、Qdrant、
 * Weaviate 都提供 hybrid 模式。
 *
 * <h2>BM25 公式</h2>
 * <pre>
 *   score(q, d) = Σ_{t∈q} IDF(t) · tf·(k1+1) / (tf + k1·(1 - b + b·dl/avgdl))
 *
 *   IDF(t) = ln(1 + (N - df + 0.5) / (df + 0.5))
 * </pre>
 *
 * <p>三个部分各自解决一个问题：
 * <ul>
 *   <li><b>IDF</b>：越罕见的词权重越高。「的」出现在所有文档里，几乎没贡献；
 *       「efSearch」只出现在少数文档里，命中它就该大幅加分。</li>
 *   <li><b>tf 饱和（k1）</b>：一个词出现 100 次不该比出现 10 次重要 10 倍。
 *       这个分式随 tf 增长但收敛到 (k1+1)，避免长文档靠堆词刷分。</li>
 *   <li><b>长度归一化（b）</b>：长文档天然包含更多词，命中概率更高，
 *       要按 dl/avgdl 打折。</li>
 * </ul>
 *
 * <p>参数取业界常用的 k1=1.2、b=0.75，不在这里调参 ——
 * 它们对最终效果的影响远小于分词质量。
 *
 * <h2>并发模型</h2>
 * 建索引加写锁、检索加读锁，和向量索引保持一致。
 * <b>打分累加器是查询内的局部变量</b>，不是共享数组 ——
 * 否则又会变成读路径写共享状态，重演线程本地的那个问题。
 */
public class Bm25Index {

    /** 词频饱和参数。越大越接近"线性 tf"，越小越早饱和。 */
    private static final double K1 = 1.2;

    /** 文档长度归一化强度。0 = 不归一化，1 = 完全归一化。 */
    private static final double B = 0.75;

    private static final int INITIAL_DOC_CAPACITY = 256;

    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();

    /** term -> 倒排表。 */
    private final Map<String, PostingList> postings = new HashMap<>();

    // ---------- 扁平存储：内部 doc 号 -> 业务 id / 文档长度 ----------
    private int[] externalIds = new int[INITIAL_DOC_CAPACITY];
    private int[] docLengths = new int[INITIAL_DOC_CAPACITY];
    private int docCount;
    private long totalTokens;

    /** 倒排表：与 doc 平铺的两个 int 数组，避免每条 posting 一个对象。 */
    private static final class PostingList {
        int[] docIds = new int[4];
        int[] termFreqs = new int[4];
        int size;

        void add(int docId, int termFreq) {
            if (size == docIds.length) {
                int newCapacity = size * 2;
                docIds = java.util.Arrays.copyOf(docIds, newCapacity);
                termFreqs = java.util.Arrays.copyOf(termFreqs, newCapacity);
            }
            docIds[size] = docId;
            termFreqs[size] = termFreq;
            size++;
        }
    }

    /**
     * 加入一篇文档。
     *
     * <p>同一个 id 重复加入会当成<b>新文档</b>追加，而不是覆盖 ——
     * 因为倒排表里删除旧文档很贵。服务层保证同一 id 不重复灌入。
     */
    public void add(int id, String text) {
        List<String> tokens = Bm25Tokenizer.tokenize(text);
        if (tokens.isEmpty()) {
            return;
        }

        // 先统计本文档内的词频
        Map<String, Integer> termFreqs = new HashMap<>();
        for (String token : tokens) {
            termFreqs.merge(token, 1, Integer::sum);
        }

        lock.writeLock().lock();
        try {
            int internal = docCount;
            ensureCapacity(internal + 1);
            externalIds[internal] = id;
            docLengths[internal] = tokens.size();
            totalTokens += tokens.size();

            for (Map.Entry<String, Integer> entry : termFreqs.entrySet()) {
                postings.computeIfAbsent(entry.getKey(), t -> new PostingList())
                        .add(internal, entry.getValue());
            }
            docCount++;
        } finally {
            lock.writeLock().unlock();
        }
    }

    /**
     * 检索 top-k。
     *
     * <p>返回的分数是 BM25 原始分（无上界），与向量检索的余弦相似度
     * <b>不可比</b>。所以融合时必须用只看排名的 RRF，
     * 而不是把两种分数加权相加。
     */
    public List<SearchResult> search(String query, int k) {
        if (k <= 0) {
            return List.of();
        }
        List<String> rawTokens = Bm25Tokenizer.tokenize(query);
        if (rawTokens.isEmpty()) {
            return List.of();
        }
        // 去重：重复的查询词不该让同一篇文档重复加分（原因见方法内注释）
        Set<String> terms = new LinkedHashSet<>(rawTokens);

        lock.readLock().lock();
        try {
            if (docCount == 0) {
                return List.of();
            }
            double averageLength = (double) totalTokens / docCount;

            // 打分累加器是局部变量：放成字段就会在读路径上写共享状态
            Map<Integer, Double> scores = new HashMap<>();

            for (String term : terms) {
                PostingList list = postings.get(term);
                if (list == null) {
                    continue;
                }
                double idf = inverseDocumentFrequency(list.size);
                for (int i = 0; i < list.size; i++) {
                    int doc = list.docIds[i];
                    int tf = list.termFreqs[i];
                    double lengthRatio = docLengths[doc] / averageLength;
                    double denominator = tf + K1 * (1 - B + B * lengthRatio);
                    double contribution = idf * (tf * (K1 + 1)) / denominator;
                    scores.merge(doc, contribution, Double::sum);
                }
            }

            if (scores.isEmpty()) {
                return List.of();
            }

            // 小顶堆保留 top-k，堆顶是当前最差的
            PriorityQueue<SearchResult> heap = new PriorityQueue<>(
                    Math.min(k, scores.size()), Comparator.comparingDouble(SearchResult::score));
            for (Map.Entry<Integer, Double> entry : scores.entrySet()) {
                // 对外用业务 id，内部 doc 号不暴露
                SearchResult hit = new SearchResult(
                        externalIds[entry.getKey()], entry.getValue().floatValue());
                if (heap.size() < k) {
                    heap.offer(hit);
                } else if (hit.score() > heap.peek().score()) {
                    heap.poll();
                    heap.offer(hit);
                }
            }

            List<SearchResult> result = new ArrayList<>(heap);
            result.sort(null);
            return result;
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * IDF：{@code ln(1 + (N - df + 0.5) / (df + 0.5))}。
     *
     * <p>外面加 1 再取对数，保证它恒为正 —— 经典 BM25 在 df > N/2 时
     * IDF 会变负，导致"包含这个词反而扣分"，是个已知的怪现象。
     */
    private double inverseDocumentFrequency(int documentFrequency) {
        return Math.log(1.0 + (docCount - documentFrequency + 0.5)
                / (documentFrequency + 0.5));
    }

    public int size() {
        lock.readLock().lock();
        try {
            return docCount;
        } finally {
            lock.readLock().unlock();
        }
    }

    /** 词表大小，用于观测分词效果（太大说明切得太碎）。 */
    public int termCount() {
        lock.readLock().lock();
        try {
            return postings.size();
        } finally {
            lock.readLock().unlock();
        }
    }

    /** 平均文档长度（token 数），BM25 的长度归一化基准。 */
    public double averageDocumentLength() {
        lock.readLock().lock();
        try {
            return docCount == 0 ? 0 : (double) totalTokens / docCount;
        } finally {
            lock.readLock().unlock();
        }
    }

    public void clear() {
        lock.writeLock().lock();
        try {
            postings.clear();
            externalIds = new int[INITIAL_DOC_CAPACITY];
            docLengths = new int[INITIAL_DOC_CAPACITY];
            docCount = 0;
            totalTokens = 0;
        } finally {
            lock.writeLock().unlock();
        }
    }

    private void ensureCapacity(int required) {
        if (required <= externalIds.length) {
            return;
        }
        int newCapacity = Math.max(required, externalIds.length * 2);
        externalIds = java.util.Arrays.copyOf(externalIds, newCapacity);
        docLengths = java.util.Arrays.copyOf(docLengths, newCapacity);
    }
}
