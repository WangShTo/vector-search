package com.vectorsearch.core.bruteforce;

import com.vectorsearch.core.DistanceMetrics;
import com.vectorsearch.core.SearchResult;
import com.vectorsearch.core.VectorIndex;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * 暴力检索：对全量向量逐个算相似度，取 top-k。
 *
 * <p><b>它的价值不是"能用"，而是"作为基准"</b>：
 * <ul>
 *   <li><b>正确性基准</b>：HNSW 是近似的，必须拿它的结果当 ground truth 算 recall。</li>
 *   <li><b>性能对照组</b>：所有"延迟从 X 降到 Y"的说法，X 就来自这里。</li>
 * </ul>
 *
 * <p>复杂度 O(n·d)，n 是向量数、d 是维度，完全无法用于大规模检索 ——
 * 这正是引入 HNSW 的动机。
 *
 * <p><b>并发策略</b>：用 {@link ReentrantReadWriteLock}。曾经的计划是换成
 * {@link java.util.concurrent.locks.StampedLock} 的乐观读，但先测了锁本身的开销：
 * 无竞争下一次读锁约 <b>16 ns</b>，而暴力检索遍历全量向量（10 万规模约 48 ms），
 * 占比不到百万分之一 —— 换锁没有任何可测收益，所以<b>不换</b>。
 * 测量见 {@code com.vectorsearch.bench.LockOverheadBenchmark}。
 */
public class BruteForceIndex implements VectorIndex {

    private static final int INITIAL_CAPACITY = 16;

    private final int dimension;
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();

    private float[][] vectors = new float[INITIAL_CAPACITY][];
    private int[] ids = new int[INITIAL_CAPACITY];
    private int size = 0;

    public BruteForceIndex(int dimension) {
        if (dimension <= 0) {
            throw new IllegalArgumentException("dimension 必须为正数: " + dimension);
        }
        this.dimension = dimension;
    }

    @Override
    public void add(int id, float[] vector) {
        validate(vector);
        lock.writeLock().lock();
        try {
            ensureCapacity(size + 1);
            vectors[size] = DistanceMetrics.normalizedCopy(vector);
            ids[size] = id;
            size++;
        } finally {
            lock.writeLock().unlock();
        }
    }

    @Override
    public List<SearchResult> search(float[] query, int k, int efSearch) {
        validate(query);
        if (k <= 0) {
            return List.of();
        }
        // 查询向量归一化一次，之后每个候选只算一次内积
        float[] q = DistanceMetrics.normalizedCopy(query);

        lock.readLock().lock();
        try {
            int n = size;
            if (n == 0) {
                return List.of();
            }

            // 小顶堆：堆顶是"当前保留结果里最差的那个"，用于 O(1) 决定要不要淘汰
            PriorityQueue<SearchResult> topK =
                    new PriorityQueue<>(Math.min(k, n), Comparator.comparingDouble(SearchResult::score));

            for (int i = 0; i < n; i++) {
                float score = DistanceMetrics.dot(q, vectors[i]);
                if (topK.size() < k) {
                    topK.offer(new SearchResult(ids[i], score));
                } else if (score > topK.peek().score()) {
                    topK.poll();
                    topK.offer(new SearchResult(ids[i], score));
                }
            }

            List<SearchResult> result = new ArrayList<>(topK);
            result.sort(null); // SearchResult 的自然序 = 相似度降序
            return result;
        } finally {
            lock.readLock().unlock();
        }
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public void forEach(IndexedVector visitor) {
        lock.readLock().lock();
        try {
            for (int i = 0; i < size; i++) {
                visitor.accept(ids[i], vectors[i]);
            }
        } finally {
            lock.readLock().unlock();
        }
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public String name() {
        return "brute-force";
    }

    private void ensureCapacity(int required) {
        if (required <= vectors.length) {
            return;
        }
        int newCapacity = Math.max(required, vectors.length * 2);
        float[][] newVectors = new float[newCapacity][];
        System.arraycopy(vectors, 0, newVectors, 0, size);
        vectors = newVectors;

        int[] newIds = new int[newCapacity];
        System.arraycopy(ids, 0, newIds, 0, size);
        ids = newIds;
    }

    private void validate(float[] vector) {
        if (vector == null) {
            throw new IllegalArgumentException("向量不能为 null");
        }
        if (vector.length != dimension) {
            throw new IllegalArgumentException(
                    "向量维度不匹配: 期望 " + dimension + "，实际 " + vector.length);
        }
    }
}
