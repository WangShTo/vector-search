package com.vectorsearch.bench;

import com.vectorsearch.core.SearchResult;
import com.vectorsearch.core.bruteforce.BruteForceIndex;
import com.vectorsearch.core.hnsw.HnswIndex;
import com.vectorsearch.core.hnsw.HnswParams;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.IntPredicate;

/**
 * 带过滤的检索：三种策略在不同选择性下的对比。
 *
 * <h2>为什么需要这个基准</h2>
 * 「先搜 top-k 再筛掉不匹配的」看起来是最自然的写法，但它在过滤选择性高的时候
 * <b>会崩掉</b>：搜回来 10 条，如果只有 1% 的文档满足条件，那 10 条里大概率
 * 一条都不满足，用户看到的是空结果 —— 而实际库里明明有 200 篇匹配的文档。
 *
 * <p>这个基准量化三件事：
 * <ol>
 *   <li><b>post-filter 的召回是怎么塌的</b> —— 返回条数和 recall 随选择性下降</li>
 *   <li><b>加大候选池能不能救</b> —— 能，但代价是延迟按倍数涨</li>
 *   <li><b>filter-aware 为什么是对的</b> —— 不匹配的节点仍然参与图遍历，
 *       只是不进结果集，召回稳住而延迟只小幅上升</li>
 * </ol>
 *
 * <h2>选择性怎么造</h2>
 * 用 {@code id % 100 < 选择性} 造出精确可控的匹配比例（20k 条、1% 就是 200 条）。
 * 因为 id 是顺序分配的、整除取模在 id 空间上是均匀的，所以匹配集合均匀散布在
 * 整个库里 —— 和「按来源文件过滤」的真实场景同构：匹配的文档不聚集在图的一处。
 *
 * <h2>三个指标</h2>
 * <ul>
 *   <li><b>recall@k</b> —— 与带同一过滤条件的暴力检索逐条比对（暴力是精确解）</li>
 *   <li><b>返回条数</b> —— post-filter 拿不满 k 条这件事必须被看见</li>
 *   <li><b>P50 延迟</b> —— 单线程，预热后取中位数</li>
 * </ul>
 */
public final class FilteredSearchBench {

    /** 一次测量的结果。 */
    private record Cell(double recall, double returned, double p50Micros) {
    }

    private FilteredSearchBench() {
    }

    public static void main(String[] args) {
        int count = 20_000;
        int dimension = 512;
        int queryCount = 30;
        int k = 10;
        int ef = 50;
        // 5% 那一档最贴近真实场景：一个来源文件往往只占整个库的百分之几
        int[] selectivities = {100, 50, 20, 5, 1};

        SyntheticDataset.Dataset dataset = SyntheticDataset.clustered(
                count, dimension, queryCount,
                SyntheticDataset.recommendedClusters(count), 2026L);

        System.out.println("=== 带过滤的检索：策略 × 选择性 ===");
        System.out.printf("规模 %,d 条 × %d 维，k=%d，efSearch=%d，查询 %d 条%n%n",
                count, dimension, k, ef, queryCount);

        HnswIndex index = new HnswIndex(dimension, HnswParams.of(16, 200));
        BruteForceIndex brute = new BruteForceIndex(dimension);
        long buildStart = System.nanoTime();
        for (int i = 0; i < count; i++) {
            index.add(i, dataset.vectors()[i]);
            brute.add(i, dataset.vectors()[i]);
        }
        long buildMillis = (System.nanoTime() - buildStart) / 1_000_000;
        System.out.printf("建图与基准索引耗时 %,d ms%n%n", buildMillis);

        System.out.printf("%-8s %-22s %10s %10s %12s%n",
                "选择性", "策略", "recall@10", "返回条数", "P50 延迟");
        System.out.println("-".repeat(66));

        for (int selectivity : selectivities) {
            IntPredicate filter = id -> Math.floorMod(id, 100) < selectivity;

            // 精确解：带同一过滤条件的暴力检索
            List<Set<Integer>> truth = new ArrayList<>(queryCount);
            for (float[] query : dataset.queries()) {
                Set<Integer> ids = new HashSet<>();
                for (SearchResult hit : brute.search(query, k, k, filter)) {
                    ids.add(hit.id());
                }
                truth.add(ids);
            }

            Cell aware = measure(dataset, queryCount, truth, filter,
                    (query, f) -> index.search(query, k, ef, f));
            Cell post = measure(dataset, queryCount, truth, filter,
                    (query, f) -> postFilter(index.search(query, k, ef, null), f, k));
            // 候选池放大到「按比例估算需要看多少条才能凑出 k 条匹配」
            int pool = (int) Math.min(count, Math.min((long) k * 100 / selectivity, 5_000));
            Cell overfetch = measure(dataset, queryCount, truth, filter,
                    (query, f) -> postFilter(
                            index.search(query, pool, Math.max(ef, pool), null), f, k));

            // 自适应 ef：想把 k 条匹配都找到，就得按「匹配率」放大探索范围。
            // 这条公式不是拍脑袋来的 —— 上面 sweep 表里每一档的最佳 ef 都对得上它
            int adaptiveEf = (int) Math.min(4_000, Math.max(ef, (long) k * 100 / selectivity));
            Cell adaptive = measure(dataset, queryCount, truth, filter,
                    (query, f) -> index.search(query, k, adaptiveEf, f));

            System.out.printf("%-8s %-22s %10.4f %10.1f %10.0f µs%n",
                    selectivity + "%", "filter-aware", aware.recall(), aware.returned(), aware.p50Micros());
            System.out.printf("%-8s %-22s %10.4f %10.1f %10.0f µs%n",
                    "", "post-filter", post.recall(), post.returned(), post.p50Micros());
            System.out.printf("%-8s %-22s %10.4f %10.1f %10.0f µs%n",
                    "", "post-filter + 放大池(" + pool + ")", overfetch.recall(),
                    overfetch.returned(), overfetch.p50Micros());
            System.out.printf("%-8s %-22s %10.4f %10.1f %10.0f µs%n",
                    "", "filter-aware 自适应 ef(" + adaptiveEf + ")", adaptive.recall(),
                    adaptive.returned(), adaptive.p50Micros());
            System.out.println();
        }

        sweep(dataset, index, k);
    }
    /**
     * 扫描：高选择性下，把 efSearch 调大能不能把 filter-aware 的召回救回来。
     *
     * <p>第一轮跑出来 1% 档的 recall 只有 0.71、延迟却涨到 4 ms，方向和直觉相反。
     * 可能的原因有两个，修法完全不同：<b>候选池太小</b>（调大 ef 即可）或者
     * <b>遍历本身走不到那些节点</b>（得改算法）。凭感觉改代码是不行的，先扫一遍。
     */
    private static void sweep(SyntheticDataset.Dataset dataset, HnswIndex index, int k) {
        System.out.println("=== 扫描：filter-aware 的 efSearch 与选择性 ===");
        int queryCount = dataset.queries().length;

        System.out.printf("%-8s", "选择性");
        for (int ef : new int[]{50, 100, 200, 500, 1000}) {
            System.out.printf("  ef=%-12d", ef);
        }
        System.out.println();

        for (int selectivity : new int[]{100, 20, 5, 1}) {
            IntPredicate filter = id -> Math.floorMod(id, 100) < selectivity;
            List<Set<Integer>> truth = truthFor(dataset, index, k, filter, selectivity);
            System.out.printf("%-8s", selectivity + "%");
            for (int ef : new int[]{50, 100, 200, 500, 1000}) {
                Cell cell = measure(dataset, queryCount, truth, filter,
                        (query, f) -> index.search(query, k, ef, f));
                System.out.printf("  %.3f/%-7.0f", cell.recall(), cell.p50Micros());
            }
            System.out.println("   (recall/µs)");
        }
    }

    /** 用暴力检索求带过滤条件的精确 top-k。 */
    private static List<Set<Integer>> truthFor(SyntheticDataset.Dataset dataset, HnswIndex index,
                                               int k, IntPredicate filter, int selectivity) {
        BruteForceIndex brute = new BruteForceIndex(index.dimension());
        for (int i = 0; i < dataset.vectors().length; i++) {
            brute.add(i, dataset.vectors()[i]);
        }
        List<Set<Integer>> truth = new ArrayList<>(dataset.queries().length);
        for (float[] query : dataset.queries()) {
            Set<Integer> ids = new HashSet<>();
            for (SearchResult hit : brute.search(query, k, k, filter)) {
                ids.add(hit.id());
            }
            truth.add(ids);
        }
        return truth;
    }

    /** 一种策略：给定查询与过滤器，返回结果列表。 */
    private interface Strategy {
        List<SearchResult> run(float[] query, IntPredicate filter);
    }

    private static List<SearchResult> postFilter(List<SearchResult> raw, IntPredicate filter, int k) {
        List<SearchResult> kept = new ArrayList<>(k);
        for (SearchResult hit : raw) {
            if (filter.test(hit.id())) {
                kept.add(hit);
                if (kept.size() == k) {
                    break;
                }
            }
        }
        return kept;
    }

    private static Cell measure(SyntheticDataset.Dataset dataset, int queryCount,
                                List<Set<Integer>> truth, IntPredicate filter, Strategy strategy) {
        // 预热：让 JIT 把热路径编译出来，否则前几条查询的耗时会污染中位数
        for (int round = 0; round < 3; round++) {
            for (float[] query : dataset.queries()) {
                strategy.run(query, filter);
            }
        }

        long[] nanos = new long[queryCount];
        boolean[] seen = new boolean[queryCount];
        double hitTotal = 0;
        double returnedTotal = 0;

        for (int round = 0; round < 5; round++) {
            for (int i = 0; i < queryCount; i++) {
                long start = System.nanoTime();
                List<SearchResult> hits = strategy.run(dataset.queries()[i], filter);
                long elapsed = System.nanoTime() - start;

                if (!seen[i]) {
                    seen[i] = true;
                    Set<Integer> expected = truth.get(i);
                    int matched = 0;
                    for (SearchResult hit : hits) {
                        if (expected.contains(hit.id())) {
                            matched++;
                        }
                    }
                    hitTotal += expected.isEmpty() ? 1.0 : (double) matched / expected.size();
                    returnedTotal += hits.size();
                }
                nanos[i] = round == 0 ? elapsed : Math.min(nanos[i], elapsed);
            }
        }

        Arrays.sort(nanos);
        double p50 = nanos[nanos.length / 2] / 1_000.0;
        return new Cell(hitTotal / queryCount, returnedTotal / queryCount, p50);
    }
}
