package com.vectorsearch.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * 文档原文存储 —— 知识库的「数据表」。
 *
 * <p>索引里只有向量，检索结果只有 id 和分数 —— 但用户要看到的是原文。
 * 这一层负责 id → 原文的映射，并且可以落盘。
 *
 * <h2>为什么用 JSONL 而不是二进制</h2>
 * 向量和图是定长数值数据，适合二进制（{@link com.vectorsearch.core.hnsw.HnswSnapshot}）；
 * 而原文是<b>变长 UTF-8 字符串</b>，二进制反而要自己处理长度前缀与编码边界。
 *
 * <p>JSONL（每行一个 JSON 对象）的好处：
 * <ul>
 *   <li>可以直接用文本工具查看和 diff，排查问题时不需写解析器</li>
 *   <li>写入是追加式的，不需要预知总长度</li>
 *   <li>坏掉一行只影响一行，不像二进制整体不可读</li>
 * </ul>
 *
 * <h2>删除为什么用墓碑（tombstone）</h2>
 * HNSW 图里删一个节点要拆掉它的所有边并修复连通性，BM25 的倒排表里也要
 * 从每个词的 posting list 里摘掉它 —— 两者都不便宜。
 *
 * <p>所以这里用<b>逻辑删除</b>：记录一个「已删除」标记，检索时过滤掉。
 * 代价是索引里仍然占着空间，要真正回收必须重建索引。这是搜索引擎的常见做法，
 * 不是偷懒 —— Lucene 也是靠 `.del` 文件做逻辑删除，靠段合并来回收。
 *
 * <h2>原子性</h2>
 * 同样先写 {@code .tmp} 再原子改名，避免中途崩溃留下半个文件。
 */
@Component
public class DocumentStore {

    private static final Logger log = LoggerFactory.getLogger(DocumentStore.class);

    private final Map<Integer, String> documents = new ConcurrentHashMap<>();
    /**
     * 文档元数据：来源、导入时间。
     *
     * <p>为什么单独一张表、而不是把来源拼进原文：元数据是<b>过滤</b>的依据。
     * 带过滤的检索要先按来源圈出候选 id，这是按 id 的哈希查表，
     * 不能顺带把原文也读出来解析一遍 —— 那条路径在每次检索都会走。
     */
    private final Map<Integer, DocMeta> metadata = new ConcurrentHashMap<>();
    /**
     * 每个来源的存活文档数。
     *
     * <p>冗余维护一份计数，而不是每次去 {@code metadata} 里数 ——
     * 选择性是每次带过滤检索都要用的输入，十万篇上现数一遍是毫秒级，
     * 比检索本身还贵。
     */
    private final Map<String, java.util.concurrent.atomic.AtomicInteger> sourceCounts =
            new ConcurrentHashMap<>();
    /** 逻辑删除标记。检索时过滤，重建索引时才真正回收。 */
    private final Set<Integer> deletedIds = ConcurrentHashMap.newKeySet();
    private final AtomicLong totalCharacters = new AtomicLong();
    /** 自增 id 生成器，保证上传新文档时不会和已有 id 撞车。 */
    private final java.util.concurrent.atomic.AtomicInteger idSequence =
            new java.util.concurrent.atomic.AtomicInteger(0);
    private final ObjectMapper objectMapper;

    public DocumentStore(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    /**
     * 分配一个新 id。
     *
     * <p>上传的文档没有外部 id，由服务端分配。用「当前最大值 + 1」而不是
     * 文档数量，否则删掉几篇之后新 id 就会和已存在的撞上。
     */
    public int nextId() {
        return idSequence.incrementAndGet();
    }

    public void put(int id, String text) {
        put(id, text, DocMeta.unknown());
    }

    public void put(int id, String text, DocMeta meta) {
        if (text == null) {
            return;
        }
        DocMeta effective = meta == null ? DocMeta.unknown() : meta;
        String previous = documents.put(id, text);
        // 先看它原来是不是「活着」的，决定要不要从旧来源里扣掉
        boolean wasLive = previous != null && !deletedIds.remove(id);
        DocMeta previousMeta = metadata.put(id, effective);

        if (previous != null) {
            totalCharacters.addAndGet(-previous.length());
            if (wasLive && previousMeta != null) {
                adjustSourceCount(previousMeta.source(), -1);
            }
        }
        totalCharacters.addAndGet(text.length());
        adjustSourceCount(effective.source(), 1);

        // 让 id 序列永远领先于已有的最大 id（恢复快照后也能正确续号）
        idSequence.accumulateAndGet(id, Math::max);
    }

    /** 元数据（来源、导入时间）。未知时返回占位值，不返回 null。 */
    public DocMeta metadataOf(int id) {
        return metadata.getOrDefault(id, DocMeta.unknown());
    }

    public String sourceOf(int id) {
        return metadataOf(id).source();
    }

    /**
     * 各来源的<b>存活</b>文档数，按来源名排序。
     *
     * <p>维护成计数器而不是每次遍历：过滤检索要靠它算选择性
     * （匹配文档数 / 总数），那条路径在每次检索都会走 ——
     * 十万篇的库上现数一遍是毫秒级开销，比检索本身还贵。
     */
    public Map<String, Integer> sourceCounts() {
        Map<String, Integer> counts = new java.util.TreeMap<>();
        for (Map.Entry<String, java.util.concurrent.atomic.AtomicInteger> entry : sourceCounts.entrySet()) {
            int value = entry.getValue().get();
            if (value > 0) {
                counts.put(entry.getKey(), value);
            }
        }
        return counts;
    }

    private void adjustSourceCount(String source, int delta) {
        sourceCounts.computeIfAbsent(source, key -> new java.util.concurrent.atomic.AtomicInteger())
                .addAndGet(delta);
    }

    /** 取原文。已删除的文档对调用方不可见。 */
    public Optional<String> get(int id) {
        if (deletedIds.contains(id)) {
            return Optional.empty();
        }
        return Optional.ofNullable(documents.get(id));
    }

    /** 这个 id 是否被逻辑删除。检索时用来过滤命中。 */
    public boolean isDeleted(int id) {
        return deletedIds.contains(id);
    }

    /**
     * 逻辑删除一篇文档。
     *
     * @return 是否真的删掉了一篇（重复删除返回 false）
     */
    public boolean remove(int id) {
        if (!documents.containsKey(id) || !deletedIds.add(id)) {
            return false;
        }
        String text = documents.get(id);
        if (text != null) {
            totalCharacters.addAndGet(-text.length());
        }
        adjustSourceCount(sourceOf(id), -1);
        return true;
    }

    /** 存活文档数（不含墓碑）。 */
    public int count() {
        return documents.size() - deletedIds.size();
    }

    /** 含墓碑的总数，用于展示「逻辑删除占了多少」。 */
    public int totalCount() {
        return documents.size();
    }

    public int deletedCount() {
        return deletedIds.size();
    }

    /**
     * 把墓碑真正删掉：从 map 里移除那些被逻辑删除的文档，并清空删除集合。
     *
     * <p>配合 {@code VectorIndexManager.compact()} 使用。压缩之前 {@code deletedIds}
     * 还留着这些 id，是为了让 {@link #getAll()} / {@link #list} 和检索结果继续把它们藏起来；
     * 一旦索引重建完成、墓碑已经不在新图里，这些 id 就没有存在意义了 ——
     * 继续留着只会让 {@link #count()} 与真实占用越来越对不上。
     *
     * <p><b>不可逆</b>：清掉之后无法恢复原文。
     *
     * @return 实际清掉的条数
     */
    public int purgeDeleted() {
        if (deletedIds.isEmpty()) {
            return 0;
        }
        int purged = 0;
        for (int id : new ArrayList<>(deletedIds)) {
            // 不调整 totalCharacters：文档在 remove() 那一刻就已经减过了，
            // 这里只是把残留的原文丢掉，不能再减一次。
            if (documents.remove(id) != null) {
                purged++;
            }
            // 来源计数同理，remove() 时已经减过，这里只清元数据本身
            metadata.remove(id);
        }
        deletedIds.clear();
        return purged;
    }

    /**
     * 分页列出存活的文档，按 id 升序。
     *
     * @param keyword 为空则不过滤；否则在原文里做大小写不敏感的子串匹配。
     *                <b>这是管理页的过滤器，不是检索</b> —— 它故意做得朴素，
     *                免得让人误以为这就是语义搜索。
     */
    public Page list(int offset, int limit, String keyword) {
        String needle = keyword == null ? "" : keyword.trim().toLowerCase();
        List<Integer> ids = new ArrayList<>();
        for (Map.Entry<Integer, String> entry : documents.entrySet()) {
            int id = entry.getKey();
            if (deletedIds.contains(id)) {
                continue;
            }
            if (!needle.isEmpty() && !entry.getValue().toLowerCase().contains(needle)) {
                continue;
            }
            ids.add(id);
        }
        ids.sort(null);

        int from = Math.min(Math.max(offset, 0), ids.size());
        int to = Math.min(from + Math.max(limit, 0), ids.size());
        List<DocumentView> items = new ArrayList<>(to - from);
        for (int i = from; i < to; i++) {
            int id = ids.get(i);
            items.add(new DocumentView(id, documents.get(id), sourceOf(id)));
        }
        return new Page(ids.size(), from, limit, items);
    }

    /**
     * 遍历全部<b>存活</b>原文。
     *
     * <p>BM25 索引启动时靠这个从原文重建 —— 见 {@link SparseIndexManager} 的说明。
     * 遍历顺序是 {@link ConcurrentHashMap} 的内部顺序，不保证稳定，
     * 但调用方（建倒排表）不依赖顺序。
     */
    public void forEach(java.util.function.BiConsumer<Integer, String> action) {
        for (Map.Entry<Integer, String> entry : documents.entrySet()) {
            if (!deletedIds.contains(entry.getKey())) {
                action.accept(entry.getKey(), entry.getValue());
            }
        }
    }

    public long totalCharacters() {
        return totalCharacters.get();
    }

    public void clear() {
        documents.clear();
        metadata.clear();
        sourceCounts.clear();
        deletedIds.clear();
        totalCharacters.set(0);
        idSequence.set(0);
    }

    // =====================================================================
    // 持久化
    // =====================================================================

    /**
     * 把所有原文写成 JSONL。每行一个 {@code {"id":..,"text":"..","deleted":..}}。
     *
     * <p>墓碑也要写出去 —— 否则重启后被删的文档会"复活"。
     */
    public void save(Path path) throws IOException {
        Path parent = path.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        Path temp = path.resolveSibling(path.getFileName() + ".tmp");

        try (BufferedWriter writer = Files.newBufferedWriter(temp, StandardCharsets.UTF_8)) {
            for (Map.Entry<Integer, String> entry : documents.entrySet()) {
                DocMeta meta = metadataOf(entry.getKey());
                writer.write(objectMapper.writeValueAsString(new DocumentRecord(
                        entry.getKey(), entry.getValue(), deletedIds.contains(entry.getKey()),
                        meta.source(), meta.importedAt())));
                writer.write('\n');
            }
        }

        try {
            Files.move(temp, path, StandardCopyOption.REPLACE_EXISTING,
                    StandardCopyOption.ATOMIC_MOVE);
        } catch (java.nio.file.AtomicMoveNotSupportedException e) {
            Files.move(temp, path, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    /**
     * 从 JSONL 读回原文。
     *
     * <p>文件不存在时<b>静默返回</b>（首次启动的正常情况）；
     * 但单行解析失败会记警告并跳过该行，而不是整体失败 ——
     * 一万篇文档里坏掉一行，不该让整个索引起不来。
     *
     * <p>老格式（只有 id/text，没有 deleted 字段）也能读：缺省的布尔值就是 false。
     *
     * @return 成功读入的条数（含墓碑）
     */
    public int load(Path path) throws IOException {
        if (!Files.exists(path)) {
            return 0;
        }
        clear();
        int loaded = 0;
        int skipped = 0;

        try (BufferedReader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) {
                    continue;
                }
                try {
                    DocumentRecord record = objectMapper.readValue(line, DocumentRecord.class);
                    // 老格式没有 source/importedAt 字段，Jackson 会给 null/0 —— DocMeta.of 负责兜底
                    put(record.id(), record.text(), DocMeta.of(record.source(), record.importedAt()));
                    if (record.deleted()) {
                        remove(record.id());
                    }
                    loaded++;
                } catch (IOException e) {
                    skipped++;
                }
            }
        }

        if (skipped > 0) {
            log.warn("文档原文加载完成，但有 {} 行无法解析，已跳过", skipped);
        }
        return loaded;
    }

    /** JSONL 里的一行。字段名即序列化后的键名。后加的两个字段在老文件里是 null/0。 */
    record DocumentRecord(int id, String text, boolean deleted, String source, long importedAt) {
    }

    /** 列表里的一项。 */
    public record DocumentView(int id, String text, String source) {
    }

    /**
     * 文档元数据。
     *
     * @param source     来源（上传时的文件名），空则记成 {@code 未标注}
     * @param importedAt 写入时间戳（毫秒），0 表示未知（老格式文件）
     */
    public record DocMeta(String source, long importedAt) {

        /** 来源未知时的占位值。过滤按来源圈候选集时，它也是一个合法来源。 */
        public static final String UNKNOWN_SOURCE = "未标注";

        public static DocMeta unknown() {
            return new DocMeta(UNKNOWN_SOURCE, 0L);
        }

        public static DocMeta of(String source, long importedAt) {
            String normalized = source == null || source.isBlank() ? UNKNOWN_SOURCE : source.trim();
            return new DocMeta(normalized, importedAt);
        }
    }

    /** 一页结果。 */
    public record Page(int total, int offset, int limit, List<DocumentView> items) {
    }
}
