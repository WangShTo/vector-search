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
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * 文档原文存储。
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
 * <p>代价是体积比二进制大（同样的文本，JSON 转义后可能多出 10%~30%），
 * 以及解析比二进制慢。对"几千到几万篇文档"的规模来说完全可以接受。
 *
 * <h2>原子性</h2>
 * 同样先写 {@code .tmp} 再原子改名，避免中途崩溃留下半个文件。
 */
@Component
public class DocumentStore {

    private static final Logger log = LoggerFactory.getLogger(DocumentStore.class);

    private final Map<Integer, String> documents = new ConcurrentHashMap<>();
    private final AtomicLong totalCharacters = new AtomicLong();
    private final ObjectMapper objectMapper;

    public DocumentStore(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    public void put(int id, String text) {
        if (text == null) {
            return;
        }
        String previous = documents.put(id, text);
        if (previous != null) {
            totalCharacters.addAndGet(-previous.length());
        }
        totalCharacters.addAndGet(text.length());
    }

    public Optional<String> get(int id) {
        return Optional.ofNullable(documents.get(id));
    }

    /** 批量取原文，顺序与传入 id 一致；不存在的 id 对应 null。 */
    public List<String> getAll(List<Integer> ids) {
        List<String> result = new ArrayList<>(ids.size());
        for (Integer id : ids) {
            result.add(documents.get(id));
        }
        return result;
    }

    public int count() {
        return documents.size();
    }

    public long totalCharacters() {
        return totalCharacters.get();
    }

    public void clear() {
        documents.clear();
        totalCharacters.set(0);
    }

    // =====================================================================
    // 持久化
    // =====================================================================

    /**
     * 把所有原文写成 JSONL。每行一个 {@code {"id":..,"text":".."}}。
     */
    public void save(Path path) throws IOException {
        Path parent = path.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        Path temp = path.resolveSibling(path.getFileName() + ".tmp");

        try (BufferedWriter writer = Files.newBufferedWriter(temp, StandardCharsets.UTF_8)) {
            for (Map.Entry<Integer, String> entry : documents.entrySet()) {
                writer.write(objectMapper.writeValueAsString(
                        new DocumentRecord(entry.getKey(), entry.getValue())));
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
     * @return 成功读入的条数
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
                    put(record.id(), record.text());
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

    /** JSONL 里的一行。字段名即序列化后的键名。 */
    record DocumentRecord(int id, String text) {
    }
}
