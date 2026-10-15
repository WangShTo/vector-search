package com.vectorsearch.service;

import org.springframework.stereotype.Component;

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
 * 这一层负责 id → 原文的映射。
 *
 * <p>当前是纯内存 {@link ConcurrentHashMap}，够 Demo 用。
 * TODO(第3周): 换成可落盘的实现，并在索引重建后能恢复原文，
 *  否则服务一重启就得重新灌数据。
 */
@Component
public class DocumentStore {

    private final Map<Integer, String> documents = new ConcurrentHashMap<>();
    private final AtomicLong totalCharacters = new AtomicLong();

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
}
