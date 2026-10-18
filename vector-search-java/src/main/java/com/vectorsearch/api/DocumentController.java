package com.vectorsearch.api;

import com.vectorsearch.api.dto.IndexDocumentRequest;
import com.vectorsearch.config.VectorSearchProperties;
import com.vectorsearch.core.TextSplitter;
import com.vectorsearch.service.DocumentStore;
import com.vectorsearch.service.SearchService;
import com.vectorsearch.service.VectorIndexManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * 知识库管理接口 —— 让这套东西能真正被「用起来」，而不只是被「看一眼」。
 *
 * <h2>为什么这些接口是必需的</h2>
 * 检索接口再强，如果用户没法把自己的资料放进来、删掉过期的、
 * 看一眼里面到底有什么，那它就是一个 tech demo。
 * 下面这几个接口对应四个最基本的动作：<b>导入 / 查看 / 删除 / 清空</b>。
 *
 * <h2>删除是逻辑删除</h2>
 * 见 {@link DocumentStore} 的说明：HNSW 删节点要修图、BM25 删词要从每条
 * posting list 里摘，两者都不便宜。所以这里只打墓碑，检索时过滤。
 * <b>要真正回收空间必须重建索引</b> —— 接口返回里会带上墓碑数量，
 * 让这件事对使用者可见，而不是悄悄堆着。
 */
@RestController
@RequestMapping("/api/documents")
public class DocumentController {

    private static final Logger log = LoggerFactory.getLogger(DocumentController.class);

    private final DocumentStore documentStore;
    private final SearchService searchService;
    private final VectorIndexManager indexManager;
    private final VectorSearchProperties properties;

    public DocumentController(DocumentStore documentStore,
                              SearchService searchService,
                              VectorIndexManager indexManager,
                              VectorSearchProperties properties) {
        this.documentStore = documentStore;
        this.searchService = searchService;
        this.indexManager = indexManager;
        this.properties = properties;
    }

    /**
     * 分页列出知识库里的文档。
     *
     * @param q 可选的<b>字面</b>过滤（子串匹配）。注意这不是语义检索 ——
     *          它只是让管理页能快速定位某一篇，故意的朴素。
     */
    @GetMapping
    public DocumentStore.Page list(@RequestParam(defaultValue = "0") int offset,
                                   @RequestParam(defaultValue = "20") int limit,
                                   @RequestParam(required = false) String q) {
        return documentStore.list(offset, Math.min(Math.max(limit, 1), 200), q);
    }

    /** 查看单篇原文。 */
    @GetMapping("/{id}")
    public ResponseEntity<DocumentDetail> get(@PathVariable int id) {
        return documentStore.get(id)
                .map(text -> ResponseEntity.ok(new DocumentDetail(
                        id, text, text.length(), documentStore.isDeleted(id))))
                .orElseGet(() -> ResponseEntity.notFound().build());
    }

    /**
     * 逻辑删除一篇文档。
     *
     * <p>删除后立刻落盘 —— 否则重启一次被删的文档就"复活"了。
     */
    @DeleteMapping("/{id}")
    public MutationResult delete(@PathVariable int id) {
        boolean removed = documentStore.remove(id);
        if (removed) {
            log.info("逻辑删除文档 id={}", id);
            persistQuietly("删除");
        }
        return new MutationResult(removed ? 1 : 0, documentStore.count(),
                documentStore.deletedCount());
    }

    /** 清空整个知识库（索引与原文一起）。 */
    @DeleteMapping
    public MutationResult clear() {
        int before = documentStore.count();
        log.warn("清空知识库，共 {} 篇文档", before);
        indexManager.reset(VectorIndexManager.TYPE_HNSW);
        documentStore.clear();
        persistQuietly("清空");
        return new MutationResult(before, 0, 0);
    }

    /**
     * 上传一个文本文件，自动切分并灌入索引。
     *
     * <p>编码自动识别：先按 UTF-8 解，出现非法字节就退回 GBK ——
     * 国内很多 .txt 是 GBK 的，硬按 UTF-8 解会得到一屏乱码，
     * 而且灌进去之后检索永远命中不了，很难排查。
     */
    @PostMapping("/upload")
    public UploadResult upload(@RequestParam("file") MultipartFile file) throws IOException {
        if (file.isEmpty()) {
            throw new IllegalArgumentException("上传的文件是空的");
        }

        String text = decode(file.getBytes());
        List<String> chunks = TextSplitter.split(
                text, properties.index().chunkSize(), properties.index().chunkOverlap());
        if (chunks.isEmpty()) {
            throw new IllegalArgumentException("文件里没有可索引的文本");
        }

        List<IndexDocumentRequest> requests = new ArrayList<>(chunks.size());
        int firstId = -1;
        int lastId = -1;
        for (String chunk : chunks) {
            int id = documentStore.nextId();
            if (firstId < 0) {
                firstId = id;
            }
            lastId = id;
            requests.add(new IndexDocumentRequest(id, chunk));
        }

        searchService.index(requests);
        persistQuietly("上传");
        log.info("已导入 {}: {} 个字符 -> {} 个分块 (id {}..{})",
                file.getOriginalFilename(), text.length(), chunks.size(), firstId, lastId);

        return new UploadResult(
                file.getOriginalFilename(),
                text.length(),
                chunks.size(),
                firstId,
                lastId,
                documentStore.count());
    }

    /**
     * 落盘。失败只记日志、不打断请求 ——
     * 用户已经成功写入了内存索引，不该因为磁盘问题报错。
     *
     * <p>每次上传/删除都全量落盘，在这个规模（几百到几万篇）是可接受的；
     * 更大的规模应该改成定期落盘或 WAL，否则每次写入都 O(n)。
     */
    private void persistQuietly(String reason) {
        try {
            indexManager.snapshot();
        } catch (RuntimeException e) {
            log.warn("{}后落盘失败（数据仍在内存中）: {}", reason, e.getMessage());
        }
    }

    /**
     * 解码上传文件。
     *
     * <p>两个坑都要处理：
     * <ol>
     *   <li><b>UTF-8 BOM</b>：Windows 上的记事本、PowerShell 的 Set-Content 都会写
     *       {@code EF BB BF}。不去掉的话它会被当成正文的一部分，混进第一个分块，
     *       既污染显示，也让开头那个词永远匹配不上。</li>
     *   <li><b>GBK</b>：国内很多 .txt 是 GBK 的，硬按 UTF-8 解会得到一屏乱码，
     *       而且灌进去之后检索永远命中不了，很难排查。</li>
     * </ol>
     */
    private static String decode(byte[] bytes) {
        String text = new String(bytes, StandardCharsets.UTF_8);
        if (text.indexOf('\uFFFD') >= 0) {
            try {
                String gbk = new String(bytes, Charset.forName("GBK"));
                if (gbk.indexOf('\uFFFD') < 0) {
                    text = gbk;
                }
            } catch (RuntimeException ignored) {
                // 某些精简 JRE 可能没有 GBK 字符集，退回 UTF-8
            }
        }
        // 顺带处理 UTF-16 的 BOM（FE FF / FF FE 解出来也会带 U+FEFF）
        if (!text.isEmpty() && text.charAt(0) == '\uFEFF') {
            text = text.substring(1);
        }
        return text;
    }

    /** 单篇文档详情。 */
    public record DocumentDetail(int id, String text, int length, boolean deleted) {
    }

    /** 删除/清空的结果。 */
    public record MutationResult(int affected, int remaining, int tombstones) {
    }

    /** 上传结果。 */
    public record UploadResult(
            String fileName,
            int characters,
            int chunks,
            int firstId,
            int lastId,
            int totalDocuments
    ) {
    }
}
