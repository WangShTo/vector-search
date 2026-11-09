package com.vectorsearch.api.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;

/**
 * 待索引的文档。
 *
 * @param id     文档 id，由调用方指定
 * @param text   文档原文，服务端负责切分与向量化
 * @param source 来源标识（通常是文件名），可空。为空时整批共用请求级来源，
 *               或者记成「未标注」
 */
public record IndexDocumentRequest(
        @NotNull(message = "id 不能为空")
        Integer id,

        @NotBlank(message = "text 不能为空")
        String text,

        String source
) {

    /**
     * 不带来源的写法，来源记成「未标注」。
     *
     * <p>保留这个重载是因为它有两个真实用途：测试里造数据不想被来源干扰；
     * 以及 {@code scripts/load_corpus.py} 之外的调用方只关心「灌进去」。
     */
    public IndexDocumentRequest(Integer id, String text) {
        this(id, text, null);
    }
}
