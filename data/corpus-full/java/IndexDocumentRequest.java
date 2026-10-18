package com.vectorsearch.api.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;

/**
 * 待索引的文档。
 *
 * @param id   文档 id，由调用方指定
 * @param text 文档原文，服务端负责切分与向量化
 */
public record IndexDocumentRequest(
        @NotNull(message = "id 不能为空")
        Integer id,

        @NotBlank(message = "text 不能为空")
        String text
) {
}
