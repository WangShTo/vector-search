package com.vectorsearch.api.dto;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotEmpty;

import java.util.List;

/**
 * 批量索引请求。
 *
 * <p>刻意做成<b>批量</b>接口而不是单条：Java 侧一次把 N 条文本发给 Python，
 * N 次模型 forward 合并成一次，N 次网络往返压缩成一次。
 * 灌 10 万条数据时，这个设计决定了是几分钟还是几小时。
 */
public record IndexDocumentsRequest(
        @NotEmpty(message = "documents 不能为空")
        @Valid
        List<IndexDocumentRequest> documents
) {
}
