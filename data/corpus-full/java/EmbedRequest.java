package com.vectorsearch.client.dto;

import java.util.List;

/**
 * 请求 Python 服务做 embedding。
 *
 * <p><b>为什么是批量接口</b>：这是本项目的核心性能设计之一。
 * 逐条调用时，一次 HTTP 往返 + 一次模型 forward 的开销全摊在一条文本上；
 * 批量调用能把 N 次往返压成 1 次，GPU/CPU 也能吃满。
 * 第 6 周会给出「单条 vs 批量」的端到端 P95 对比数据。
 *
 * @param texts 待编码文本，不应为空
 */
public record EmbedRequest(List<String> texts) {
}
