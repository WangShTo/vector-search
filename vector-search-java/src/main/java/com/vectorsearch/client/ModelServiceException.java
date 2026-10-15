package com.vectorsearch.client;

/**
 * 调用 Python 模型服务失败。
 *
 * <p>单独定义而不是抛 {@code RuntimeException}：模型服务是外部依赖，
 * 它的失败应该能被上层识别并降级（例如退化为纯 BM25 检索），
 * 而不是和代码 bug 混在一起。
 */
public class ModelServiceException extends RuntimeException {

    private final int statusCode;

    public ModelServiceException(String message) {
        this(message, -1, null);
    }

    public ModelServiceException(String message, Throwable cause) {
        this(message, -1, cause);
    }

    public ModelServiceException(String message, int statusCode, Throwable cause) {
        super(message, cause);
        this.statusCode = statusCode;
    }

    /** HTTP 状态码；连接失败等非 HTTP 错误为 -1。 */
    public int statusCode() {
        return statusCode;
    }
}
