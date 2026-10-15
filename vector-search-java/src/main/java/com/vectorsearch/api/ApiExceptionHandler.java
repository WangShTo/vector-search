package com.vectorsearch.api;

import com.vectorsearch.client.ModelServiceException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ProblemDetail;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

/**
 * 统一异常处理。
 *
 * <p>用 Spring 6 的 {@link ProblemDetail}（RFC 7807），返回结构化错误
 * 而不是默认的 Whitelabel 页面 —— 压测脚本要能区分「服务挂了」和
 * 「参数传错了」，否则失败原因全靠猜。
 */
@RestControllerAdvice
public class ApiExceptionHandler {

    private static final Logger log = LoggerFactory.getLogger(ApiExceptionHandler.class);

    /**
     * 模型服务不可用 → 503。
     *
     * <p>单独处理而不是混进 500：这是<b>外部依赖故障</b>，
     * 调用方重试可能就好了，和代码 bug 是不同的语义。
     */
    @ExceptionHandler(ModelServiceException.class)
    public ProblemDetail handleModelService(ModelServiceException e) {
        log.error("模型服务调用失败 (status={}): {}", e.statusCode(), e.getMessage());
        ProblemDetail problem = ProblemDetail.forStatusAndDetail(
                HttpStatus.SERVICE_UNAVAILABLE,
                "向量模型服务不可用：" + e.getMessage());
        problem.setTitle("模型服务异常");
        problem.setProperty("statusCode", e.statusCode());
        return problem;
    }

    /** 参数问题 → 400。 */
    @ExceptionHandler(IllegalArgumentException.class)
    public ProblemDetail handleIllegalArgument(IllegalArgumentException e) {
        log.warn("非法请求参数: {}", e.getMessage());
        ProblemDetail problem = ProblemDetail.forStatusAndDetail(
                HttpStatus.BAD_REQUEST, e.getMessage());
        problem.setTitle("请求参数错误");
        return problem;
    }
}
