package com.vectorsearch.api;

import com.vectorsearch.config.VectorSearchProperties;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;

/**
 * 管理密钥校验 —— 保护「能把东西弄没」的接口。
 *
 * <h2>为什么需要这一层</h2>
 * 这个项目带 {@code POST /api/admin/reset} 和 {@code DELETE /api/documents}，
 * 而 {@code docs/DEPLOY.md} 教的部署方式是「公网 IP + 8080 直接访问」。
 * 两者相加意味着：**任何人拿到地址就能清空知识库**。
 * 之前这个问题只是被写进文档当成已知风险，现在把它做成应用自带的能力。
 *
 * <h2>保护范围为什么只有两条规则</h2>
 * 只拦 <b>删除</b> 和 <b>管理</b>，不拦检索 / 统计 / 列表 / 上传。
 * 理由是这个项目要能当公开 demo 用：全都鉴权的话访客连搜都搜不了，
 * 而真正危险的是「能把东西弄没」的操作 —— 弄没之后没有撤销键。
 *
 * <h2>密钥为什么默认不启用</h2>
 * 留空 = 不鉴权，是本地开发的默认值。本地跑着玩还要先配个密钥是纯粹的摩擦，
 * 而且会让人为了省事去改代码。启动日志会明确告诉你当前处于哪种状态。
 */
@Component
public class ApiKeyFilter extends OncePerRequestFilter {

    /** 密钥请求头。用自定义头而不是 {@code Authorization}，避免和未来的标准认证方案抢同一个头。 */
    public static final String HEADER = "X-API-Key";

    private static final Logger log = LoggerFactory.getLogger(ApiKeyFilter.class);

    private final String apiKey;

    public ApiKeyFilter(VectorSearchProperties properties) {
        this.apiKey = properties.security().apiKey();
        if (apiKey.isEmpty()) {
            log.info("未配置管理密钥，删除与管理接口【不鉴权】。仅限本地开发 —— "
                    + "部署到公网前请设置环境变量 VECTOR_SEARCH_API_KEY");
        } else {
            log.info("已启用管理密钥：删除与管理接口需要 {} 请求头", HEADER);
        }
    }

    /** 密钥是否启用。没配就等于完全不干预请求。 */
    boolean enabled() {
        return !apiKey.isEmpty();
    }

    /**
     * 这个请求是否必须带密钥。
     *
     * <p>两条规则：<b>删除</b>（任何 {@code DELETE}）和<b>管理</b>（{@code /api/admin/**}）。
     * 其余一律放行。
     */
    static boolean requiresKey(String method, String path) {
        if (path == null) {
            return false;
        }
        if (path.startsWith("/api/admin")) {
            return true;
        }
        return "DELETE".equalsIgnoreCase(method);
    }

    /**
     * 常数时间比较。
     *
     * <p>{@code equals} 会在第一个不同字节处返回，比较耗时可被用来逐字节猜出密钥。
     * 演示项目也照这个写 —— 这才是「知道自己在做什么」，而不是等上线了再补。
     */
    private boolean matches(String provided) {
        if (provided == null) {
            return false;
        }
        return MessageDigest.isEqual(
                provided.getBytes(StandardCharsets.UTF_8),
                apiKey.getBytes(StandardCharsets.UTF_8));
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response,
                                    FilterChain chain) throws ServletException, IOException {
        if (!enabled() || !requiresKey(request.getMethod(), request.getRequestURI())) {
            chain.doFilter(request, response);
            return;
        }
        if (matches(request.getHeader(HEADER))) {
            chain.doFilter(request, response);
            return;
        }

        log.warn("拒绝未授权请求: {} {}（缺少或错误的 {}）",
                request.getMethod(), request.getRequestURI(), HEADER);
        response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
        response.setContentType("application/json;charset=UTF-8");
        response.getWriter().write("{\"error\":\"unauthorized\","
                + "\"message\":\"该操作需要管理密钥，请在请求头 " + HEADER + " 中提供\"}");
    }
}
