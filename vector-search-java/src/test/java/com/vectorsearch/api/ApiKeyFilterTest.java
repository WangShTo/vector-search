package com.vectorsearch.api;

import com.vectorsearch.config.VectorSearchProperties;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.mock.web.MockFilterChain;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.mock.web.MockHttpServletResponse;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link ApiKeyFilter} 的规则测试。
 *
 * <p>用 Mock 的请求/响应，不起 Spring 上下文 —— 这个类没有任何依赖注入，
 * 起上下文只是让测试慢一个数量级。
 */
class ApiKeyFilterTest {

    private static VectorSearchProperties props(String key) {
        return new VectorSearchProperties(
                null, 0, null, null, new VectorSearchProperties.Security(key));
    }

    private static MockHttpServletResponse run(ApiKeyFilter filter, String method, String path,
                                               String providedKey) throws Exception {
        MockHttpServletRequest request = new MockHttpServletRequest(method, path);
        if (providedKey != null) {
            request.addHeader(ApiKeyFilter.HEADER, providedKey);
        }
        MockHttpServletResponse response = new MockHttpServletResponse();
        filter.doFilter(request, response, new MockFilterChain());
        return response;
    }

    @Test
    @DisplayName("没配密钥时完全不干预：删除请求照常放行")
    void disabledByDefault() throws Exception {
        ApiKeyFilter filter = new ApiKeyFilter(props(""));
        assertFalse(filter.enabled());

        MockHttpServletRequest request = new MockHttpServletRequest("DELETE", "/api/documents");
        MockHttpServletResponse response = new MockHttpServletResponse();
        MockFilterChain chain = new MockFilterChain();
        filter.doFilter(request, response, chain);

        assertEquals(200, response.getStatus());
        assertNotNull(chain.getRequest(), "未启用鉴权时请求必须继续往下走");
    }

    @Test
    @DisplayName("配了密钥：删除没带密钥 -> 401")
    void deleteWithoutKeyRejected() throws Exception {
        ApiKeyFilter filter = new ApiKeyFilter(props("s3cret"));
        assertTrue(filter.enabled());

        MockHttpServletResponse response = run(filter, "DELETE", "/api/documents/12", null);
        assertEquals(401, response.getStatus());
    }

    @Test
    @DisplayName("配了密钥：管理接口没带密钥 -> 401")
    void adminWithoutKeyRejected() throws Exception {
        ApiKeyFilter filter = new ApiKeyFilter(props("s3cret"));

        assertEquals(401, run(filter, "POST", "/api/admin/reset", null).getStatus());
        assertEquals(401, run(filter, "POST", "/api/admin/snapshot", null).getStatus());
    }

    @Test
    @DisplayName("配了密钥：带对密钥 -> 放行")
    void correctKeyAccepted() throws Exception {
        ApiKeyFilter filter = new ApiKeyFilter(props("s3cret"));

        MockHttpServletRequest request = new MockHttpServletRequest("DELETE", "/api/documents/12");
        request.addHeader(ApiKeyFilter.HEADER, "s3cret");
        MockHttpServletResponse response = new MockHttpServletResponse();
        MockFilterChain chain = new MockFilterChain();
        filter.doFilter(request, response, chain);

        assertEquals(200, response.getStatus());
        assertNotNull(chain.getRequest(), "密钥正确时请求必须继续往下走");
    }

    @Test
    @DisplayName("配了密钥：带错密钥 -> 401")
    void wrongKeyRejected() throws Exception {
        ApiKeyFilter filter = new ApiKeyFilter(props("s3cret"));

        assertEquals(401, run(filter, "DELETE", "/api/documents/12", "wrong").getStatus());
        // 前缀相同也不行 —— 不允许「猜中开头就算过」
        assertEquals(401, run(filter, "DELETE", "/api/documents/12", "s3cre").getStatus());
        assertEquals(401, run(filter, "DELETE", "/api/documents/12", "s3cretx").getStatus());
    }

    @Test
    @DisplayName("配了密钥：检索与统计不被拦（公开 demo 要能搜）")
    void readsStayOpen() throws Exception {
        ApiKeyFilter filter = new ApiKeyFilter(props("s3cret"));

        MockHttpServletRequest search = new MockHttpServletRequest("POST", "/api/search");
        MockHttpServletResponse searchRes = new MockHttpServletResponse();
        MockFilterChain searchChain = new MockFilterChain();
        filter.doFilter(search, searchRes, searchChain);
        assertEquals(200, searchRes.getStatus());
        assertNotNull(searchChain.getRequest(), "检索是公开读，必须放行");

        assertEquals(200, run(filter, "GET", "/api/stats", null).getStatus());
        assertEquals(200, run(filter, "GET", "/api/documents", null).getStatus());
    }

    @Test
    @DisplayName("需要密钥的路径规则")
    void pathRules() {
        assertTrue(ApiKeyFilter.requiresKey("DELETE", "/api/documents"));
        assertTrue(ApiKeyFilter.requiresKey("DELETE", "/api/documents/7"));
        assertTrue(ApiKeyFilter.requiresKey("POST", "/api/admin/snapshot"));
        assertTrue(ApiKeyFilter.requiresKey("POST", "/api/admin/reset"));

        assertFalse(ApiKeyFilter.requiresKey("POST", "/api/search"));
        assertFalse(ApiKeyFilter.requiresKey("GET", "/api/stats"));
        assertFalse(ApiKeyFilter.requiresKey("GET", "/api/documents"));
        // 上传是写操作但不算危险操作：公开 demo 要让访客能添加文档
        assertFalse(ApiKeyFilter.requiresKey("POST", "/api/documents/upload"));
        assertFalse(ApiKeyFilter.requiresKey("GET", "/"));
        assertFalse(ApiKeyFilter.requiresKey("GET", null));
    }

    @Test
    @DisplayName("401 响应体是可解析的 JSON")
    void unauthorizedBodyIsJson() throws Exception {
        ApiKeyFilter filter = new ApiKeyFilter(props("s3cret"));
        MockHttpServletResponse response = run(filter, "DELETE", "/api/documents", null);

        String body = response.getContentAsString();
        assertTrue(body.startsWith("{"), "前端要能 JSON.parse，不能返回 HTML 错误页");
        assertTrue(body.contains("unauthorized"));
        assertTrue(response.getContentType().startsWith("application/json"));
    }
}
