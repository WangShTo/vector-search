package com.vectorsearch.client;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.vectorsearch.client.dto.EmbedRequest;
import com.vectorsearch.client.dto.EmbedResponse;
import com.vectorsearch.client.dto.RerankRequest;
import com.vectorsearch.client.dto.RerankResponse;
import com.vectorsearch.config.VectorSearchProperties;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

/**
 * 调用 Python 模型服务的客户端。
 *
 * <p>用 JDK 自带的 {@link HttpClient} 而不是引入 OkHttp/RestTemplate ——
 * 少一个依赖，而且它自带连接池，对我们的场景够用。
 *
 * <p><b>这里是本项目最容易被低估的性能点</b>：Java 和 Python 是两个进程，
 * 每次调用都是一次网络往返。逐条 embedding 时，网络开销经常比模型推理还大。
 * 所以接口从设计上就是<b>批量</b>的 —— 一次请求带上 N 条文本。
 * 第 6 周会量出「单条 vs 批量」的端到端 P95 差距，这个数字通常很惊人。
 *
 * <p>TODO(第4周): rerank 接入后，这里会多一层降级逻辑 ——
 * Python 服务不可用时退化为纯向量检索，而不是整个查询失败。
 */
@Component
public class ModelServiceClient {

    private static final Logger log = LoggerFactory.getLogger(ModelServiceClient.class);

    private final HttpClient httpClient;
    private final ObjectMapper objectMapper;
    private final VectorSearchProperties properties;

    public ModelServiceClient(ObjectMapper objectMapper, VectorSearchProperties properties) {
        this.objectMapper = objectMapper;
        this.properties = properties;
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(3))
                .version(HttpClient.Version.HTTP_1_1)
                .build();
    }

    /**
     * 批量编码文本。
     *
     * @param texts 文本列表，不应为空
     * @return 与 texts 顺序一一对应的向量数组
     */
    public float[][] embed(List<String> texts) {
        if (texts == null || texts.isEmpty()) {
            return new float[0][];
        }

        EmbedResponse response = post("/embed", new EmbedRequest(texts), EmbedResponse.class);

        if (response.vectors() == null || response.vectors().size() != texts.size()) {
            throw new ModelServiceException(String.format(
                    "模型服务返回的向量条数(%s)与请求文本条数(%d)不一致",
                    response.vectors() == null ? "null" : response.vectors().size(), texts.size()));
        }

        // 维度不一致如果拖到写入索引时才报错，排查会很痛苦 —— 在这里就拦下
        int expectedDimension = properties.index().dimension();
        if (response.dimension() > 0 && response.dimension() != expectedDimension) {
            throw new ModelServiceException(String.format(
                    "向量维度不匹配：模型输出 %d 维，索引配置 %d 维。"
                            + "检查 vector-search.index.dimension 与 Python 侧 EMBEDDING_MODEL 是否对应",
                    response.dimension(), expectedDimension));
        }

        float[][] vectors = new float[response.vectors().size()][];
        for (int i = 0; i < vectors.length; i++) {
            float[] vector = response.vectors().get(i);
            if (vector == null || vector.length == 0) {
                throw new ModelServiceException("第 " + i + " 条文本的向量为空");
            }
            vectors[i] = vector;
        }
        return vectors;
    }

    /**
     * 对候选文档做 cross-encoder 精排。
     *
     * <p>第 4 周接入两阶段检索时使用。
     */
    public List<RerankResponse.Item> rerank(String query, List<String> documents, int topN) {
        if (documents == null || documents.isEmpty()) {
            return List.of();
        }
        int effectiveTopN = topN <= 0 ? documents.size() : Math.min(topN, documents.size());
        RerankResponse response = post("/rerank",
                new RerankRequest(query, documents, effectiveTopN), RerankResponse.class);

        List<RerankResponse.Item> results = response.results();
        if (results == null) {
            return List.of();
        }

        // 防御：下越界会让上层拿到不存在的文档
        List<RerankResponse.Item> safe = new ArrayList<>(results.size());
        for (RerankResponse.Item item : results) {
            if (item.index() < 0 || item.index() >= documents.size()) {
                log.warn("rerank 返回了越界下标 {}，已忽略（候选数 {}）", item.index(), documents.size());
                continue;
            }
            safe.add(item);
        }
        return safe;
    }

    /** 探活，供 /api/stats 展示模型服务是否可达。 */
    public boolean isHealthy() {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(properties.modelServiceUrl() + "/health"))
                    .timeout(Duration.ofSeconds(2))
                    .GET()
                    .build();
            HttpResponse<Void> response = httpClient.send(request, HttpResponse.BodyHandlers.discarding());
            return response.statusCode() / 100 == 2;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return false;
        } catch (Exception e) {
            return false;
        }
    }

    private <T> T post(String path, Object body, Class<T> responseType) {
        String url = properties.modelServiceUrl() + path;
        try {
            String payload = objectMapper.writeValueAsString(body);
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .timeout(Duration.ofMillis(properties.modelTimeoutMs()))
                    .header("Content-Type", "application/json; charset=UTF-8")
                    .POST(HttpRequest.BodyPublishers.ofString(payload, StandardCharsets.UTF_8))
                    .build();

            HttpResponse<String> response = httpClient.send(
                    request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));

            if (response.statusCode() / 100 != 2) {
                throw new ModelServiceException(
                        "模型服务 " + path + " 返回 HTTP " + response.statusCode() + ": " + response.body(),
                        response.statusCode(), null);
            }
            return objectMapper.readValue(response.body(), responseType);

        } catch (JsonProcessingException e) {
            throw new ModelServiceException("解析模型服务响应失败: " + url, e);
        } catch (java.io.IOException e) {
            throw new ModelServiceException("调用模型服务失败: " + url
                    + "（确认 Python 服务已启动，或检查 vector-search.model-service-url）", e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ModelServiceException("调用模型服务被中断: " + url, e);
        }
    }
}
