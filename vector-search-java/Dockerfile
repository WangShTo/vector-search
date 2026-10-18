# ---- 构建阶段 ----
#
# 构建和运行都必须用 JDK 21。
# 项目的 <java.version> 是 17（源码语法目标是 17），但它通过 --add-modules
# 引入了 jdk.incubator.vector —— 这是孵化器模块，只有在 JDK 21 里才存在。
# 用 JDK 17 构建会直接失败（找不到模块），用 JDK 17 运行会在
# DistanceMetricsSimd 初始化时抛 NoClassDefFoundError。
FROM maven:3.9-eclipse-temurin-21 AS build

WORKDIR /build

# 先只拷 pom 再预拉依赖：只要 pom 没变，改业务代码就不会重新下载依赖。
# 首次构建约几分钟，之后重建镜像十几秒。
COPY pom.xml .
RUN mvn -B -ntp dependency:go-offline

COPY src ./src
RUN mvn -B -ntp package -DskipTests

# ---- 运行阶段 ----
FROM eclipse-temurin:21-jre

WORKDIR /app

# spring-boot-maven-plugin 会打出一个可执行 jar；repkg 之前的原件是 .jar.original，
# 不会被 *.jar 匹配到
COPY --from=build /build/target/*.jar app.jar

# 运行期同样要开孵化器模块，否则 SIMD 那条路径起不来
ENV JAVA_TOOL_OPTIONS="--add-modules jdk.incubator.vector"

# 索引落盘目录。compose 里挂成 named volume ——
# 不挂的话每次重建容器索引都会丢，持久化就白做了
ENV VECTOR_SEARCH_DATA_DIR=/app/data/index

EXPOSE 8080

ENTRYPOINT ["java", "-jar", "app.jar"]
