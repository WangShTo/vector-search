package com.vectorsearch.core.hnsw;

import com.vectorsearch.core.quant.ScalarQuantizer;

import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.ShortBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.zip.CRC32;

/**
 * HNSW 索引的二进制快照格式 —— 把索引存盘、原样读回。
 *
 * <h2>为什么可以做得这么简单</h2>
 * 因为索引的存储已经是<b>扁平数组</b>了（见 {@link HnswIndex} 的类注释）。
 * 七个连续数组落盘就是写七段字节，读回来就是读七段字节 —— 没有任何对象图遍历、
 * 没有指针修正、没有序列化框架。
 *
 * <p>如果用原来的 {@code List<float[]>} + {@code List<int[][]>} 对象化布局，
 * 序列化会痛苦得多：每条向量要单独写长度、每个节点的每一层要单独处理，
 * 而且反序列化要重建几十万个对象。**扁平化在这里又一次付了红利。**
 *
 * <h2>文件布局</h2>
 * <pre>
 *   偏移 0    64 字节   头部（魔数 / 版本 / 参数 / 计数）
 *   偏移 64   196 字节  段表：7 项 × (kind:int, offset:long, length:long, crc32:long)
 *   偏移 260  ...       7 个数据段，顺序与段表一致
 * </pre>
 *
 * <h2>四个关键设计决策（都是面试会问的）</h2>
 * <ol>
 *   <li><b>魔数 + 版本号</b>：拿错文件能立刻拒绝，而不是算出垃圾结果。
 *       格式将来演进时，老文件能被识别出来。</li>
 *   <li><b>每段独立 CRC32</b>：磁盘损坏或截断能被发现。没有校验的话，
 *       损坏的索引会安静地返回错误结果 —— 这类 bug 几乎无法排查。</li>
 *   <li><b>参数一致性校验</b>：加载时比对 dimension 与 m，不一致直接拒绝。
 *       这和之前踩过的「模型维度 768/512 写错」是同一类防御 ——
 *       维度错了拖到检索结果不对才发现，代价极高。</li>
 *   <li><b>只写已用长度，不写容量</b>：容量是 2 的幂（10 万节点时是 131072），
 *       按容量写会多存 31% 的空白。这里按 {@code nodeCount} 截断，
 *       加载后容量按需重新增长。</li>
 * </ol>
 *
 * <h2>原子性</h2>
 * 写盘先落到 {@code <name>.tmp}，写完再 {@code ATOMIC_MOVE} 改名。
 * 这样中途崩溃不会留下半个文件被下次启动当成有效索引读进来。
 */
public final class HnswSnapshot {

    /** 文件魔数，8 字节。换格式时改这个也改版本号。 */
    private static final byte[] MAGIC = {'V', 'S', 'R', 'C', 'H', 'I', 'D', 'X'};

    /** 格式版本。字段布局变化时递增。 */
    /**
     * 当前写出的格式版本。
     *
     * <ul>
     *   <li><b>v1</b>：向量段固定是 float32。</li>
     *   <li><b>v2</b>：头部第 52 字节多一个「存储精度」标记，
     *       向量段可能是 float32（0）或 int8（1）。v1 文件该位置是预留的 0，
     *       所以按 v1 读出来自然就是 float32 —— 旧快照仍然可读。</li>
     *   <li><b>v3</b>：头部第 56 字节多一个 SQ8 量化刻度（float）。v2 的 SQ8 文件
     *       用的是固定刻度 127，读取时按 127 解码，同样兼容。</li>
     * </ul>
     */
    public static final int FORMAT_VERSION = 3;

    private static final int HEADER_BYTES = 64;
    /**
     * 段表每项的字节数：kind(int) + offset(long) + length(long) + crc32(long)。
     *
     * <p>这里踩过一次错：最初把它写成 {@code 4 + 8 + 8}（漏了 offset），
     * 而实际每项写的是 28 字节，结果缓冲区第 3 项就溢出了。
     * 常量必须与真正写入的字段严格对应 —— 差一个字段，
     * 后面所有偏移量都会跟着错。
     */
    private static final int SEGMENT_ENTRY_BYTES = 4 + 8 + 8 + 8;
    private static final int SEGMENT_COUNT = 7;

    /** 分段标识，写进段表，读的时候校验顺序与类型相符。 */
    private static final int SEG_VECTORS = 0;
    private static final int SEG_EXTERNAL_IDS = 1;
    private static final int SEG_LEVELS = 2;
    private static final int SEG_LAYER0_LINKS = 3;
    private static final int SEG_LAYER0_COUNT = 4;
    private static final int SEG_UPPER_STORE = 5;
    private static final int SEG_UPPER_OFFSET = 6;

    /** 读写缓冲区。64 KB 在吞吐和内存之间比较平衡。 */
    private static final int BUFFER_BYTES = 64 * 1024;

    /** 一个已写出/待读入的数据段。 */
    private record Segment(long offset, long length, long crc) {
    }

    /**
     * 快照内容 —— 就是索引的七个数组加上它们的元数据。
     *
     * <p>注意这里传的是<b>已截断到实际使用长度</b>的数组，不是带空余容量的数组。
     */
    public record Data(
            int dimension,
            int m,
            int efConstruction,
            double levelMultiplier,
            /** SQ8 的量化刻度；f32 或 v1/v2 旧文件为 0（读取时回落到固定刻度）。 */
            float quantScale,
            int nodeCount,
            int entryPoint,
            int maxLevel,
            int upperStoreSize,
            float[] vectorStore,
            byte[] quantizedStore,
            int[] externalIdStore,
            byte[] levelStore,
            int[] layer0Links,
            short[] layer0Count,
            int[] upperStore,
            int[] upperOffset
    ) {
    }

    private HnswSnapshot() {
    }

    // =====================================================================
    // 写
    // =====================================================================

    /**
     * 把索引写到文件。先写临时文件再原子改名，中途崩溃不会产生半个有效文件。
     */
    public static void write(Path path, Data data) throws IOException {
        Path parent = path.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        Path temp = path.resolveSibling(path.getFileName() + ".tmp");

        Segment[] segments = new Segment[SEGMENT_COUNT];

        try (FileChannel ch = FileChannel.open(temp,
                StandardOpenOption.CREATE, StandardOpenOption.WRITE,
                StandardOpenOption.TRUNCATE_EXISTING)) {

            // 先把头和段表的空间留出来，数据段从 204 字节开始
            long dataStart = HEADER_BYTES + (long) SEGMENT_COUNT * SEGMENT_ENTRY_BYTES;
            ch.position(dataStart);

            // 逐段写数据，同时算出每段的偏移、长度与 CRC
            long offset = dataStart;
            segments[SEG_VECTORS] = data.quantizedStore() != null
                    ? writeBytes(ch, data.quantizedStore())
                    : writeFloats(ch, data.vectorStore());
            segments[SEG_EXTERNAL_IDS] = writeInts(ch, data.externalIdStore());
            segments[SEG_LEVELS] = writeBytes(ch, data.levelStore());
            segments[SEG_LAYER0_LINKS] = writeInts(ch, data.layer0Links());
            segments[SEG_LAYER0_COUNT] = writeShorts(ch, data.layer0Count());
            segments[SEG_UPPER_STORE] = writeInts(ch, data.upperStore());
            segments[SEG_UPPER_OFFSET] = writeInts(ch, data.upperOffset());

            // 回填偏移：上面的 write 只返回长度与 CRC，偏移在这里统一算
            long cursor = dataStart;
            for (int i = 0; i < SEGMENT_COUNT; i++) {
                Segment s = segments[i];
                segments[i] = new Segment(cursor, s.length(), s.crc());
                cursor += s.length();
            }

            // 写完数据再回填头部与段表（单次数据遍历，不用为了算 CRC 读两遍）
            ch.position(0);
            writeHeader(ch, data);
            writeSegmentTable(ch, segments);
        }

        // 原子替换：要么是旧的完整文件，要么是新的完整文件，不存在中间态
        try {
            Files.move(temp, path, StandardCopyOption.REPLACE_EXISTING,
                    StandardCopyOption.ATOMIC_MOVE);
        } catch (java.nio.file.AtomicMoveNotSupportedException e) {
            // 少数文件系统不支持原子移动，退化成普通替换并记录
            Files.move(temp, path, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private static void writeHeader(FileChannel ch, Data data) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        buf.put(MAGIC);
        buf.putInt(FORMAT_VERSION);
        buf.putInt(data.dimension());
        buf.putInt(data.m());
        buf.putInt(data.efConstruction());
        buf.putDouble(data.levelMultiplier());
        buf.putInt(data.nodeCount());
        buf.putInt(data.entryPoint());
        buf.putInt(data.maxLevel());
        buf.putInt(data.upperStoreSize());
        buf.putInt(SEGMENT_COUNT);
        // 52 字节处：向量段精度。0 = float32，1 = int8（SQ8）
        buf.putInt(data.quantizedStore() != null ? 1 : 0);
        // 56 字节处：SQ8 的量化刻度。刻度不同就是两套编码，
        // 不写进文件的话读回来的向量会「看起来正常但全错」
        buf.putFloat(data.quantScale());
        // 剩余字节留作将来扩展，现在是 0
        buf.position(HEADER_BYTES);
        buf.flip();
        while (buf.hasRemaining()) {
            ch.write(buf);
        }
    }

    private static void writeSegmentTable(FileChannel ch, Segment[] segments) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(SEGMENT_COUNT * SEGMENT_ENTRY_BYTES)
                .order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < SEGMENT_COUNT; i++) {
            buf.putInt(i);
            buf.putLong(segments[i].offset());
            buf.putLong(segments[i].length());
            buf.putLong(segments[i].crc());
        }
        buf.flip();
        while (buf.hasRemaining()) {
            ch.write(buf);
        }
    }

    // =====================================================================
    // 读
    // =====================================================================

    /** 从文件读回索引数据。任何一致性校验失败都抛异常，绝不静默降级。 */
    public static Data read(Path path) throws IOException {
        if (!Files.exists(path)) {
            throw new IOException("索引快照不存在: " + path);
        }
        try (FileChannel ch = FileChannel.open(path, StandardOpenOption.READ)) {
            if (ch.size() < HEADER_BYTES) {
                throw new IOException("文件太短，不是有效的索引快照: " + path);
            }

            ByteBuffer header = ByteBuffer.allocate(HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
            readFully(ch, header);
            header.flip();

            byte[] magic = new byte[MAGIC.length];
            header.get(magic);
            for (int i = 0; i < MAGIC.length; i++) {
                if (magic[i] != MAGIC[i]) {
                    throw new IOException("魔数不匹配，这不是索引快照文件: " + path);
                }
            }

            int version = header.getInt();
            if (version < 1 || version > FORMAT_VERSION) {
                throw new IOException("快照格式版本不支持: 文件是 v" + version
                        + "，当前支持 v1 ~ v" + FORMAT_VERSION + "。请重建索引。");
            }

            int dimension = header.getInt();
            int m = header.getInt();
            int efConstruction = header.getInt();
            double levelMultiplier = header.getDouble();
            int nodeCount = header.getInt();
            int entryPoint = header.getInt();
            int maxLevel = header.getInt();
            int upperStoreSize = header.getInt();
            int segmentCount = header.getInt();

            // v2 起多一个「向量存储精度」标记；v1 文件这个位置是预留字节，读出来是 0
            int storageMode = header.getInt();
            if (storageMode != 0 && storageMode != 1) {
                throw new IOException("未知的向量存储精度标记: " + storageMode
                        + "（0 = float32，1 = int8）");
            }

            // v3 起多了量化刻度。v2 文件用的是固定 127（分量按 [-1,1] 理解），
            // 必须按老刻度解码，否则读回来的向量会整体缩放错
            float quantScale = version >= 3 ? header.getFloat() : ScalarQuantizer.LEGACY_SCALE;
            if (storageMode == 1 && quantScale <= 0) {
                throw new IOException("SQ8 快照的量化刻度无效: " + quantScale);
            }

            if (segmentCount != SEGMENT_COUNT) {
                throw new IOException("段数不匹配: 文件 " + segmentCount
                        + "，期望 " + SEGMENT_COUNT);
            }

            // 段表
            ByteBuffer table = ByteBuffer.allocate(segmentCount * SEGMENT_ENTRY_BYTES)
                    .order(ByteOrder.LITTLE_ENDIAN);
            readFully(ch, table);
            table.flip();

            Segment[] segments = new Segment[segmentCount];
            for (int i = 0; i < segmentCount; i++) {
                int kind = table.getInt();
                if (kind != i) {
                    throw new IOException("段顺序不匹配: 第 " + i + " 项写的是 kind=" + kind);
                }
                segments[i] = new Segment(table.getLong(), table.getLong(), table.getLong());
            }

            // 按段读回数组。长度由元数据推导，并与段表里的长度交叉校验。
            int maxLinks0 = m * 2;
            float[] vectorStore = storageMode == 0
                    ? readFloats(ch, segments[SEG_VECTORS], (long) nodeCount * dimension)
                    : null;
            byte[] quantizedStore = storageMode == 1
                    ? readBytes(ch, segments[SEG_VECTORS], (long) nodeCount * dimension)
                    : null;
            int[] externalIdStore = readInts(ch, segments[SEG_EXTERNAL_IDS], nodeCount);
            byte[] levelStore = readBytes(ch, segments[SEG_LEVELS], nodeCount);
            int[] layer0Links = readInts(ch, segments[SEG_LAYER0_LINKS], (long) nodeCount * maxLinks0);
            short[] layer0Count = readShorts(ch, segments[SEG_LAYER0_COUNT], nodeCount);
            int[] upperStore = readInts(ch, segments[SEG_UPPER_STORE], upperStoreSize);
            int[] upperOffset = readInts(ch, segments[SEG_UPPER_OFFSET], nodeCount);

            return new Data(dimension, m, efConstruction, levelMultiplier, quantScale,
                    nodeCount, entryPoint, maxLevel, upperStoreSize,
                    vectorStore, quantizedStore, externalIdStore, levelStore,
                    layer0Links, layer0Count, upperStore, upperOffset);
        }
    }

    // =====================================================================
    // 分类型的分段读写
    //
    // 每段写入时边写边算 CRC；读取时边读边算，读完比对。
    // 之所以分类型写这么多方法，是因为 ByteBuffer 的类型视图无法泛型化，
    // 用统一接口包一层反而更难读。
    // =====================================================================

    private static Segment writeFloats(FileChannel ch, float[] src) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(BUFFER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        FloatBuffer view = buf.asFloatBuffer();
        CRC32 crc = new CRC32();
        int i = 0;
        while (i < src.length) {
            view.clear();
            int n = Math.min(src.length - i, view.capacity());
            view.put(src, i, n);
            buf.clear();
            buf.limit(n * Float.BYTES);
            crc.update(buf.array(), 0, buf.limit());
            while (buf.hasRemaining()) {
                ch.write(buf);
            }
            i += n;
        }
        return new Segment(0, (long) src.length * Float.BYTES, crc.getValue());
    }

    private static Segment writeInts(FileChannel ch, int[] src) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(BUFFER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        IntBuffer view = buf.asIntBuffer();
        CRC32 crc = new CRC32();
        int i = 0;
        while (i < src.length) {
            view.clear();
            int n = Math.min(src.length - i, view.capacity());
            view.put(src, i, n);
            buf.clear();
            buf.limit(n * Integer.BYTES);
            crc.update(buf.array(), 0, buf.limit());
            while (buf.hasRemaining()) {
                ch.write(buf);
            }
            i += n;
        }
        return new Segment(0, (long) src.length * Integer.BYTES, crc.getValue());
    }

    private static Segment writeShorts(FileChannel ch, short[] src) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(BUFFER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        ShortBuffer view = buf.asShortBuffer();
        CRC32 crc = new CRC32();
        int i = 0;
        while (i < src.length) {
            view.clear();
            int n = Math.min(src.length - i, view.capacity());
            view.put(src, i, n);
            buf.clear();
            buf.limit(n * Short.BYTES);
            crc.update(buf.array(), 0, buf.limit());
            while (buf.hasRemaining()) {
                ch.write(buf);
            }
            i += n;
        }
        return new Segment(0, (long) src.length * Short.BYTES, crc.getValue());
    }

    private static Segment writeBytes(FileChannel ch, byte[] src) throws IOException {
        CRC32 crc = new CRC32();
        crc.update(src);
        ByteBuffer buf = ByteBuffer.wrap(src);
        while (buf.hasRemaining()) {
            ch.write(buf);
        }
        return new Segment(0, src.length, crc.getValue());
    }

    private static float[] readFloats(FileChannel ch, Segment seg, long expectedCount)
            throws IOException {
        float[] dst = new float[(int) checkLength(seg, expectedCount, Float.BYTES)];
        ch.position(seg.offset());
        ByteBuffer buf = ByteBuffer.allocate(BUFFER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        CRC32 crc = new CRC32();
        int i = 0;
        while (i < dst.length) {
            int want = (int) Math.min((long) BUFFER_BYTES, (long) (dst.length - i) * Float.BYTES);
            readChunk(ch, buf, want, crc);
            FloatBuffer view = buf.asFloatBuffer();
            int n = view.remaining();
            view.get(dst, i, n);
            i += n;
        }
        verifyCrc(seg, crc);
        return dst;
    }

    private static int[] readInts(FileChannel ch, Segment seg, long expectedCount)
            throws IOException {
        int[] dst = new int[(int) checkLength(seg, expectedCount, Integer.BYTES)];
        ch.position(seg.offset());
        ByteBuffer buf = ByteBuffer.allocate(BUFFER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        CRC32 crc = new CRC32();
        int i = 0;
        while (i < dst.length) {
            int want = (int) Math.min((long) BUFFER_BYTES, (long) (dst.length - i) * Integer.BYTES);
            readChunk(ch, buf, want, crc);
            IntBuffer view = buf.asIntBuffer();
            int n = view.remaining();
            view.get(dst, i, n);
            i += n;
        }
        verifyCrc(seg, crc);
        return dst;
    }

    private static short[] readShorts(FileChannel ch, Segment seg, long expectedCount)
            throws IOException {
        short[] dst = new short[(int) checkLength(seg, expectedCount, Short.BYTES)];
        ch.position(seg.offset());
        ByteBuffer buf = ByteBuffer.allocate(BUFFER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        CRC32 crc = new CRC32();
        int i = 0;
        while (i < dst.length) {
            int want = (int) Math.min((long) BUFFER_BYTES, (long) (dst.length - i) * Short.BYTES);
            readChunk(ch, buf, want, crc);
            ShortBuffer view = buf.asShortBuffer();
            int n = view.remaining();
            view.get(dst, i, n);
            i += n;
        }
        verifyCrc(seg, crc);
        return dst;
    }

    private static byte[] readBytes(FileChannel ch, Segment seg, long expectedCount)
            throws IOException {
        byte[] dst = new byte[(int) checkLength(seg, expectedCount, 1)];
        ch.position(seg.offset());
        int read = 0;
        ByteBuffer buf = ByteBuffer.allocate(Math.max(1, Math.min(BUFFER_BYTES, dst.length)));
        while (read < dst.length) {
            buf.clear();
            buf.limit(Math.min(buf.capacity(), dst.length - read));
            int n = ch.read(buf);
            if (n < 0) {
                throw new EOFException("读取第 " + SEG_LEVELS + " 段时文件提前结束");
            }
            buf.flip();
            buf.get(dst, read, n);
            read += n;
        }
        CRC32 crc = new CRC32();
        crc.update(dst);
        verifyCrc(seg, crc);
        return dst;
    }

    // =====================================================================
    // 校验辅助
    // =====================================================================

    /** 段表里的长度必须与元数据推导出的长度严格一致。 */
    private static long checkLength(Segment seg, long expectedCount, int elementBytes)
            throws IOException {
        long expectedBytes = expectedCount * elementBytes;
        if (seg.length() != expectedBytes) {
            throw new IOException(String.format(
                    "段长度不一致: 文件记录 %d 字节，按元数据应为 %d 字节"
                            + "（可能是维度或 HNSW 参数与快照不匹配，请重建索引）",
                    seg.length(), expectedBytes));
        }
        return expectedCount;
    }

    private static void verifyCrc(Segment seg, CRC32 crc) throws IOException {
        if (crc.getValue() != seg.crc()) {
            throw new IOException(String.format(
                    "CRC 校验失败: 期望 %d，实际 %d。文件可能已损坏，请重建索引。",
                    seg.crc(), crc.getValue()));
        }
    }

    /** 读一整块；循环到缓冲区填满为止，避免单次 read 返回不足。 */
    private static void readChunk(FileChannel ch, ByteBuffer buf, int want, CRC32 crc)
            throws IOException {
        buf.clear();
        buf.limit(want);
        readFully(ch, buf);
        buf.position(0);
        crc.update(buf.array(), 0, want);
    }

    private static void readFully(FileChannel ch, ByteBuffer buf) throws IOException {
        while (buf.hasRemaining()) {
            if (ch.read(buf) < 0) {
                throw new EOFException("文件提前结束");
            }
        }
    }
}
