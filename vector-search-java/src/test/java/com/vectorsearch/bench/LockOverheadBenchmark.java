package com.vectorsearch.bench;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Threads;
import org.openjdk.jmh.annotations.Warmup;

import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.concurrent.locks.StampedLock;

/**
 * 读锁开销的微基准：值不值得把 {@link ReentrantReadWriteLock} 换成 {@link StampedLock}。
 *
 * <h2>这个基准要回答什么</h2>
 *
 * <p>{@code HnswIndex} 的 {@code search} 每次都要拿一次读锁，而一次检索在 10 万规模上
 * 约 <b>104 µs</b>（JMH，efSearch=10）。读锁本身的竞争代价是几十到几百纳秒的量级 ——
 * 如果占比只有千分之一，那么换锁就是纯粹的工程表演：代码更复杂、更容易写错，
 * 而收益小到任何测量都分不出来。
 *
 * <p>所以这里不猜，直接测四种写法的单次开销：
 * <ul>
 *   <li>{@code noLock}       —— 基准线，同时包含 JMH 循环本身的开销</li>
 *   <li>{@code rwReadLock}   —— 当前实现（{@link ReentrantReadWriteLock} 读锁）</li>
 *   <li>{@code stampedPessimistic} —— {@link StampedLock} 的悲观读锁（等价替换）</li>
 *   <li>{@code stampedOptimistic}  —— 乐观读 + 版本校验（不做任何共享写，无需加锁）</li>
 * </ul>
 *
 * <h2>怎么读这些数字</h2>
 *
 * <p>关键不是 ns 的绝对值，而是 <b>（带锁的读数 − noLock 的读数）÷ 104 µs</b>。
 * 用 1 线程和 8 线程两组数据，是因为读锁只有一个共享计数器，线程越多
 * 缓存行弹跳越厉害 —— 8 线程那组是「锁竞争最坏情况」的上界。
 *
 * <p>还有一层保守估计：真实检索持锁约 100 µs，也就是说这把锁每秒只被拿一万次左右，
 * 而本基准每轮拿锁的频率高好几个数量级。所以这里测出的相对开销
 * <b>比真实情况还偏高</b>，结论只会更站得住。
 *
 * <h2>正文为什么这么小</h2>
 *
 * <p>正文做得越少，锁的开销占比越大、越容易分辨。这里只累加 8 个 long，
 * 目的只是让 JIT 不能把整段临界区优化掉，而不是模拟检索。
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 3, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Fork(1)
@State(Scope.Benchmark)
public class LockOverheadBenchmark {

    /** 被所有线程共享的读锁 —— 这正是竞争发生的那个对象。 */
    private final ReentrantReadWriteLock rwLock = new ReentrantReadWriteLock();
    private final StampedLock stampedLock = new StampedLock();

    /** 临界区里要读的数据。故意用普通字段，避免 volatile 语义自己就贡献一份开销。 */
    private long[] data;

    @Setup
    public void setup() {
        data = new long[8];
        for (int i = 0; i < data.length; i++) {
            data[i] = i;
        }
    }

    /** 临界区正文：8 次数组读，几纳秒的量级。 */
    private long readBody() {
        long sum = 0;
        for (long v : data) {
            sum += v;
        }
        return sum;
    }

    // ---------------- 基准线 ----------------

    @Benchmark
    @Threads(1)
    public long noLock1() {
        return readBody();
    }

    @Benchmark
    @Threads(8)
    public long noLock8() {
        return readBody();
    }

    // ---------------- 当前实现：ReentrantReadWriteLock ----------------

    @Benchmark
    @Threads(1)
    public long rwReadLock1() {
        var lock = rwLock.readLock();
        lock.lock();
        try {
            return readBody();
        } finally {
            lock.unlock();
        }
    }

    @Benchmark
    @Threads(8)
    public long rwReadLock8() {
        var lock = rwLock.readLock();
        lock.lock();
        try {
            return readBody();
        } finally {
            lock.unlock();
        }
    }

    // ---------------- StampedLock 悲观读 ----------------

    @Benchmark
    @Threads(1)
    public long stampedPessimistic1() {
        long stamp = stampedLock.readLock();
        try {
            return readBody();
        } finally {
            stampedLock.unlockRead(stamp);
        }
    }

    @Benchmark
    @Threads(8)
    public long stampedPessimistic8() {
        long stamp = stampedLock.readLock();
        try {
            return readBody();
        } finally {
            stampedLock.unlockRead(stamp);
        }
    }

    // ---------------- StampedLock 乐观读 ----------------

    @Benchmark
    @Threads(1)
    public long stampedOptimistic1() {
        long stamp = stampedLock.tryOptimisticRead();
        long sum = readBody();
        if (!stampedLock.validate(stamp)) {
            // 真的撞上写入了：退化成悲观读重来一遍
            long readStamp = stampedLock.readLock();
            try {
                sum = readBody();
            } finally {
                stampedLock.unlockRead(readStamp);
            }
        }
        return sum;
    }

    @Benchmark
    @Threads(8)
    public long stampedOptimistic8() {
        long stamp = stampedLock.tryOptimisticRead();
        long sum = readBody();
        if (!stampedLock.validate(stamp)) {
            long readStamp = stampedLock.readLock();
            try {
                sum = readBody();
            } finally {
                stampedLock.unlockRead(readStamp);
            }
        }
        return sum;
    }

    // ================= 第二组：真实量级的临界区 =================
    //
    // 上面那组把临界区压到 2 ns，是为了让锁的开销占比最大化 —— 它能给出
    // 「这把锁本身多贵」，但**不能**直接换算成检索里的占比，因为那样测出来的
    // 拿锁频率比真实检索高两个数量级（1.1 µs 一次 vs 104 µs 一次）。
    //
    // 这一组换了个问法：让临界区真的干约 100 µs 的活（和一次检索同量级），
    // 再看加锁比不加锁多花多少。这才是能直接下结论的数字。

    /** 临界区里被反复扫的数组，模拟检索时的图遍历/距离计算负载。 */
    private float[] payload;

    @Setup
    public void setupPayload() {
        payload = new float[512];
        for (int i = 0; i < payload.length; i++) {
            payload[i] = (i % 7) * 0.5f;
        }
    }

    /**
     * 约 100 µs 的正文：对 512 维数组反复累加。
     *
     * <p>不追求精确等于 104 µs —— 量级对了就够。结论看的是「锁相对正文的占比」，
     * 正文快慢只会线性缩放这个比值，不影响「值不值得换锁」的判断。
     */
    private long heavyBody() {
        float sum = 0;
        for (int rep = 0; rep < 300; rep++) {
            for (int i = 0; i < payload.length; i++) {
                sum += payload[i];
            }
        }
        // 用返回值消费掉，避免整段被 JIT 当死代码删掉
        return (long) sum;
    }

    @Benchmark
    @Threads(1)
    public long noLockHeavy1() {
        return heavyBody();
    }

    @Benchmark
    @Threads(8)
    public long noLockHeavy8() {
        return heavyBody();
    }

    @Benchmark
    @Threads(8)
    public long rwReadLockHeavy8() {
        var lock = rwLock.readLock();
        lock.lock();
        try {
            return heavyBody();
        } finally {
            lock.unlock();
        }
    }

    @Benchmark
    @Threads(8)
    public long stampedOptimisticHeavy8() {
        long stamp = stampedLock.tryOptimisticRead();
        long sum = heavyBody();
        if (!stampedLock.validate(stamp)) {
            long readStamp = stampedLock.readLock();
            try {
                sum = heavyBody();
            } finally {
                stampedLock.unlockRead(readStamp);
            }
        }
        return sum;
    }
}
