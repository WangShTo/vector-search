# 绑核跑并发基准 —— 专治混合架构（P 核 + E 核）上的吞吐抖动
#
#   .\scripts\bench_concurrency_pinned.ps1                    # 默认绑逻辑核 0-7（8 个 P 核主线程）
#   .\scripts\bench_concurrency_pinned.ps1 -Mask 0xFFFF       # 绑 16 个 P 核线程（含超线程）
#   .\scripts\bench_concurrency_pinned.ps1 -Mask 0xFFFFFF -NoPin   # 不绑，做对照
#
# 为什么需要它：
#   这台机器是 i7-14650HX（8 P 核 + 8 E 核，24 逻辑核）。不绑核时，8 线程的基准
#   会在 P 核和 E 核之间迁移，同一配置不同轮次能差 4 倍。实测 E 核算力只有 P 核的
#   0.55 倍（见 README），所以「加线程反而更慢」有很大一部分不是锁争用，是调度。
#
# 绑核的两个层面：
#   1. JMH 主进程 —— Start-Process 之后立刻设 ProcessorAffinity；
#   2. JMH fork 出来的 worker 进程 —— 它是 JMH 自己 spawn 的，只能轮询发现后补设。
#   少了第 2 步等于没绑：真正跑基准的是 worker。
#
# ⚠️ 跑之前先停掉 vector-search 的两个服务，否则它们会抢核心污染数据。

[CmdletBinding()]
param(
    # 逻辑 CPU 掩码。0xFF = 0-7（8 个 P 核主线程）；0xFFFF = 0-15（P 核含超线程）
    [int64]$Mask = 0xFF,
    [switch]$NoPin,
    # 注意方法名：1 线程是单数的 searchWith1Thread，其余是复数 Threads
    [string]$Benchmarks = 'ConcurrentSearchBenchmark\.searchWith(1Thread|4Threads|8Threads)$',
    [int]$NodeCount = 20000,
    [int]$Dimension = 512,
    [int]$EfSearch = 50,
    [int]$TopK = 10,
    [int]$WarmupIterations = 3,
    [int]$MeasureIterations = 8,
    [string]$IterationTime = '1s',
    [string]$OutFile = 'bench/concurrent-pinned.json'
)

$ErrorActionPreference = 'Continue'
. "$PSScriptRoot\env.ps1"

$javaDir = Join-Path $RepoRoot 'vector-search-java'
$logDir = Join-Path $RepoRoot 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stdout = Join-Path $logDir 'concurrent-pinned.out.log'
$stderr = Join-Path $logDir 'concurrent-pinned.err.log'

$rule = '=' * 74
Write-Host $rule
Write-Host '并发基准（绑核）'
Write-Host $rule
if ($NoPin) {
    Write-Host '  亲和性      : 不绑（对照组）'
} else {
    Write-Host ('  亲和性      : 0x{0:X}（逻辑核 {1}）' -f $Mask, (($Mask -band 0xFF).ToString('X2')))
}
Write-Host "  规模        : n=$NodeCount dim=$Dimension efSearch=$EfSearch topK=$TopK"
Write-Host "  轮次        : 预热 $WarmupIterations × $IterationTime，测量 $MeasureIterations × $IterationTime"
Write-Host "  过滤器      : $Benchmarks"
Write-Host ''

Push-Location $javaDir
try {
    # ---------- 1. 编译 + classpath ----------
    Write-Host '[1/3] 编译'
    & $MvnCmd '-B' '-ntp' '-q' 'test-compile' 2>&1 | Where-Object { $_ -notmatch 'incubator' } | ForEach-Object { Write-Host "      $_" }
    if ($LASTEXITCODE -ne 0) { throw "编译失败，退出码 $LASTEXITCODE" }

    if (-not (Test-Path 'target\cp.txt')) {
        Write-Host '[2/3] 导出依赖 classpath'
        & $MvnCmd '-B' '-ntp' '-q' 'dependency:build-classpath' '-Dmdep.outputFile=target/cp.txt' '-Dmdep.includeScope=test' 2>&1 |
            Where-Object { $_ -notmatch 'incubator' } | ForEach-Object { Write-Host "      $_" }
        if ($LASTEXITCODE -ne 0) { throw "导出 classpath 失败" }
    } else {
        Write-Host '[2/3] 复用已有的 target\cp.txt'
    }
    $classpath = "target\classes;target\test-classes;$((Get-Content 'target\cp.txt' -Raw).Trim())"

    # ---------- 2. 组装 JMH 参数 ----------
    $jmhArgs = @(
        '--add-modules', 'jdk.incubator.vector',
        '-cp', $classpath,
        'org.openjdk.jmh.Main',
        '-p', "nodeCount=$NodeCount",
        '-p', "dimension=$Dimension",
        '-p', "m=16",
        '-p', 'efConstruction=200',
        '-p', "efSearch=$EfSearch",
        '-p', "topK=$TopK",
        '-f', '1',
        '-wi', "$WarmupIterations",
        '-i', "$MeasureIterations",
        '-w', $IterationTime,
        '-r', $IterationTime,
        '-rf', 'json',
        # 路径含空格（G:\all workspace\...），必须自带引号：
        # Start-Process 的 -ArgumentList 数组会用空格拼接，不会替元素加引号，
        # 不加的话 JMH 只收到 "-rff G:\all"，剩下半截当额外参数 —— 结果文件写到 G:\all。
        '-rff', "`"$(Join-Path $RepoRoot $OutFile)`"",
        $Benchmarks
    )

    # ---------- 3. 启动并绑核 ----------
    Write-Host '[3/3] 运行 JMH（会先建图，约 30 秒后才出第一行结果）'
    Remove-Item $stdout, $stderr -ErrorAction SilentlyContinue
    $launch = Get-Date
    $proc = Start-Process -FilePath $JavaExe -ArgumentList $jmhArgs -PassThru `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden

    if (-not $NoPin) {
        Start-Sleep -Milliseconds 400
        try {
            $proc.ProcessorAffinity = [IntPtr]$Mask
            # 基准跑 1-2 分钟，临时提到 High：后台进程（浏览器、这个 agent）抢不到核心。
            # 结束后进程自然退出，优先级随之消失，不需要还原。
            $proc.PriorityClass = 'High'
        } catch { Write-Host "      主进程绑核失败: $($_.Exception.Message)" }
    }

    $pinned = @{}
    $lastSize = 0
    while (-not $proc.HasExited) {
        if (-not $NoPin) {
            # JMH 会 fork 一个 worker JVM —— 真正跑基准的是它，必须也绑上
            foreach ($p in (Get-Process -Name java -ErrorAction SilentlyContinue)) {
                try {
                    if ($p.StartTime -lt $launch) { continue }
                    if ([int64]$p.ProcessorAffinity -ne $Mask) {
                        $p.ProcessorAffinity = [IntPtr]$Mask
                        try { $p.PriorityClass = 'High' } catch { }
                        if (-not $pinned.ContainsKey($p.Id)) {
                            $pinned[$p.Id] = $true
                            Write-Host ("      已绑核 PID {0}（{1}）" -f $p.Id, $p.ProcessName)
                        }
                    }
                } catch { }
            }
        }
        # 把日志增量转出来，免得看起来像卡死
        if (Test-Path $stdout) {
            $lines = @(Get-Content $stdout -ErrorAction SilentlyContinue)
            if ($lines.Count -gt $lastSize) {
                $lines[($lastSize)..($lines.Count - 1)] | Where-Object { $_ -notmatch 'incubator' } | ForEach-Object { Write-Host "      $_" }
                $lastSize = $lines.Count
            }
        }
        Start-Sleep -Milliseconds 500
    }
    $proc.WaitForExit()
    if (Test-Path $stdout) {
        $lines = @(Get-Content $stdout)
        if ($lines.Count -gt $lastSize) { $lines[$lastSize..($lines.Count - 1)] | ForEach-Object { Write-Host "      $_" } }
    }
    # Start-Process 配输出重定向时 $proc.ExitCode 常为 null（进程对象早于退出就释放了），
    # 所以不拿退出码判成败，直接看结果文件 —— 那才是真正要的产物。
    $resultPath = Join-Path $RepoRoot $OutFile
    if (-not (Test-Path $resultPath)) {
        Write-Host 'JMH 没有产出结果文件，stderr 末尾：'
        if (Test-Path $stderr) { Get-Content $stderr | Select-Object -Last 20 | ForEach-Object { Write-Host "      $_" } }
        throw 'JMH 运行失败'
    }
    Write-Host ''
    Write-Host ("完成。原始结果: " + $resultPath)
} finally {
    Pop-Location
}
