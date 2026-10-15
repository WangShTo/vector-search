# 跑基准并生成报告
#
#   .\scripts\run_bench.ps1                           # 全量（10 万条，耗时较长）
#   .\scripts\run_bench.ps1 -NodeCount 5000 -Quick     # 快速验证链路
#   .\scripts\run_bench.ps1 -SkipJmh                   # 只测 recall/延迟
#
# 产出：
#   bench/results/recall.json     召回率与延迟扫描（RecallRunner）
#   bench/results/jmh.json        JMH 原始结果
#   bench/REPORT.md               渲染后的 Markdown 报告（可直接贴进简历/博客）
#
# ⚠️ 跑基准前请先停掉正在运行的 Java / Python 服务，否则它们会争抢 CPU 污染数据。

[CmdletBinding()]
param(
    [int]$NodeCount = 100000,
    [int]$Dimension = 512,
    [int]$TopK = 10,
    [int]$M = 16,
    [int]$EfConstruction = 200,
    [string]$EfSearch = "10,20,50,100,200,400",
    [int]$Queries = 200,
    [switch]$SkipJmh,
    [switch]$SkipRecall,
    # 快速模式：大幅削减 JMH 的预热/测量轮次。
    # 数字会变粗糙，但能在几十秒内验证整条链路 —— 调脚本时很有用。
    # 正式出数据不要加这个开关。
    [switch]$Quick
)

# 这里刻意用 Continue 而不是 Stop。
#
# 原因：本脚本要调用 java / mvn 这些原生程序，而 JVM 会把
# "WARNING: Using incubator modules" 写到 stderr。在 Stop 语义下，
# PowerShell 会把这行 stderr 当成「终止错误」直接抛出 —— 脚本会因为这个
# 完全无害的警告而死掉。（这个坑在 env.ps1 里踩过一次，这里是第二次。）
#
# 代价是必须自己检查每一步的 $LASTEXITCODE，所以下面每个原生命令后面都有显式判断。
$ErrorActionPreference = 'Continue'

. "$PSScriptRoot\env.ps1"

$javaDir = Join-Path $RepoRoot 'vector-search-java'
$benchDir = Join-Path $RepoRoot 'bench'
$resultDir = Join-Path $benchDir 'results'
New-Item -ItemType Directory -Force -Path $resultDir | Out-Null

# JVM 每次启动都会打的噪音，过滤掉以免淹没真正的输出
$noisePattern = 'Using incubator modules'

function Invoke-Native {
    param(
        [Parameter(Mandatory)][string]$Exe,
        [Parameter(Mandatory)][string[]]$Arguments,
        [Parameter(Mandatory)][string]$StepName
    )
    # "$_" 把 ErrorRecord 转成普通字符串，否则管道里混着错误对象会让
    # Where-Object 的匹配行为变得难以预料
    & $Exe @Arguments 2>&1 |
        ForEach-Object { "$_" } |
        Where-Object { $_ -notmatch $noisePattern } |
        ForEach-Object { Write-Host "      $_" }
    if ($LASTEXITCODE -ne 0) {
        throw "$StepName 失败，退出码 $LASTEXITCODE"
    }
}

$rule = '=' * 74 -join ''
Write-Host $rule
Write-Host "向量检索基准"
Write-Host $rule
Write-Host "  JDK          : $env:JAVA_HOME"
Write-Host "  Maven        : $MvnCmd"
Write-Host "  规模         : n=$NodeCount  dim=$Dimension  topK=$TopK"
Write-Host "  HNSW 参数    : m=$M  efConstruction=$EfConstruction"
Write-Host "  efSearch 扫描: $EfSearch"
Write-Host ""

Push-Location $javaDir
try {
    # ---------- 1. 编译 ----------
    Write-Host "[1/4] 编译（含基准代码）"
    Invoke-Native -Exe $MvnCmd -StepName '编译' -Arguments @('-B', '-ntp', '-q', 'test-compile')
    Write-Host "      OK"

    # ---------- 2. 依赖 classpath ----------
    Write-Host "[2/4] 导出依赖 classpath"
    Invoke-Native -Exe $MvnCmd -StepName '导出 classpath' -Arguments @(
        '-B', '-ntp', '-q', 'dependency:build-classpath',
        '-Dmdep.outputFile=target/cp.txt', '-Dmdep.includeScope=test'
    )
    $deps = (Get-Content 'target\cp.txt' -Raw).Trim()
    $classpath = "target\classes;target\test-classes;$deps"
    Write-Host "      OK"

    $jvmArgs = @('--add-modules', 'jdk.incubator.vector')

    # ---------- 3. 召回率扫描 ----------
    if (-not $SkipRecall) {
        Write-Host "[3/4] 召回率与延迟扫描（RecallRunner）"
        $recallOut = Join-Path $resultDir 'recall.json'
        Invoke-Native -Exe $JavaExe -StepName 'RecallRunner' -Arguments ($jvmArgs + @(
            "-Dbench.n=$NodeCount",
            "-Dbench.dim=$Dimension",
            "-Dbench.topK=$TopK",
            "-Dbench.m=$M",
            "-Dbench.efConstruction=$EfConstruction",
            "-Dbench.queries=$Queries",
            "-Dbench.efSearch=$EfSearch",
            "-Dbench.out=$recallOut",
            '-cp', $classpath,
            'com.vectorsearch.bench.RecallRunner'
        ))
    } else {
        Write-Host "[3/4] 跳过召回率扫描"
    }

    # ---------- 4. JMH ----------
    if (-not $SkipJmh) {
        if ($Quick) {
            Write-Host "[4/4] JMH 微基准（快速模式：轮次已削减，数字仅供参考）"
        } else {
            Write-Host "[4/4] JMH 微基准（这一步最慢，可能需要十几分钟）"
        }
        $jmhOut = Join-Path $resultDir 'jmh.json'

        # 顺序有讲究：-cp 必须在主类之前（那是给 JVM 的），
        # 其余参数在主类之后（那是 JMH 自己解析的程序参数）
        $jmhArguments = $jvmArgs + @(
            '-cp', $classpath,
            'org.openjdk.jmh.Main',
            '-p', "nodeCount=$NodeCount",
            '-p', "dimension=$Dimension",
            '-p', "topK=$TopK",
            '-p', "m=$M",
            '-p', "efConstruction=$EfConstruction",
            '-rf', 'json', '-rff', $jmhOut
        )
        if ($Quick) {
            $jmhArguments += @('-f', '1', '-wi', '1', '-i', '2', '-w', '300ms', '-r', '500ms')
        }
        Invoke-Native -Exe $JavaExe -Arguments $jmhArguments -StepName 'JMH'
    } else {
        Write-Host "[4/4] 跳过 JMH"
    }
} finally {
    Pop-Location
}

# ---------- 渲染报告 ----------
Write-Host ""
Write-Host "渲染报告..."
$py = Join-Path $RepoRoot 'model-service\.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { $py = 'python' }
& $py (Join-Path $PSScriptRoot 'render_bench_report.py') `
    --results $resultDir `
    --out (Join-Path $benchDir 'REPORT.md') 2>&1 |
    ForEach-Object { Write-Host "      $_" }
if ($LASTEXITCODE -ne 0) { throw "报告渲染失败" }

Write-Host ""
Write-Host "完成。报告: $(Join-Path $benchDir 'REPORT.md')"
