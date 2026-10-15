# 开发入口 —— 一条命令管理两个服务
#
#   .\scripts\dev.ps1 start     启动 Python 模型服务 + Java 检索服务
#   .\scripts\dev.ps1 stop      停止两个服务
#   .\scripts\dev.ps1 status    查看健康状态与索引统计
#   .\scripts\dev.ps1 load      把 data/eval/corpus.jsonl 灌进索引
#   .\scripts\dev.ps1 smoke     跑端到端冒烟测试
#   .\scripts\dev.ps1 eval      跑评测并出报告
#
# 服务日志写到 logs/ 目录（不入版本控制）。

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('start', 'stop', 'status', 'load', 'smoke', 'eval', 'restart', 'help')]
    [string]$Action = 'help'
)

$ErrorActionPreference = 'Continue'
. "$PSScriptRoot\env.ps1"

$modelDir = Join-Path $RepoRoot 'model-service'
$javaDir = Join-Path $RepoRoot 'vector-search-java'
$logDir = Join-Path $RepoRoot 'logs'
$pidFile = Join-Path $logDir 'services.json'
$javaUrl = 'http://127.0.0.1:8080'
$modelUrl = 'http://127.0.0.1:8000'

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Get-VenvPython {
    $candidate = Join-Path $modelDir '.venv\Scripts\python.exe'
    if (Test-Path $candidate) { return $candidate }
    # 没建虚拟环境就退回系统 python，并给出提示
    Write-Host "  [提示] 未找到虚拟环境，使用系统 python。建议先执行：" -ForegroundColor Yellow
    Write-Host "         cd model-service; python -m venv .venv; .venv\Scripts\python.exe -m pip install -r requirements.txt" -ForegroundColor Yellow
    return 'python'
}

function Test-Endpoint {
    param([string]$Url, [int]$TimeoutSec = 3)
    try {
        $response = Invoke-WebRequest -Uri $Url -TimeoutSec $TimeoutSec -UseBasicParsing
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Wait-Endpoint {
    param([string]$Url, [string]$Name, [int]$TimeoutSec = 180)
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (Test-Endpoint -Url $Url) { return $true }
        Start-Sleep -Seconds 3
    }
    Write-Host "  [失败] $Name 在 $TimeoutSec 秒内没有就绪" -ForegroundColor Red
    return $false
}

function Save-Pids {
    param($ModelPid, $JavaPid)
    @{ model = $ModelPid; java = $JavaPid } | ConvertTo-Json | Set-Content -Path $pidFile -Encoding UTF8
}

function Stop-Services {
    Write-Host "停止服务..."
    if (Test-Path $pidFile) {
        $pids = Get-Content $pidFile -Raw | ConvertFrom-Json
        foreach ($name in @('model', 'java')) {
            $processId = $pids.$name
            if ($processId) {
                $proc = Get-Process -Id $processId -ErrorAction SilentlyContinue
                if ($proc) {
                    # 杀整棵进程树：mvn spring-boot:run 会再拉起一个 java 子进程，
                    # 只杀父进程会留下占着 8080 端口的孤儿
                    & taskkill /PID $processId /T /F 2>&1 | Out-Null
                    Write-Host "  已停止 $name (PID $processId)"
                }
            }
        }
        Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    } else {
        Write-Host "  没有记录在案的进程"
    }

    # 兜底：按端口清理可能残留的孤儿进程
    foreach ($port in @(8000, 8080)) {
        $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
        foreach ($conn in $conns) {
            & taskkill /PID $conn.OwningProcess /T /F 2>&1 | Out-Null
            Write-Host "  清理端口 $port 上的残留进程 (PID $($conn.OwningProcess))"
        }
    }
}

function Start-Services {
    Write-Host "启动服务..."

    if (Test-Endpoint -Url "$modelUrl/health") {
        Write-Host "  Python 模型服务已在运行，跳过"
    } else {
        $python = Get-VenvPython
        $pyOut = Join-Path $logDir 'model-service.log'
        $pyErr = Join-Path $logDir 'model-service.err.log'
        $pyProc = Start-Process -FilePath $python `
            -ArgumentList @('-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', '8000') `
            -WorkingDirectory $modelDir `
            -RedirectStandardOutput $pyOut -RedirectStandardError $pyErr `
            -WindowStyle Hidden -PassThru
        Write-Host "  Python 模型服务已拉起 (PID $($pyProc.Id))，首次启动要下载模型，请耐心等待"
        Write-Host "  日志: $pyOut"
    }

    if (Test-Endpoint -Url "$javaUrl/api/stats") {
        Write-Host "  Java 检索服务已在运行，跳过"
    } else {
        $javaOut = Join-Path $logDir 'vector-search.log'
        $javaErr = Join-Path $logDir 'vector-search.err.log'
        $javaProc = Start-Process -FilePath 'cmd.exe' `
            -ArgumentList @('/c', "`"$MvnCmd`" -B -ntp spring-boot:run") `
            -WorkingDirectory $javaDir `
            -RedirectStandardOutput $javaOut -RedirectStandardError $javaErr `
            -WindowStyle Hidden -PassThru
        Write-Host "  Java 检索服务已拉起 (PID $($javaProc.Id))"
        Write-Host "  日志: $javaOut"
    }

    Save-Pids -ModelPid $pyProc.Id -JavaPid $javaProc.Id

    Write-Host ""
    Write-Host "等待服务就绪（首次启动需下载模型，可能要几分钟）..."
    $modelOk = Wait-Endpoint -Url "$modelUrl/health" -Name '模型服务'
    if (-not $modelOk) { Write-Host "  查看日志: $(Join-Path $logDir 'model-service.err.log')" -ForegroundColor Red }
    $javaOk = Wait-Endpoint -Url "$javaUrl/api/stats" -Name 'Java 服务'
    if (-not $javaOk) { Write-Host "  查看日志: $(Join-Path $logDir 'vector-search.err.log')" -ForegroundColor Red }

    if ($modelOk -and $javaOk) {
        Write-Host ""
        Write-Host "两个服务都已就绪：" -ForegroundColor Green
        Write-Host "  Java   $javaUrl/api/search"
        Write-Host "  Python $modelUrl/docs"
    }
}

function Show-Status {
    Write-Host "服务状态"
    Write-Host "  Python 模型服务: $(if (Test-Endpoint "$modelUrl/health") { '正常' } else { '不可达' })"
    Write-Host "  Java 检索服务  : $(if (Test-Endpoint "$javaUrl/api/stats") { '正常' } else { '不可达' })"
    if (Test-Endpoint "$javaUrl/api/stats") {
        try {
            $stats = Invoke-WebRequest -Uri "$javaUrl/api/stats" -UseBasicParsing | Select-Object -ExpandProperty Content | ConvertFrom-Json
            Write-Host ""
            Write-Host "索引统计"
            Write-Host "  实现      : $($stats.indexType)"
            Write-Host "  向量数    : $($stats.indexedCount)"
            Write-Host "  维度      : $($stats.dimension)"
            Write-Host "  文档数    : $($stats.storedDocuments)"
            if ($stats.hnsw) {
                Write-Host "  最大层    : $($stats.hnsw.maxLevel)"
                Write-Host "  总边数    : $($stats.hnsw.totalLinks)"
            }
        } catch {
            Write-Host "  （解析 stats 失败）"
        }
    }
}

function Invoke-Load {
    $python = Get-VenvPython
    & $python (Join-Path $PSScriptRoot 'load_corpus.py') --service $javaUrl
}

function Invoke-Smoke {
    $python = Get-VenvPython
    & $python (Join-Path $PSScriptRoot 'smoke_test.py')
}

function Invoke-Eval {
    $python = Get-VenvPython
    $queries = Join-Path $RepoRoot 'data\eval\queries.jsonl'
    if (-not (Test-Path $queries)) {
        Write-Host "  评测集不存在，先构建：" -ForegroundColor Yellow
        Write-Host "    python eval/build_dataset.py --corpus-dir data/corpus --out data/eval"
        return
    }
    & $python (Join-Path $RepoRoot 'eval\evaluate.py') --queries $queries --top-k 10 --out (Join-Path $RepoRoot 'bench\results')
}

switch ($Action) {
    'start'   { Start-Services }
    'stop'    { Stop-Services }
    'restart' { Stop-Services; Start-Sleep -Seconds 2; Start-Services }
    'status'  { Show-Status }
    'load'    { Invoke-Load }
    'smoke'   { Invoke-Smoke }
    'eval'    { Invoke-Eval }
    default {
        Write-Host "用法: .\scripts\dev.ps1 <动作>"
        Write-Host ""
        Write-Host "  start    启动两个服务（后台运行，日志在 logs/）"
        Write-Host "  stop     停止两个服务"
        Write-Host "  restart  重启"
        Write-Host "  status   查看健康状态与索引统计"
        Write-Host "  load     把 data/eval/corpus.jsonl 灌进索引"
        Write-Host "  smoke    端到端冒烟测试"
        Write-Host "  eval     跑检索质量评测"
    }
}
