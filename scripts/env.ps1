# 工具链探测 —— 被其他脚本 dot-source 使用
#
#   . "$PSScriptRoot\env.ps1"
#
# 设置：$env:JAVA_HOME、$env:PATH，并导出 $JavaExe / $MvnCmd / $RepoRoot
#
# 为什么需要这个文件：Java / Maven 可能装在 PATH 里、也可能装在便携目录里，
# 脚本不应该硬编码路径 —— 那会让项目在别人机器上跑不起来（简历项目的大忌）。

# 注意：这里刻意**不**设置 $ErrorActionPreference。
#
# 本文件是被 dot-source 的，设置它会污染调用方的全局偏好 —— 而且是在调用方
# 设完之后执行的，会把调用方的设定悄悄覆盖掉。
#
# 这个坑真实发生过：run_bench.ps1 先设了 'Continue'（因为它要调用 java，
# 而 JVM 会往 stderr 打 "Using incubator modules" 警告，在 'Stop' 语义下
# 会被当成终止错误），结果紧接着 dot-source 本文件，偏好被这里改回 'Stop'，
# 脚本于是死在那个无害的警告上。
#
# 被 dot-source 的库脚本应该只导出变量和函数，不产生全局副作用。
# 需要严格模式的调用方自己设，本文件内部的探测全部显式指定 -ErrorAction。

$script:RepoRoot = Split-Path -Parent $PSScriptRoot

function Find-Jdk {
    # 1) JAVA_HOME 优先
    if ($env:JAVA_HOME -and (Test-Path (Join-Path $env:JAVA_HOME 'bin\java.exe'))) {
        return $env:JAVA_HOME
    }
    # 2) 便携工具目录（本项目安装位置）
    $tools = Join-Path $env:USERPROFILE 'tools'
    if (Test-Path $tools) {
        $candidate = Get-ChildItem $tools -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '^jdk' -and (Test-Path (Join-Path $_.FullName 'bin\java.exe')) } |
            Sort-Object Name -Descending | Select-Object -First 1
        if ($candidate) { return $candidate.FullName }
    }
    # 3) 常见安装位置
    foreach ($root in @('C:\Program Files\Eclipse Adoptium', 'C:\Program Files\Java',
                        'C:\Program Files\Microsoft', 'C:\Program Files\Amazon Corretto')) {
        if (-not (Test-Path $root)) { continue }
        $candidate = Get-ChildItem $root -Directory -ErrorAction SilentlyContinue |
            Where-Object { Test-Path (Join-Path $_.FullName 'bin\java.exe') } |
            Sort-Object Name -Descending | Select-Object -First 1
        if ($candidate) { return $candidate.FullName }
    }
    # 4) PATH 上的 java
    $onPath = Get-Command java -ErrorAction SilentlyContinue
    if ($onPath) { return (Split-Path -Parent (Split-Path -Parent $onPath.Source)) }
    return $null
}

function Find-Maven {
    $onPath = Get-Command mvn -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }
    if ($env:MAVEN_HOME -and (Test-Path (Join-Path $env:MAVEN_HOME 'bin\mvn.cmd'))) {
        return (Join-Path $env:MAVEN_HOME 'bin\mvn.cmd')
    }
    $tools = Join-Path $env:USERPROFILE 'tools'
    $candidate = Join-Path $tools 'maven\bin\mvn.cmd'
    if (Test-Path $candidate) { return $candidate }
    foreach ($root in @('C:\Program Files\apache-maven*', 'C:\apache-maven*')) {
        $found = Get-ChildItem $root -ErrorAction SilentlyContinue |
            ForEach-Object { Join-Path $_.FullName 'bin\mvn.cmd' } |
            Where-Object { Test-Path $_ } | Select-Object -First 1
        if ($found) { return $found }
    }
    return $null
}

$jdkHome = Find-Jdk
if (-not $jdkHome) {
    throw "找不到 JDK。请设置 JAVA_HOME，或把 JDK 解压到 $env:USERPROFILE\tools\jdk-*"
}
$mvnCmd = Find-Maven
if (-not $mvnCmd) {
    throw "找不到 Maven。请把 Maven 加入 PATH，或解压到 $env:USERPROFILE\tools\maven"
}

$env:JAVA_HOME = $jdkHome
$javaBin = Join-Path $jdkHome 'bin'
if ($env:PATH -notlike "*$javaBin*") {
    $env:PATH = "$javaBin;$env:PATH"
}

$script:JavaExe = Join-Path $javaBin 'java.exe'
$script:MvnCmd = $mvnCmd

# 版本自检：Spring Boot 3 需要 17+，低于此版本会在编译时给出难懂的错误。
#
# 从 JDK 的 release 文件读版本，而不是解析 `java -version`。两个原因：
#   1. java -version 把版本号写到 stderr，在本脚本 ErrorActionPreference='Stop'
#      的设定下，PowerShell 会把这行 stderr 当成终止错误直接抛出（踩过这个坑）
#   2. release 文件的 "JAVA_VERSION=..." 格式稳定，不随 JDK 发行版变化
$major = $null
$releaseFile = Join-Path $jdkHome 'release'
if (Test-Path $releaseFile) {
    $match = Select-String -Path $releaseFile -Pattern '^JAVA_VERSION="(\d+)' -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($match) { $major = [int]$match.Matches[0].Groups[1].Value }
}
if ($null -ne $major -and $major -lt 17) {
    throw "检测到 Java $major，但本项目需要 Java 17 及以上（JAVA_HOME=$jdkHome）"
}
