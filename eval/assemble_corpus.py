"""组装两份语料 —— 一次产出「演示用」和「评测用」。

## 为什么要分两份

同一个项目，两种用途，需要的语料**不一样**：

| 用途 | 语料 | 要求 |
|---|---|---|
| 演示页面 | `data/corpus-demo/` 只含项目文档 | 结果要**人能读**，像知识库问答 |
| 评测 | `data/corpus-full/` 文档 + 源码 | 要有 `efSearch` 这类标识符，BM25 才有靶子 |

第一版演示直接拿源码当语料，搜出来全是 `able>`、`*.` 这种碎片 ——
页面看起来像系统坏了，但问题在语料，不在检索。

## 为什么不把这些文件提交进仓库

它们是<b>副本</b>。副本会随源文件改动而过期，而且白白多占一份体积。
两个目录都 gitignore 了，用这个脚本随时重建。

## 用法

    python eval/assemble_corpus.py

然后按脚本打印的提示分别建数据集：演示那份加 `--strip-code`，
评测那份不加（标识符本身是检索目标）。
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data" / "corpus-full"


def copy_file(source: Path, destination: Path) -> bool:
    if not source.exists():
        print(f"  [跳过] {source} 不存在", file=sys.stderr)
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def build_demo_corpus() -> None:
    """演示语料：只要项目文档，**不要源码**。

    为什么单独做一份：演示页面要看起来像「知识库问答」。第一版直接拿源码当语料，
    搜出来全是 `able>`、`*.` 这种碎片 —— 页面看起来像系统坏了，但问题在语料。
    再配合 `build_dataset.py --strip-code` 剥掉代码块与表格，就只剩可读的散文。

    评测语料（corpus-full）不能这么做：评测**需要** `efSearch`、`HnswIndex`
    这类标识符作为精确匹配目标，剥掉反而失真。
    """
    target = ROOT / "data" / "corpus-demo"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    copied = 0
    for path in sorted((ROOT / "data" / "corpus").glob("*.md")):
        copied += copy_file(path, target / path.name)
    for source, name in [
        (ROOT / "README.md", "项目说明.md"),
        (ROOT / "docs" / "RESUME.md", "项目复盘.md"),
        (ROOT / "bench" / "REPORT.md", "性能报告.md"),
    ]:
        copied += copy_file(source, target / name)

    total_bytes = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
    print(f"演示语料已组装: {copied} 个文件, {total_bytes / 1024:.0f} KB -> {target}")


def main() -> int:
    build_demo_corpus()
    print()

    if TARGET.exists():
        shutil.rmtree(TARGET)
    TARGET.mkdir(parents=True)

    copied = 0

    # 1. 手工编写的中文技术语料（语义查询主要靠它）
    for path in sorted((ROOT / "data" / "corpus").glob("*.md")):
        copied += copy_file(path, TARGET / path.name)

    # 2. 项目文档
    for source, name in [
        (ROOT / "README.md", "project-readme.md"),
        (ROOT / "docs" / "RESUME.md", "project-resume.md"),
        (ROOT / "bench" / "REPORT.md", "project-benchmark-report.md"),
    ]:
        copied += copy_file(source, TARGET / name)

    # 3. Java 源码（含大量标识符，BM25 的主场）
    java_root = ROOT / "vector-search-java" / "src" / "main" / "java"
    for path in sorted(java_root.rglob("*.java")):
        copied += copy_file(path, TARGET / "java" / path.name)

    # 4. Python 源码
    for path in sorted((ROOT / "model-service" / "app").glob("*.py")):
        copied += copy_file(path, TARGET / "python" / path.name)

    total_bytes = sum(p.stat().st_size for p in TARGET.rglob("*") if p.is_file())
    print(f"评测语料已组装: {copied} 个文件, {total_bytes / 1024:.0f} KB -> {TARGET}")
    print()
    print("下一步:")
    print("  # 演示（散文，剥掉代码与表格）")
    print("  python eval/build_dataset.py --corpus-dir data/corpus-demo --out data/eval-demo \\")
    print("      --chunk-size 260 --strip-code")
    print()
    print("  # 评测（混合语料，保留标识符）")
    print("  python eval/build_dataset.py --corpus-dir data/corpus-full --out data/eval \\")
    print("      --query-count 150 --chunk-size 400")
    return 0


if __name__ == "__main__":
    sys.exit(main())
