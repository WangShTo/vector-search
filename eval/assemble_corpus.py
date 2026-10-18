"""组装评测语料 —— 把项目里的文档与源码收集成一个目录，供 build_dataset.py 切分。

## 为什么需要这一步

要公平对比 vector / bm25 / hybrid，语料必须**足够大且足够杂**：

- 太小（几十个分块）时三种模式都是 recall@10 = 1.0，看不出差别
- 只有自然语言、没有标识符时，BM25 的优势体现不出来

所以这里把项目自己的文档和源码收进来：既有中文技术散文，
又有 `efSearch`、`HnswIndex` 这类精确标识符 —— 正好覆盖两类查询。

## 为什么不直接把这些文件提交进仓库

它们是<b>副本</b>。副本会随源文件改动而过期，而且白白多占一份体积。
所以 `data/corpus-full/` 是 gitignore 的，用这个脚本随时重建。

## 用法

    python eval/assemble_corpus.py
    python eval/build_dataset.py --corpus-dir data/corpus-full --out data/eval \
        --query-count 150 --chunk-size 400
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


def main() -> int:
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
    print(f"语料已组装: {copied} 个文件, {total_bytes / 1024:.0f} KB -> {TARGET}")
    print()
    print("下一步:")
    print("  python eval/build_dataset.py --corpus-dir data/corpus-full --out data/eval \\")
    print("      --query-count 150 --chunk-size 400")
    return 0


if __name__ == "__main__":
    sys.exit(main())
