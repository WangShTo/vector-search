"""构建评测集：语料 -> 分块 -> 带标注的查询。

## 为什么评测集要自己建

网上的中文检索数据集要么太大、要么和你的语料领域不匹配。
而"recall@10 达到 0.91"这句话，只有在<b>你的语料 + 你的标注</b>上才有意义。
面试官问"这个 0.91 是怎么测的"，你要能答出评测集的构造方法 —— 这就是答案。

## 评测方法：已知项检索（known-item search）

对语料里的一个分块，从它自己身上取一段文字当查询，那么"正确答案"
就是它自己所在的文档。这样不需要人工标注相关性，但要求检索系统
真的能把它找回来 —— 是一个干净、可复现、无争议的评测协议。

代价是它比真实查询简单（查询词就是文档里的词）。所以这个数字
应该理解为<b>上界</b>，不能直接等同于线上效果。这一点必须写进报告，
否则就是在误导人。

## 用法

    # 1. 把你的语料（课程讲义 / 论文 / 笔记）放进 data/corpus/
    # 2. 构建评测集
    python build_dataset.py --corpus-dir ../data/corpus --out ../data/eval --strategy paragraph
    # 3. 启动 Java 服务并灌入语料后，跑评测
    python evaluate.py --queries ../data/eval/queries.jsonl --out results/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

# Windows 控制台默认 GBK，print 非 ASCII 字符会抛 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".java", ".py", ".json"}


# ----------------------------------------------------------------------
# 分块策略 —— 第 5 周消融实验的对照变量
# ----------------------------------------------------------------------

def chunk_fixed(text: str, chunk_size: int, overlap: int) -> list[str]:
    """固定窗口切分。

    实现简单、块大小均匀（对模型友好），但会在句子中间切断，
    导致一个完整的语义单元被拆到两个块里，两边都检索不好。
    """
    text = text.strip()
    if not text:
        return []
    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks


def chunk_paragraph(text: str, chunk_size: int, overlap: int) -> list[str]:
    """按段落切分，超长段落再退化为固定窗口。

    尽量保持语义单元完整 —— 这是它相对固定窗口的全部理由，
    第 5 周要用 recall 数据验证这个直觉是否成立。
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            chunks.extend(chunk_fixed(paragraph, chunk_size, overlap))
            continue
        if len(buffer) + len(paragraph) + 1 <= chunk_size:
            buffer = f"{buffer}\n{paragraph}" if buffer else paragraph
        else:
            if buffer:
                chunks.append(buffer)
            buffer = paragraph
    if buffer:
        chunks.append(buffer)
    return chunks


STRATEGIES = {
    "fixed": chunk_fixed,
    "paragraph": chunk_paragraph,
}


# ----------------------------------------------------------------------
# 查询生成
# ----------------------------------------------------------------------

def make_query(chunk: str, target_length: int, rng: random.Random) -> str | None:
    """从分块里取一段文字当查询。

    优先取完整句子（更接近真实查询的形态），句子太短就向后拼接。
    """
    sentences = [s.strip() for s in re.split(r"(?<=[。！？!?.\n])", chunk) if s.strip()]
    if not sentences:
        return None

    start = rng.randrange(len(sentences))
    query = ""
    for sentence in sentences[start:] + sentences[:start]:
        query += sentence
        if len(query) >= target_length:
            break
        if len(query) >= target_length * 2:
            break

    query = re.sub(r"\s+", " ", query).strip()
    if len(query) < 8:
        return None
    return query[: target_length * 2]


# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# 剥离不适合做演示语料的内容
# ----------------------------------------------------------------------

FENCED_CODE = re.compile(r"```.*?```", re.S)
TABLE_ROW = re.compile(r"^[ \t]*\|.*$", re.M)
RULE = re.compile(r"^[ \t]*(?:-{3,}|\*{3,}|_{3,})[ \t]*$", re.M)
QUOTE = re.compile(r"^[ \t]*>[ \t]?", re.M)
INLINE_CODE = re.compile(r"`([^`\n]{1,120})`")
HEADING = re.compile(r"^[ \t]*#{1,6}[ \t]*", re.M)
# 粗体可能跨行，所以 . 要能吃换行（re.S）
BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)
LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")


def strip_non_prose(text: str) -> str:
    """去掉代码块、表格行与 Markdown 标记，只留下可读的散文。

    **为什么需要这一步**：项目的文档里混着大量代码与表格，直接切块会得到
    「able>」「*.」「| 字段 | 说明 |」这种碎片。拿它当演示语料，
    检索结果看起来就像系统坏了 —— 但问题在语料，不在检索。

    **注意这只用于演示语料**，不要用于评测语料：评测时标识符本身是有意义的
    检索目标（BM25 的优势正是精确匹配 `efSearch` 这类词），剥掉反而失真。
    """
    text = FENCED_CODE.sub("\n", text)
    text = TABLE_ROW.sub("", text)
    text = RULE.sub("", text)
    text = QUOTE.sub("", text)
    text = INLINE_CODE.sub(r"\1", text)   # 保留行内 code 的文字，去掉反引号
    text = LINK.sub(r"\1", text)
    text = HEADING.sub("", text)
    text = BOLD.sub(r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_corpus(corpus_dir: Path, strip_code: bool = False) -> list[dict]:
    files = sorted(
        p for p in corpus_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES
    )
    if not files:
        print(f"[错误] {corpus_dir} 下没有找到可读文本文件", file=sys.stderr)
        print(f"       支持的扩展名: {', '.join(sorted(TEXT_SUFFIXES))}", file=sys.stderr)
        sys.exit(1)

    documents = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            print(f"[跳过] {path}: {exc}", file=sys.stderr)
            continue
        if strip_code:
            text = strip_non_prose(text)
        documents.append({"source": str(path.relative_to(corpus_dir)), "text": text})
    return documents


def main() -> int:
    parser = argparse.ArgumentParser(description="构建检索评测集")
    parser.add_argument("--corpus-dir", required=True, help="语料目录")
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument("--strategy", default="paragraph", choices=sorted(STRATEGIES),
                        help="分块策略（消融实验的对照变量）")
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--query-count", type=int, default=150, help="查询条数")
    parser.add_argument("--seed", type=int, default=42, help="固定种子，保证评测集可复现")
    parser.add_argument("--strip-code", action="store_true",
                        help="剥离代码块、表格与 Markdown 标记，只留散文。"
                             "适合生成演示语料；评测语料不要用 —— "
                             "会丢掉标识符这个 BM25 的优势目标")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    documents = read_corpus(corpus_dir, strip_code=args.strip_code)
    chunker = STRATEGIES[args.strategy]

    chunks: list[dict] = []
    for doc_index, document in enumerate(documents):
        for piece in chunker(document["text"], args.chunk_size, args.overlap):
            chunk_id = len(chunks)
            chunks.append({
                "id": chunk_id,
                "source": document["source"],
                "doc_index": doc_index,
                "text": piece,
            })

    if not chunks:
        print("[错误] 分块结果为空，检查 --chunk-size 是否过大", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    sample_size = min(args.query_count, len(chunks))
    sampled = rng.sample(chunks, sample_size)

    queries = []
    for chunk in sampled:
        query_text = make_query(chunk["text"], target_length=24, rng=rng)
        if query_text is None:
            continue
        queries.append({
            "query": query_text,
            "relevant_ids": [chunk["id"]],
            # 记录来源，便于人工抽查"这个查询到底合不合理"
            "source": chunk["source"],
        })

    corpus_path = out_dir / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    queries_path = out_dir / "queries.jsonl"
    with queries_path.open("w", encoding="utf-8") as f:
        for query in queries:
            f.write(json.dumps(query, ensure_ascii=False) + "\n")

    # 记录构建参数：换了分块策略或种子，两次评测的数字就没法比
    meta = {
        "strategy": args.strategy,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "seed": args.seed,
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "query_count": len(queries),
        "corpus_fingerprint": hashlib.sha256(
            "\n".join(c["text"] for c in chunks).encode("utf-8")
        ).hexdigest()[:16],
    }
    (out_dir / "dataset_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"语料文件    : {len(documents)} 个")
    print(f"分块策略    : {args.strategy} (size={args.chunk_size}, overlap={args.overlap})")
    print(f"分块总数    : {len(chunks)}")
    print(f"查询条数    : {len(queries)}")
    print(f"语料指纹    : {meta['corpus_fingerprint']}")
    print(f"输出        : {corpus_path}")
    print(f"              {queries_path}")
    print()
    print("提示：语料指纹变了的话，前后的 recall 数字不可比。")
    print("     做消融实验时，只有 --strategy / 索引参数可以变，语料和种子必须固定。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
