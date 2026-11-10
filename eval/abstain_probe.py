#!/usr/bin/env python3
"""拒答（abstention）实验：哪些信号能区分「相关问题」和「不相关问题」。

## 为什么需要这个

面试里被追问最多的一句是「如果我问一个跟知识库不相关的问题，你的表现是什么」。
系统如果永远返回 top-k，它就永远不会承认自己不知道 —— 这在检索产品里是致命的。

更要命的是：**RRF 分数不能当阈值**。k=60 时单路 rank1 贡献 1/61≈0.0164，
融合分被结构性地压在 0.016~0.033 的窄带里，它只编码「排第几」，不编码「有多相关」。
所以「给融合分设个下限」这句话是错的 —— 这个脚本就是要用数据证明它错。

## 做法

对每条查询收四个信号（同一次运行、同一个索引）：

| 信号 | 来源 | 量纲 |
|---|---|---|
| `vectorTop1` | `mode=vector` topK=1 | 余弦相似度，归一化后 [-1, 1] |
| `bm25Top1`   | `mode=bm25` topK=1   | BM25，无上界 |
| `rrfTop1`    | `mode=hybrid` topK=1 | RRF，≈0.016~0.033 |
| `ceTop1`     | `mode=hybrid&rerank=true` topK=10 | cross-encoder 打分，[0, 1] |

正样本用 `data/eval/queries.jsonl`（已知项检索，每条都是「库里有答案」的问句），
负样本用 `eval/negatives.jsonl`（手写、跟代码一起提交，因为 `data/eval/` 被 gitignore 忽略，
放那里会让别人克隆下来跑不出数；两档难度：`off-domain` 完全跨领域、
`adjacent-tech` 同领域但库里没有答案）。

然后对每个信号扫阈值，算：
- **拒答率** = 负样本里被判为「不相关」的比例（越高越好）
- **误拒率** = 正样本里被判为「不相关」的比例（越低越好，这是用户真实痛点）
- **AUC** = 正负样本分数的可分性（不看阈值，只看信号本身有没有判别力）

## 用法

    python eval/abstain_probe.py --load            # 切到评测语料再跑（602 分块）
    python eval/abstain_probe.py --skip-load       # 索引已就绪
    python eval/abstain_probe.py --restore-demo    # 跑完把演示语料灌回去
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVICE = "http://127.0.0.1:8080"
DEV_PS1 = ROOT / "scripts" / "dev.ps1"


def ensure_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass


# ----------------------------------------------------------------------
# 服务
# ----------------------------------------------------------------------

def run(cmd: list[str], env: dict[str, str] | None = None, capture: bool = True,
        timeout: float = 900.0) -> str:
    """capture=False 用于 dev.ps1 start：守护进程会持有继承的 stdout 管道。"""
    kwargs: dict = {"cwd": str(ROOT), "env": env, "timeout": timeout}
    if capture:
        kwargs.update(capture_output=True, text=True, encoding="utf-8", errors="replace")
    proc = subprocess.run(cmd, **kwargs)
    if proc.returncode != 0:
        detail = ""
        if capture:
            detail = (proc.stdout or "")[-1200:] + (proc.stderr or "")[-1200:]
        raise RuntimeError(f"命令失败（exit={proc.returncode}）：{' '.join(cmd)}\n{detail}")
    return (proc.stdout or "") if capture else ""


def service_ready(timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(f"{SERVICE}/api/stats", timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def restart() -> None:
    try:
        run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(DEV_PS1), "stop"],
            capture=False)
    except RuntimeError:
        pass
    run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(DEV_PS1), "start"],
        capture=False)
    deadline = time.time() + 120
    while time.time() < deadline:
        if service_ready():
            return
        time.sleep(1.0)
    raise RuntimeError("重启后 /api/stats 不可达")


def load_corpus(corpus: Path) -> None:
    run([sys.executable, str(ROOT / "scripts" / "load_corpus.py"),
         "--service", SERVICE, "--corpus", str(corpus), "--reset"])
    run(["curl.exe", "-s", "-X", "POST", f"{SERVICE}/api/admin/snapshot"])


# ----------------------------------------------------------------------
# 检索
# ----------------------------------------------------------------------

def search(body: dict) -> dict:
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{SERVICE}/api/search",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last: Exception | None = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last = exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"检索失败：{exc if last is None else last}")


def top_scores(query: str, mode: str, top_k: int, rerank: bool) -> list[float]:
    out = search({"query": query, "mode": mode, "topK": top_k, "rerank": rerank})
    hits = out.get("hits") or []
    return [float(h.get("score", 0.0)) for h in hits]


def snoop(query: str) -> dict:
    """一条查询收齐四个信号。"""
    vec = top_scores(query, "vector", 1, False)
    bm = top_scores(query, "bm25", 1, False)
    rrf = top_scores(query, "hybrid", 1, False)
    out = search({"query": query, "mode": "hybrid", "topK": 10, "rerank": True})
    hits = out.get("hits") or []
    ce = [float(h.get("score", 0.0)) for h in hits]
    top = hits[0] if hits else {}
    return {
        "vectorTop1": vec[0] if vec else None,
        "bm25Top1": bm[0] if bm else None,
        "rrfTop1": rrf[0] if rrf else None,
        "ceTop1": ce[0] if ce else None,
        "ceMargin": (ce[0] - ce[1]) if len(ce) > 1 else None,
        "reranked": out.get("reranked"),
        "topId": top.get("id"),
        "topSnippet": (top.get("text") or "")[:60].replace("\n", " "),
    }


# ----------------------------------------------------------------------
# 分析
# ----------------------------------------------------------------------

def auc_of(positives: list[float], negatives: list[float]) -> float:
    """AUC = P(正样本分数 > 负样本分数)，用秩和公式算，不依赖第三方库。"""
    scores = [(s, 1) for s in positives] + [(s, 0) for s in negatives]
    scores.sort(key=lambda item: item[0])
    rank_sum = 0.0
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j + 1 < n and scores[j + 1][0] == scores[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 平均秩处理并列，1-based
        for k in range(i, j + 1):
            if scores[k][1] == 1:
                rank_sum += avg_rank
        i = j + 1
    n_pos = len(positives)
    n_neg = len(negatives)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def sweep(positives: list[float], negatives: list[float]) -> dict:
    """扫阈值，返回最优工作点与全量曲线。"""
    candidates = sorted({round(v, 6) for v in positives + negatives})
    curve = []
    best = None
    for t in candidates:
        refused_pos = sum(1 for s in positives if s < t)
        refused_neg = sum(1 for s in negatives if s < t)
        refusal_rate = refused_neg / len(negatives) if negatives else 0.0
        false_refusal = refused_pos / len(positives) if positives else 0.0
        youden = refusal_rate - false_refusal
        row = {
            "threshold": t,
            "refusalRate": round(refusal_rate, 4),
            "falseRefusalRate": round(false_refusal, 4),
            "accuracy": round((refused_neg + (len(positives) - refused_pos))
                              / (len(positives) + len(negatives)), 4),
        }
        curve.append(row)
        if best is None or youden > best["youden"]:
            best = dict(row, youden=round(youden, 4))
    return {"best": best, "curve": curve}


def analyse(signal: str, positives: list[float], negatives: list[float]) -> dict:
    pos = [v for v in positives if v is not None]
    neg = [v for v in negatives if v is not None]
    res = {
        "signal": signal,
        "positiveCount": len(pos),
        "negativeCount": len(neg),
        "auc": round(auc_of(pos, neg), 4) if pos and neg else None,
        "positiveMin": round(min(pos), 4) if pos else None,
        "positiveP50": round(sorted(pos)[len(pos) // 2], 4) if pos else None,
        "positiveMax": round(max(pos), 4) if pos else None,
        "negativeMin": round(min(neg), 4) if neg else None,
        "negativeP50": round(sorted(neg)[len(neg) // 2], 4) if neg else None,
        "negativeMax": round(max(neg), 4) if neg else None,
        "overlap": bool(pos and neg and min(pos) <= max(neg)),
    }
    if pos and neg:
        res.update(sweep(pos, neg))
    return res


# ----------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    ensure_utf8_stdout()

    parser = argparse.ArgumentParser(description="拒答信号与阈值实验")
    parser.add_argument("--positives", default="data/eval/queries.jsonl")
    parser.add_argument("--negatives", default="eval/negatives.jsonl")
    parser.add_argument("--limit-positives", type=int, default=120)
    parser.add_argument("--corpus", default="data/eval/corpus.jsonl")
    parser.add_argument("--out", default="bench/results/abstain-scores.json")
    parser.add_argument("--load", action="store_true", help="切到评测语料再跑")
    parser.add_argument("--skip-load", action="store_true", help="索引已就绪，不动数据")
    parser.add_argument("--restore-demo", action="store_true", help="跑完灌回演示语料")
    args = parser.parse_args()

    pos_rows = read_jsonl(ROOT / args.positives)[: args.limit_positives]
    neg_rows = read_jsonl(ROOT / args.negatives)
    print(f"正样本 {len(pos_rows)} 条 / 负样本 {len(neg_rows)} 条", flush=True)

    if args.load and not args.skip_load:
        print("准备索引：重启 → 清空 → 灌评测语料 …", flush=True)
        restart()
        load_corpus(ROOT / args.corpus)
        stats = json.loads(urllib.request.urlopen(f"{SERVICE}/api/stats", timeout=10).read())
        print(f"  索引就绪：{stats['indexedCount']} 分块 / {stats['sparse']['terms']} 词条", flush=True)

    if not service_ready():
        raise SystemExit("服务不可达，先 dev.ps1 start")

    records = []
    total = len(pos_rows) + len(neg_rows)
    for idx, row in enumerate(pos_rows + neg_rows, start=1):
        label = "pos" if idx <= len(pos_rows) else "neg"
        category = row.get("category", "related" if label == "pos" else "unknown")
        try:
            scores = snoop(row["query"])
        except RuntimeError as exc:
            print(f"  [{idx}/{total}] 失败，跳过：{exc}", flush=True)
            continue
        records.append({"query": row["query"], "label": label, "category": category, **scores})
        if idx % 10 == 0 or idx == total:
            print(f"  [{idx}/{total}] {label} {row['query'][:24]}  ce={scores['ceTop1']}", flush=True)

    pos = [r for r in records if r["label"] == "pos"]
    neg = [r for r in records if r["label"] == "neg"]
    neg_off = [r for r in neg if r["category"] == "off-domain"]
    neg_adj = [r for r in neg if r["category"] == "adjacent-tech"]

    def col(rows: list[dict], key: str) -> list[float]:
        return [float(r[key]) for r in rows if r.get(key) is not None]

    analyses = {
        "ceTop1": analyse("ceTop1（cross-encoder 最高分）", col(pos, "ceTop1"), col(neg, "ceTop1")),
        "vectorTop1": analyse("vectorTop1（余弦）", col(pos, "vectorTop1"), col(neg, "vectorTop1")),
        "bm25Top1": analyse("bm25Top1", col(pos, "bm25Top1"), col(neg, "bm25Top1")),
        "rrfTop1": analyse("rrfTop1（RRF 融合分）", col(pos, "rrfTop1"), col(neg, "rrfTop1")),
        "ceMargin": analyse("ceMargin（top1−top2）", col(pos, "ceMargin"), col(neg, "ceMargin")),
    }

    # 两档负样本分开看：难负样本（同领域无答案）才是真实场景
    by_difficulty = {}
    for key in ("ceTop1", "vectorTop1"):
        if neg_off and neg_adj:
            by_difficulty[key] = {
                "off-domain": analyse(key, col(pos, key), col(neg_off, key)),
                "adjacent-tech": analyse(key, col(pos, key), col(neg_adj, key)),
            }

    print("\n=== 判别力（AUC，越高越能分开） ===", flush=True)
    for key, res in analyses.items():
        print(f"  {key:<12} AUC {res['auc']}   正样本中位 {res['positiveP50']}  "
              f"负样本中位 {res['negativeP50']}  重叠 {res['overlap']}", flush=True)

    print("\n=== 最优阈值（最大化 拒答率−误拒率） ===", flush=True)
    for key, res in analyses.items():
        best = res.get("best")
        if best:
            print(f"  {key:<12} 阈值 {best['threshold']:<8} 拒答率 {best['refusalRate']:.3f}  "
                  f"误拒率 {best['falseRefusalRate']:.3f}  准确率 {best['accuracy']:.3f}", flush=True)

    if by_difficulty:
        print("\n=== 两档负样本分开看（ceTop1） ===", flush=True)
        for name, res in by_difficulty["ceTop1"].items():
            best = res.get("best") or {}
            print(f"  {name:<14} AUC {res['auc']}  最优阈值 {best.get('threshold')} "
                  f"拒答率 {best.get('refusalRate')} 误拒率 {best.get('falseRefusalRate')}", flush=True)

    payload = {
        "setup": {
            "service": SERVICE,
            "positives": args.positives,
            "negatives": args.negatives,
            "positiveCount": len(pos),
            "negativeCount": len(neg),
            "note": "正样本取自已知项检索评测集（库里有答案）；负样本离线构造，分 off-domain / adjacent-tech 两档",
        },
        "analyses": analyses,
        "byDifficulty": by_difficulty,
        "records": records,
    }

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n结果已写入 {out_path}", flush=True)

    if args.restore_demo:
        print("恢复演示语料 …", flush=True)
        restart()
        load_corpus(ROOT / "data" / "eval-demo" / "corpus.jsonl")
        print("完成。", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
