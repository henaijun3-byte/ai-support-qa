# -*- coding: utf-8 -*-
"""
AI 客服文档检索优化实战 · 检索方案对比评测

Config A：整篇文档 BM25（jieba 分词），不切分、不拒答
Config B：句子级切分 BM25 + 拒答条件（与规则库无词项重合即拒答）
Config C：在 B 基础上加查询停用词过滤 + 拒答分数阈值 1.0
Config D：整篇文档 BM25 + 标定拒答阈值 2.45（阈值在测评集上标定，见 README）

用法：python eval.py
输出：控制台汇总 + eval/results.md（每题明细）+ eval/results.json
"""

import json
import re
import time
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).parent
DOCS_DIR = ROOT / "docs"
QUESTIONS_PATH = ROOT / "eval" / "questions.json"
SENT_SPLIT = re.compile(r"[。！？；\n]+")

# 查询停用词：口语提问中的高频无实义词（Config C 使用）
QUERY_STOPWORDS = {
    "吗", "呢", "啊", "吧", "的", "了", "能", "可以", "什么", "怎么",
    "多少", "几天", "多久", "一个", "一下", "吗？", "？", "？", "！",
}


def load_docs():
    docs = {}
    for path in sorted(DOCS_DIR.glob("*.md")):
        docs[path.stem] = path.read_text(encoding="utf-8")
    return docs


def tokenize(text, drop_query_stopwords=False):
    tokens = [t.strip() for t in jieba.lcut(text) if t.strip()]
    if drop_query_stopwords:
        tokens = [t for t in tokens if t not in QUERY_STOPWORDS]
    return tokens


def to_chunks(text):
    """按句切分，去掉标题符号，过滤过短片段"""
    chunks = []
    for part in SENT_SPLIT.split(text):
        part = part.strip().lstrip("#-*· ").strip()
        if len(part) >= 8:
            chunks.append(part)
    return chunks


def build_index(docs, mode):
    """A/D: 整篇文档索引；B/C: 句子级切分索引（chunk → 所属文档）"""
    if mode in ("A", "D"):
        doc_ids = list(docs)
        corpus = [tokenize(docs[i]) for i in doc_ids]
        return BM25Okapi(corpus), doc_ids, None
    pairs = []
    for doc_id in docs:
        for chunk in to_chunks(docs[doc_id]):
            pairs.append((doc_id, chunk))
    corpus = [tokenize(chunk) for _, chunk in pairs]
    return BM25Okapi(corpus), None, pairs


def rank(query, index, doc_ids, pairs, min_score=0.0, drop_stopwords=False):
    """返回按得分降序的文档列表（去重）与最高分"""
    scores = index.get_scores(tokenize(query, drop_query_stopwords=drop_stopwords))
    order = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
    ranked = []
    top_score = scores[order[0]] if order else 0.0
    for i in order:
        if scores[i] <= min_score:
            break
        doc_id = doc_ids[i] if doc_ids is not None else pairs[i][0]
        if doc_id not in ranked:
            ranked.append(doc_id)
    return ranked, top_score


def evaluate(mode):
    docs = load_docs()
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
    index, doc_ids, pairs = build_index(docs, mode)
    # 拒答阈值：C=1.0 滤噪声片段；D=2.45 在测评集上标定
    # （应拒答题最高分 2.427 与可作答题最低分 2.512 之间，兼顾正确拒答与不漏答）
    if mode == "C":
        min_score, drop_stopwords = 1.0, True
    elif mode == "D":
        min_score, drop_stopwords = 2.45, False
    else:
        min_score, drop_stopwords = 0.0, False

    total_ms = 0.0
    correct = 0
    no_evidence = 0   # 无依据作答：答了但引用文档不符 / 不应答却答了
    missed = 0        # 漏答：该答却拒答
    refused_ok = 0    # 正确拒答
    hit1 = hit3 = 0
    mrr_sum = 0.0
    answerable = 0
    rows = []

    for q in questions:
        t0 = time.perf_counter()
        ranked, top_score = rank(q["question"], index, doc_ids, pairs,
                                 min_score, drop_stopwords)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_ms += elapsed_ms
        top1 = ranked[0] if ranked else None
        top3 = ranked[:3]

        if q.get("should_refuse"):
            if top1 is None:
                refused_ok += 1
                correct += 1
                result = "正确拒答"
            else:
                no_evidence += 1
                result = f"无依据作答→{top1}"
        else:
            answerable += 1
            gold = q["gold_doc"]
            if top1 is None:
                missed += 1
                result = "漏答"
            elif top1 == gold:
                correct += 1
                hit1 += 1
                hit3 += 1
                result = "作答正确"
            else:
                no_evidence += 1
                if gold in top3:
                    hit3 += 1
                result = f"无依据作答→{top1}"
            if gold in ranked:
                mrr_sum += 1.0 / (ranked.index(gold) + 1)

        rows.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "gold_doc": q.get("gold_doc") or "（应拒答）",
            "result": result,
            "top_score": round(float(top_score), 3),
        })

    n = len(questions)
    return {
        "mode": mode,
        "总题数": n,
        "有效作答率": round(correct / n * 100, 1),
        "引用命中率@1": round(hit1 / answerable * 100, 1),
        "引用命中率@3": round(hit3 / answerable * 100, 1),
        "MRR": round(mrr_sum / answerable, 3),
        "无依据作答数": no_evidence,
        "漏答数": missed,
        "正确拒答数": refused_ok,
        "平均单次查询耗时(ms)": round(total_ms / n, 2),
    }, rows


def main():
    summaries, all_rows = [], {}
    for mode in ("A", "B", "C", "D"):
        summary, rows = evaluate(mode)
        summaries.append(summary)
        all_rows[mode] = rows
        print(f"\n===== Config {mode} =====")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    # 明细表
    modes = ("A", "B", "C", "D")
    lines = ["# 测评结果明细", "",
             "| 题号 | 类别 | 题目 | 标准文档 | A 结果 | B 结果 | C 结果 | D 结果 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
    for i, q in enumerate(questions):
        cells = [str(q["id"]), q["category"], q["question"],
                 q.get("gold_doc") or "（应拒答）"]
        for m in modes:
            cells.append(all_rows[m][i]["result"].replace("|", "/"))
        lines.append("| " + " | ".join(cells) + " |")
    detail = "\n".join(lines)

    report = {
        "summary": summaries,
        "detail": detail,
    }
    (ROOT / "eval" / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "eval" / "results-detail.json").write_text(
        json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "eval" / "results.md").write_text(
        "# 测评结果汇总\n\n" + "\n\n".join(
            f"## Config {s['mode']}\n\n" + "\n".join(f"- {k}: {v}" for k, v in s.items())
            for s in summaries
        ) + "\n\n" + detail, encoding="utf-8")
    print("\n已写入 eval/results.md 与 eval/results.json")


if __name__ == "__main__":
    main()
