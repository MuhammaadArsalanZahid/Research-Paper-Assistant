"""evaluate.py - Retrieval evaluation (Step 12) with and without re-ranking.

Usage (from the project folder, after adding your papers in the app so that data/ exists):
    python evaluation/evaluate.py                          # uses evaluation/questions.json
    python evaluation/evaluate.py evaluation/questions_sample.json

questions file = list of
    {"question": "...", "paper_file": "name.pdf", "page": 5}      question answered in a paper
    {"question": "...", "paper_file": null}                        question NOT answered in any paper
It reports how often the correct PAPER (and the correct page) is in the final top 4, with and
without re-ranking, and suggests a value for RERANK_MIN_SCORE (retriever.py) from the
"not in any paper" questions.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from store import KnowledgeBase          # noqa: E402
from retriever import retrieve           # noqa: E402


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "questions.json")
    with open(path, encoding="utf-8") as f:
        cases = json.load(f)
    kb = KnowledgeBase()
    if not kb.chunks:
        sys.exit("The library is empty. Add your papers in the app first (Library tab).")
    positives = [c for c in cases if c.get("paper_file")]
    negatives = [c for c in cases if not c.get("paper_file")]
    print(f"Library: {len(kb.papers)} papers, {len(kb.chunks)} chunks | "
          f"{len(positives)} questions with answers, {len(negatives)} without\n")

    stats = {"with": [0, 0], "without": [0, 0]}      # [paper hits, page hits]
    rows, best_scores = [], []
    for i, case in enumerate(positives, start=1):
        row = [i, case["question"], f"{case['paper_file']} p.{case['page']}"]
        for label, rerank in (("without", False), ("with", True)):
            top = retrieve(kb, case["question"], rerank=rerank)
            paper_hit = any(c["paper_id"] == case["paper_file"] for c in top)
            page_hit = any(c["paper_id"] == case["paper_file"] and c["page"] == case["page"] for c in top)
            stats[label][0] += paper_hit
            stats[label][1] += page_hit
            row.append(("paper OK" if paper_hit else "MISS") + (", page OK" if page_hit else ""))
            if rerank and top:
                best_scores.append(top[0]["rerank_score"])
        rows.append(row)
        print(f"{i}. {case['question']}\n   no re-rank: {row[3]} | re-rank: {row[4]}")

    neg_scores = []
    for case in negatives:
        top = retrieve(kb, case["question"])
        if top:
            neg_scores.append(top[0]["rerank_score"])
            print(f"[not in papers] best re-rank score {top[0]['rerank_score']:.1f}: {case['question']}")

    n = len(positives)
    print("\n=== SUMMARY (final top 4) ===")
    for label in ("without", "with"):
        print(f"{label:>8} re-ranking: paper hit-rate {stats[label][0]}/{n} ({100 * stats[label][0] / n:.0f}%), "
              f"page hit-rate {stats[label][1]}/{n} ({100 * stats[label][1] / n:.0f}%)")
    if neg_scores and best_scores:
        hi, lo = max(neg_scores), min(best_scores)
        if hi < lo:
            print(f"Suggested RERANK_MIN_SCORE: about {(hi + lo) / 2:.1f} "
                  f"(highest 'not in papers' score {hi:.1f} < lowest real best score {lo:.1f})")
        else:
            print(f"Scores overlap (highest 'not in papers' {hi:.1f} >= lowest real best {lo:.1f}); "
                  "choose RERANK_MIN_SCORE carefully and mention this in your README.")

    print("\n=== README TABLE (copy) ===")
    print("| # | Question | Correct location | Without re-ranking | With re-ranking |\n|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(map(str, r)) + " |")


if __name__ == "__main__":
    main()
