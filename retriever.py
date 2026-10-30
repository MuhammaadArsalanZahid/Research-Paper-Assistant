"""retriever.py - Two-stage retrieval: FAISS (20 candidates) -> cross-encoder re-rank (best 4)."""

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FETCH_K = 10      # candidates from FAISS (kam = fast)
TOP_N = 3         # kept after re-ranking (kam = fast LLM)
# If even the best re-ranked chunk scores below this, the app answers "not found"
# (cross-encoder scores: clearly relevant is above 0, unrelated is about -9 or lower).
# This is a starting value: evaluation/evaluate.py suggests a better one from your own questions.
RERANK_MIN_SCORE = -8.0
# If the model says 'not found' although the best passage scores above this, the app shows the
# relevant passages instead of a flat 'not found' (small models are often too cautious).
RERANK_STRONG = 0.0

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def retrieve(kb, query, paper_id=None, fetch_k=FETCH_K, top_n=TOP_N, rerank=True):
    """Return the best chunks (dicts with faiss_score and rerank_score).
    paper_id: restrict the search to one paper. rerank=False: FAISS order only."""
    if kb.index is None or not kb.chunks:
        return []
    qv = kb.embed_query(query)
    n = kb.index.ntotal if paper_id else min(fetch_k, kb.index.ntotal)
    scores, ids = kb.index.search(qv, n)

    candidates = []
    for s, i in zip(scores[0], ids[0]):
        if i < 0:
            continue
        chunk = kb.chunks[i]
        if paper_id and chunk["paper_id"] != paper_id:
            continue
        candidates.append({**chunk, "faiss_score": float(s), "rerank_score": None})
        if len(candidates) >= fetch_k:
            break
    if not candidates:
        return []

    if rerank:
        pairs = [[query, c["text"]] for c in candidates]
        scores_pred = get_reranker().predict(pairs)
        for c, s in zip(candidates, scores_pred):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda c: c["rerank_score"] if c["rerank_score"] is not None else -1e9, reverse=True)
    else:
        candidates.sort(key=lambda c: c["faiss_score"], reverse=True)
    return candidates[:top_n]
