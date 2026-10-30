"""store.py - Embeddings (BGE) and the FAISS knowledge base saved on disk.

The index file and a metadata file are saved after every change and loaded
automatically when the app starts. Deleting a paper removes its chunks and
rebuilds the index from the remaining vectors (no need to embed again).
"""
import json
import os
import numpy as np
import faiss
from ingest import ingest_pdf

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# bge models need this prefix for SEARCH QUERIES only (not for the passages)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


class KnowledgeBase:
    def __init__(self, data_dir=DATA_DIR):
        self.index_file = os.path.join(data_dir, "index.faiss")
        self.meta_file = os.path.join(data_dir, "metadata.json")
        self.data_dir = data_dir
        self.index = None
        self.chunks = []      # same order as the vectors in the index
        self.papers = {}      # paper_id -> {"title", "pages"}
        self.load()

    # ---------------- embeddings ----------------
    def embed_passages(self, texts, batch_size=32, callback=None):
        out = []
        for i in range(0, len(texts), batch_size):
            v = get_embedder().encode(texts[i:i + batch_size], normalize_embeddings=True,
                                      convert_to_numpy=True)
            out.append(np.asarray(v, dtype="float32"))
            if callback:
                callback(min(i + batch_size, len(texts)) / len(texts))
        return np.vstack(out)

    def embed_query(self, query):
        v = get_embedder().encode([QUERY_PREFIX + query], normalize_embeddings=True,
                                  convert_to_numpy=True)
        return np.asarray(v, dtype="float32")

    # ---------------- save / load ----------------
    def save(self):
        os.makedirs(self.data_dir, exist_ok=True)
        if self.index is not None and self.index.ntotal > 0:
            faiss.write_index(self.index, self.index_file)
        elif os.path.exists(self.index_file):
            os.remove(self.index_file)
        tmp = self.meta_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"papers": self.papers, "chunks": self.chunks}, f, ensure_ascii=False)
        os.replace(tmp, self.meta_file)

    def load(self):
        try:
            if os.path.exists(self.index_file) and os.path.exists(self.meta_file):
                with open(self.meta_file, encoding="utf-8") as f:
                    meta = json.load(f)
                self.index = faiss.read_index(self.index_file)
                self.chunks, self.papers = meta["chunks"], meta["papers"]
                if self.index.ntotal != len(self.chunks):
                    raise ValueError("index and metadata do not match")
        except Exception as e:
            print(f"[store] Could not load saved knowledge base ({e}). Starting empty.")
            self.index, self.chunks, self.papers = None, [], {}

    # ---------------- add / delete ----------------
    def add_paper(self, path, callback=None):
        info = ingest_pdf(path)
        if info["paper_id"] in self.papers:
            raise ValueError(f"'{info['paper_id']}' is already in the library.")
        vectors = self.embed_passages([c["text"] for c in info["chunks"]], callback=callback)
        if self.index is None:
            self.index = faiss.IndexFlatIP(vectors.shape[1])   # vectors are normalized -> cosine
        self.index.add(vectors)
        self.chunks.extend(info["chunks"])
        self.papers[info["paper_id"]] = {"title": info["title"], "pages": info["pages"]}
        self.save()
        return info

    def delete_paper(self, paper_id):
        if paper_id not in self.papers:
            raise KeyError("Paper not found in the library.")
        keep = [i for i, c in enumerate(self.chunks) if c["paper_id"] != paper_id]
        if keep and self.index is not None:
            vectors = self.index.reconstruct_n(0, self.index.ntotal)[keep]
            self.index = faiss.IndexFlatIP(vectors.shape[1])
            self.index.add(np.ascontiguousarray(vectors, dtype="float32"))
        else:
            self.index = None
        self.chunks = [self.chunks[i] for i in keep]
        del self.papers[paper_id]
        self.save()

    # ---------------- helpers ----------------
    def paper_list(self):
        counts = {}
        for c in self.chunks:
            counts[c["paper_id"]] = counts.get(c["paper_id"], 0) + 1
        rows = [{"paper_id": pid, "title": p["title"], "pages": p["pages"],
                 "chunks": counts.get(pid, 0)} for pid, p in self.papers.items()]
        return sorted(rows, key=lambda r: r["title"].lower())

    def paper_chunks(self, paper_id):
        return [c for c in self.chunks if c["paper_id"] == paper_id]

    def summary_chunks(self, paper_id, limit=5):
        """Abstract + Conclusion chunks of a paper (or first and last chunks as fallback)."""
        chunks = self.paper_chunks(paper_id)
        picked = [c for c in chunks if c["section"] in ("Abstract", "Conclusion")]
        if not picked:
            picked = chunks[:2] + chunks[-2:]
        return picked[:limit]
