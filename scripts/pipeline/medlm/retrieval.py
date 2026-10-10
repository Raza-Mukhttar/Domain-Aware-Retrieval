# -*- coding: utf-8 -*-
"""Dense exemplar retrieval, Eqs. (6)-(8).

A frozen all-MiniLM-L6-v2 sentence encoder feeds a FAISS IndexFlatIP index built
once from the training split. Embeddings are L2-normalised, so inner product is
cosine similarity.
"""
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from . import paths


class Retriever:
    """Optionally stratified by question type.

    use_question_type=True restricts search to the query's own type bucket and
    falls back to the global index when that bucket holds fewer than k exemplars.
    The fallback count is reported rather than hidden.
    """

    def __init__(self, train, device, use_question_type, embedder=None):
        self.train = train
        self.use_qt = use_question_type
        self.encoder = SentenceTransformer(embedder or paths.EMBEDDER, device=device)
        emb = self.encode([t["question"] for t in train])
        self.emb = emb
        self.global_index = faiss.IndexFlatIP(emb.shape[1])
        self.global_index.add(emb)
        self.buckets = {}
        self.bucket_index = {}
        if use_question_type:
            for i, t in enumerate(train):
                self.buckets.setdefault(t["qtype"], []).append(i)
            for qt, ids in self.buckets.items():
                index = faiss.IndexFlatIP(emb.shape[1])
                index.add(emb[ids])
                self.bucket_index[qt] = index
        self.fallbacks = 0

    def encode(self, questions):
        e = self.encoder.encode(questions, convert_to_numpy=True, batch_size=256,
                                show_progress_bar=False).astype("float32")
        faiss.normalize_L2(e)
        return e

    def search(self, q_emb, qtype, k):
        if self.use_qt:
            ids = self.buckets.get(qtype, [])
            if len(ids) >= k:
                _, I = self.bucket_index[qtype].search(q_emb[None, :], k)
                return [ids[j] for j in I[0] if j >= 0]
            self.fallbacks += 1
        _, I = self.global_index.search(q_emb[None, :], k)
        return [int(j) for j in I[0] if j >= 0]

    def search_excluding(self, q_emb, qtype, k, banned):
        """Nearest exemplars that are NOT in `banned`, for the source-disjoint
        evaluation. Over-fetches from the global index and filters, so the result
        is the k nearest admissible exemplars rather than the k nearest overall."""
        if not banned:
            return self.search(q_emb, qtype, k)
        want = min(len(self.train), max(k * 40, 200))
        _, I = self.global_index.search(q_emb[None, :], want)
        out = [int(j) for j in I[0] if j >= 0 and int(j) not in banned]
        return out[:k]

    def save(self, directory):
        import json
        import os
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.global_index, os.path.join(directory, "global.index"))
        np.save(os.path.join(directory, "embeddings.npy"), self.emb)
        for qt, index in self.bucket_index.items():
            faiss.write_index(index, os.path.join(directory, f"bucket_{qt}.index"))
        with open(os.path.join(directory, "buckets.json"), "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in self.buckets.items()}, f)
