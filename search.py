"""저장된 FAISS 인덱스로 검색. 사용법: python search.py "음주운전 기준" [-k 3]"""
import argparse
import json

import faiss
from sentence_transformers import SentenceTransformer

from build_index import INDEX_PATH, META_PATH, MODEL_NAME


class Searcher:
    def __init__(self):
        self.index = faiss.read_index(str(INDEX_PATH))
        self.chunks = json.loads(META_PATH.read_text(encoding="utf-8"))
        self.model = SentenceTransformer(MODEL_NAME)

    def search(self, query, k=3):
        q = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(q, k)
        return [{**self.chunks[i], "score": float(s)} for s, i in zip(scores[0], ids[0])]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=3)
    args = ap.parse_args()
    for rank, r in enumerate(Searcher().search(args.query, args.k), start=1):
        print(f"\n#{rank}  score={r['score']:.4f}  {r['article']}({r['title']})  p.{r['page']}")
        print(r["text"][:300])
