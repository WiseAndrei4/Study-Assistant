import json
import math
import os
import re
from collections import Counter, defaultdict


class BM25Index:
    _TOKEN_RE = re.compile(r"\\[a-zA-Z]+|[a-zA-Z0-9_]+")

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: list[int] = []
        self.doc_len: dict[int, int] = {}
        self.avgdl: float = 0.0
        self.inverted: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.idf: dict[str, float] = {}

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        return [tok.lower() for tok in cls._TOKEN_RE.findall(text)]

    def add_documents(self, docs: list[tuple[int, str]]) -> None:
        for doc_id, text in docs:
            tokens = self.tokenize(text)
            self.doc_ids.append(doc_id)
            self.doc_len[doc_id] = len(tokens)
            for term, freq in Counter(tokens).items():
                self.inverted[term].append((doc_id, freq))

        n_docs = len(self.doc_ids)
        self.avgdl = (sum(self.doc_len.values()) / n_docs) if n_docs else 0.0
        self.idf = {
            term: math.log(1.0 + (n_docs - len(postings) + 0.5) / (len(postings) + 0.5))
            for term, postings in self.inverted.items()
        }

    def search(self, query: str, k: int = 30) -> list[tuple[int, float]]:
        if not self.doc_ids:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term in self.tokenize(query):
            if term not in self.inverted:
                continue
            idf = self.idf[term]
            for doc_id, tf in self.inverted[term]:
                dl = self.doc_len[doc_id]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / max(self.avgdl, 1e-9)))
                scores[doc_id] += idf * ((tf * (self.k1 + 1.0)) / denom)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

    def save(self, path: str) -> None:
        payload = {
            "k1": self.k1,
            "b": self.b,
            "doc_ids": self.doc_ids,
            "doc_len": {str(k): v for k, v in self.doc_len.items()},
            "avgdl": self.avgdl,
            "inverted": dict(self.inverted),
            "idf": self.idf,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.k1 = payload["k1"]
        self.b = payload["b"]
        self.doc_ids = payload["doc_ids"]
        self.doc_len = {int(k): v for k, v in payload["doc_len"].items()}
        self.avgdl = payload["avgdl"]
        self.inverted = defaultdict(list, {k: [tuple(p) for p in v] for k, v in payload["inverted"].items()})
        self.idf = payload["idf"]