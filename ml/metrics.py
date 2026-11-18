import math
import numpy as np

def ndcg_at_k(ranked_doc_ids: list[int], qrels: dict[int, int], k: int = 10) -> float:
    dcg = sum(
        (2 ** qrels.get(doc_id, 0) - 1) / math.log2(idx + 2)
        for idx, doc_id in enumerate(ranked_doc_ids[:k])
    )
    ideal_rels = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0

def mrr_at_k(ranked_doc_ids: list[int], relevant_ids: set[int], k: int = 10) -> float:
    for idx, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / idx
    return 0.0