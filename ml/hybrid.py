from collections import defaultdict

def reciprocal_rank_fusion(
    dense_hits: list[tuple[int, float]],
    sparse_hits: list[tuple[int, float]],
    k: int = 60,
    top_n: int = 30,
) -> list[tuple[int, float, dict]]:
    rrf_scores = defaultdict(float)
    provenance = defaultdict(lambda: {"dense_rank": None, "sparse_rank": None})

    for rank, (doc_id, _) in enumerate(dense_hits, start=1):
        rrf_scores[doc_id] += 1.0 / (k + rank)
        provenance[doc_id]["dense_rank"] = rank

    for rank, (doc_id, _) in enumerate(sparse_hits, start=1):
        rrf_scores[doc_id] += 1.0 / (k + rank)
        provenance[doc_id]["sparse_rank"] = rank

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [(doc_id, score, provenance[doc_id]) for doc_id, score in fused]