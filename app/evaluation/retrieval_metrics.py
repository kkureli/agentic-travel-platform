import math


def recall_at_k(
    correct_rank: int | None,
    k: int,
) -> float:
    if correct_rank is None:
        return 0.0

    return float(correct_rank <= k)


def reciprocal_rank(
    correct_rank: int | None,
) -> float:
    if correct_rank is None:
        return 0.0

    return 1.0 / correct_rank


def precision_at_k(
    relevance_scores: list[int],
    k: int,
) -> float:
    if k <= 0:
        return 0.0

    top_k_scores = relevance_scores[:k]

    relevant_count = sum(1 for score in top_k_scores if score > 0)

    return relevant_count / k


def dcg_at_k(
    relevance_scores: list[int],
    k: int,
) -> float:
    score = 0.0

    for rank, relevance in enumerate(
        relevance_scores[:k],
        start=1,
    ):
        score += relevance / math.log2(rank + 1)

    return score


def ndcg_at_k(
    relevance_scores: list[int],
    k: int,
) -> float:
    actual_dcg = dcg_at_k(
        relevance_scores,
        k,
    )

    ideal_scores = sorted(
        relevance_scores,
        reverse=True,
    )

    ideal_dcg = dcg_at_k(
        ideal_scores,
        k,
    )

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def mean(
    values: list[float],
) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)
