from app.schemas.document import Document

EVAL_DATASET = [
    {
        "query": ("When can I cancel a standard hotel reservation without penalty?"),
        "expected_source": "hotel_cancellation_policy.md",
        "relevant_sources": {
            "hotel_cancellation_policy.md": 3,
            "hotel_booking_policy.md": 1,
        },
    },
    {
        "query": ("Can a non-refundable hotel reservation be refunded?"),
        "expected_source": "hotel_cancellation_policy.md",
        "relevant_sources": {
            "hotel_cancellation_policy.md": 3,
        },
    },
    {
        "query": ("What information does a guest need to provide when booking?"),
        "expected_source": "hotel_booking_policy.md",
        "relevant_sources": {
            "hotel_booking_policy.md": 3,
        },
    },
    {
        "query": ("What protection can travel insurance provide?"),
        "expected_source": "travel_insurance.md",
        "relevant_sources": {
            "travel_insurance.md": 3,
        },
    },
    {
        "query": ("What can families do in Antalya?"),
        "expected_source": "antalya_guide.md",
        "relevant_sources": {
            "antalya_guide.md": 3,
        },
    },
    {
        "query": ("What benefits are available to loyalty program members?"),
        "expected_source": "loyalty_program.md",
        "relevant_sources": {
            "loyalty_program.md": 3,
        },
    },
    {
        "query": (
            "A guest booked a room that normally cannot be refunded. "
            "Are there any exceptional cases where support may still review it?"
        ),
        "expected_source": "hotel_cancellation_policy.md",
    },
    {
        "query": (
            "The hotel cancelled the reservation rather than the guest. "
            "What happens in that case?"
        ),
        "expected_source": "hotel_cancellation_policy.md",
    },
    {
        "query": (
            "What details must be provided before a hotel reservation can be confirmed?"
        ),
        "expected_source": "hotel_booking_policy.md",
    },
    {
        "query": (
            "If a traveler becomes ill before the trip, "
            "what kind of protection might cover the loss?"
        ),
        "expected_source": "travel_insurance.md",
    },
    {
        "query": (
            "I'm traveling with children and want a family-friendly "
            "area around Antalya. What should I consider?"
        ),
        "expected_source": "antalya_guide.md",
    },
    {
        "query": (
            "Does being a repeat customer provide any rewards or benefits when booking?"
        ),
        "expected_source": "loyalty_program.md",
    },
]


def get_relevance_scores(
    results: list[Document],
    relevant_sources: dict[str, int],
) -> list[int]:
    scores: list[int] = []

    for result in results:
        source = result.metadata.get(
            "source",
            "",
        )

        relevance = 0

        for expected_source, score in relevant_sources.items():
            if source.endswith(expected_source):
                relevance = score
                break

        scores.append(relevance)

    return scores
