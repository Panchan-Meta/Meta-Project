"""Utility functions for keyword selection and search query generation."""

from __future__ import annotations

import random
from typing import Iterable, List


def generate_search_query(keyword: str) -> str:
    """Create a deterministic search query for a given keyword.

    The function does not rely on global state to ease unit testing.
    """

    base = keyword.strip()
    if not base:
        return ""
    return f"{base} insights"


def select_keywords(
    keywords: Iterable[str],
    *,
    limit: int = 1,
    seed: int | None = None,
) -> List[str]:
    """Shuffle and pick keywords with an optional seed.

    Args:
        keywords: Candidate keyword collection.
        limit: Number of keywords to pick.
        seed: Random seed for reproducible sampling.
    """

    population = [k for k in keywords if str(k).strip()]
    rng = random.Random(seed)
    rng.shuffle(population)
    return population[:limit]

