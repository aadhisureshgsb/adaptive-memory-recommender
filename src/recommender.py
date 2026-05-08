"""
recommender.py — Adaptive Memory Recommender with Personalised Forgetting Rates.

This is the implementation of the core paper contribution.

Three models compared:

  Baseline 1: Static CF (no temporal weighting)
    Standard collaborative filtering using all historical ratings equally.
    Equivalent to infinite memory — nothing is ever forgotten.

  Baseline 2: Fixed decay (uniform forgetting rate)
    All users share the same exponential time-decay parameter.
    This is the current state-of-practice at most platforms.
    lambda = 0.003 (half-life ~231 days, matching literature defaults)

  Ours: Adaptive decay (personalised forgetting rate)
    Each user gets their own decay rate derived from their
    volatility score. Volatile users = fast forgetting.
    Stable users = slow forgetting.
    decay_lambda = base_lambda * (1 + volatility_score * amplifier)

The key evaluation metric: NDCG@10 on held-out recent ratings.
We specifically evaluate on ratings made AFTER a detected drift event
to test whether personalised forgetting improves post-drift recommendations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity

from src.drift_detector import UserVolatilityProfile


@dataclass
class RecommenderConfig:
    base_decay_lambda: float = 0.003    # fixed decay baseline (~231-day half-life)
    volatility_amplifier: float = 3.0   # how much volatility scales the decay
    n_factors: int = 20                 # latent factors for matrix factorisation
    top_k: int = 10                     # for NDCG@K evaluation
    test_fraction: float = 0.20         # hold out last 20% of each user's ratings
    min_ratings: int = 20


class AdaptiveMemoryRecommender:
    """
    Item-based collaborative filter with adaptive time-decay weighting.

    The time-decay weight for a rating made t days ago by user u is:

        w(t, u) = exp(-lambda_u * t)

    where lambda_u is personalised:

        lambda_u = base_lambda * (1 + volatility_u * amplifier)

    For a stable user (volatility=0.1):
        lambda = 0.003 * 1.3 = 0.0039 → half-life ~178 days

    For a volatile user (volatility=0.9):
        lambda = 0.003 * 3.7 = 0.011 → half-life ~63 days

    This means a volatile user's 6-month-old ratings are weighted
    ~5x less than a stable user's — directly encoding the intuition
    that volatile users' old preferences are less predictive.
    """

    def __init__(self, config: RecommenderConfig | None = None):
        self.config = config or RecommenderConfig()
        self._item_matrix: Optional[np.ndarray] = None
        self._item_ids: Optional[list] = None
        self._user_profiles: dict[int, UserVolatilityProfile] = {}

    def fit(
        self,
        train_df: pd.DataFrame,
        volatility_profiles: dict[int, UserVolatilityProfile],
        mode: str = "adaptive",
    ) -> "AdaptiveMemoryRecommender":
        """
        Fit the recommender using time-weighted ratings.

        Args:
            train_df: Training ratings (user_id, movie_id, rating, timestamp)
            volatility_profiles: Per-user volatility from drift_detector
            mode: "static" | "fixed_decay" | "adaptive"
        """
        self._user_profiles = volatility_profiles
        self._mode = mode

        df = train_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        reference_date = df["timestamp"].max()

        # Apply time-decay weights
        df["days_ago"] = (reference_date - df["timestamp"]).dt.total_seconds() / 86400

        if mode == "static":
            df["weight"] = 1.0

        elif mode == "fixed_decay":
            lam = self.config.base_decay_lambda
            df["weight"] = np.exp(-lam * df["days_ago"])

        elif mode == "adaptive":
            def get_lambda(row):
                uid = row["user_id"]
                if uid in volatility_profiles:
                    v = volatility_profiles[uid].volatility_score
                    return self.config.base_decay_lambda * (
                        1 + v * self.config.volatility_amplifier
                    )
                return self.config.base_decay_lambda

            df["lambda"] = df.apply(get_lambda, axis=1)
            df["weight"] = np.exp(-df["lambda"] * df["days_ago"])

        # Build weighted user-item matrix
        df["weighted_rating"] = df["rating"] * df["weight"]

        pivot = df.pivot_table(
            index="user_id",
            columns="movie_id",
            values="weighted_rating",
            aggfunc="sum",
        ).fillna(0)

        self._user_ids = pivot.index.tolist()
        self._item_ids = pivot.columns.tolist()
        self._user_matrix = pivot.values
        self._item_matrix = cosine_similarity(pivot.T)

        return self

    def recommend(self, user_id: int, n: int = 10, exclude_seen: bool = True) -> list[int]:
        """Return top-n recommended item IDs for a user."""
        if user_id not in self._user_ids:
            return []

        u_idx = self._user_ids.index(user_id)
        user_vec = self._user_matrix[u_idx]
        scores = self._item_matrix.T @ user_vec

        seen = set(np.where(user_vec > 0)[0]) if exclude_seen else set()
        ranked = [
            self._item_ids[i]
            for i in np.argsort(scores)[::-1]
            if i not in seen
        ]
        return ranked[:n]


# ------------------------------------------------------------------ #
#  Evaluation                                                          #
# ------------------------------------------------------------------ #

def train_test_split_temporal(
    df: pd.DataFrame,
    test_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal train/test split — test set is each user's most recent ratings.
    This is the correct split for recommendation evaluation.
    Leaks future info if you do random split.
    """
    train_rows, test_rows = [], []

    for uid, user_df in df.groupby("user_id"):
        user_df = user_df.sort_values("timestamp")
        n_test = max(1, int(len(user_df) * test_fraction))
        train_rows.append(user_df.iloc[:-n_test])
        test_rows.append(user_df.iloc[-n_test:])

    return pd.concat(train_rows), pd.concat(test_rows)


def ndcg_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    """
    Normalised Discounted Cumulative Gain at K.
    Standard IR metric for recommendation quality.
    """
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(recommended[:k])
        if item in relevant
    )
    ideal_dcg = sum(
        1.0 / np.log2(i + 2)
        for i in range(min(len(relevant), k))
    )
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def evaluate(
    model: AdaptiveMemoryRecommender,
    test_df: pd.DataFrame,
    k: int = 10,
    post_drift_only: bool = False,
    drift_timestamps: dict[int, pd.Timestamp] | None = None,
) -> dict:
    """
    Evaluate a recommender model on the test set.

    Args:
        post_drift_only: If True, only evaluate on users who had a drift event,
                         and only on ratings made after the drift.
                         This is the key evaluation in the paper.
    """
    ndcg_scores = []
    hit_rates = []
    n_evaluated = 0

    for uid, user_test in test_df.groupby("user_id"):
        if post_drift_only and drift_timestamps:
            if uid not in drift_timestamps:
                continue
            drift_ts = drift_timestamps[uid]
            user_test = user_test[user_test["timestamp"] > drift_ts]
            if len(user_test) == 0:
                continue

        relevant = set(user_test["movie_id"].tolist())
        recommended = model.recommend(uid, n=k)

        if not recommended:
            continue

        ndcg = ndcg_at_k(recommended, relevant, k)
        hit = 1.0 if any(item in relevant for item in recommended) else 0.0

        ndcg_scores.append(ndcg)
        hit_rates.append(hit)
        n_evaluated += 1

    return {
        "ndcg_at_k": round(float(np.mean(ndcg_scores)), 4) if ndcg_scores else 0.0,
        "hit_rate": round(float(np.mean(hit_rates)), 4) if hit_rates else 0.0,
        "n_users_evaluated": n_evaluated,
        "k": k,
    }
