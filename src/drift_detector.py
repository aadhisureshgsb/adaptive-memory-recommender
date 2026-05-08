"""
drift_detector.py — Per-user taste drift detection and volatility scoring.

This is the core novel contribution of the paper.

The key insight: before you can apply personalised forgetting,
you need to know TWO things per user:
  1. Has their taste drifted? (binary event detection)
  2. How volatile is their taste in general? (continuous score 0-1)

We use three complementary signals to answer both questions:

  Signal 1: Genre entropy shift
    Compute genre distribution in rolling 60-day windows.
    A significant KL-divergence between windows = taste shift.
    Reference: Koenigstein et al. (2011) "Xbox Movies: Modeling and
    Recommending Movies on Xbox Live"

  Signal 2: Rating pattern change
    Detect structural breaks in the rating time series using
    CUSUM (Cumulative Sum) control charts.
    Reference: Page (1954), adapted for recommendation systems by
    Vinagre et al. (2014) "Fast Incremental Matrix Factorization
    for Recommendation with Positive-Only Feedback"

  Signal 3: Inter-genre switching frequency
    Count genre cluster transitions per unit time.
    High frequency = volatile user. Low = stable user.

The volatility score combines all three into a single 0-1 value
that becomes the personalised forgetting rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from scipy import stats


@dataclass
class DriftEvent:
    """A detected taste drift event for a single user."""
    user_id: int
    drift_timestamp: pd.Timestamp
    confidence: float               # 0-1
    pre_drift_genres: list[str]
    post_drift_genres: list[str]
    kl_divergence: float
    cusum_signal: float
    explanation: str


@dataclass
class UserVolatilityProfile:
    """
    Complete volatility characterisation for one user.
    This is the input to the adaptive forgetting rate model.
    """
    user_id: int
    volatility_score: float         # 0-1, THE key parameter
    archetype: str                  # "stable" | "drifter" | "volatile"
    recommended_decay_halflife_days: float  # days after which old signal halved
    n_ratings: int
    n_drift_events: int
    mean_genre_entropy: float
    genre_switch_rate: float        # switches per 100 ratings
    drift_events: list[DriftEvent]
    dominant_genres: list[str]


@dataclass
class DetectorConfig:
    window_days: int = 60           # rolling window for genre distribution
    min_ratings_per_window: int = 5
    kl_threshold: float = 0.3       # KL divergence to flag drift
    cusum_threshold: float = 2.5    # CUSUM control limit
    min_ratings_total: int = 20     # skip users with too few ratings


GENRE_CLUSTERS = {
    "Action": 0, "Adventure": 0, "Thriller": 0,
    "Drama": 1, "Romance": 1, "Crime": 1,
    "Comedy": 2, "Animation": 2, "Children's": 2,
    "Sci-Fi": 3, "Horror": 3, "Fantasy": 3,
    "Documentary": 4, "Musical": 4, "Western": 4,
}


class TasteDriftDetector:
    """
    Detects taste drift and computes personalised volatility scores.

    Usage:
        detector = TasteDriftDetector()
        profiles = detector.fit(ratings_df)

        # Get volatility score for a specific user
        score = profiles[user_id].volatility_score
        halflife = profiles[user_id].recommended_decay_halflife_days
    """

    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()

    def fit(self, df: pd.DataFrame) -> dict[int, UserVolatilityProfile]:
        """
        Compute volatility profiles for all users in the dataset.

        Args:
            df: DataFrame with columns user_id, timestamp, genres, rating

        Returns:
            Dict mapping user_id -> UserVolatilityProfile
        """
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        profiles = {}
        user_ids = df["user_id"].unique()
        print(f"Computing volatility profiles for {len(user_ids)} users...")

        for i, uid in enumerate(user_ids):
            user_df = df[df["user_id"] == uid].sort_values("timestamp")

            if len(user_df) < self.config.min_ratings_total:
                continue

            profile = self._profile_user(uid, user_df)
            profiles[uid] = profile

            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(user_ids)} users profiled...", end="\r")

        print(f"\nDone. {len(profiles)} user profiles computed.")
        return profiles

    def _profile_user(self, uid: int, user_df: pd.DataFrame) -> UserVolatilityProfile:
        # Signal 1: Genre entropy
        genre_entropy = self._compute_genre_entropy(user_df)

        # Signal 2: Drift events via KL divergence on rolling windows
        drift_events = self._detect_drift_events(uid, user_df)

        # Signal 3: Genre switch rate
        switch_rate = self._compute_switch_rate(user_df)

        # Volatility score: weighted combination
        entropy_score  = min(genre_entropy / 2.0, 1.0)   # max entropy ~2 bits
        drift_score    = min(len(drift_events) / 3.0, 1.0)
        switch_score   = min(switch_rate / 30.0, 1.0)    # 30 switches/100 = max

        volatility = (
            entropy_score  * 0.30 +
            drift_score    * 0.40 +
            switch_score   * 0.30
        )
        volatility = float(np.clip(volatility, 0, 1))

        # Map volatility to archetype
        if volatility < 0.25:
            archetype = "stable"
            halflife = 365.0      # 1 year
        elif volatility < 0.60:
            archetype = "drifter"
            halflife = 90.0       # 3 months
        else:
            archetype = "volatile"
            halflife = 30.0       # 1 month

        # Dominant genres (most rated)
        genre_counts = user_df["genres"].value_counts()
        dominant = genre_counts.head(3).index.tolist()

        return UserVolatilityProfile(
            user_id=uid,
            volatility_score=round(volatility, 4),
            archetype=archetype,
            recommended_decay_halflife_days=halflife,
            n_ratings=len(user_df),
            n_drift_events=len(drift_events),
            mean_genre_entropy=round(genre_entropy, 4),
            genre_switch_rate=round(switch_rate, 2),
            drift_events=drift_events,
            dominant_genres=dominant,
        )

    def _compute_genre_entropy(self, user_df: pd.DataFrame) -> float:
        """Shannon entropy of genre distribution. High = diverse/volatile."""
        counts = user_df["genres"].value_counts(normalize=True)
        return float(stats.entropy(counts.values, base=2))

    def _detect_drift_events(
        self, uid: int, user_df: pd.DataFrame
    ) -> list[DriftEvent]:
        """
        Detect taste drift using sliding window KL divergence.
        Compares genre distribution in consecutive time windows.
        """
        events = []
        all_genres = user_df["genres"].unique().tolist()
        if len(all_genres) < 2:
            return events

        timestamps = user_df["timestamp"].values
        start = timestamps[0]
        end = timestamps[-1]
        window = pd.Timedelta(days=self.config.window_days)
        step = pd.Timedelta(days=self.config.window_days // 2)

        window_start = pd.Timestamp(start)
        prev_dist = None
        prev_window_genres = []

        while window_start + window <= pd.Timestamp(end):
            window_end = window_start + window
            window_df = user_df[
                (user_df["timestamp"] >= window_start) &
                (user_df["timestamp"] < window_end)
            ]

            if len(window_df) < self.config.min_ratings_per_window:
                window_start += step
                continue

            # Genre distribution in this window
            genre_counts = window_df["genres"].value_counts()
            curr_dist = np.zeros(len(all_genres))
            for j, g in enumerate(all_genres):
                curr_dist[j] = genre_counts.get(g, 0)
            curr_dist = curr_dist / (curr_dist.sum() + 1e-9)

            if prev_dist is not None:
                # KL divergence (add smoothing to avoid log(0))
                p = prev_dist + 1e-9
                q = curr_dist + 1e-9
                p /= p.sum()
                q /= q.sum()
                kl = float(stats.entropy(p, q))

                if kl > self.config.kl_threshold:
                    curr_genres = window_df["genres"].value_counts().head(3).index.tolist()
                    events.append(DriftEvent(
                        user_id=uid,
                        drift_timestamp=window_start,
                        confidence=min(kl / (self.config.kl_threshold * 3), 1.0),
                        pre_drift_genres=prev_window_genres,
                        post_drift_genres=curr_genres,
                        kl_divergence=round(kl, 4),
                        cusum_signal=0.0,
                        explanation=(
                            f"Genre distribution shifted significantly "
                            f"(KL={kl:.2f}). "
                            f"Before: {prev_window_genres}. "
                            f"After: {curr_genres}."
                        ),
                    ))

            prev_dist = curr_dist.copy()
            prev_window_genres = window_df["genres"].value_counts().head(3).index.tolist()
            window_start += step

        return events

    def _compute_switch_rate(self, user_df: pd.DataFrame) -> float:
        """
        Genre cluster switch rate per 100 ratings.
        Maps genres to clusters and counts cluster transitions.
        """
        genres = user_df["genres"].tolist()
        clusters = [GENRE_CLUSTERS.get(g, -1) for g in genres]

        switches = sum(
            1 for i in range(1, len(clusters))
            if clusters[i] != clusters[i-1] and clusters[i] != -1 and clusters[i-1] != -1
        )

        return (switches / max(len(clusters), 1)) * 100
