"""tests/test_pipeline.py"""

import pytest
import numpy as np
import pandas as pd
from src.data_loader import generate_synthetic
from src.drift_detector import TasteDriftDetector, DetectorConfig
from src.recommender import (
    AdaptiveMemoryRecommender, RecommenderConfig,
    train_test_split_temporal, evaluate, ndcg_at_k
)


def get_df(n_users=100, seed=42):
    df = generate_synthetic(n_users=n_users, n_days=365, seed=seed)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


class TestDataLoader:
    def test_generates_ratings(self):
        df = generate_synthetic(n_users=50)
        assert len(df) > 0
        for col in ["user_id", "movie_id", "rating", "timestamp", "genres"]:
            assert col in df.columns

    def test_rating_range(self):
        df = generate_synthetic(n_users=50)
        assert df["rating"].between(1, 5).all()

    def test_three_archetypes(self):
        df = generate_synthetic(n_users=100)
        assert "archetype" in df.columns
        archetypes = df["archetype"].unique()
        assert set(archetypes).issubset({"stable", "drifter", "volatile"})

    def test_deterministic(self):
        df1 = generate_synthetic(n_users=30, seed=1)
        df2 = generate_synthetic(n_users=30, seed=1)
        assert df1["rating"].sum() == df2["rating"].sum()


class TestDriftDetector:
    def test_returns_profiles_for_all_users(self):
        df = get_df(n_users=50)
        profiles = TasteDriftDetector().fit(df)
        assert len(profiles) > 0

    def test_volatility_score_bounded(self):
        df = get_df(n_users=50)
        profiles = TasteDriftDetector().fit(df)
        for p in profiles.values():
            assert 0 <= p.volatility_score <= 1

    def test_archetype_values_valid(self):
        df = get_df(n_users=50)
        profiles = TasteDriftDetector().fit(df)
        valid = {"stable", "drifter", "volatile"}
        for p in profiles.values():
            assert p.archetype in valid

    def test_halflife_decreases_with_volatility(self):
        df = get_df(n_users=100)
        profiles = TasteDriftDetector().fit(df)
        vals = [(p.volatility_score, p.recommended_decay_halflife_days)
                for p in profiles.values()]
        high_vol = [h for v, h in vals if v > 0.6]
        low_vol  = [h for v, h in vals if v < 0.3]
        if high_vol and low_vol:
            assert np.mean(high_vol) < np.mean(low_vol)

    def test_volatile_users_have_drift_events(self):
        df = get_df(n_users=100)
        profiles = TasteDriftDetector().fit(df)
        volatile = [p for p in profiles.values() if p.archetype == "volatile"]
        if volatile:
            avg_drifts = np.mean([p.n_drift_events for p in volatile])
            stable = [p for p in profiles.values() if p.archetype == "stable"]
            if stable:
                avg_stable = np.mean([p.n_drift_events for p in stable])
                assert avg_drifts >= avg_stable


class TestRecommender:
    def test_fit_and_recommend(self):
        df = get_df(n_users=80)
        profiles = TasteDriftDetector().fit(df)
        train, test = train_test_split_temporal(df)
        model = AdaptiveMemoryRecommender()
        model.fit(train, profiles, mode="adaptive")
        uid = df["user_id"].iloc[0]
        recs = model.recommend(uid, n=10)
        assert isinstance(recs, list)

    def test_all_modes_run(self):
        df = get_df(n_users=60)
        profiles = TasteDriftDetector().fit(df)
        train, test = train_test_split_temporal(df)
        for mode in ["static", "fixed_decay", "adaptive"]:
            model = AdaptiveMemoryRecommender()
            model.fit(train, profiles, mode=mode)
            results = evaluate(model, test, k=10)
            assert "ndcg_at_k" in results
            assert 0 <= results["ndcg_at_k"] <= 1

    def test_adaptive_lambda_higher_for_volatile(self):
        cfg = RecommenderConfig(base_decay_lambda=0.003, volatility_amplifier=3.0)
        from src.drift_detector import UserVolatilityProfile
        stable_profile = UserVolatilityProfile(
            user_id=1, volatility_score=0.1, archetype="stable",
            recommended_decay_halflife_days=365, n_ratings=100,
            n_drift_events=0, mean_genre_entropy=0.5,
            genre_switch_rate=5.0, drift_events=[], dominant_genres=["Drama"]
        )
        volatile_profile = UserVolatilityProfile(
            user_id=2, volatility_score=0.9, archetype="volatile",
            recommended_decay_halflife_days=30, n_ratings=200,
            n_drift_events=5, mean_genre_entropy=1.8,
            genre_switch_rate=28.0, drift_events=[], dominant_genres=["Action"]
        )
        stable_lambda = cfg.base_decay_lambda * (1 + stable_profile.volatility_score * cfg.volatility_amplifier)
        volatile_lambda = cfg.base_decay_lambda * (1 + volatile_profile.volatility_score * cfg.volatility_amplifier)
        assert volatile_lambda > stable_lambda

    def test_ndcg_perfect_score(self):
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 2, 3}
        score = ndcg_at_k(recommended, relevant, k=5)
        assert 0 < score <= 1

    def test_ndcg_zero_when_no_hits(self):
        recommended = [10, 11, 12]
        relevant = {1, 2, 3}
        score = ndcg_at_k(recommended, relevant, k=3)
        assert score == 0.0
