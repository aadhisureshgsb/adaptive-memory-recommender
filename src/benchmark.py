"""
benchmark.py — Full experimental evaluation.

Reproduces the main results table from the paper.

Experiment 1: Overall NDCG@10 across all users
Experiment 2: NDCG@10 restricted to post-drift ratings (the key result)
Experiment 3: NDCG improvement by volatility tier

Usage:
    python -m src.benchmark --data data/ratings.parquet --output benchmarks/results.json
"""

from __future__ import annotations

import json
import click
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict

from src.data_loader import generate_synthetic
from src.drift_detector import TasteDriftDetector, DetectorConfig
from src.recommender import (
    AdaptiveMemoryRecommender, RecommenderConfig,
    train_test_split_temporal, evaluate
)


@dataclass
class ExperimentResult:
    method: str
    ndcg_at_10_overall: float
    ndcg_at_10_post_drift: float
    hit_rate_overall: float
    hit_rate_post_drift: float
    n_users: int


def run_experiments(data_path: str) -> list[ExperimentResult]:
    df = pd.read_parquet(data_path)
    print(f"Dataset: {len(df):,} ratings, {df['user_id'].nunique()} users")

    # Compute volatility profiles
    detector = TasteDriftDetector(DetectorConfig())
    profiles = detector.fit(df)

    # Build drift timestamp lookup for post-drift evaluation
    drift_timestamps = {}
    for uid, profile in profiles.items():
        if profile.drift_events:
            drift_timestamps[uid] = profile.drift_events[0].drift_timestamp

    # Temporal train/test split
    train_df, test_df = train_test_split_temporal(df, test_fraction=0.20)

    cfg = RecommenderConfig()
    results = []

    for method in ["static", "fixed_decay", "adaptive"]:
        print(f"\nFitting {method} model...")
        model = AdaptiveMemoryRecommender(cfg)
        model.fit(train_df, profiles, mode=method)

        overall = evaluate(model, test_df, k=10)
        post_drift = evaluate(
            model, test_df, k=10,
            post_drift_only=True,
            drift_timestamps=drift_timestamps,
        )

        label = {
            "static":      "Baseline 1: Static CF (no decay)",
            "fixed_decay": "Baseline 2: Fixed decay (uniform lambda)",
            "adaptive":    "Ours: Adaptive decay (personalised lambda)",
        }[method]

        results.append(ExperimentResult(
            method=label,
            ndcg_at_10_overall=overall["ndcg_at_k"],
            ndcg_at_10_post_drift=post_drift["ndcg_at_k"],
            hit_rate_overall=overall["hit_rate"],
            hit_rate_post_drift=post_drift["hit_rate"],
            n_users=overall["n_users_evaluated"],
        ))
        print(f"  Overall NDCG@10: {overall['ndcg_at_k']:.4f}")
        print(f"  Post-drift NDCG@10: {post_drift['ndcg_at_k']:.4f}")

    return results


def print_report(results: list[ExperimentResult]) -> None:
    width = 74
    bar = "─" * width
    print(f"\n{'ADAPTIVE MEMORY — BENCHMARK RESULTS':^{width}}")
    print(bar)
    print(f"  {'Method':<42} {'NDCG@10':>8} {'Post-drift':>11} {'HR':>6}")
    print(bar)
    best_ndcg = max(r.ndcg_at_10_overall for r in results)
    best_pd   = max(r.ndcg_at_10_post_drift for r in results)
    for r in results:
        m1 = " *" if r.ndcg_at_10_overall == best_ndcg else "  "
        m2 = " *" if r.ndcg_at_10_post_drift == best_pd else "  "
        print(
            f"  {r.method:<42} "
            f"{r.ndcg_at_10_overall:>7.4f}{m1} "
            f"{r.ndcg_at_10_post_drift:>9.4f}{m2} "
            f"{r.hit_rate_overall:>5.3f}"
        )
    print(bar)

    our = results[-1]
    baseline = results[1]  # fixed decay is the fairest comparison
    if baseline.ndcg_at_10_post_drift > 0:
        improvement = (our.ndcg_at_10_post_drift - baseline.ndcg_at_10_post_drift) / baseline.ndcg_at_10_post_drift * 100
        print(f"\n  Post-drift NDCG improvement vs fixed decay: {improvement:+.1f}%")
        print(f"  This is the headline result: personalised forgetting improves")
        print(f"  recommendation quality specifically after taste drift events.\n")


@click.command()
@click.option("--data", required=True)
@click.option("--output", default=None)
def main(data, output):
    results = run_experiments(data)
    print_report(results)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"Saved to {output}")


if __name__ == "__main__":
    main()
