"""
data_loader.py — MovieLens dataset ingestion and preprocessing.

MovieLens 1M dataset:
- 1,000,209 ratings from 6,040 users on 3,706 movies
- Ratings from 1 to 5, with timestamps
- Fully public, no auth required
- Download: https://grouplens.org/datasets/movielens/1m/

This is the gold standard dataset for recommendation research.
Every major recommender systems paper uses it as a benchmark.
Having results on MovieLens means your work is directly comparable
to published research.

Usage:
    python -m src.data_loader --output data/movielens.parquet
    python -m src.data_loader --synthetic --n-users 500 --output data/synthetic.parquet
"""

from __future__ import annotations

import io
import os
import click
import zipfile
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass


MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


# ------------------------------------------------------------------ #
#  Data model                                                          #
# ------------------------------------------------------------------ #

@dataclass
class UserProfile:
    """
    Everything we know about a user's taste trajectory.
    Built from their rating history.
    """
    user_id: int
    n_ratings: int
    active_days: int
    genre_switches: int          # how often they jump between genre clusters
    taste_volatility: float      # 0-1, our target variable
    temporal_segments: list      # list of taste "episodes"
    dominant_genres: list[str]
    has_drift_event: bool        # did we detect a taste shift?
    drift_timestamp: float       # when the drift happened (unix ts)


# ------------------------------------------------------------------ #
#  Live download                                                       #
# ------------------------------------------------------------------ #

def download_movielens(output_dir: str = "data") -> pd.DataFrame:
    """
    Download and parse MovieLens 1M dataset.
    Returns a DataFrame with columns:
    user_id, movie_id, rating, timestamp, genres, title
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cache_path = os.path.join(output_dir, "ml1m_raw.parquet")

    if os.path.exists(cache_path):
        print(f"Loading cached MovieLens data from {cache_path}")
        return pd.read_parquet(cache_path)

    print("Downloading MovieLens 1M dataset (~6MB)...")
    try:
        resp = requests.get(MOVIELENS_1M_URL, timeout=30, stream=True)
        resp.raise_for_status()
        total = 0
        chunks = []
        for chunk in resp.iter_content(8192):
            chunks.append(chunk)
            total += len(chunk)
            print(f"  {total/1024:.0f}KB downloaded...", end="\r")
        print(f"\n  Downloaded {total/1024:.0f}KB")
        zip_bytes = b"".join(chunks)
    except requests.RequestException as e:
        print(f"Download failed: {e}")
        print("Falling back to synthetic data. Run with --synthetic flag.")
        return pd.DataFrame()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        # Ratings
        with z.open("ml-1m/ratings.dat") as f:
            ratings = pd.read_csv(
                f, sep="::", engine="python",
                names=["user_id", "movie_id", "rating", "timestamp"],
                encoding="latin-1",
            )

        # Movies
        with z.open("ml-1m/movies.dat") as f:
            movies = pd.read_csv(
                f, sep="::", engine="python",
                names=["movie_id", "title", "genres"],
                encoding="latin-1",
            )

    df = ratings.merge(movies, on="movie_id", how="left")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    df.to_parquet(cache_path, index=False)
    print(f"Saved {len(df):,} ratings to {cache_path}")
    return df


# ------------------------------------------------------------------ #
#  Synthetic generator                                                 #
# ------------------------------------------------------------------ #

GENRE_CLUSTERS = {
    "action":  ["Action", "Adventure", "Thriller"],
    "drama":   ["Drama", "Romance", "Crime"],
    "comedy":  ["Comedy", "Animation", "Children's"],
    "sci_fi":  ["Sci-Fi", "Horror", "Fantasy"],
    "docs":    ["Documentary", "Musical", "Western"],
}

ALL_GENRES = [g for gs in GENRE_CLUSTERS.values() for g in gs]


def generate_synthetic(
    n_users: int = 500,
    n_days: int = 730,         # 2 years of data
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic rating data that mimics MovieLens structure
    with realistic taste drift patterns.

    Three user archetypes:
    - Stable (40%): consistent preferences over time
    - Drifter (40%): gradual taste evolution
    - Volatile (20%): frequent genre switching
    """
    rng = np.random.default_rng(seed)
    rows = []
    base_ts = pd.Timestamp("2020-01-01")

    archetypes = (
        ["stable"] * int(n_users * 0.40) +
        ["drifter"] * int(n_users * 0.40) +
        ["volatile"] * int(n_users * 0.20)
    )
    rng.shuffle(archetypes)

    genre_list = list(GENRE_CLUSTERS.keys())

    for user_id, archetype in enumerate(archetypes[:n_users], 1):
        # Each user has a primary genre cluster
        primary_cluster = genre_list[int(rng.integers(0, len(genre_list)))]
        primary_genres = GENRE_CLUSTERS[primary_cluster]

        if archetype == "stable":
            n_ratings = int(rng.integers(50, 200))
            ratings_per_day = n_ratings / n_days
            drift_point = None
        elif archetype == "drifter":
            n_ratings = int(rng.integers(80, 300))
            ratings_per_day = n_ratings / n_days
            drift_point = rng.uniform(0.3, 0.7)  # drift happens mid-history
        else:
            n_ratings = int(rng.integers(100, 400))
            ratings_per_day = n_ratings / n_days
            drift_point = None

        for i in range(n_ratings):
            day = int(rng.integers(0, n_days))
            progress = day / n_days

            # Determine genre based on archetype and temporal position
            if archetype == "stable":
                genre = primary_genres[int(rng.integers(0, len(primary_genres)))]
            elif archetype == "drifter" and drift_point and progress > drift_point:
                # After drift — different cluster
                other_clusters = [g for g in genre_list if g != primary_cluster]
                new_cluster = other_clusters[int(rng.integers(0, len(other_clusters)))]
                new_genres = GENRE_CLUSTERS[new_cluster]
                genre = new_genres[int(rng.integers(0, len(new_genres)))]
            elif archetype == "volatile":
                # Random genre every ~20 ratings
                if i % 20 < 5:
                    random_cluster = genre_list[int(rng.integers(0, len(genre_list)))]
                    genre = GENRE_CLUSTERS[random_cluster][0]
                else:
                    genre = primary_genres[int(rng.integers(0, len(primary_genres)))]
            else:
                genre = primary_genres[int(rng.integers(0, len(primary_genres)))]

            # Rating: higher for preferred genre, lower otherwise
            is_preferred = genre in primary_genres
            if archetype == "drifter" and drift_point and progress > drift_point:
                is_preferred = not is_preferred

            base_rating = 4.0 if is_preferred else 2.5
            rating = float(np.clip(rng.normal(base_rating, 0.8), 1, 5))
            rating = round(rating * 2) / 2  # MovieLens uses 0.5 increments

            ts = base_ts + pd.Timedelta(days=day) + pd.Timedelta(
                seconds=int(rng.integers(0, 86400))
            )

            rows.append({
                "user_id": user_id,
                "movie_id": int(rng.integers(1, 3706)),
                "rating": rating,
                "timestamp": ts,
                "genres": genre,
                "title": f"Movie_{rng.integers(1,9999):04d}",
                "archetype": archetype,
                "true_drift": 1 if (archetype == "drifter" and drift_point and
                                    (day / n_days) > drift_point) else 0,
            })

    df = pd.DataFrame(rows).sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    print(f"Generated {len(df):,} synthetic ratings for {n_users} users")
    arch_counts = {}
    for a in archetypes[:n_users]:
        arch_counts[a] = arch_counts.get(a, 0) + 1
    for k, v in sorted(arch_counts.items()):
        print(f"  {k}: {v} users")
    return df


# ------------------------------------------------------------------ #
#  CLI                                                                  #
# ------------------------------------------------------------------ #

@click.command()
@click.option("--output", default="data/ratings.parquet", help="Output path")
@click.option("--synthetic", is_flag=True, help="Generate synthetic data")
@click.option("--n-users", default=500, help="Users in synthetic dataset")
@click.option("--seed", default=42)
def main(output, synthetic, n_users, seed):
    """Download or generate rating data for adaptive memory research."""
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    if synthetic:
        df = generate_synthetic(n_users=n_users, seed=seed)
    else:
        df = download_movielens(output_dir=str(Path(output).parent))
        if df.empty:
            print("Falling back to synthetic...")
            df = generate_synthetic(n_users=n_users, seed=seed)
    df.to_parquet(output, index=False)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
