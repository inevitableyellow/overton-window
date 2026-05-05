"""
overton_pipeline.py
===================
Overton Window Discourse Shift Analysis Pipeline
SI 630: Natural Language Processing — Winter 2026
Paris Heard, School of Information, University of Michigan

Measures lexical and semantic drift in U.S. news headlines (2001–2024)
across five major outlets using TF-IDF cosine distance, sentence embeddings,
permutation testing, and VADER sentiment analysis.

DATA SOURCE:
    dess-mannheim/US_Multi_Outlet_News_Headlines2001_2024 (HuggingFace, restricted)
    Request access: https://huggingface.co/datasets/dess-mannheim/US_Multi_Outlet_News_Headlines2001_2024

EXAMPLE USAGE:
    # Run full pipeline with defaults
    python overton_pipeline.py

    # Custom date split and outlets
    python overton_pipeline.py \\
        --baseline-start 2001 --baseline-end 2015 \\
        --comparison-start 2016 --comparison-end 2024 \\
        --outlets nyt lat wsj foxnews msnbc \\
        --output-dir results/

    # Skip slow steps for quick runs
    python overton_pipeline.py --skip-embeddings --skip-permutation-test

REQUIREMENTS:
    pip install datasets huggingface-hub scikit-learn pandas numpy
                matplotlib scipy vaderSentiment sentence-transformers umap-learn tqdm
"""

import re
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_OUTLETS        = ["nyt", "lat", "wsj", "foxnews", "msnbc"]
DEFAULT_BASELINE_START = 2001
DEFAULT_BASELINE_END   = 2015
DEFAULT_COMP_START     = 2016
DEFAULT_COMP_END       = 2024
MAX_FEATURES           = 10_000
NGRAM_RANGE            = (1, 2)
MIN_DF                 = 5
MIN_WORDS              = 4

BLUE   = "steelblue"
ORANGE = "darkorange"
GRAY   = "gray"

ARTIFACT_TERMS = {
    "said", "says", "new", "year", "years", "day", "days",
    "week", "time", "just", "like", "make", "made", "say",
}

TOPIC_FILTERS = {
    "covid":    ["covid", "coronavirus", "pandemic", "vaccine", "omicron"],
    "election": ["election", "vote", "ballot", "voter", "voting", "primary"],
    "trump":    ["trump", "trumps", "donald"],
    "combined": ["covid", "coronavirus", "pandemic", "vaccine", "election",
                 "vote", "ballot", "trump", "trumps"],
}


# ── 1. Preprocessing ──────────────────────────────────────────────────────────

def parse_date(date_str: str) -> str | None:
    """
    Parse a date string in YYYY/MM or YYYY/MM/DD format.

    Returns a normalized 'YYYY-MM' string, or None if the date is invalid
    or outside plausible range (1990–2030).
    """
    try:
        parts = str(date_str).strip().split("/")
        if len(parts) >= 2:
            year, month = int(parts[0]), int(parts[1])
            if 1990 <= year <= 2030 and 1 <= month <= 12:
                return f"{year:04d}-{month:02d}"
    except (ValueError, AttributeError):
        pass
    return None


def preprocess(text: str) -> str:
    """
    Normalize a headline for TF-IDF vectorization.

    Steps: lowercase → strip URLs → keep only letters/digits/hyphens
    → collapse whitespace.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+",        "", text)
    text = re.sub(r"[^a-z0-9\s\-]", "", text)
    text = re.sub(r"\s+",           " ", text).strip()
    return text


def is_valid(text: str, min_words: int = MIN_WORDS) -> bool:
    """Return True if the preprocessed headline meets the minimum word count."""
    return len(str(text).split()) >= min_words


# ── 2. Data Loading ───────────────────────────────────────────────────────────

def load_and_split(
    outlets: list[str]       = DEFAULT_OUTLETS,
    baseline_start: int      = DEFAULT_BASELINE_START,
    baseline_end: int        = DEFAULT_BASELINE_END,
    comparison_start: int    = DEFAULT_COMP_START,
    comparison_end: int      = DEFAULT_COMP_END,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the HuggingFace dataset and split into baseline and comparison corpora.

    Requires authenticated HuggingFace access:
        from huggingface_hub import login; login()

    Parameters
    ----------
    outlets : list of str
        Outlet split names to include. Default: all five outlets.
    baseline_start / baseline_end : int
        Inclusive year range for the baseline corpus.
    comparison_start / comparison_end : int
        Inclusive year range for the comparison corpus.

    Returns
    -------
    df_baseline, df_comparison : pd.DataFrame
        Each with columns: ['title', 'title_clean', 'source', 'year_month']
    """
    from datasets import load_dataset

    print("Loading dataset from HuggingFace (requires authenticated access)...")
    ds = load_dataset(
        "dess-mannheim/US_Multi_Outlet_News_Headlines2001_2024", "raw"
    )

    baseline_records, comparison_records = [], []

    for outlet in outlets:
        print(f"  Processing {outlet}...", end="", flush=True)
        b_count = c_count = 0

        for row in ds[outlet]:
            date = parse_date(row["date"])
            if date is None:
                continue
            year = int(date[:4])

            headline = preprocess(row["headline"])
            if not is_valid(headline):
                continue

            record = {"title": headline, "source": outlet, "year_month": date}

            if baseline_start <= year <= baseline_end:
                baseline_records.append(record)
                b_count += 1
            elif comparison_start <= year <= comparison_end:
                comparison_records.append(record)
                c_count += 1

        print(f" baseline={b_count:,}  comparison={c_count:,}")

    df_baseline   = pd.DataFrame(baseline_records)
    df_comparison = pd.DataFrame(comparison_records)

    # clean column is identical to title here (already preprocessed)
    df_baseline["title_clean"]   = df_baseline["title"]
    df_comparison["title_clean"] = df_comparison["title"]

    print(
        f"\nBaseline   : {len(df_baseline):,} headlines  "
        f"({df_baseline['year_month'].nunique()} monthly bins)"
    )
    print(
        f"Comparison : {len(df_comparison):,} headlines  "
        f"({df_comparison['year_month'].nunique()} monthly bins)"
    )
    return df_baseline, df_comparison


# ── 3. TF-IDF Vectorization ───────────────────────────────────────────────────

def build_tfidf(
    df_baseline: pd.DataFrame,
    df_comparison: pd.DataFrame,
    max_features: int  = MAX_FEATURES,
    ngram_range: tuple = NGRAM_RANGE,
    min_df: int        = MIN_DF,
) -> tuple:
    """
    Fit a TF-IDF vectorizer on the baseline corpus and transform both corpora.

    The vectorizer is fit *only* on the baseline so that the comparison corpus
    is measured against the baseline's vocabulary — a deliberate methodological
    choice that mirrors how Overton Window drift is conceptualized.

    Returns
    -------
    vectorizer : TfidfVectorizer (fitted)
    X_baseline : sparse matrix  (n_baseline_docs × max_features)
    X_comparison : sparse matrix
    feature_names : list of str
    centroid_baseline : np.ndarray (max_features,)
    """
    print("Fitting TF-IDF vectorizer on baseline corpus...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
    )
    X_baseline   = vectorizer.fit_transform(df_baseline["title_clean"])
    X_comparison = vectorizer.transform(df_comparison["title_clean"])

    feature_names      = vectorizer.get_feature_names_out()
    centroid_baseline  = np.asarray(X_baseline.mean(axis=0)).flatten()

    print(f"  Vocabulary size : {len(feature_names):,}")
    print(f"  Baseline matrix : {X_baseline.shape}")
    print(f"  Comparison matrix: {X_comparison.shape}")

    return vectorizer, X_baseline, X_comparison, feature_names, centroid_baseline


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine distance (1 − cosine similarity) between two vectors."""
    sim = cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0]
    return float(1.0 - sim)


# ── 4. Monthly Drift ──────────────────────────────────────────────────────────

def compute_monthly_drift(
    df: pd.DataFrame,
    X: object,
    centroid_baseline: np.ndarray,
    corpus_label: str,
) -> pd.DataFrame:
    """
    Compute per-month cosine distance from the baseline centroid and
    intra-month spread (std of per-doc cosine similarities to centroid).

    Parameters
    ----------
    df : DataFrame with a 'year_month' column aligned to rows of X
    X  : sparse TF-IDF matrix (same row order as df)
    centroid_baseline : baseline centroid vector
    corpus_label : 'baseline' or 'comparison'

    Returns
    -------
    DataFrame with columns:
        month, cosine_distance_baseline, intra_month_spread, n_headlines, corpus
    """
    records = []
    for month, group in df.groupby("year_month"):
        idx  = group.index.tolist()
        mat  = X[idx]
        ctrd = np.asarray(mat.mean(axis=0)).flatten()

        dist  = cosine_distance(ctrd, centroid_baseline)
        sims  = cosine_similarity(mat, centroid_baseline.reshape(1, -1)).flatten()
        spread = float(np.std(sims))

        records.append({
            "month":                    month,
            "cosine_distance_baseline": dist,
            "intra_month_spread":       spread,
            "n_headlines":              len(idx),
            "corpus":                   corpus_label,
        })

    return pd.DataFrame(records).sort_values("month").reset_index(drop=True)


# ── 5. Permutation Test ───────────────────────────────────────────────────────

def permutation_test(
    df_baseline: pd.DataFrame,
    df_comparison: pd.DataFrame,
    X_baseline: object,
    X_comparison: object,
    centroid_baseline: np.ndarray,
    n_permutations: int = 1000,
    sample_size: int    = 5000,
    random_state: int   = 42,
) -> dict:
    """
    Non-parametric permutation test for observed Overton Window drift.

    Pools a sample from both corpora, randomly splits n_permutations times,
    and computes the mean cosine distance from the baseline centroid for each
    permuted 'comparison' half. The empirical p-value is the proportion of
    permuted means >= the observed mean.

    Returns
    -------
    dict with keys:
        observed_mean, permuted_means (array), p_value, significant
    """
    rng = np.random.default_rng(random_state)

    # observed mean
    observed_mean = float(
        np.mean([
            cosine_distance(
                np.asarray(X_comparison[i].mean(axis=0)).flatten(),
                centroid_baseline,
            )
            for i in range(min(sample_size, X_comparison.shape[0]))
        ])
    )

    print(f"Observed comparison mean cosine distance: {observed_mean:.4f}")
    print(f"Running {n_permutations} permutations...")

    # pool
    n_base = min(sample_size // 2, X_baseline.shape[0])
    n_comp = min(sample_size // 2, X_comparison.shape[0])
    base_idx = rng.choice(X_baseline.shape[0], n_base, replace=False)
    comp_idx = rng.choice(X_comparison.shape[0], n_comp, replace=False)

    import scipy.sparse as sp
    pool = sp.vstack([X_baseline[base_idx], X_comparison[comp_idx]])
    n_pool = pool.shape[0]

    permuted_means = []
    for i in range(n_permutations):
        perm = rng.permutation(n_pool)
        perm_comp = pool[perm[:n_comp]]
        dists = cosine_similarity(
            perm_comp, centroid_baseline.reshape(1, -1)
        ).flatten()
        permuted_means.append(float(1.0 - dists.mean()))

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_permutations} done...")

    permuted_means = np.array(permuted_means)
    p_value = float(np.mean(permuted_means >= observed_mean))

    print(f"\n=== Permutation Test Results ===")
    print(f"Observed mean cosine distance : {observed_mean:.4f}")
    print(f"Permuted mean (mean)          : {permuted_means.mean():.4f}")
    print(f"Permuted mean (std)           : {permuted_means.std():.4f}")
    print(f"Permuted mean (95th pct)      : {np.percentile(permuted_means, 95):.4f}")
    print(f"p-value                       : {p_value:.4f}")
    print(f"Significant (p < 0.05)        : {p_value < 0.05}")

    return {
        "observed_mean":  observed_mean,
        "permuted_means": permuted_means,
        "p_value":        p_value,
        "significant":    p_value < 0.05,
    }


# ── 6. Sentiment Analysis ─────────────────────────────────────────────────────

def compute_sentiment(df: pd.DataFrame, corpus_label: str) -> pd.DataFrame:
    """
    Score each headline with VADER and aggregate monthly statistics.

    VADER (Valence Aware Dictionary and Sentiment Reasoner) is well-suited
    to short, informal text like news headlines. The compound score ranges
    from -1 (most negative) to +1 (most positive).

    Returns
    -------
    DataFrame with columns:
        month, mean_sentiment, sentiment_std, sentiment_volatility, corpus
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    print(f"Scoring {corpus_label} sentiment ({len(df):,} headlines)...")
    df = df.copy()
    df["compound"] = df["title"].apply(
        lambda t: analyzer.polarity_scores(t)["compound"]
    )
    df["sentiment_bin"] = pd.cut(
        df["compound"],
        bins=[-1.01, -0.05, 0.05, 1.01],
        labels=["negative", "neutral", "positive"],
    )

    monthly = (
        df.groupby("year_month")
        .agg(
            mean_sentiment     = ("compound", "mean"),
            sentiment_std      = ("compound", "std"),
            sentiment_volatility = ("compound", lambda x: x.std()),
            n_headlines        = ("compound", "count"),
        )
        .reset_index()
        .rename(columns={"year_month": "month"})
    )
    monthly["corpus"] = corpus_label
    return monthly, df


def sentiment_conditioned_drift(
    df_baseline: pd.DataFrame,
    df_comparison: pd.DataFrame,
    X_baseline: object,
    X_comparison: object,
    centroid_baseline: np.ndarray,
) -> dict:
    """
    Compute monthly cosine drift separately for negative, neutral, and positive
    headlines in each corpus.

    Returns
    -------
    dict mapping sentiment bin → DataFrame of monthly drift stats
    """
    results = {}
    for sent_bin in ["negative", "neutral", "positive"]:
        records = []
        for corpus_label, df, X in [
            ("baseline",   df_baseline,   X_baseline),
            ("comparison", df_comparison, X_comparison),
        ]:
            mask = df["sentiment_bin"] == sent_bin
            sub  = df[mask].reset_index(drop=True)
            orig_idx = df[mask].index.tolist()
            if len(orig_idx) < 10:
                continue

            for month, group in sub.groupby("year_month"):
                local_idx = group.index.tolist()
                global_idx = [orig_idx[i] for i in local_idx]
                mat  = X[global_idx]
                ctrd = np.asarray(mat.mean(axis=0)).flatten()
                dist = cosine_distance(ctrd, centroid_baseline)
                records.append({
                    "month":       month,
                    "cosine_dist": dist,
                    "corpus":      corpus_label,
                    "n":           len(local_idx),
                })

        if records:
            results[sent_bin] = (
                pd.DataFrame(records).sort_values("month").reset_index(drop=True)
            )

    return results


# ── 7. Topic Filtering ────────────────────────────────────────────────────────

def apply_topic_filter(
    df: pd.DataFrame,
    keywords: list[str],
) -> pd.DataFrame:
    """Return df with rows containing any of the keywords removed."""
    pattern = "|".join(keywords)
    mask = df["title"].str.contains(pattern, case=False, na=False)
    return df[~mask].reset_index(drop=True)


def run_filtered_analysis(
    df_baseline: pd.DataFrame,
    df_comparison: pd.DataFrame,
    X_baseline: object,
    X_comparison: object,
    centroid_baseline: np.ndarray,
    filter_name: str,
    keywords: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove headlines matching keywords from both corpora, refit TF-IDF,
    and return monthly drift DataFrames for baseline and comparison.
    """
    df_b_f = apply_topic_filter(df_baseline,   keywords)
    df_c_f = apply_topic_filter(df_comparison, keywords)

    _, X_b_f, X_c_f, _, ctrd_f = build_tfidf(df_b_f, df_c_f)

    df_b_drift = compute_monthly_drift(df_b_f, X_b_f, ctrd_f, "baseline")
    df_c_drift = compute_monthly_drift(df_c_f, X_c_f, ctrd_f, "comparison")

    removed_b = len(df_baseline) - len(df_b_f)
    removed_c = len(df_comparison) - len(df_c_f)
    print(
        f"  [{filter_name}] removed {removed_b:,} baseline / "
        f"{removed_c:,} comparison headlines"
    )
    return df_b_drift, df_c_drift


# ── 8. Top TF-IDF Terms ───────────────────────────────────────────────────────

def top_tfidf_terms(
    centroid: np.ndarray,
    feature_names: np.ndarray,
    n: int = 15,
    exclude: set = ARTIFACT_TERMS,
) -> list[tuple[str, float]]:
    """
    Return the top-n TF-IDF terms from a centroid vector, excluding artifacts.

    Parameters
    ----------
    centroid      : centroid vector (max_features,)
    feature_names : vocabulary array from the fitted vectorizer
    n             : number of terms to return
    exclude       : set of terms to skip (stopword-like artifacts)

    Returns
    -------
    list of (term, weight) tuples sorted descending by weight
    """
    top_idx = centroid.argsort()[::-1]
    results = []
    for i in top_idx:
        term = feature_names[i]
        if term not in exclude:
            results.append((term, round(float(centroid[i]), 4)))
        if len(results) >= n:
            break
    return results


# ── 9. Figures ────────────────────────────────────────────────────────────────

def _shared_x_setup(axes, all_months: list, step_divisor: int = 24):
    """Apply shared x-axis tick labels to a list of matplotlib Axes."""
    step  = max(1, len(all_months) // step_divisor)
    ticks = list(range(0, len(all_months), step))
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels(
        [all_months[t] for t in ticks], rotation=45, ha="right", fontsize=8
    )


def plot_drift(
    df_baseline_drift: pd.DataFrame,
    df_comparison_drift: pd.DataFrame,
    output_path: str = "outputs/figures/fig1_tfidf_drift.png",
):
    """Figure 1: Monthly TF-IDF cosine distance and intra-month spread."""
    all_months = sorted(
        set(df_baseline_drift["month"]) | set(df_comparison_drift["month"])
    )
    m2x = {m: i for i, m in enumerate(all_months)}

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig.suptitle(
        "Figure 1: TF-IDF Cosine Distance from Baseline Centroid\n"
        "(NYT · LAT · WSJ · Fox News · MSNBC  |  2001–2024)",
        fontsize=13,
    )

    for corpus, df, color in [
        ("Baseline",   df_baseline_drift,   BLUE),
        ("Comparison", df_comparison_drift, ORANGE),
    ]:
        x = [m2x[m] for m in df["month"]]
        axes[0].plot(x, df["cosine_distance_baseline"], color=color,
                     linewidth=1.5, marker="o", markersize=2, label=corpus)
        axes[1].plot(x, df["intra_month_spread"],       color=color,
                     linewidth=1.5, marker="o", markersize=2, label=corpus)

    b_s = m2x[df_baseline_drift["month"].iloc[0]]
    b_e = m2x[df_baseline_drift["month"].iloc[-1]]
    for ax in axes:
        ax.axvspan(b_s, b_e, color=GRAY, alpha=0.08, label="Baseline Period")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Cosine Distance\nfrom Baseline Centroid")
    axes[0].set_title("Monthly Cosine Distance to Baseline Centroid")
    axes[1].set_ylabel("Intra-Month Spread\nof Representations")
    axes[1].set_xlabel("Month (YYYY-MM)")

    _shared_x_setup(axes, all_months)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_permutation_test(
    perm_results: dict,
    output_path: str = "outputs/figures/fig2_permutation_test.png",
):
    """Figure 2: Permutation test null distribution vs observed drift."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Figure 2: Permutation Test — Null Distribution of Mean Cosine Distance",
                 fontsize=13)

    ax.hist(perm_results["permuted_means"], bins=50, color=BLUE,
            alpha=0.7, label="Permuted means (n=1,000)")
    ax.axvline(perm_results["observed_mean"], color=ORANGE, linewidth=2,
               label=f"Observed mean = {perm_results['observed_mean']:.4f}")
    ax.set_xlabel("Mean Cosine Distance from Baseline Centroid")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_sentiment(
    sent_baseline: pd.DataFrame,
    sent_comparison: pd.DataFrame,
    output_path: str = "outputs/figures/fig3_sentiment.png",
):
    """Figure 3: Mean sentiment, std, and volatility over time."""
    all_months = sorted(
        set(sent_baseline["month"]) | set(sent_comparison["month"])
    )
    m2x = {m: i for i, m in enumerate(all_months)}

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(
        "Figure 3: Headline Sentiment Over Time\n"
        "(NYT · LAT · WSJ · Fox News · MSNBC  |  2001–2024)",
        fontsize=13,
    )

    metrics = ["mean_sentiment", "sentiment_std", "sentiment_volatility"]
    titles  = ["Mean Sentiment (VADER compound)",
               "Sentiment Std Dev",
               "Intra-Month Sentiment Volatility"]

    for ax, metric, title in zip(axes, metrics, titles):
        for corpus, df, color in [
            ("Baseline",   sent_baseline,   BLUE),
            ("Comparison", sent_comparison, ORANGE),
        ]:
            x = [m2x[m] for m in df["month"] if m in m2x]
            y = df[metric].tolist()
            ax.plot(x, y, color=color, linewidth=1.5,
                    marker="o", markersize=2, label=corpus)

        b_s = m2x[sent_baseline["month"].iloc[0]]
        b_e = m2x[sent_baseline["month"].iloc[-1]]
        ax.axvspan(b_s, b_e, alpha=0.06, color=GRAY)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    _shared_x_setup(axes, all_months)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_sentiment_conditioned_drift(
    results_by_sentiment: dict,
    sent_baseline: pd.DataFrame,
    output_path: str = "outputs/figures/fig4_sentiment_conditioned.png",
):
    """Figure 4: Cosine drift conditioned on headline sentiment valence."""
    all_months = sorted(set().union(*[
        df["month"].unique()
        for df in results_by_sentiment.values()
    ]))
    m2x = {m: i for i, m in enumerate(all_months)}

    sent_labels = {
        "negative": "Negative Headlines",
        "neutral":  "Neutral Headlines",
        "positive": "Positive Headlines",
    }

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True, sharey=True)
    fig.suptitle(
        "Figure 4: Sentiment-Conditioned Cosine Distance from Baseline Centroid\n"
        "(NYT · LAT · WSJ · Fox News · MSNBC  |  2001–2024)",
        fontsize=13,
    )

    for ax, sent_bin in zip(axes, ["negative", "neutral", "positive"]):
        if sent_bin not in results_by_sentiment:
            continue
        df_s = results_by_sentiment[sent_bin]

        for corpus, color in [("baseline", BLUE), ("comparison", ORANGE)]:
            sub = df_s[df_s["corpus"] == corpus]
            x   = [m2x[m] for m in sub["month"] if m in m2x]
            ax.plot(x, sub["cosine_dist"].tolist(), color=color, linewidth=1.5,
                    marker="o", markersize=2,
                    label=f"{corpus.capitalize()} "
                          f"({'2001–2015' if corpus == 'baseline' else '2016–2024'})")

        base_months = df_s[df_s["corpus"] == "baseline"]["month"]
        ax.axvspan(m2x[base_months.min()], m2x[base_months.max()],
                   alpha=0.08, color=GRAY)
        ax.set_ylabel("Cosine Distance\nfrom Baseline Centroid")
        ax.set_title(sent_labels[sent_bin], fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    _shared_x_setup(axes, all_months)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_topic_filters(
    df_comparison_drift: pd.DataFrame,
    filter_results: dict,
    output_path: str = "outputs/figures/fig5_topic_filters.png",
):
    """Figure 5: Comparison corpus drift with topic-filtered variants overlaid."""
    all_months = sorted(df_comparison_drift["month"].unique())
    m2x = {m: i for i, m in enumerate(all_months)}

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.suptitle(
        "Figure 5: Topic-Filtered Drift — Comparison Corpus (2016–2024)",
        fontsize=13,
    )

    x_base = [m2x[m] for m in df_comparison_drift["month"] if m in m2x]
    ax.plot(x_base, df_comparison_drift["cosine_distance_baseline"],
            color="black", linewidth=2, label="Unfiltered", zorder=5)

    filter_colors = {"covid": "red", "election": "blue",
                     "trump": "green",  "combined": "orange"}

    for name, (_, cs) in filter_results.items():
        x = [m2x[m] for m in cs["month"] if m in m2x]
        ax.plot(x, cs["cosine_distance_baseline"], linewidth=1.5,
                color=filter_colors.get(name, "gray"),
                linestyle="--", label=f"Filtered: {name}")

    ax.set_xlabel("Month (YYYY-MM)")
    ax.set_ylabel("Cosine Distance from Baseline Centroid")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    step  = max(1, len(all_months) // 24)
    ticks = list(range(0, len(all_months), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([all_months[t] for t in ticks],
                       rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ── 10. Save Results ──────────────────────────────────────────────────────────

def save_results(
    df_baseline_drift: pd.DataFrame,
    df_comparison_drift: pd.DataFrame,
    sent_baseline: pd.DataFrame,
    sent_comparison: pd.DataFrame,
    output_dir: str = "outputs/results",
):
    """Save monthly drift and sentiment CSVs to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df_baseline_drift.to_csv(out / "baseline_monthly_drift.csv", index=False)
    df_comparison_drift.to_csv(out / "comparison_monthly_drift.csv", index=False)
    pd.concat([df_baseline_drift, df_comparison_drift]).to_csv(
        out / "all_monthly_drift.csv", index=False
    )
    sent_baseline.to_csv(out / "baseline_monthly_sentiment.csv", index=False)
    sent_comparison.to_csv(out / "comparison_monthly_sentiment.csv", index=False)

    print(f"Results saved to {output_dir}/")


# ── 11. CLI ───────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Overton Window Discourse Shift Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baseline-start",   type=int, default=DEFAULT_BASELINE_START)
    parser.add_argument("--baseline-end",     type=int, default=DEFAULT_BASELINE_END)
    parser.add_argument("--comparison-start", type=int, default=DEFAULT_COMP_START)
    parser.add_argument("--comparison-end",   type=int, default=DEFAULT_COMP_END)
    parser.add_argument(
        "--outlets", nargs="+", default=DEFAULT_OUTLETS,
        choices=["nyt", "lat", "wsj", "foxnews", "msnbc"],
        help="Outlets to include",
    )
    parser.add_argument("--max-features",  type=int, default=MAX_FEATURES)
    parser.add_argument("--min-df",        type=int, default=MIN_DF)
    parser.add_argument("--n-permutations",type=int, default=1000)
    parser.add_argument("--output-dir",    type=str, default="outputs")
    parser.add_argument(
        "--skip-embeddings",      action="store_true",
        help="Skip sentence embedding analysis (slow)",
    )
    parser.add_argument(
        "--skip-permutation-test", action="store_true",
        help="Skip permutation test (slow)",
    )
    parser.add_argument(
        "--skip-sentiment",       action="store_true",
        help="Skip VADER sentiment analysis",
    )
    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    out_figures = Path(args.output_dir) / "figures"
    out_results = Path(args.output_dir) / "results"

    # ── Step 1: Load data
    print("\n=== Step 1: Loading Data ===")
    df_baseline, df_comparison = load_and_split(
        outlets          = args.outlets,
        baseline_start   = args.baseline_start,
        baseline_end     = args.baseline_end,
        comparison_start = args.comparison_start,
        comparison_end   = args.comparison_end,
    )

    # ── Step 2: TF-IDF
    print("\n=== Step 2: TF-IDF Vectorization ===")
    vectorizer, X_baseline, X_comparison, feature_names, centroid_baseline = build_tfidf(
        df_baseline, df_comparison,
        max_features = args.max_features,
        min_df       = args.min_df,
    )

    # ── Step 3: Monthly drift
    print("\n=== Step 3: Monthly Drift ===")
    df_base_drift = compute_monthly_drift(
        df_baseline, X_baseline, centroid_baseline, "baseline"
    )
    df_comp_drift = compute_monthly_drift(
        df_comparison, X_comparison, centroid_baseline, "comparison"
    )

    print(f"\nBaseline drift mean  : {df_base_drift['cosine_distance_baseline'].mean():.4f}")
    print(f"Comparison drift mean: {df_comp_drift['cosine_distance_baseline'].mean():.4f}")

    plot_drift(df_base_drift, df_comp_drift,
               str(out_figures / "fig1_tfidf_drift.png"))

    # ── Step 4: Top terms
    print("\n=== Step 4: Top TF-IDF Terms ===")
    print("Baseline:")
    for term, w in top_tfidf_terms(centroid_baseline, feature_names):
        print(f"  {term:<25} {w:.4f}")

    comp_centroid = np.asarray(X_comparison.mean(axis=0)).flatten()
    print("Comparison:")
    for term, w in top_tfidf_terms(comp_centroid, feature_names):
        print(f"  {term:<25} {w:.4f}")

    # ── Step 5: Permutation test
    if not args.skip_permutation_test:
        print("\n=== Step 5: Permutation Test ===")
        perm_results = permutation_test(
            df_baseline, df_comparison,
            X_baseline, X_comparison,
            centroid_baseline,
            n_permutations=args.n_permutations,
        )
        plot_permutation_test(perm_results,
                              str(out_figures / "fig2_permutation_test.png"))

    # ── Step 6: Topic filtering
    print("\n=== Step 6: Topic Filtering ===")
    filter_results = {}
    for name, keywords in TOPIC_FILTERS.items():
        bs, cs = run_filtered_analysis(
            df_baseline, df_comparison,
            X_baseline, X_comparison,
            centroid_baseline, name, keywords,
        )
        filter_results[name] = (bs, cs)

    plot_topic_filters(df_comp_drift, filter_results,
                       str(out_figures / "fig5_topic_filters.png"))

    # ── Step 7: Sentiment
    if not args.skip_sentiment:
        print("\n=== Step 7: Sentiment Analysis ===")
        sent_base, df_baseline = compute_sentiment(df_baseline, "baseline")
        sent_comp, df_comparison = compute_sentiment(df_comparison, "comparison")

        b_sent = df_baseline["compound"].values
        c_sent = df_comparison["compound"].values
        t_stat, p_val = stats.ttest_ind(b_sent, c_sent)
        cohen_d = (c_sent.mean() - b_sent.mean()) / \
                  np.sqrt((b_sent.std()**2 + c_sent.std()**2) / 2)

        print(f"Baseline mean sentiment  : {b_sent.mean():.4f} (std={b_sent.std():.4f})")
        print(f"Comparison mean sentiment: {c_sent.mean():.4f} (std={c_sent.std():.4f})")
        print(f"t-statistic : {t_stat:.4f}")
        print(f"p-value     : {p_val:.4f}")
        print(f"Cohen's d   : {cohen_d:.4f}")

        plot_sentiment(sent_base, sent_comp,
                       str(out_figures / "fig3_sentiment.png"))

        sent_conditioned = sentiment_conditioned_drift(
            df_baseline, df_comparison,
            X_baseline, X_comparison, centroid_baseline,
        )
        plot_sentiment_conditioned_drift(
            sent_conditioned, sent_base,
            str(out_figures / "fig4_sentiment_conditioned.png"),
        )

        save_results(df_base_drift, df_comp_drift, sent_base, sent_comp,
                     str(out_results))
    else:
        save_results(df_base_drift, df_comp_drift,
                     pd.DataFrame(), pd.DataFrame(), str(out_results))

    print("\n✓ Pipeline complete.")


if __name__ == "__main__":
    main()
