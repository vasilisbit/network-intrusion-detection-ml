"""
Shared utilities for the CIC-IDS-2017 project.

Design rules:
  - No notebook-specific logic lives here. Only pure, reusable functions.
  - A single RANDOM_STATE constant is imported by every notebook.
  - Loading/cleaning is idempotent: you can call load_cic_ids_2017 multiple
    times and get the same result (important for caching).
"""

from __future__ import annotations

import glob
import os
from itertools import product
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd


# =============================================================================
# Global reproducibility seed. Import this everywhere.
# =============================================================================
RANDOM_STATE = 42


# =============================================================================
# Paths (relative to project root). Notebooks live in ./notebooks, so they
# resolve to parent for data/figures/results.
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"

for _p in (DATA_DIR, FIGURES_DIR, RESULTS_DIR):
    _p.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Loading the CIC-IDS-2017 CSV files
# =============================================================================
def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Column names in the CIC-IDS-2017 CSVs have leading spaces on every
    column (an artifact of CICFlowMeter's CSV export). We strip them so
    df["Flow Duration"] works instead of df[" Flow Duration"].

    The raw schema also contains a *genuinely* duplicated column:
    "Fwd Header Length" AND "Fwd Header Length.1" both appear as
    separate columns carrying the exact same values (verified against
    the raw data). This is a confirmed CICFlowMeter export bug. We drop
    the ".1" copy to keep the schema clean.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    if "Fwd Header Length.1" in df.columns:
        df = df.drop(columns=["Fwd Header Length.1"])

    return df


def _clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    The raw CSVs contain a UTF-8 replacement character (U+FFFD, bytes
    0xEF 0xBF 0xBD) in the Web Attack label names, where there should
    have been an en-dash. This corruption is baked into the raw files;
    no codec can recover the original character.

    We normalize all Label strings by:
      - Stripping surrounding whitespace.
      - Removing the replacement character (�) and any ASCII
        control/non-printable bytes that occasionally slip through.
      - Collapsing repeated whitespace to a single space.

    After this step, labels look like "Web Attack Brute Force" rather
    than "Web Attack � Brute Force" (or the even worse double-encoded
    "Web Attack ï¿½ Brute Force" you get from printing naively).
    """
    if "Label" not in df.columns:
        return df
    df = df.copy()
    labels = df["Label"].astype(str)
    labels = labels.str.replace("�", " ", regex=False)
    labels = labels.str.replace(r"[^\x20-\x7E]", " ", regex=True)
    labels = labels.str.replace(r"\s+", " ", regex=True).str.strip()
    df["Label"] = labels
    return df


def _optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    The raw files default every numeric column to float64. For ~2.8M rows
    and ~80 columns that's ~1.7 GB. Downcasting ints to int32 and floats
    to float32 cuts this by roughly half with no loss of analytical value
    (we don't need 15 digits of precision for packet counts).
    """
    df = df.copy()
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def load_cic_ids_2017(
    data_dir: str | os.PathLike | None = None,
    files: Iterable[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """
    Load and concatenate the CIC-IDS-2017 CSV files.

    Parameters
    ----------
    data_dir : path, optional
        Where the CSVs live. Defaults to DATA_DIR.
    files : iterable of str, optional
        Subset of filenames to load (e.g. for quick iteration).
    nrows : int, optional
        If given, read only the first N rows from EACH file. Handy for
        fast prototyping before committing to a full run.

    Returns
    -------
    pd.DataFrame
        A single concatenated, column-standardized, memory-optimized frame.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR

    if files is None:
        csv_paths = sorted(glob.glob(str(data_dir / "*.csv")))
    else:
        csv_paths = [str(data_dir / f) for f in files]

    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}. "
            "Download the CIC-IDS-2017 dataset from Kaggle and place the "
            "CSVs into the data/ directory."
        )

    frames = []
    for path in csv_paths:
        # low_memory=False avoids dtype guessing per-chunk, which causes
        # mixed-type warnings on these files.
        df = pd.read_csv(path, low_memory=False, nrows=nrows,
                         encoding="latin1")
        df = _standardize_columns(df)
        df = _clean_labels(df)
        df["__source_file"] = Path(path).name  # keeps provenance for debugging
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    full = _optimize_memory(full)
    return full


# =============================================================================
# Cached clean dataset — Parquet snapshot after Q1 preprocessing
# =============================================================================
CLEAN_PARQUET = DATA_DIR / "cic_ids_2017_clean.parquet"


def load_clean_cached() -> pd.DataFrame:
    """
    Return the cleaned dataset produced by Q1, loading from a Parquet
    snapshot on disk. Raises if Q1 hasn't been run yet.

    Q2 and Q3 both call this so they don't re-run the full Q1 pipeline.
    """
    if not CLEAN_PARQUET.exists():
        raise FileNotFoundError(
            f"{CLEAN_PARQUET} not found. Run notebooks/q1_eda.ipynb "
            "end-to-end first — it writes the cleaned snapshot."
        )
    return pd.read_parquet(CLEAN_PARQUET)


def save_clean_cached(df: pd.DataFrame) -> Path:
    """Persist the cleaned dataset so Q2/Q3 can load it instantly."""
    df.to_parquet(CLEAN_PARQUET, index=False)
    return CLEAN_PARQUET


# =============================================================================
# Cleaning helpers used in Q1 and reused in Q2/Q3
# =============================================================================
def handle_infinities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace +/- inf with NaN. The CIC-IDS-2017 dataset has many Infs in
    rate columns (e.g. Flow Bytes/s) because those are computed as
    bytes / duration and duration can be 0 microseconds for single-packet
    flows. Leaving them as Inf breaks any scaler or model.
    """
    return df.replace([np.inf, -np.inf], np.nan)


def drop_duplicates_report(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop exact-duplicate rows and return (cleaned_df, removed_count)."""
    before = len(df)
    cleaned = df.drop_duplicates().reset_index(drop=True)
    return cleaned, before - len(cleaned)


def kmeans_result_is_pathological(
    X, labels, sample_size: int = 20_000, seed: int = 42,
    min_frac: float = 0.001, sil_threshold: float = 0.9,
) -> bool:
    """
    Detect the *specific* cuML/intelex KMeans degeneracy: a clustering where a
    near-empty cluster (a handful of outliers) produces an artificially HIGH
    silhouette that would hijack best-k selection.

    Why both conditions are required
    --------------------------------
    A tiny cluster ALONE is not pathological: at k >= 10 even stock sklearn
    produces sub-0.1% clusters here, because the rare attack classes are
    genuinely tiny (Heartbleed = 11 rows = 0.002% of 500K). Flagging on tiny
    cluster alone caused pointless CPU refits at every high k.

    The real poison is a tiny cluster *combined with* an implausibly high
    silhouette (e.g. cuML k=2 -> [149998, 2] with silhouette ~0.98). That
    fake score beats the true peak and corrupts the argmax. So we fire only
    when a cluster is < ``min_frac`` of the points AND the silhouette exceeds
    ``sil_threshold`` (0.9 is far above anything this dataset legitimately
    reaches, ~0.45-0.57).

    Returns True only when BOTH hold.
    """
    from sklearn.metrics import silhouette_score

    labels = np.asarray(labels)
    counts = np.bincount(labels)
    counts = counts[counts > 0]
    if len(counts) < 2:
        return False  # single cluster -> not this pathology
    tiny = (counts.min() / len(labels)) < min_frac
    if not tiny:
        return False  # no near-empty cluster -> healthy regardless of score
    X_arr = np.asarray(X)
    n = min(sample_size, len(X_arr))
    sil = silhouette_score(X_arr, labels, sample_size=n, random_state=seed)
    return bool(sil > sil_threshold)


# =============================================================================
# Subsampling to a target size (rare-class-aware)
# =============================================================================
def subsample_to_target(
    y: np.ndarray,
    target_rows: int | None = None,
    target_frac: float | None = None,
    rare_floor: int = 1_000,
    balanced: bool = False,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """
    Pick row indices so the kept set has ~``target_rows`` rows, while
    protecting rare classes. Returns a sorted array of integer indices into
    ``y`` (so callers do ``X.iloc[idx]``, ``y[idx]``).

    Strategy (why this and not plain proportional sampling)
    -------------------------------------------------------
    CIC-IDS-2017 is pathologically imbalanced: BENIGN is ~83% while three
    attack classes have 11-36 rows total. A *purely* proportional sample to,
    say, 134K rows would reduce Heartbleed (11) and SQL Injection (21) to
    ~1 row each - unsplittable for train/test and fatal for StratifiedKFold
    (which needs >= n_splits members per class). It would also spend ~83% of
    the budget re-copying BENIGN, i.e. keep the expensive majority and throw
    away the cheap, informative minority.

    So we use a *floored* proportional scheme:
      1. Classes with <= ``rare_floor`` rows are kept IN FULL (reserved).
      2. The remaining budget (target - reserved) is split among the large
         classes IN PROPORTION to their true sizes (preserves the realistic
         prior among the common classes).
      3. If a large class is smaller than its proportional share, it is taken
         in full and its leftover budget is redistributed to the others
         (iteratively), so the final total still hits the target.
      4. Largest-remainder rounding makes the per-class integer allocations
         sum exactly to the achievable target.

    Class-level imbalance during *training* is handled separately by
    ``class_weight='balanced'`` - this function only controls dataset SIZE.

    Parameters
    ----------
    y : array of encoded labels (ints).
    target_rows : desired total rows. Mutually exclusive with target_frac.
    target_frac : desired fraction of len(y) in (0, 1]. Used if target_rows
        is None.
    rare_floor : classes with count <= this are always kept fully.
    balanced : if True, the budget for the large classes is split EVENLY
        (equal cap per big class) instead of proportionally to their true
        sizes. Rare classes (<= rare_floor) are still kept in full. This
        produces a roughly class-balanced sample of ~target_rows rows, which
        is what Q3 clustering wants: a proportional sample is ~83% BENIGN, so
        the PCA scatter is one overlapping blob and K-Means' best silhouette
        degenerates to k=2 (normal-vs-anomalous). An even split exposes the
        per-attack structure (best k ~ #classes) and gives readable plots.
        Q2 classification leaves this False (the realistic prior is the point).
    random_state : seed for the per-class random choice (reproducible).

    Returns
    -------
    np.ndarray : sorted indices to keep. If neither target is given (or the
        target >= len(y)), returns all indices (no subsampling).
    """
    y = np.asarray(y)
    n_total = len(y)

    # Resolve the target -----------------------------------------------------
    if target_rows is None and target_frac is None:
        return np.arange(n_total)
    if target_rows is None:
        if not (0 < target_frac <= 1):
            raise ValueError("target_frac must be in (0, 1].")
        target_rows = int(round(target_frac * n_total))
    if target_rows >= n_total:
        return np.arange(n_total)
    if target_rows <= 0:
        raise ValueError("target_rows must be positive.")

    classes, counts = np.unique(y, return_counts=True)
    count_of = dict(zip(classes.tolist(), counts.tolist()))

    # 1) Reserve rare classes in full ---------------------------------------
    rare = [c for c in classes if count_of[c] <= rare_floor]
    big = [c for c in classes if count_of[c] > rare_floor]
    alloc = {c: count_of[c] for c in rare}
    reserved = sum(alloc.values())

    budget = target_rows - reserved
    if budget <= 0:
        # Even the rare classes alone meet/exceed the target. Keep rares
        # fully (we never drop rares) and take nothing extra from big classes.
        alloc.update({c: 0 for c in big})
    elif balanced:
        # EVEN split: every big class gets the same cap (~budget/n_big). If a
        # class is smaller than the cap it is taken in full and its leftover is
        # redistributed to the others, so the total still hits the target.
        remaining_big = list(big)
        big_budget = budget
        final_big = {}
        while remaining_big:
            cap = big_budget / len(remaining_big)
            small = [c for c in remaining_big if count_of[c] <= cap]
            if small:
                for c in small:
                    final_big[c] = count_of[c]
                    big_budget -= count_of[c]
                    remaining_big.remove(c)
                continue
            base = int(np.floor(cap))
            alloc_big = {c: base for c in remaining_big}
            short = big_budget - base * len(remaining_big)
            # hand out the rounding remainder deterministically (largest class first)
            for c in sorted(remaining_big, key=lambda c: count_of[c], reverse=True)[:short]:
                alloc_big[c] += 1
            final_big.update(alloc_big)
            break
        alloc.update(final_big)
    else:
        # 2-3) Proportional split among big classes, with overflow
        #      redistribution when a class is smaller than its share.
        remaining_big = list(big)
        big_budget = budget
        final_big = {}
        while remaining_big:
            pool = sum(count_of[c] for c in remaining_big)
            # ideal real-valued share for each remaining class
            ideal = {c: big_budget * count_of[c] / pool for c in remaining_big}
            # any class whose ideal share exceeds its size -> cap it, recurse
            capped = [c for c in remaining_big if ideal[c] >= count_of[c]]
            if capped:
                for c in capped:
                    final_big[c] = count_of[c]
                    big_budget -= count_of[c]
                    remaining_big.remove(c)
                continue
            # no more capping needed: largest-remainder rounding on `ideal`
            floors = {c: int(np.floor(ideal[c])) for c in remaining_big}
            short = big_budget - sum(floors.values())
            remainders = sorted(
                remaining_big, key=lambda c: ideal[c] - floors[c], reverse=True
            )
            for c in remainders[:short]:
                floors[c] += 1
            final_big.update(floors)
            break
        alloc.update(final_big)

    # 4) Draw the actual indices --------------------------------------------
    rng = np.random.default_rng(random_state)
    keep = []
    for c in classes:
        idx = np.where(y == c)[0]
        k = alloc.get(c, 0)
        if k < len(idx):
            idx = rng.choice(idx, size=k, replace=False)
        keep.append(idx)
    return np.sort(np.concatenate(keep)) if keep else np.array([], dtype=int)


def save_figure(fig, name: str, dpi: int = 120) -> Path:
    """Save a matplotlib figure under outputs/figures/ with consistent settings."""
    out = FIGURES_DIR / name
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    return out


def save_results(df: pd.DataFrame, name: str) -> Path:
    """Save a results DataFrame under outputs/results/."""
    out = RESULTS_DIR / name
    df.to_csv(out, index=False)
    return out


# =============================================================================
# Feature-analysis helpers used by Q1 feature selection
# =============================================================================
def top_correlated_pairs(corr: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Given a square correlation matrix, return the top_n pairs of distinct
    features ranked by |r|, as a DataFrame with columns (feature_a,
    feature_b, corr, abs_corr).

    We take only the upper triangle to avoid (a,b)+(b,a) duplicates and to
    skip the diagonal (r=1.0 self-correlation).
    """
    m = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = (
        m.stack()
         .rename("corr")
         .reset_index()
         .rename(columns={"level_0": "feature_a", "level_1": "feature_b"})
    )
    pairs["abs_corr"] = pairs["corr"].abs()
    return pairs.sort_values("abs_corr", ascending=False).head(top_n).reset_index(drop=True)


def find_low_variance_features(
    df: pd.DataFrame, threshold: float = 0.0
) -> list[str]:
    """
    Return numeric columns whose variance is <= threshold.

    A zero-variance column is constant — it carries no information and
    must be removed before any model. A near-zero-variance column (tiny
    threshold, e.g. 1e-6) is almost-constant; useful to catch the
    bulk-rate columns in CIC-IDS-2017 which are near-zero for >99% of
    rows.
    """
    numeric = df.select_dtypes(include=np.number)
    variances = numeric.var(numeric_only=True)
    return variances[variances <= threshold].index.tolist()


def find_highly_correlated_features(
    corr: pd.DataFrame, threshold: float = 0.95
) -> list[str]:
    """
    Given a correlation matrix, return a set of features to DROP so that
    no remaining pair has |r| > threshold.

    Strategy: walk the upper triangle; for every pair above threshold,
    mark the second column (by order in the matrix) for removal. This
    keeps the first occurrence of each highly-correlated cluster and
    drops its near-duplicates.
    """
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    to_drop: set[str] = set()
    for col in upper.columns:
        partners = upper.index[upper[col].abs() > threshold].tolist()
        for p in partners:
            # keep p (the earlier column in the matrix), drop col
            if p not in to_drop:
                to_drop.add(col)
                break
    return sorted(to_drop)


# =============================================================================
# Q2 classification helpers — used by notebooks/q2_classification.ipynb.
# Imports are lazy so importing utils stays cheap when only Q1 helpers are
# needed.
# =============================================================================
def evaluate_classifier(y_true, y_pred) -> dict:
    """
    Standard multi-class classification metrics: accuracy, weighted
    precision/recall, F1-weighted, F1-macro.

    Why both F1-weighted and F1-macro:
      - weighted reflects how the model handles the *bulk* of traffic
        (favors majority classes).
      - macro gives every class equal voice, exposing whether rare attack
        types are being missed.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
    )
    return {
        "accuracy":    accuracy_score(y_true, y_pred),
        "precision_w": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_w":    recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro":    f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def manual_grid_search(
    make_estimator: Callable,
    grid: dict,
    X_tr, y_tr, X_va, y_va,
    scoring: str = "f1_weighted",
) -> tuple[dict, float, pd.DataFrame]:
    """
    Scenario-A grid search: for every combination in `grid`, fit a fresh
    estimator on (X_tr, y_tr) and score on (X_va, y_va).

    Uses ``estimator.set_params(**combo)`` rather than passing the combo
    into the factory — that way grids whose keys carry Pipeline prefixes
    (e.g. ``algo__C`` for ``Pipeline([..., ('algo', LR())])``) work the
    same as plain-estimator grids (e.g. ``max_depth``).

    Returns (best_params, best_score, full_results_df).
    """
    from sklearn.metrics import f1_score
    keys = list(grid.keys())
    rows: list[dict] = []
    best_score = -np.inf
    best_params: dict | None = None
    for combo in product(*[grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        est = make_estimator()
        est.set_params(**params)
        est.fit(X_tr, y_tr)
        if scoring == "f1_weighted":
            s = f1_score(y_va, est.predict(X_va), average="weighted", zero_division=0)
        elif scoring == "f1_macro":
            s = f1_score(y_va, est.predict(X_va), average="macro", zero_division=0)
        else:
            raise ValueError(f"unsupported scoring: {scoring}")
        rows.append({**params, f"val_{scoring}": s})
        if s > best_score:
            best_score, best_params = s, params
    results = (
        pd.DataFrame(rows)
        .sort_values(f"val_{scoring}", ascending=False)
        .reset_index(drop=True)
    )
    return best_params, best_score, results


def plot_confusion(
    y_true, y_pred, labels: list[str], title: str, fname: str,
    normalize: bool = True,
):
    """
    Row-normalized confusion matrix saved under outputs/figures/.

    Row-normalization is essential here: without it, the BENIGN row dwarfs
    every rare-class row in absolute counts and we cannot see whether,
    say, Heartbleed is being detected.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    n_classes = len(labels)
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    if normalize:
        with np.errstate(invalid="ignore"):
            cm_show = cm / cm.sum(axis=1, keepdims=True)
            cm_show = np.nan_to_num(cm_show)
        fmt, vmax, cbar_label = ".2f", 1.0, "fraction of true row"
    else:
        cm_show, fmt, vmax, cbar_label = cm, "d", None, "count"

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cm_show, annot=True, fmt=fmt, cmap="Blues", vmin=0, vmax=vmax,
        xticklabels=labels, yticklabels=labels,
        cbar_kws={"label": cbar_label}, ax=ax, annot_kws={"size": 7},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_figure(fig, fname)
    plt.show()
    plt.close(fig)
    return cm


def plot_feature_importance(
    model, feature_names, top_k: int, title: str, fname: str,
) -> pd.Series:
    """
    Horizontal bar chart of the top-k ``model.feature_importances_``
    values. Returns the (sorted-ascending) Series so the caller can also
    persist it as CSV.
    """
    import matplotlib.pyplot as plt

    imp = (
        pd.Series(model.feature_importances_, index=feature_names)
        .sort_values(ascending=True)
        .tail(top_k)
    )
    fig, ax = plt.subplots(figsize=(8, max(4, top_k * 0.3)))
    imp.plot(kind="barh", color="steelblue", ax=ax)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    plt.tight_layout()
    save_figure(fig, fname)
    plt.show()
    plt.close(fig)
    return imp


def plot_lr_coefficients(
    lr_model, feature_names, class_names, top_k: int,
    title: str, fname: str,
) -> pd.DataFrame:
    """
    Heatmap of Logistic Regression coefficients (classes × top-k features
    by mean |coef| across classes).

    For multinomial LR the coefficient matrix has shape
    ``(n_classes, n_features)`` — each row tells the model how strongly
    each feature pushes a sample *toward* that class (positive) or *away*
    from it (negative). We pick the top-k features by mean absolute
    coefficient so the visualization highlights the most influential
    features.

    For Pipeline-wrapped models, pass ``pipeline.named_steps['algo']``.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    coef = pd.DataFrame(
        lr_model.coef_,
        index=list(class_names),
        columns=list(feature_names),
    )
    top_features = (
        coef.abs().mean(axis=0).sort_values(ascending=False).head(top_k).index
    )
    coef_top = coef[top_features]

    fig, ax = plt.subplots(figsize=(max(8, top_k * 0.55),
                                    max(5, len(class_names) * 0.4)))
    vmax = float(coef_top.abs().values.max())
    sns.heatmap(
        coef_top, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        vmin=-vmax, vmax=vmax,
        cbar_kws={"label": "coefficient (sign = direction, |·| = strength)"},
        ax=ax, annot_kws={"size": 7},
    )
    ax.set_xlabel("Feature")
    ax.set_ylabel("Class")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_figure(fig, fname)
    plt.show()
    plt.close(fig)
    return coef_top


def plot_decision_tree(
    tree_model, feature_names, class_names,
    title: str, fname: str,
    max_depth_display: int = 3,
    figsize: tuple = (22, 11),
):
    """
    Render a fitted ``DecisionTreeClassifier`` (or one tree from a
    ``RandomForestClassifier``) using ``sklearn.tree.plot_tree``.

    For trees deeper than ~4 levels the full graph is unreadable on a
    PDF page; we limit the *display* to ``max_depth_display`` levels via
    sklearn's built-in ``max_depth`` parameter on ``plot_tree``. The
    underlying model is unaffected — this only crops what gets drawn.

    Each box shows: split rule (e.g. ``Destination Port <= 53.5``), the
    impurity (gini/entropy), the per-class sample weights, and the
    majority predicted class (color-coded).
    """
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree

    fig, ax = plt.subplots(figsize=figsize)
    plot_tree(
        tree_model,
        feature_names=list(feature_names),
        class_names=list(class_names),
        max_depth=max_depth_display,
        filled=True,
        rounded=True,
        impurity=True,
        proportion=False,
        fontsize=8,
        ax=ax,
    )
    ax.set_title(
        f"{title}\n(visualizing top {max_depth_display} levels — full tree may be deeper)"
    )
    plt.tight_layout()
    save_figure(fig, fname)
    plt.show()
    plt.close(fig)


def plot_one_forest_tree(
    rf_model, tree_index: int, feature_names, class_names,
    title: str, fname: str,
    max_depth_display: int = 3,
):
    """
    Plot a single tree from a fitted ``RandomForestClassifier``. Useful
    to *show* what one of the forest's voters looks like — keep in mind
    each tree was trained on its own bootstrap sample with random
    feature subsets at each split, so trees disagree.
    """
    if not hasattr(rf_model, "estimators_"):
        raise AttributeError("rf_model has no estimators_ — was it fitted?")
    if not (0 <= tree_index < len(rf_model.estimators_)):
        raise IndexError(
            f"tree_index {tree_index} out of range [0, {len(rf_model.estimators_)})"
        )
    plot_decision_tree(
        rf_model.estimators_[tree_index],
        feature_names=feature_names,
        class_names=class_names,
        title=f"{title} — tree #{tree_index} of {len(rf_model.estimators_)}",
        fname=fname,
        max_depth_display=max_depth_display,
    )


# =============================================================================
# Q3 clustering helpers — used by notebooks/q3_clustering.ipynb.
# =============================================================================
def evaluate_clustering(
    X, labels_pred, sample_size: int = 20_000,
    seed: int = 42,
) -> dict:
    """
    Silhouette score (sampled for speed) + Davies-Bouldin index for a
    clustering result.

    Why silhouette is sampled but Davies-Bouldin is not: silhouette is
    O(n^2) (it needs all pairwise distances), so at hundreds of thousands /
    millions of rows the full computation is infeasible on ANY hardware -
    the sample is a statistical estimate (within ~+/-0.01 at 20K), not a
    hardware shortcut. Davies-Bouldin is O(n*k*d) (linear in n), so it runs
    on the FULL data here with no cap, even at multi-million rows.

    Why sampled silhouette: the standard silhouette is
    :math:`O(n^2)` in memory and pairwise distances. On 134K rows that
    is ~18B comparisons; a 10K stratified sample gets us within ±0.01
    in seconds instead of minutes.

    DBSCAN's noise label (``-1``) is excluded from both metrics — those
    points aren't assigned to any cluster, so including them would
    force sklearn to treat "noise" as its own cluster and distort the
    scores.

    Returns dict with: ``n_clusters``, ``n_noise``, ``silhouette``,
    ``davies_bouldin``. Returns ``nan`` metrics when there is only one
    cluster (or all points are noise), where the metrics are
    mathematically undefined.
    """
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    labels = np.asarray(labels_pred)
    X_arr = np.asarray(X)

    noise_mask = labels == -1
    valid_mask = ~noise_mask

    n_noise = int(noise_mask.sum())
    valid_labels = labels[valid_mask]
    unique_clusters = np.unique(valid_labels)
    n_clusters = int(len(unique_clusters))

    if n_clusters < 2:
        return {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "silhouette": float("nan"),
            "davies_bouldin": float("nan"),
        }

    X_valid = X_arr[valid_mask]
    sample_size_eff = min(sample_size, len(X_valid))
    sil = silhouette_score(
        X_valid, valid_labels,
        sample_size=sample_size_eff, random_state=seed,
    )
    db = davies_bouldin_score(X_valid, valid_labels)
    return {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "silhouette": float(sil),
        "davies_bouldin": float(db),
    }


def plot_elbow_silhouette(
    ks, inertias, silhouettes, title: str, fname: str,
):
    """
    Dual-axis plot: K-Means inertia (elbow method) on the left axis,
    mean silhouette score on the right axis, both over a range of k.

    The elbow method looks for the point where the inertia curve
    "bends" — beyond which adding more clusters gives diminishing
    returns. Silhouette peaks at the k that gives best-separated,
    most-cohesive clusters. The two usually point to similar k.
    """
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(9, 5))
    color_inertia = "tab:blue"
    color_sil = "tab:red"

    ax1.plot(ks, inertias, marker="o", color=color_inertia, label="Inertia")
    ax1.set_xlabel("Number of clusters k")
    ax1.set_ylabel("Inertia (sum of squared distances)", color=color_inertia)
    ax1.tick_params(axis="y", labelcolor=color_inertia)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(ks, silhouettes, marker="s", linestyle="--",
             color=color_sil, label="Silhouette")
    ax2.set_ylabel("Silhouette score (higher = better)", color=color_sil)
    ax2.tick_params(axis="y", labelcolor=color_sil)

    ax1.set_title(title)
    ax1.set_xticks(list(ks))
    fig.tight_layout()
    save_figure(fig, fname)
    plt.show()
    plt.close(fig)


def plot_k_distance(X, k: int, title: str, fname: str,
                    sample_size: int | None = 20_000,
                    seed: int = 42) -> np.ndarray:
    """
    k-distance graph for DBSCAN ε selection: for each point, find the
    distance to its k-th nearest neighbor, sort ascending, and plot.
    The "knee" of the resulting curve is a reasonable choice for ε —
    below the knee are dense regions, above are outliers.

    For large X we sample (``sample_size`` rows) because the full
    k-NN query is :math:`O(n \\log n)` but still slow on 134K+ rows;
    the shape of the curve is stable under sampling.

    Returns the sorted k-distances (the y-values of the plot).
    """
    import matplotlib.pyplot as plt
    from sklearn.neighbors import NearestNeighbors

    X_arr = np.asarray(X)
    if sample_size is not None and len(X_arr) > sample_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_arr), size=sample_size, replace=False)
        X_sample = X_arr[idx]
    else:
        X_sample = X_arr

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X_sample)
    distances, _ = nbrs.kneighbors(X_sample)
    # Column k (0-indexed) is the distance to the k-th nearest neighbor
    # (excluding the point itself which is column 0).
    kth = np.sort(distances[:, k])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(kth, linewidth=1.5, color="steelblue")
    ax.set_xlabel("Points sorted by increasing k-distance")
    ax.set_ylabel(f"Distance to {k}-th nearest neighbor")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, fname)
    plt.show()
    plt.close(fig)
    return kth


def plot_pca_scatter_comparison(
    X_pca_2d, cluster_labels, true_labels, class_names,
    title: str, fname: str,
    cluster_palette: str = "tab20",
    plot_sample_size: int | None = 50_000,
    seed: int = 42,
):
    """
    Side-by-side 2-D PCA scatter: left colored by cluster label, right
    colored by true class label. Same point positions in both panels —
    visual check of whether the unsupervised clusters align with the
    supervised classes.

    DBSCAN noise points (label ``-1``) are drawn as grey x-markers.

    For large datasets we plot only a uniform random sample of
    ``plot_sample_size`` points (default 50K). This is a *rendering* limit,
    not an analytical one: a 2-D scatter saturates visually at a few tens of
    thousands of points (everything overlaps into a blob) and a million-point
    PNG is huge and slow. All metrics (silhouette, Davies-Bouldin, the
    cluster-to-label heatmap) are computed elsewhere on the FULL data, so the
    plot sample changes nothing that gets reported. Set to ``None`` to plot
    every point.
    """
    import matplotlib.pyplot as plt

    cluster_labels = np.asarray(cluster_labels)
    true_labels = np.asarray(true_labels)
    X_pca_2d = np.asarray(X_pca_2d)

    # Rendering subsample (uniform, reproducible). Metrics are unaffected -
    # they are computed on the full arrays before this function is called.
    if plot_sample_size is not None and len(X_pca_2d) > plot_sample_size:
        rng = np.random.default_rng(seed)
        sub = rng.choice(len(X_pca_2d), size=plot_sample_size, replace=False)
        X_pca_2d = X_pca_2d[sub]
        cluster_labels = cluster_labels[sub]
        true_labels = true_labels[sub]

    fig, (ax_c, ax_t) = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    # Left: colored by cluster
    noise_mask = cluster_labels == -1
    if noise_mask.any():
        ax_c.scatter(
            X_pca_2d[noise_mask, 0], X_pca_2d[noise_mask, 1],
            c="lightgrey", marker="x", s=10, alpha=0.5, label="noise",
        )
    valid_clusters = sorted(set(cluster_labels[~noise_mask].tolist()))
    cmap_c = plt.get_cmap(cluster_palette)
    for i, cl in enumerate(valid_clusters):
        mask = cluster_labels == cl
        ax_c.scatter(
            X_pca_2d[mask, 0], X_pca_2d[mask, 1],
            s=6, alpha=0.6, color=cmap_c(i % cmap_c.N), label=f"c{cl}",
        )
    ax_c.set_title("Colored by cluster")
    ax_c.set_xlabel("PC 1")
    ax_c.set_ylabel("PC 2")
    if len(valid_clusters) <= 15:
        ax_c.legend(fontsize=7, markerscale=2, loc="best")

    # Right: colored by true class
    cmap_t = plt.get_cmap("tab20")
    for i, name in enumerate(class_names):
        mask = true_labels == i
        if mask.any():
            ax_t.scatter(
                X_pca_2d[mask, 0], X_pca_2d[mask, 1],
                s=6, alpha=0.6, color=cmap_t(i % cmap_t.N), label=str(name),
            )
    ax_t.set_title("Colored by true Label")
    ax_t.set_xlabel("PC 1")
    ax_t.legend(fontsize=6, markerscale=2, loc="center left",
                bbox_to_anchor=(1.02, 0.5))

    fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, fname)
    plt.show()
    plt.close(fig)


def plot_cluster_label_heatmap(
    cluster_labels, true_labels, class_names,
    title: str, fname: str,
    normalize: str = "cluster",
) -> pd.DataFrame:
    """
    Heatmap: rows = clusters, columns = true classes, values = counts
    normalized per row (if ``normalize="cluster"``) or per column
    (``normalize="class"``) or raw counts (``normalize="none"``).

    Row-normalized (default): "what is this cluster mostly made of?"
    Column-normalized: "where did this true class end up?"

    Returns the displayed DataFrame so the caller can save it as CSV.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    cluster_labels = np.asarray(cluster_labels)
    true_labels = np.asarray(true_labels)

    cluster_ids = sorted(set(cluster_labels.tolist()))
    ct = pd.DataFrame(
        0,
        index=[f"noise" if c == -1 else f"c{c}" for c in cluster_ids],
        columns=list(class_names),
        dtype=float,
    )
    for ci, cid in enumerate(cluster_ids):
        mask = cluster_labels == cid
        if not mask.any():
            continue
        counts = pd.Series(true_labels[mask]).value_counts()
        for class_idx, cnt in counts.items():
            if 0 <= class_idx < len(class_names):
                ct.iloc[ci, class_idx] = cnt

    if normalize == "cluster":
        row_sums = ct.sum(axis=1).replace(0, np.nan)
        ct_show = ct.div(row_sums, axis=0).fillna(0.0)
        fmt, vmax, cbar_label = ".2f", 1.0, "fraction of cluster"
    elif normalize == "class":
        col_sums = ct.sum(axis=0).replace(0, np.nan)
        ct_show = ct.div(col_sums, axis=1).fillna(0.0)
        fmt, vmax, cbar_label = ".2f", 1.0, "fraction of class"
    else:
        ct_show = ct
        fmt, vmax, cbar_label = ".0f", None, "count"

    fig, ax = plt.subplots(figsize=(
        max(9, len(class_names) * 0.5),
        max(5, len(cluster_ids) * 0.35),
    ))
    sns.heatmap(
        ct_show, annot=True, fmt=fmt, cmap="Blues",
        vmin=0, vmax=vmax, cbar_kws={"label": cbar_label},
        ax=ax, annot_kws={"size": 7},
    )
    ax.set_xlabel("True class")
    ax.set_ylabel("Cluster")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_figure(fig, fname)
    plt.show()
    plt.close(fig)
    return ct_show


def plot_dendrogram(
    linkage_matrix, title: str, fname: str,
    truncate_mode: str = "lastp", p: int = 30,
):
    """
    Plot a hierarchical-clustering dendrogram from a precomputed
    ``scipy.cluster.hierarchy.linkage`` matrix. Truncation keeps the
    figure readable — showing every merge on >1000 samples is noise.
    """
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(
        linkage_matrix,
        truncate_mode=truncate_mode,
        p=p,
        leaf_rotation=90,
        leaf_font_size=8,
        show_contracted=True,
        ax=ax,
    )
    ax.set_xlabel("Sample index or (cluster size)")
    ax.set_ylabel("Linkage distance")
    ax.set_title(title)
    plt.tight_layout()
    save_figure(fig, fname)
    plt.show()
    plt.close(fig)
