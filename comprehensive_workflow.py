# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 15:25:16 2026

@author: rjing
"""

from __future__ import annotations

import json
import logging
import pickle
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import norm, rankdata, spearmanr
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


DATA_DIR = Path.cwd()

EQUITY_FILE = DATA_DIR / "sp500_top126_close_usd_aligned_final.csv"
CRYPTO_FILE = DATA_DIR / "crypto_top126_close_usd_aligned_final.csv"

OUTPUT_DIR = DATA_DIR / "complete_updated_analysis_2020_2025"

QUICK_MODE = False

RUN_PRIMARY_INFERENCE = True
RUN_TIER1_ROBUSTNESS = True
RUN_TIER2_ROBUSTNESS = True
RUN_SEED_STABILITY = True

SAVE_INTERMEDIATE_OUTPUTS = True
SAVE_SUMMARY_WORKBOOK = True
SAVE_LATEX_TABLES = True
SHOW_FIGURES = False

@dataclass(frozen=True)
class AnalysisConfig:
    years: Tuple[int, ...] = tuple(range(2020, 2026))
    expected_input_assets_per_market: int = 126
    expected_price_sessions: int = 1509

    shock_alpha: float = 0.05
    max_lag: int = 5
    eta_grid: Tuple[float, ...] = (0.1, 1.0, 10.0)

    train0: int = 147
    validation_block: int = 21
    n_folds: int = 4
    maxiter_cv: int = 200
    maxiter_final: int = 400

    kmeans_n_init: int = 50
    kmeans_seed: int = 42

    permutations: int = 5000
    bootstrap_repetitions: int = 5000
    random_seed: int = 20240601
    seed_stability_repetitions: int = 20

    lag_weight_epsilon: float = 1e-6
    graph_distance_epsilon: float = 1e-8

    primary_structural_predictors: Tuple[str, ...] = (
        "WithinMean",
        "BetweenMean",
        "CommBetweenness",
    )
    strength_predictors: Tuple[str, ...] = (
        "WithinMean",
        "BetweenMean",
    )
    descriptive_predictors: Tuple[str, ...] = ("BridgeRatio",)
    propagation_outcomes: Tuple[str, ...] = (
        "WithinOutgoing",
        "CrossOutgoing",
    )


    run_primary_inference: bool = RUN_PRIMARY_INFERENCE
    run_tier1: bool = RUN_TIER1_ROBUSTNESS
    run_tier2: bool = RUN_TIER2_ROBUSTNESS
    run_seed_stability: bool = RUN_SEED_STABILITY
    save_intermediate: bool = SAVE_INTERMEDIATE_OUTPUTS
    save_summary_workbook: bool = SAVE_SUMMARY_WORKBOOK
    save_latex_tables: bool = SAVE_LATEX_TABLES
    show_figures: bool = SHOW_FIGURES


CONFIG = AnalysisConfig(
    permutations=500 if QUICK_MODE else 5000,
    bootstrap_repetitions=500 if QUICK_MODE else 5000,
    seed_stability_repetitions=5 if QUICK_MODE else 20,
    maxiter_cv=80 if QUICK_MODE else 200,
    maxiter_final=160 if QUICK_MODE else 400,
)


BASELINE_DIR = OUTPUT_DIR / "baseline"
BASELINE_WINDOW_DIR = BASELINE_DIR / "annual_windows"
BASELINE_MATRIX_DIR = BASELINE_DIR / "A_matrices"
BASELINE_MEMBERSHIP_DIR = BASELINE_DIR / "memberships"
BASELINE_FIGURE_DIR = BASELINE_DIR / "figures"

ROBUSTNESS_DIR = OUTPUT_DIR / "robustness"
TIER1_DIR = ROBUSTNESS_DIR / "tier1_propagation"
TIER2_DIR = ROBUSTNESS_DIR / "tier2_dependence"
SEED_DIR = ROBUSTNESS_DIR / "seed_stability"
ROBUSTNESS_FIGURE_DIR = ROBUSTNESS_DIR / "figures"

SUMMARY_DIR = OUTPUT_DIR / "summary"


def prepare_output_directories() -> None:
    directories = [
        OUTPUT_DIR,
        BASELINE_DIR,
        BASELINE_WINDOW_DIR,
        BASELINE_MATRIX_DIR,
        BASELINE_MEMBERSHIP_DIR,
        BASELINE_FIGURE_DIR,
        ROBUSTNESS_DIR,
        TIER1_DIR,
        TIER2_DIR,
        SEED_DIR,
        ROBUSTNESS_FIGURE_DIR,
        SUMMARY_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


prepare_output_directories()

LOG_FILE = OUTPUT_DIR / "analysis_run.log"

logger = logging.getLogger("structure_propagation")
logger.setLevel(logging.INFO)
logger.handlers.clear()

log_formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def market_of(label: str) -> str:
    return str(label).split("_")[0]


def effective_folds(
    n_obs: int,
    train0: int,
    validation_block: int,
    n_folds: int,
) -> int:
    return sum(
        1
        for fold in range(n_folds)
        if train0 + (fold + 1) * validation_block <= n_obs
    )


def resolve_input_path(preferred: Path, fallback_pattern: str) -> Path:
    if preferred.exists():
        return preferred

    matches = sorted(preferred.parent.glob(fallback_pattern))

    if len(matches) == 1:
        logger.warning(
            "Preferred input %s was not found. Using %s.",
            preferred.name,
            matches[0].name,
        )
        return matches[0]

    if len(matches) > 1:
        raise FileNotFoundError(
            f"{preferred} was not found and the fallback pattern "
            f"{fallback_pattern!r} matched multiple files:\n"
            + "\n".join(str(path) for path in matches)
        )

    raise FileNotFoundError(f"Input file not found: {preferred}")


def read_price_matrix(
    path: Path,
    sheet_name: Optional[str] = None,
) -> pd.DataFrame:

    suffix = path.suffix.lower()

    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(
            path,
            sheet_name=sheet_name if sheet_name is not None else 0,
            engine="openpyxl",
        )
    else:
        raise ValueError(f"Unsupported price-file format: {suffix}")

    frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed:")]

    if "Date" not in frame.columns:
        frame = frame.rename(columns={frame.columns[0]: "Date"})

    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise").dt.normalize()
    frame = frame.set_index("Date").sort_index()
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).astype(float)
    frame.index.name = "Date"

    if frame.index.duplicated().any():
        duplicated = frame.index[frame.index.duplicated(keep=False)]
        raise ValueError(f"Duplicate dates in {path}: {duplicated.tolist()[:10]}")

    if frame.columns.duplicated().any():
        duplicated = frame.columns[frame.columns.duplicated(keep=False)]
        raise ValueError(f"Duplicate asset identifiers in {path}: {duplicated.tolist()}")

    return frame


def load_and_validate_aligned_prices(
    equity_path: Path,
    crypto_path: Path,
    config: AnalysisConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    equity = read_price_matrix(equity_path)
    crypto = read_price_matrix(crypto_path)

    if not equity.index.equals(crypto.index):
        only_equity = equity.index.difference(crypto.index)
        only_crypto = crypto.index.difference(equity.index)
        raise ValueError(
            "The two final matrices are not aligned. "
            f"Equity-only dates: {only_equity.tolist()[:10]}. "
            f"Crypto-only dates: {only_crypto.tolist()[:10]}."
        )

    if equity.shape[1] != config.expected_input_assets_per_market:
        raise ValueError(
            f"Expected {config.expected_input_assets_per_market} equity columns, "
            f"found {equity.shape[1]}."
        )

    if crypto.shape[1] != config.expected_input_assets_per_market:
        raise ValueError(
            f"Expected {config.expected_input_assets_per_market} crypto columns, "
            f"found {crypto.shape[1]}."
        )

    if len(equity) != config.expected_price_sessions:
        raise ValueError(
            f"Expected {config.expected_price_sessions} aligned price sessions, "
            f"found {len(equity)}."
        )

    if equity.isna().any().any() or crypto.isna().any().any():
        raise ValueError("The aligned input matrices contain missing prices.")

    if (equity <= 0).any().any() or (crypto <= 0).any().any():
        raise ValueError("All input prices must be strictly positive.")

    if (equity.index.dayofweek >= 5).any():
        raise ValueError("Weekend dates were found in the aligned equity calendar.")

    interval_controls = pd.DataFrame(index=equity.index)
    interval_controls.index.name = "Date"
    interval_controls["PreviousSession"] = equity.index.to_series().shift(1)
    interval_controls["CalendarGapDays"] = equity.index.to_series().diff().dt.days
    interval_controls["ExtendedInterval"] = (
        interval_controls["CalendarGapDays"] > 1
    ).astype(int)
    interval_controls["Monday"] = (equity.index.dayofweek == 0).astype(int)
    interval_controls["Year"] = equity.index.year

    return equity, crypto, interval_controls

#---------------------ANNUAL RETURNS, STANDARDISATION, AND SHOCK EVENTS------------------------------

def annual_return_window(
    equity_prices: pd.DataFrame,
    crypto_prices: pd.DataFrame,
    year: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:

    year_dates = equity_prices.index[equity_prices.index.year == int(year)]

    if len(year_dates) == 0:
        raise ValueError(f"No observations were found in {year}.")

    previous_dates = equity_prices.index[equity_prices.index < year_dates[0]]

    if len(previous_dates) == 0:
        raise ValueError(f"No preceding session is available before {year}.")

    previous_date = previous_dates[-1]
    price_dates = pd.DatetimeIndex([previous_date]).append(year_dates)

    equity_returns_all = np.log(equity_prices.loc[price_dates]).diff().iloc[1:]
    crypto_returns_all = np.log(crypto_prices.loc[price_dates]).diff().iloc[1:]

    finite_equity = np.isfinite(equity_returns_all.to_numpy()).all(axis=0)
    finite_crypto = np.isfinite(crypto_returns_all.to_numpy()).all(axis=0)

    variance_equity = equity_returns_all.var(axis=0, ddof=1).to_numpy()
    variance_crypto = crypto_returns_all.var(axis=0, ddof=1).to_numpy()

    keep_equity = finite_equity & np.isfinite(variance_equity) & (variance_equity > 0)
    keep_crypto = finite_crypto & np.isfinite(variance_crypto) & (variance_crypto > 0)

    equity_returns = equity_returns_all.loc[:, keep_equity].copy()
    crypto_returns = crypto_returns_all.loc[:, keep_crypto].copy()

    calendar_gaps = pd.Series(
        np.diff(price_dates.values).astype("timedelta64[D]").astype(int),
        index=year_dates,
        name="CalendarGapDays",
        dtype=int,
    )

    interval = pd.DataFrame(index=year_dates)
    interval.index.name = "Date"
    interval["PreviousSession"] = pd.DatetimeIndex(price_dates[:-1]).to_numpy()
    interval["CalendarGapDays"] = calendar_gaps
    interval["ExtendedInterval"] = (calendar_gaps > 1).astype(int)
    interval["Monday"] = (year_dates.dayofweek == 0).astype(int)
    interval["Year"] = int(year)

    diagnostics = {
        "window": int(year),
        "price_sessions": int(len(year_dates)),
        "return_days": int(len(equity_returns)),
        "previous_session": previous_date,
        "first_session": year_dates[0],
        "last_session": year_dates[-1],
        "eq_assets_before_variance_screen": int(equity_returns_all.shape[1]),
        "cr_assets_before_variance_screen": int(crypto_returns_all.shape[1]),
        "eq_assets": int(equity_returns.shape[1]),
        "cr_assets": int(crypto_returns.shape[1]),
        "dropped_zero_or_invalid_eq": ",".join(
            equity_returns_all.columns[~keep_equity].tolist()
        ),
        "dropped_zero_or_invalid_cr": ",".join(
            crypto_returns_all.columns[~keep_crypto].tolist()
        ),
        "regular_intervals": int((interval["ExtendedInterval"] == 0).sum()),
        "extended_intervals": int((interval["ExtendedInterval"] == 1).sum()),
        "monday_intervals": int(interval["Monday"].sum()),
        "non_monday_extended_intervals": int(
            (
                (interval["ExtendedInterval"] == 1)
                & (interval["Monday"] == 0)
            ).sum()
        ),
    }

    return equity_returns, crypto_returns, interval, diagnostics


def standardize_equity(returns: pd.DataFrame) -> pd.DataFrame:
    mean = returns.mean(axis=0)
    standard_deviation = returns.std(axis=0, ddof=1).replace(0, np.nan)
    standardized = (returns - mean) / standard_deviation

    if standardized.isna().any().any():
        raise ValueError("Equity standardisation produced missing values.")

    return standardized


def standardize_crypto(
    returns: pd.DataFrame,
    extended_interval: pd.Series,
) -> pd.DataFrame:

    indicator = extended_interval.reindex(returns.index)

    if indicator.isna().any():
        raise ValueError("The extended-interval indicator is incomplete.")

    standardized = pd.DataFrame(
        index=returns.index,
        columns=returns.columns,
        dtype=float,
    )

    for state in (0, 1):
        mask = indicator.eq(state)
        segment = returns.loc[mask]
        mean = segment.mean(axis=0)
        standard_deviation = segment.std(axis=0, ddof=1).replace(0, np.nan)
        segment_standardized = (segment - mean) / standard_deviation

        constant_columns = segment_standardized.columns[
            segment_standardized.isna().all(axis=0)
        ]

        if len(constant_columns) > 0:
            segment_standardized.loc[:, constant_columns] = 0.0

        standardized.loc[mask] = segment_standardized

    if standardized.isna().any().any():
        raise ValueError("Cryptocurrency standardisation produced missing values.")

    return standardized


def shock_events(
    standardized_returns: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    thresholds = standardized_returns.quantile(alpha)
    return standardized_returns.lt(thresholds).astype(int)


#----------------STRUCTURAL LAYER: MG3 FILTERING, SPECTRAL CLUSTERING, AND SUMMARIES----------------------

def mg3_filter(
    returns: pd.DataFrame,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
]:

    correlation = returns.corr().to_numpy(dtype=float)
    n_observations, n_assets = returns.shape

    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    q_ratio = n_observations / n_assets
    lambda_plus = (1.0 + np.sqrt(1.0 / q_ratio)) ** 2
    lambda_tilde = (1.0 - eigenvalues[0] / n_assets) * lambda_plus

    informative = eigenvalues > lambda_tilde
    informative[0] = False

    removed_component = np.zeros_like(correlation)

    for mode_index in np.where(~informative)[0]:
        removed_component += eigenvalues[mode_index] * np.outer(
            eigenvectors[:, mode_index],
            eigenvectors[:, mode_index],
        )

    residual = correlation - removed_component
    np.fill_diagonal(residual, 0.0)

    return (
        correlation,
        residual,
        eigenvalues,
        eigenvectors,
        informative,
        float(lambda_plus),
        float(lambda_tilde),
    )


def informative_mask_with_fallback(
    informative: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    used = np.asarray(informative, dtype=bool).copy()
    fallback_fired = int(used.sum()) < 2

    if fallback_fired:
        used[:] = False
        used[1:min(3, len(used))] = True

    if int(used.sum()) < 2:
        raise ValueError("At least two non-market modes are required for clustering.")

    return used, fallback_fired


def spectral_embedding(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    used_informative: np.ndarray,
) -> np.ndarray:
    mode_indices = np.where(used_informative)[0]

    return np.column_stack(
        [
            np.sqrt(max(float(eigenvalues[index]), 0.0))
            * eigenvectors[:, index]
            for index in mode_indices
        ]
    )


def detect_communities(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    informative: np.ndarray,
    asset_names: Sequence[str],
    market_prefix: str,
    config: AnalysisConfig,
) -> Tuple[pd.Series, np.ndarray, np.ndarray, bool]:
    used_informative, fallback_fired = informative_mask_with_fallback(informative)
    embedding = spectral_embedding(eigenvalues, eigenvectors, used_informative)
    n_clusters = int(used_informative.sum())

    model = KMeans(
        n_clusters=n_clusters,
        n_init=config.kmeans_n_init,
        random_state=config.kmeans_seed,
    )

    labels = model.fit_predict(embedding)

    named = pd.Series(
        [f"{market_prefix}_C{label}" for label in labels],
        index=list(asset_names),
        name="Community",
    )

    return named, embedding, used_informative, fallback_fired


def compute_community_event_matrices(
    equity_events: pd.DataFrame,
    crypto_events: pd.DataFrame,
    equity_communities: pd.Series,
    crypto_communities: pd.Series,
) -> Tuple[Dict[str, List[str]], pd.DataFrame, pd.DataFrame]:
    groups: Dict[str, List[str]] = {}

    for group in sorted(equity_communities.unique()):
        groups[group] = equity_communities[
            equity_communities == group
        ].index.tolist()

    for group in sorted(crypto_communities.unique()):
        groups[group] = crypto_communities[
            crypto_communities == group
        ].index.tolist()

    counts = pd.DataFrame(index=equity_events.index)
    proportions = pd.DataFrame(index=equity_events.index)

    for group, assets in groups.items():
        event_matrix = (
            equity_events[assets]
            if group.startswith("EQ")
            else crypto_events[assets]
        )

        counts[group] = event_matrix.sum(axis=1)
        proportions[group] = counts[group] / len(assets)

    counts.index.name = "Date"
    proportions.index.name = "Date"

    return groups, counts, proportions


def community_structural_summaries(
    structural_matrix: np.ndarray,
    named_communities: pd.Series,
) -> pd.DataFrame:
    asset_names = list(named_communities.index)
    index_map = {name: position for position, name in enumerate(asset_names)}

    absolute_matrix = np.abs(structural_matrix).copy()
    np.fill_diagonal(absolute_matrix, 0.0)

    rows: List[Dict[str, Any]] = []

    for group in sorted(named_communities.unique()):
        members = named_communities[named_communities == group].index.tolist()
        member_indices = [index_map[member] for member in members]
        other_indices = [
            index
            for index in range(len(asset_names))
            if index not in member_indices
        ]

        if len(member_indices) > 1:
            upper_indices = np.triu_indices(len(member_indices), 1)
            within_absolute = absolute_matrix[
                np.ix_(member_indices, member_indices)
            ][upper_indices]
            within_signed = structural_matrix[
                np.ix_(member_indices, member_indices)
            ][upper_indices]
        else:
            within_absolute = np.array([], dtype=float)
            within_signed = np.array([], dtype=float)

        within_mean = (
            float(within_absolute.mean())
            if len(within_absolute) > 0
            else np.nan
        )

        between_mean = (
            float(absolute_matrix[np.ix_(member_indices, other_indices)].mean())
            if len(other_indices) > 0
            else np.nan
        )

        strength = float(
            absolute_matrix[member_indices, :].sum(axis=1).mean()
        )

        bridge_ratio = (
            between_mean / (within_mean + between_mean)
            if pd.notna(within_mean)
            and pd.notna(between_mean)
            and (within_mean + between_mean) > 0
            else np.nan
        )

        signed_within_mean = (
            float(within_signed.mean())
            if len(within_signed) > 0
            else np.nan
        )

        rows.append(
            {
                "Community": group,
                "size": len(member_indices),
                "WithinMean": within_mean,
                "BetweenMean": between_mean,
                "Strength": strength,
                "BridgeRatio": bridge_ratio,
                "B_signed": signed_within_mean,
            }
        )

    return pd.DataFrame(rows).set_index("Community")


def community_betweenness(
    structural_matrix: np.ndarray,
    named_communities: pd.Series,
    epsilon: float,
) -> pd.Series:
    """Weighted betweenness on the full community-contraction network."""

    asset_names = list(named_communities.index)
    index_map = {name: position for position, name in enumerate(asset_names)}

    absolute_matrix = np.abs(structural_matrix).copy()
    np.fill_diagonal(absolute_matrix, 0.0)

    communities = sorted(named_communities.unique())
    member_indices = {
        group: [
            index_map[member]
            for member in named_communities[
                named_communities == group
            ].index
        ]
        for group in communities
    }

    graph = nx.Graph()
    graph.add_nodes_from(communities)

    for first in range(len(communities)):
        for second in range(first + 1, len(communities)):
            group_first = communities[first]
            group_second = communities[second]
            indices_first = member_indices[group_first]
            indices_second = member_indices[group_second]

            edge_weight = float(
                absolute_matrix[np.ix_(indices_first, indices_second)].mean()
            )

            graph.add_edge(
                group_first,
                group_second,
                weight=1.0 / (edge_weight + epsilon),
            )

    if len(communities) < 3:
        values = {group: 0.0 for group in communities}
    else:
        values = nx.betweenness_centrality(
            graph,
            weight="weight",
            normalized=True,
        )

    return pd.Series(values, name="CommBetweenness")


def community_contagion_summaries(
    excitation_matrix: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for source in excitation_matrix.index:
        same_market = [
            target
            for target in excitation_matrix.columns
            if market_of(target) == market_of(source)
        ]
        other_market = [
            target
            for target in excitation_matrix.columns
            if market_of(target) != market_of(source)
        ]

        rows.append(
            {
                "Community": source,
                "SelfExcitation": float(excitation_matrix.loc[source, source]),
                "WithinOutgoing": float(
                    excitation_matrix.loc[
                        source,
                        [target for target in same_market if target != source],
                    ].sum()
                ),
                "CrossOutgoing": float(
                    excitation_matrix.loc[source, other_market].sum()
                ),
                "TotalOutgoing": float(excitation_matrix.loc[source, :].sum()),
            }
        )

    return pd.DataFrame(rows).set_index("Community")


#-----------------COMMUNITY-LEVEL DISCRETE-TIME HAWKES-TYPE MODEL----------------------

def rolling_origin_folds(
    n_observations: int,
    train0: int,
    validation_block: int,
    n_folds: int,
) -> List[Tuple[slice, slice]]:
    folds: List[Tuple[slice, slice]] = []

    for fold in range(n_folds):
        train_end = train0 + fold * validation_block
        validation_end = train_end + validation_block

        if validation_end > n_observations:
            break

        folds.append(
            (
                slice(0, train_end),
                slice(train_end, validation_end),
            )
        )

    return folds


def build_group_lagged_design(
    group_proportions: pd.DataFrame,
    max_lag: int,
) -> Tuple[np.ndarray, List[Tuple[str, int]], pd.DatetimeIndex]:
    source_columns = list(group_proportions.columns)
    n_observations = len(group_proportions)
    blocks: List[np.ndarray] = []
    lag_columns: List[Tuple[str, int]] = []

    for lag in range(1, max_lag + 1):
        block = group_proportions.iloc[
            max_lag - lag:n_observations - lag
        ].to_numpy(dtype=float)
        blocks.append(block)
        lag_columns.extend((source, lag) for source in source_columns)

    design = np.hstack(blocks)
    trimmed_index = group_proportions.iloc[max_lag:].index

    return design, lag_columns, trimmed_index


def poisson_quasi_negative_log_likelihood(
    observed_counts: np.ndarray,
    expected_counts: np.ndarray,
) -> float:
    expected = np.clip(expected_counts, 1e-12, None)
    return float(np.sum(expected - observed_counts * np.log(expected)))


def fit_poisson_ridge_target(
    lagged_design: np.ndarray,
    target_counts: np.ndarray,
    is_extended: np.ndarray,
    target_size: int,
    eta: float,
    initial_parameters: Optional[np.ndarray] = None,
    max_iterations: int = 400,
):
    n_features = lagged_design.shape[1]

    if initial_parameters is None or len(initial_parameters) != n_features + 2:
        baseline = max(
            float(np.mean(target_counts)) / max(target_size, 1),
            1e-4,
        )
        initial_parameters = np.concatenate(
            [[baseline, baseline], np.zeros(n_features)]
        )

    bounds = [
        (1e-10, None),
        (1e-10, None),
    ] + [(0.0, None)] * n_features

    extended = np.asarray(is_extended, dtype=float)
    regular = 1.0 - extended

    def objective_and_gradient(parameters: np.ndarray) -> Tuple[float, np.ndarray]:
        mu_regular, mu_extended = parameters[:2]
        excitation_coefficients = parameters[2:]

        per_asset_intensity = (
            mu_regular * regular
            + mu_extended * extended
            + lagged_design.dot(excitation_coefficients)
        )

        safe_intensity = np.clip(per_asset_intensity, 1e-12, None)
        expected_counts = target_size * safe_intensity

        objective = poisson_quasi_negative_log_likelihood(
            target_counts,
            expected_counts,
        ) + eta * np.dot(excitation_coefficients, excitation_coefficients)

        gradient_intensity = target_size - target_counts / safe_intensity

        gradient = np.empty_like(parameters)
        gradient[0] = np.dot(gradient_intensity, regular)
        gradient[1] = np.dot(gradient_intensity, extended)
        gradient[2:] = (
            lagged_design.T.dot(gradient_intensity)
            + 2.0 * eta * excitation_coefficients
        )

        return objective, gradient

    return minimize(
        fun=lambda parameters: objective_and_gradient(parameters)[0],
        x0=initial_parameters,
        jac=lambda parameters: objective_and_gradient(parameters)[1],
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iterations},
    )


def select_ridge_penalty_via_cv(
    lagged_design: np.ndarray,
    target_counts: np.ndarray,
    is_extended: np.ndarray,
    target_size: int,
    eta_grid: Sequence[float],
    train0: int,
    validation_block: int,
    n_folds: int,
    max_iterations: int,
) -> Tuple[float, pd.DataFrame, np.ndarray]:
    folds = rolling_origin_folds(
        n_observations=len(target_counts),
        train0=train0,
        validation_block=validation_block,
        n_folds=n_folds,
    )

    if not folds:
        raise ValueError("No expanding-window validation folds are available.")

    records: List[Dict[str, float]] = []
    best_eta: Optional[float] = None
    best_score = np.inf
    best_warm_start: Optional[np.ndarray] = None

    for eta in eta_grid:
        validation_losses: List[float] = []
        warm_start: Optional[np.ndarray] = None

        for training_slice, validation_slice in folds:
            fit = fit_poisson_ridge_target(
                lagged_design=lagged_design[training_slice],
                target_counts=target_counts[training_slice],
                is_extended=is_extended[training_slice],
                target_size=target_size,
                eta=float(eta),
                initial_parameters=warm_start,
                max_iterations=max_iterations,
            )

            warm_start = fit.x.copy()
            mu_regular, mu_extended = warm_start[:2]
            excitation_coefficients = warm_start[2:]

            validation_intensity = (
                mu_regular * (1.0 - is_extended[validation_slice])
                + mu_extended * is_extended[validation_slice]
                + lagged_design[validation_slice].dot(excitation_coefficients)
            )

            expected_validation_counts = target_size * np.clip(
                validation_intensity,
                1e-12,
                None,
            )

            validation_losses.append(
                poisson_quasi_negative_log_likelihood(
                    target_counts[validation_slice],
                    expected_validation_counts,
                )
            )

        mean_validation_loss = float(np.mean(validation_losses))

        records.append(
            {
                "eta": float(eta),
                "mean_val_qnll": mean_validation_loss,
            }
        )

        if mean_validation_loss < best_score:
            best_score = mean_validation_loss
            best_eta = float(eta)
            best_warm_start = warm_start.copy() if warm_start is not None else None

    if best_eta is None or best_warm_start is None:
        raise RuntimeError("Ridge-penalty selection did not produce a valid model.")

    table = pd.DataFrame(records).sort_values(
        ["mean_val_qnll", "eta"]
    ).reset_index(drop=True)

    return best_eta, table, best_warm_start


def fit_community_layer(
    group_counts: pd.DataFrame,
    group_proportions: pd.DataFrame,
    groups: Mapping[str, List[str]],
    max_lag: int,
    eta_grid: Sequence[float],
    train0: int,
    validation_block: int,
    n_folds: int,
    maxiter_cv: int,
    maxiter_final: int,
    extended_interval: pd.Series,
    use_interval_baselines: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lagged_design, lag_columns, trimmed_index = build_group_lagged_design(
        group_proportions,
        max_lag,
    )

    trimmed_counts = group_counts.loc[trimmed_index]
    interval_indicator = (
        extended_interval.reindex(trimmed_index).astype(float).to_numpy()
    )

    if not use_interval_baselines:
        interval_indicator = np.zeros_like(interval_indicator)

    communities = list(group_counts.columns)

    integrated_excitation = pd.DataFrame(
        0.0,
        index=communities,
        columns=communities,
    )

    lag_one_excitation = pd.DataFrame(
        0.0,
        index=communities,
        columns=communities,
    )

    mean_lag = pd.DataFrame(
        np.nan,
        index=communities,
        columns=communities,
    )

    lag_index = pd.MultiIndex.from_tuples(
        lag_columns,
        names=["source", "lag"],
    )

    fit_rows: List[Dict[str, Any]] = []

    for target in communities:
        target_counts = trimmed_counts[target].to_numpy(dtype=float)

        best_eta, cv_table, warm_start = select_ridge_penalty_via_cv(
            lagged_design=lagged_design,
            target_counts=target_counts,
            is_extended=interval_indicator,
            target_size=len(groups[target]),
            eta_grid=eta_grid,
            train0=train0,
            validation_block=validation_block,
            n_folds=n_folds,
            max_iterations=maxiter_cv,
        )

        final_fit = fit_poisson_ridge_target(
            lagged_design=lagged_design,
            target_counts=target_counts,
            is_extended=interval_indicator,
            target_size=len(groups[target]),
            eta=best_eta,
            initial_parameters=warm_start,
            max_iterations=maxiter_final,
        )

        mu_regular, mu_extended = final_fit.x[:2]
        coefficients = pd.Series(final_fit.x[2:], index=lag_index)

        for source in communities:
            source_values = coefficients.loc[source].to_numpy(dtype=float)
            total_excitation = float(source_values.sum())

            integrated_excitation.loc[source, target] = total_excitation
            lag_one_excitation.loc[source, target] = float(source_values[0])

            if total_excitation > 1e-8:
                mean_lag.loc[source, target] = float(
                    np.dot(
                        np.arange(1, max_lag + 1),
                        source_values,
                    )
                    / total_excitation
                )

        fit_rows.append(
            {
                "target": target,
                "eta": best_eta,
                "cv_mean_val_qnll": float(cv_table.iloc[0]["mean_val_qnll"]),
                "mu_regular": float(mu_regular),
                "mu_extended": float(mu_extended),
                "extended_minus_regular": float(mu_extended - mu_regular),
                "interval_baselines_used": bool(use_interval_baselines),
                "success": bool(final_fit.success),
                "status": int(final_fit.status),
                "message": str(final_fit.message),
                "nonzero_source_lag_coeffs": int((coefficients > 1e-10).sum()),
            }
        )

    return (
        integrated_excitation,
        lag_one_excitation,
        mean_lag,
        pd.DataFrame(fit_rows),
    )

# -----------------BASELINE ANNUAL ESTIMATION------------------


def fit_baseline_window(
    year: int,
    equity_prices: pd.DataFrame,
    crypto_prices: pd.DataFrame,
    config: AnalysisConfig,
) -> Dict[str, Any]:
    equity_returns, crypto_returns, interval, data_diagnostics = annual_return_window(
        equity_prices,
        crypto_prices,
        year,
    )

    extended_interval = interval["ExtendedInterval"]

    standardized_equity = standardize_equity(equity_returns)
    standardized_crypto = standardize_crypto(
        crypto_returns,
        extended_interval,
    )

    equity_events = shock_events(standardized_equity, config.shock_alpha)
    crypto_events = shock_events(standardized_crypto, config.shock_alpha)

    (
        equity_correlation,
        equity_residual,
        equity_eigenvalues,
        equity_eigenvectors,
        equity_informative_raw,
        equity_lambda_plus,
        equity_lambda_tilde,
    ) = mg3_filter(equity_returns)

    (
        crypto_correlation,
        crypto_residual,
        crypto_eigenvalues,
        crypto_eigenvectors,
        crypto_informative_raw,
        crypto_lambda_plus,
        crypto_lambda_tilde,
    ) = mg3_filter(crypto_returns)

    (
        equity_communities,
        equity_embedding,
        equity_informative_used,
        equity_fallback_fired,
    ) = detect_communities(
        equity_eigenvalues,
        equity_eigenvectors,
        equity_informative_raw,
        equity_returns.columns.tolist(),
        "EQ",
        config,
    )

    (
        crypto_communities,
        crypto_embedding,
        crypto_informative_used,
        crypto_fallback_fired,
    ) = detect_communities(
        crypto_eigenvalues,
        crypto_eigenvectors,
        crypto_informative_raw,
        crypto_returns.columns.tolist(),
        "CR",
        config,
    )

    groups, group_counts, group_proportions = compute_community_event_matrices(
        equity_events,
        crypto_events,
        equity_communities,
        crypto_communities,
    )

    integrated_excitation, lag_one_excitation, mean_lag, fit_diagnostics = (
        fit_community_layer(
            group_counts=group_counts,
            group_proportions=group_proportions,
            groups=groups,
            max_lag=config.max_lag,
            eta_grid=config.eta_grid,
            train0=config.train0,
            validation_block=config.validation_block,
            n_folds=config.n_folds,
            maxiter_cv=config.maxiter_cv,
            maxiter_final=config.maxiter_final,
            extended_interval=extended_interval,
            use_interval_baselines=True,
        )
    )

    equity_structure = community_structural_summaries(
        equity_residual,
        equity_communities,
    )
    equity_structure["CommBetweenness"] = community_betweenness(
        equity_residual,
        equity_communities,
        config.graph_distance_epsilon,
    )

    crypto_structure = community_structural_summaries(
        crypto_residual,
        crypto_communities,
    )
    crypto_structure["CommBetweenness"] = community_betweenness(
        crypto_residual,
        crypto_communities,
        config.graph_distance_epsilon,
    )

    structural_summary = pd.concat([equity_structure, crypto_structure])
    community_summary = structural_summary.join(
        community_contagion_summaries(integrated_excitation)
    )
    community_summary["window"] = int(year)
    community_summary["market"] = [
        market_of(community)
        for community in community_summary.index
    ]
    community_summary = community_summary.reset_index()

    data_diagnostics.update(
        {
            "eq_communities": int(equity_communities.nunique()),
            "cr_communities": int(crypto_communities.nunique()),
            "eq_informative_modes_raw": int(equity_informative_raw.sum()),
            "cr_informative_modes_raw": int(crypto_informative_raw.sum()),
            "eq_informative_modes_used": int(equity_informative_used.sum()),
            "cr_informative_modes_used": int(crypto_informative_used.sum()),
            "eq_fallback_fired": bool(equity_fallback_fired),
            "cr_fallback_fired": bool(crypto_fallback_fired),
            "lambda_plus_eq": float(equity_lambda_plus),
            "lambda_plus_cr": float(crypto_lambda_plus),
            "lambda_tilde_eq": float(equity_lambda_tilde),
            "lambda_tilde_cr": float(crypto_lambda_tilde),
            "leading_eigenvalue_eq": float(equity_eigenvalues[0]),
            "leading_eigenvalue_cr": float(crypto_eigenvalues[0]),
            "market_mode_share_eq": float(
                equity_eigenvalues[0] / equity_returns.shape[1]
            ),
            "market_mode_share_cr": float(
                crypto_eigenvalues[0] / crypto_returns.shape[1]
            ),
            "n_obs_after_lag_trim": int(len(group_counts) - config.max_lag),
            "effective_cv_folds": int(
                effective_folds(
                    len(group_counts) - config.max_lag,
                    config.train0,
                    config.validation_block,
                    config.n_folds,
                )
            ),
            "fit_success_all": bool(fit_diagnostics["success"].all()),
            "mean_crypto_shock_rate": float(crypto_events.to_numpy().mean()),
            "mean_equity_shock_rate": float(equity_events.to_numpy().mean()),
        }
    )

    membership = pd.concat(
        [
            pd.DataFrame(
                {
                    "asset": equity_communities.index,
                    "community": equity_communities.values,
                    "market": "EQ",
                    "window": int(year),
                }
            ),
            pd.DataFrame(
                {
                    "asset": crypto_communities.index,
                    "community": crypto_communities.values,
                    "market": "CR",
                    "window": int(year),
                }
            ),
        ],
        ignore_index=True,
    )

    return {
        "year": int(year),
        "ret_eq": equity_returns,
        "ret_cr": crypto_returns,
        "interval": interval,
        "std_eq": standardized_equity,
        "std_cr": standardized_crypto,
        "E_eq": equity_events,
        "E_cr": crypto_events,
        "rho_eq": equity_correlation,
        "rho_cr": crypto_correlation,
        "B_eq": equity_residual,
        "B_cr": crypto_residual,
        "evals_eq": equity_eigenvalues,
        "evals_cr": crypto_eigenvalues,
        "evecs_eq": equity_eigenvectors,
        "evecs_cr": crypto_eigenvectors,
        "info_eq_raw": equity_informative_raw,
        "info_cr_raw": crypto_informative_raw,
        "info_eq_used": equity_informative_used,
        "info_cr_used": crypto_informative_used,
        "named_eq": equity_communities,
        "named_cr": crypto_communities,
        "Phi_eq": equity_embedding,
        "Phi_cr": crypto_embedding,
        "groups": groups,
        "Y": group_counts,
        "X": group_proportions,
        "A": integrated_excitation,
        "A1": lag_one_excitation,
        "MeanLag": mean_lag,
        "fit_info": fit_diagnostics.assign(window=int(year), spec="baseline"),
        "struct": structural_summary,
        "summary": community_summary,
        "membership": membership,
        "diag": data_diagnostics,
    }


def excitation_partition_single(
    excitation_matrix: pd.DataFrame,
    mean_lag_matrix: Optional[pd.DataFrame] = None,
    lag_epsilon: float = 1e-6,
) -> Dict[str, Any]:
    source_market = np.array([market_of(label) for label in excitation_matrix.index])
    target_market = np.array([market_of(label) for label in excitation_matrix.columns])
    values = excitation_matrix.to_numpy(dtype=float)

    self_mask = np.eye(len(excitation_matrix), dtype=bool)
    within_mask = (source_market[:, None] == target_market[None, :]) & ~self_mask
    cross_mask = source_market[:, None] != target_market[None, :]
    equity_to_crypto = (
        (source_market[:, None] == "EQ")
        & (target_market[None, :] == "CR")
    )
    crypto_to_equity = (
        (source_market[:, None] == "CR")
        & (target_market[None, :] == "EQ")
    )

    total = float(values.sum())

    row: Dict[str, Any] = {
        "Total": total,
        "SelfMass": float(values[self_mask].sum()),
        "WithinMass": float(values[within_mask].sum()),
        "CrossMass": float(values[cross_mask].sum()),
        "EQtoCR": float(values[equity_to_crypto].sum()),
        "CRtoEQ": float(values[crypto_to_equity].sum()),
    }

    if total > 0:
        row.update(
            {
                "SelfShare": row["SelfMass"] / total,
                "WithinShare": row["WithinMass"] / total,
                "CrossShare": row["CrossMass"] / total,
                "EQtoCR_share": row["EQtoCR"] / total,
                "CRtoEQ_share": row["CRtoEQ"] / total,
            }
        )
    else:
        row.update(
            {
                "SelfShare": np.nan,
                "WithinShare": np.nan,
                "CrossShare": np.nan,
                "EQtoCR_share": np.nan,
                "CRtoEQ_share": np.nan,
            }
        )

    if mean_lag_matrix is not None:
        retained = values > lag_epsilon
        row["MeanLag"] = (
            float(
                np.nansum(mean_lag_matrix.to_numpy()[retained] * values[retained])
                / values[retained].sum()
            )
            if retained.any()
            else np.nan
        )
        row["n_links_retained"] = int(retained.sum())

    return row


def excitation_partition_table(
    window_results: Mapping[int, Dict[str, Any]],
    config: AnalysisConfig,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for year in config.years:
        row = excitation_partition_single(
            window_results[year]["A"],
            window_results[year]["MeanLag"],
            config.lag_weight_epsilon,
        )
        row["window"] = int(year)
        rows.append(row)

    columns = ["window"] + [column for column in rows[0] if column != "window"]
    return pd.DataFrame(rows)[columns]


def community_structure_table(
    combined_summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for (year, market), subset in combined_summary.groupby(
        ["window", "market"],
        sort=True,
    ):
        sizes = subset["size"].astype(int).to_numpy()
        rows.append(
            {
                "window": int(year),
                "market": market,
                "N_assets": int(sizes.sum()),
                "K_communities": int(len(sizes)),
                "largest_share": float(sizes.max() / sizes.sum()),
                "median_size": float(np.median(sizes)),
                "minimum_size": int(sizes.min()),
                "maximum_size": int(sizes.max()),
            }
        )

    long_table = pd.DataFrame(rows)
    wide = long_table.pivot(index="window", columns="market")
    wide.columns = [
        f"{statistic}_{market}"
        for statistic, market in wide.columns
    ]
    wide = wide.reset_index().sort_values("window")

    diagnostic_index = diagnostics.set_index("window")
    wide["lambda_tilde_EQ"] = wide["window"].map(
        diagnostic_index["lambda_tilde_eq"]
    )
    wide["lambda_tilde_CR"] = wide["window"].map(
        diagnostic_index["lambda_tilde_cr"]
    )
    wide["market_mode_share_EQ"] = wide["window"].map(
        diagnostic_index["market_mode_share_eq"]
    )
    wide["market_mode_share_CR"] = wide["window"].map(
        diagnostic_index["market_mode_share_cr"]
    )
    wide["T_w"] = wide["window"].map(diagnostic_index["return_days"])

    return wide


def save_annual_intermediate_outputs(
    result: Dict[str, Any],
    output_directory: Path,
) -> None:
    year = int(result["year"])
    year_directory = output_directory / str(year)
    year_directory.mkdir(parents=True, exist_ok=True)

    result["ret_eq"].to_csv(year_directory / "equity_log_returns.csv")
    result["ret_cr"].to_csv(year_directory / "crypto_log_returns.csv")
    result["interval"].to_csv(year_directory / "interval_controls.csv")
    result["std_eq"].to_csv(year_directory / "equity_standardized_returns.csv")
    result["std_cr"].to_csv(year_directory / "crypto_standardized_returns.csv")
    result["E_eq"].to_csv(year_directory / "equity_shock_events.csv")
    result["E_cr"].to_csv(year_directory / "crypto_shock_events.csv")
    result["Y"].to_csv(year_directory / "community_shock_counts.csv")
    result["X"].to_csv(year_directory / "community_shock_proportions.csv")

    for market, label in (("eq", "equity"), ("cr", "crypto")):
        asset_names = result[f"ret_{market}"].columns

        pd.DataFrame(
            result[f"rho_{market}"],
            index=asset_names,
            columns=asset_names,
        ).to_csv(year_directory / f"{label}_correlation_matrix.csv")

        pd.DataFrame(
            result[f"B_{market}"],
            index=asset_names,
            columns=asset_names,
        ).to_csv(year_directory / f"{label}_residual_matrix_B.csv")

        eigen_table = pd.DataFrame(
            {
                "mode_rank": np.arange(1, len(result[f"evals_{market}"]) + 1),
                "eigenvalue": result[f"evals_{market}"],
                "informative_raw": result[f"info_{market}_raw"].astype(int),
                "informative_used": result[f"info_{market}_used"].astype(int),
            }
        )
        eigen_table.to_csv(
            year_directory / f"{label}_eigenspectrum.csv",
            index=False,
        )

        used_modes = np.where(result[f"info_{market}_used"])[0] + 1
        embedding = pd.DataFrame(
            result[f"Phi_{market}"],
            index=asset_names,
            columns=[f"mode_{mode}" for mode in used_modes],
        )
        embedding.index.name = "asset"
        embedding.to_csv(year_directory / f"{label}_spectral_embedding.csv")

    result["A"].to_csv(year_directory / "A_integrated_excitation.csv")
    result["A1"].to_csv(year_directory / "A_lag1_excitation.csv")
    result["MeanLag"].to_csv(year_directory / "A_mean_lag.csv")
    result["membership"].to_csv(year_directory / "community_membership.csv", index=False)
    result["summary"].to_csv(year_directory / "community_summary.csv", index=False)
    result["fit_info"].to_csv(year_directory / "fit_diagnostics.csv", index=False)

    group_rows = [
        {
            "community": group,
            "market": market_of(group),
            "size": len(assets),
            "assets": ",".join(assets),
        }
        for group, assets in result["groups"].items()
    ]
    pd.DataFrame(group_rows).to_csv(
        year_directory / "community_asset_lists.csv",
        index=False,
    )


def save_baseline_figures(
    combined_summary: pd.DataFrame,
    window_results: Mapping[int, Dict[str, Any]],
    output_directory: Path,
    config: AnalysisConfig,
) -> None:

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(12, 8),
        gridspec_kw={"hspace": 0.32, "wspace": 0.28},
    )
    maximum_value = max(
        window_results[year]["A"].to_numpy().max()
        for year in config.years
    )
    image = None
    panel_labels = ["A", "B", "C", "D", "E", "F"]

    for panel_index, (axis, year) in enumerate(zip(axes.ravel(), config.years)):
        excitation = window_results[year]["A"]
        n_equity = sum(market_of(label) == "EQ" for label in excitation.index)
        image = axis.imshow(
            excitation.to_numpy(),
            cmap="viridis",
            vmin=0,
            vmax=maximum_value,
            aspect="equal",
        )

        for lower, upper in ((0, n_equity), (n_equity, len(excitation))):
            axis.add_patch(
                plt.Rectangle(
                    (lower - 0.5, lower - 0.5),
                    upper - lower,
                    upper - lower,
                    fill=False,
                    edgecolor="white",
                    linewidth=1.2,
                )
            )

        axis.text(
            -0.12,
            1.03,
            panel_labels[panel_index],
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontweight="bold",
        )
        axis.set_xticks(
            [n_equity / 2 - 0.5, n_equity + (len(excitation) - n_equity) / 2 - 0.5],
            ["EQ", "CR"],
        )
        axis.set_yticks(
            [n_equity / 2 - 0.5, n_equity + (len(excitation) - n_equity) / 2 - 0.5],
            ["EQ", "CR"],
        )
        axis.set_xlabel("Target")
        axis.set_ylabel("Source", labelpad=10)

    if image is not None:
        figure.colorbar(
            image,
            ax=axes.ravel().tolist(),
            shrink=0.75,
            label="integrated excitation",
        )

    figure.savefig(
        output_directory / "fig_excitation_heatmaps.png",
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        output_directory / "fig_excitation_heatmaps.pdf",
        bbox_inches="tight",
    )

    if config.show_figures:
        plt.show()
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.2))

    equity_color = "C0"
    crypto_color = "C1"

    def stacked_positions(base_level: float, sizes: np.ndarray, spread: float = 0.05) -> np.ndarray:
        positions = np.full(len(sizes), base_level, dtype=float)
        if len(sizes) == 0:
            return positions

        for size_value in np.unique(sizes):
            index = np.where(sizes == size_value)[0]
            count = len(index)
            if count == 1:
                positions[index] = base_level
            else:
                offsets = np.linspace(-spread, spread, count)
                positions[index] = base_level + offsets
        return positions

    for vertical_position, year in enumerate(config.years):
        for offset, market, marker, color, legend_name in (
            (0.16, "EQ", "o", equity_color, "Equity"),
            (-0.16, "CR", "s", crypto_color, "Crypto"),
        ):
            sizes = combined_summary.loc[
                (combined_summary["window"] == year)
                & (combined_summary["market"] == market),
                "size",
            ].to_numpy(dtype=int)

            sizes = np.sort(sizes)
            base_level = vertical_position + offset
            positions = stacked_positions(base_level, sizes)

            axis.scatter(
                sizes,
                positions,
                s=35,
                alpha=0.85,
                marker=marker,
                color=color,
                label=legend_name if vertical_position == 0 else None,
            )
            axis.plot(
                [sizes.min(), sizes.max()],
                [base_level, base_level],
                linewidth=0.8,
                alpha=0.4,
                color=color,
            )
            axis.plot(
                [np.median(sizes)] * 2,
                [base_level - 0.08, base_level + 0.08],
                linewidth=2,
                color=color,
            )

    axis.set_yticks(range(len(config.years)), config.years)
    axis.set_xlabel("Community size")
    axis.set_ylabel("Year")
    axis.legend(frameon=False, loc="lower right")
    figure.tight_layout()

    figure.savefig(
        output_directory / "fig_community_sizes.png",
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        output_directory / "fig_community_sizes.pdf",
        bbox_inches="tight",
    )

    if config.show_figures:
        plt.show()
    plt.close(figure)


def save_baseline_results(
    window_results: Mapping[int, Dict[str, Any]],
    combined_summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    memberships: pd.DataFrame,
    fit_diagnostics: pd.DataFrame,
    output_directory: Path,
    config: AnalysisConfig,
) -> Dict[str, pd.DataFrame]:
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "A_matrices").mkdir(exist_ok=True)
    (output_directory / "memberships").mkdir(exist_ok=True)

    partition = excitation_partition_table(window_results, config)
    community_structure = community_structure_table(combined_summary, diagnostics)

    combined_summary.to_csv(
        output_directory / "community_summary_all_windows.csv",
        index=False,
    )
    diagnostics.to_csv(output_directory / "window_diagnostics.csv", index=False)
    memberships.to_csv(
        output_directory / "community_membership_all_windows.csv",
        index=False,
    )
    fit_diagnostics.to_csv(
        output_directory / "baseline_fit_diagnostics.csv",
        index=False,
    )
    partition.to_csv(output_directory / "excitation_partition.csv", index=False)
    community_structure.to_csv(
        output_directory / "community_structure_by_window.csv",
        index=False,
    )

    for year in config.years:
        result = window_results[year]
        result["A"].to_csv(
            output_directory / "A_matrices" / f"A_comm_{year}.csv"
        )
        result["A1"].to_csv(
            output_directory / "A_matrices" / f"A1_comm_{year}.csv"
        )
        result["MeanLag"].to_csv(
            output_directory / "A_matrices" / f"MeanLag_comm_{year}.csv"
        )
        result["membership"].to_csv(
            output_directory / "memberships" / f"community_membership_{year}.csv",
            index=False,
        )

    return {
        "excitation_partition": partition,
        "community_structure": community_structure,
    }


def run_baseline_analysis(
    equity_prices: pd.DataFrame,
    crypto_prices: pd.DataFrame,
    output_directory: Path,
    config: AnalysisConfig,
    save_intermediate: bool,
) -> Dict[str, Any]:
    window_results: Dict[int, Dict[str, Any]] = {}
    summaries: List[pd.DataFrame] = []
    diagnostics: List[Dict[str, Any]] = []
    memberships: List[pd.DataFrame] = []
    fits: List[pd.DataFrame] = []

    for year in config.years:
        logger.info("Fitting baseline annual window %s", year)
        result = fit_baseline_window(year, equity_prices, crypto_prices, config)
        window_results[year] = result
        summaries.append(result["summary"])
        diagnostics.append(result["diag"])
        memberships.append(result["membership"])
        fits.append(result["fit_info"])

        if save_intermediate:
            save_annual_intermediate_outputs(
                result,
                output_directory / "annual_windows",
            )

        logger.info(
            "%s complete. EQ assets=%s, CR assets=%s, EQ communities=%s, "
            "CR communities=%s, all fits converged=%s",
            year,
            result["diag"]["eq_assets"],
            result["diag"]["cr_assets"],
            result["diag"]["eq_communities"],
            result["diag"]["cr_communities"],
            result["diag"]["fit_success_all"],
        )

    combined_summary = pd.concat(summaries, ignore_index=True)
    diagnostic_table = pd.DataFrame(diagnostics)
    membership_table = pd.concat(memberships, ignore_index=True)
    fit_table = pd.concat(fits, ignore_index=True)

    saved_tables = save_baseline_results(
        window_results,
        combined_summary,
        diagnostic_table,
        membership_table,
        fit_table,
        output_directory,
        config,
    )

    figure_directory = (
        output_directory
        / "figures"
    )
    
    figure_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    
    save_baseline_figures(
        combined_summary,
        window_results,
        figure_directory,
        config,
    )

    with (output_directory / "baseline_cache.pkl").open("wb") as handle:
        pickle.dump(window_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    manifest = {
        "years": list(config.years),
        "equity_assets_input": int(equity_prices.shape[1]),
        "crypto_assets_input": int(crypto_prices.shape[1]),
        "price_sessions_total": int(len(equity_prices)),
        "alpha": config.shock_alpha,
        "L": config.max_lag,
        "eta_grid": list(config.eta_grid),
        "train0": config.train0,
        "validation_block": config.validation_block,
        "n_folds": config.n_folds,
        "interval_control": "regular versus extended equity-session interval",
        "annual_variance_screen": True,
    }
    (output_directory / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    return {
        "window_results": window_results,
        "community_summary": combined_summary,
        "window_diagnostics": diagnostic_table,
        "memberships": membership_table,
        "fit_diagnostics": fit_table,
        **saved_tables,
    }


#---------------PRIMARY INFERENCE AND CROSS-MARKET DIAGNOSTICS------------

def safe_spearman(
    first: Iterable[float],
    second: Iterable[float],
    minimum_n: int = 4,
) -> Tuple[float, float, int]:
    first_array = np.asarray(first, dtype=float)
    second_array = np.asarray(second, dtype=float)
    valid = np.isfinite(first_array) & np.isfinite(second_array)
    first_array = first_array[valid]
    second_array = second_array[valid]
    n_observations = len(first_array)

    if (
        n_observations < minimum_n
        or np.std(first_array) == 0
        or np.std(second_array) == 0
    ):
        return np.nan, np.nan, n_observations

    result = spearmanr(first_array, second_array)
    return float(result.correlation), float(result.pvalue), n_observations


def benjamini_hochberg_keep_nan(values: Iterable[float]) -> np.ndarray:
    p_values = np.asarray(list(values), dtype=float)
    missing = ~np.isfinite(p_values)
    working = np.where(missing, 1.0, p_values)
    order = np.argsort(working)

    adjusted_sorted = (
        working[order]
        * len(working)
        / (np.arange(len(working)) + 1)
    )
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]

    adjusted = np.empty(len(working))
    adjusted[order] = np.clip(adjusted_sorted, 0.0, 1.0)
    adjusted[missing] = np.nan

    return adjusted


def fisher_z_combine(
    correlations: Sequence[float],
    sample_sizes: Sequence[int],
    number_of_controls: int = 0,
) -> Tuple[float, float, float, int]:
    correlations_array = np.asarray(correlations, dtype=float)
    sample_sizes_array = np.asarray(sample_sizes, dtype=float)
    weights = sample_sizes_array - 3.0 - number_of_controls

    valid = (
        np.isfinite(correlations_array)
        & np.isfinite(sample_sizes_array)
        & (weights > 0)
    )

    if not valid.any():
        return np.nan, np.nan, 0.0, 0

    transformed = np.arctanh(
        np.clip(correlations_array[valid], -0.999999, 0.999999)
    )
    valid_weights = weights[valid]
    weighted_mean = np.sum(valid_weights * transformed) / np.sum(valid_weights)
    normal_statistic = weighted_mean * np.sqrt(np.sum(valid_weights))

    combined_correlation = float(np.tanh(weighted_mean))
    approximate_p_value = float(2.0 * norm.sf(abs(normal_statistic)))

    return (
        combined_correlation,
        approximate_p_value,
        float(np.sum(valid_weights)),
        int(valid.sum()),
    )


def correlations_with_fixed_first_vector(
    first: np.ndarray,
    second_matrix: np.ndarray,
) -> np.ndarray:
    first = np.asarray(first, dtype=float)
    second_matrix = np.asarray(second_matrix, dtype=float)

    centered_first = first - first.mean()
    centered_second = second_matrix - second_matrix.mean(axis=1, keepdims=True)

    denominator = np.sqrt(
        np.sum(centered_first * centered_first)
        * np.sum(centered_second * centered_second, axis=1)
    )

    correlations = np.full(second_matrix.shape[0], np.nan)
    valid = denominator > 0
    correlations[valid] = (
        centered_second[valid] @ centered_first
    ) / denominator[valid]

    return correlations


def random_permutations(
    values: np.ndarray,
    repetitions: int,
    random_generator: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(values)
    random_order = np.argsort(
        random_generator.random((repetitions, len(values))),
        axis=1,
    )
    return values[random_order]


def permutation_meta_p_value(
    frame: pd.DataFrame,
    predictor: str,
    outcome: str,
    observed_meta_correlation: float,
    repetitions: int,
    seed: int,
) -> float:
    if not np.isfinite(observed_meta_correlation):
        return np.nan

    random_generator = np.random.default_rng(seed)
    permutation_correlations: List[np.ndarray] = []
    sample_sizes: List[int] = []

    for _, subset in frame.groupby("window", sort=True):
        first = subset[predictor].to_numpy(dtype=float)
        second = subset[outcome].to_numpy(dtype=float)
        valid = np.isfinite(first) & np.isfinite(second)
        first = first[valid]
        second = second[valid]

        if len(first) < 4 or np.std(first) == 0 or np.std(second) == 0:
            continue

        ranked_first = rankdata(first)
        ranked_second = rankdata(second)
        permuted_second = random_permutations(
            ranked_second,
            repetitions,
            random_generator,
        )

        permutation_correlations.append(
            correlations_with_fixed_first_vector(ranked_first, permuted_second)
        )
        sample_sizes.append(len(first))

    if not permutation_correlations:
        return np.nan

    correlation_matrix = np.column_stack(permutation_correlations)
    weights = np.asarray(sample_sizes, dtype=float) - 3.0
    transformed = np.arctanh(
        np.clip(correlation_matrix, -0.999999, 0.999999)
    )
    permuted_meta = np.tanh(
        np.sum(transformed * weights, axis=1) / np.sum(weights)
    )

    return float(
        (
            np.sum(np.abs(permuted_meta) >= abs(observed_meta_correlation))
            + 1
        )
        / (repetitions + 1)
    )


def partial_spearman(
    first: Iterable[float],
    second: Iterable[float],
    controls: Sequence[Iterable[float]],
) -> Tuple[float, float, int]:
    first_array = np.asarray(first, dtype=float)
    second_array = np.asarray(second, dtype=float)
    control_arrays = [np.asarray(control, dtype=float) for control in controls]

    valid = np.isfinite(first_array) & np.isfinite(second_array)
    for control in control_arrays:
        valid &= np.isfinite(control)

    first_array = first_array[valid]
    second_array = second_array[valid]
    control_arrays = [control[valid] for control in control_arrays]
    n_observations = len(first_array)

    if (
        n_observations < 4 + len(control_arrays)
        or np.std(first_array) == 0
        or np.std(second_array) == 0
    ):
        return np.nan, np.nan, n_observations

    design_columns = [np.ones(n_observations)]

    for control in control_arrays:
        if np.std(control) == 0:
            return np.nan, np.nan, n_observations
        design_columns.append(rankdata(control))

    design = np.column_stack(design_columns)
    ranked_first = rankdata(first_array)
    ranked_second = rankdata(second_array)

    residual_first = ranked_first - design @ np.linalg.lstsq(
        design,
        ranked_first,
        rcond=None,
    )[0]
    residual_second = ranked_second - design @ np.linalg.lstsq(
        design,
        ranked_second,
        rcond=None,
    )[0]

    if np.std(residual_first) == 0 or np.std(residual_second) == 0:
        return np.nan, np.nan, n_observations

    correlation = float(np.corrcoef(residual_first, residual_second)[0, 1])
    degrees_of_freedom = n_observations - len(control_arrays) - 2
    statistic = correlation * np.sqrt(
        degrees_of_freedom / max(1.0 - correlation * correlation, 1e-12)
    )
    p_value = float(2.0 * stats.t.sf(abs(statistic), degrees_of_freedom))

    return correlation, p_value, n_observations


def permutation_partial_market_p_value(
    frame: pd.DataFrame,
    predictor: str,
    outcome: str,
    observed_meta_correlation: float,
    repetitions: int,
    seed: int,
) -> float:
    if not np.isfinite(observed_meta_correlation):
        return np.nan

    random_generator = np.random.default_rng(seed)
    permutation_correlations: List[np.ndarray] = []
    sample_sizes: List[int] = []

    for _, subset in frame.groupby("window", sort=True):
        first = subset[predictor].to_numpy(dtype=float)
        second = subset[outcome].to_numpy(dtype=float)
        market = subset["market"].eq("CR").astype(float).to_numpy()

        valid = np.isfinite(first) & np.isfinite(second) & np.isfinite(market)
        first = first[valid]
        second = second[valid]
        market = market[valid]
        n_observations = len(first)

        if (
            n_observations < 5
            or np.std(first) == 0
            or np.std(second) == 0
            or np.std(market) == 0
        ):
            continue

        ranked_first = rankdata(first)
        ranked_second = rankdata(second)
        ranked_market = rankdata(market)

        design = np.column_stack([np.ones(n_observations), ranked_market])
        residual_maker = np.eye(n_observations) - design @ np.linalg.pinv(design)
        residual_first = residual_maker @ ranked_first

        permuted_ranked_second = np.tile(
            ranked_second,
            (repetitions, 1),
        ).astype(float)

        for market_value in np.unique(market):
            indices = np.where(market == market_value)[0]

            if len(indices) > 1:
                random_order = np.argsort(
                    random_generator.random((repetitions, len(indices))),
                    axis=1,
                )
                permuted_ranked_second[:, indices] = ranked_second[indices][
                    random_order
                ]

        residual_second = (
            residual_maker @ permuted_ranked_second.T
        ).T

        permutation_correlations.append(
            correlations_with_fixed_first_vector(residual_first, residual_second)
        )
        sample_sizes.append(n_observations)

    if not permutation_correlations:
        return np.nan

    correlation_matrix = np.column_stack(permutation_correlations)
    weights = np.asarray(sample_sizes, dtype=float) - 4.0
    transformed = np.arctanh(
        np.clip(correlation_matrix, -0.999999, 0.999999)
    )
    permuted_meta = np.tanh(
        np.sum(transformed * weights, axis=1) / np.sum(weights)
    )

    return float(
        (
            np.sum(np.abs(permuted_meta) >= abs(observed_meta_correlation))
            + 1
        )
        / (repetitions + 1)
    )


def bootstrap_complete_window_interval(
    frame: pd.DataFrame,
    predictor: str,
    outcome: str,
    repetitions: int,
    seed: int,
) -> Tuple[float, float, float]:
    random_generator = np.random.default_rng(seed)
    windows = np.array(sorted(frame["window"].unique()))
    blocks = {window: frame[frame["window"] == window] for window in windows}
    values: List[float] = []

    for _ in range(repetitions):
        sampled_windows = random_generator.choice(
            windows,
            size=len(windows),
            replace=True,
        )
        sample = pd.concat(
            [blocks[int(window)] for window in sampled_windows],
            ignore_index=True,
        )
        correlation, _, _ = safe_spearman(sample[predictor], sample[outcome])

        if np.isfinite(correlation):
            values.append(correlation)

    if not values:
        return np.nan, np.nan, np.nan

    lower, median, upper = np.percentile(values, [2.5, 50.0, 97.5])
    return float(lower), float(median), float(upper)


def plot_crosslayer_six_panel(
    frame: pd.DataFrame,
    per_window_table: pd.DataFrame,
    output_directory: Path,
    config: AnalysisConfig,
) -> None:

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    colours = dict(
        zip(
            config.years,
            plt.cm.tab10(
                np.linspace(
                    0,
                    1,
                    len(config.years),
                )
            ),
        )
    )

    figure, axes = plt.subplots(
        3,
        2,
        figsize=(11, 13),
    )

    predictor_labels = {
        "WithinMean":
            "Internal cohesion (WithinMean)",

        "BetweenMean":
            "External coupling (BetweenMean)",

        "CommBetweenness":
            "Topological bridge position (CommBetweenness)",
    }

    outcome_labels = {
        "WithinOutgoing":
            "Within-market outgoing excitation",

        "CrossOutgoing":
            "Cross-market outgoing excitation",
    }

    panel_labels = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
    ]

    panel_index = 0

    for row_position, predictor in enumerate(
        config.primary_structural_predictors
    ):

        for column_position, outcome in enumerate(
            config.propagation_outcomes
        ):

            axis = axes[
                row_position,
                column_position,
            ]

            for year in config.years:

                subset = frame.loc[
                    frame["window"] == year
                ]

                axis.scatter(
                    subset[predictor],
                    subset[outcome],
                    s=32,
                    alpha=0.75,
                    color=colours[year],
                    label=str(year),
                )

            pooled_rho, _, _ = safe_spearman(
                frame[predictor],
                frame[outcome],
            )

            annual_correlations = (
                per_window_table.loc[
                    (
                        per_window_table[
                            "predictor"
                        ]
                        == predictor
                    )
                    & (
                        per_window_table[
                            "outcome"
                        ]
                        == outcome
                    ),
                    "rho",
                ]
            )

            median_window_rho = (
                annual_correlations.median()
            )

            axis.text(
                0.02,
                0.98,
                panel_labels[panel_index],
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=13,
                fontweight="bold",
            )

            correlation_text = (
                rf"median window $\rho_S$ = "
                f"{median_window_rho:+.2f}"
                "\n"
                rf"pooled $\rho_S$ = "
                f"{pooled_rho:+.2f}"
            )

            axis.text(
                0.02,
                0.89,
                correlation_text,
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=9.5,
            )

            axis.set_xlabel(
                predictor_labels[predictor]
            )

            axis.set_ylabel(
                outcome_labels[outcome]
            )

            panel_index += 1

    axes[0, 0].legend(
        title="Window",
        loc="upper right",
        frameon=True,
        fontsize=8.5,
        title_fontsize=8.5,
        markerscale=0.9,
        borderpad=0.5,
        labelspacing=0.3,
    )

    figure.tight_layout(
        h_pad=2.0,
        w_pad=2.0,
    )

    figure.savefig(
        output_directory
        / "fig_crosslayer_six_panel.png",
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        output_directory
        / "fig_crosslayer_six_panel.pdf",
        bbox_inches="tight",
    )

    if config.show_figures:
        plt.show()

    plt.close(figure)


def run_primary_inference_and_diagnostics(
    community_summary: pd.DataFrame,
    output_directory: Path,
    config: AnalysisConfig,
    file_prefix: str = "",
    save_figure: bool = True,
) -> Dict[str, pd.DataFrame]:
    output_directory.mkdir(parents=True, exist_ok=True)
    frame = community_summary.copy()

    all_predictors = (
        list(config.primary_structural_predictors)
        + list(config.descriptive_predictors)
    )

    per_window_rows: List[Dict[str, Any]] = []
    aggregate_rows: List[Dict[str, Any]] = []
    relationship_index = 0

    for predictor in all_predictors:
        for outcome in config.propagation_outcomes:
            annual_correlations: List[float] = []
            annual_sizes: List[int] = []

            for year, subset in frame.groupby("window", sort=True):
                correlation, p_value, n_communities = safe_spearman(
                    subset[predictor],
                    subset[outcome],
                )
                annual_correlations.append(correlation)
                annual_sizes.append(n_communities)
                per_window_rows.append(
                    {
                        "predictor": predictor,
                        "outcome": outcome,
                        "window": int(year),
                        "rho": correlation,
                        "p": p_value,
                        "n_communities": n_communities,
                    }
                )

            meta_correlation, normal_p, sum_weight, n_windows = fisher_z_combine(
                annual_correlations,
                annual_sizes,
            )

            permutation_p = (
                permutation_meta_p_value(
                    frame,
                    predictor,
                    outcome,
                    meta_correlation,
                    config.permutations,
                    config.random_seed + relationship_index,
                )
                if predictor in config.primary_structural_predictors
                else np.nan
            )

            pooled_correlation, pooled_p, pooled_n = safe_spearman(
                frame[predictor],
                frame[outcome],
            )

            valid_annual = np.asarray(
                [value for value in annual_correlations if np.isfinite(value)]
            )

            aggregate_rows.append(
                {
                    "predictor": predictor,
                    "outcome": outcome,
                    "median_rho": float(np.nanmedian(valid_annual)),
                    "iqr_rho": float(
                        np.nanpercentile(valid_annual, 75)
                        - np.nanpercentile(valid_annual, 25)
                    ),
                    "frac_windows_positive": float(np.mean(valid_annual > 0)),
                    "n_windows": n_windows,
                    "sum_fisher_weight": sum_weight,
                    "meta_rho": meta_correlation,
                    "meta_p": normal_p,
                    "perm_p": permutation_p,
                    "pooled_rho_descr": pooled_correlation,
                    "pooled_p_descr": pooled_p,
                    "pooled_n": pooled_n,
                    "family": (
                        "inferential"
                        if predictor in config.primary_structural_predictors
                        else "descriptive"
                    ),
                }
            )

            relationship_index += 1

    per_window = pd.DataFrame(per_window_rows)
    aggregate = pd.DataFrame(aggregate_rows)

    inferential = aggregate["family"].eq("inferential")
    aggregate["meta_q_bh"] = np.nan
    aggregate["perm_q_bh"] = np.nan
    aggregate.loc[inferential, "meta_q_bh"] = benjamini_hochberg_keep_nan(
        aggregate.loc[inferential, "meta_p"]
    )
    aggregate.loc[inferential, "perm_q_bh"] = benjamini_hochberg_keep_nan(
        aggregate.loc[inferential, "perm_p"]
    )

    per_window.to_csv(
        output_directory / f"{file_prefix}crosslayer_spearman_per_window.csv",
        index=False,
    )
    aggregate.to_csv(
        output_directory / f"{file_prefix}crosslayer_spearman_aggregate.csv",
        index=False,
    )

    predictor_correlation = frame[
        [
            "WithinMean",
            "BetweenMean",
            "Strength",
            "BridgeRatio",
            "CommBetweenness",
        ]
    ].corr(method="spearman")
    predictor_correlation.to_csv(
        output_directory / f"{file_prefix}predictor_correlation_spearman.csv"
    )

    work = frame.copy()
    work["OutgoingToOthers"] = work["WithinOutgoing"] + work["CrossOutgoing"]
    work["CrossShare_NoSelf"] = work["CrossOutgoing"] / work[
        "OutgoingToOthers"
    ].replace(0, np.nan)

    cross_rows: List[Dict[str, Any]] = []

    for predictor_index, predictor in enumerate(config.primary_structural_predictors):
        pooled_correlation, pooled_p, pooled_n = safe_spearman(
            work[predictor],
            work["CrossOutgoing"],
        )
        lower, median, upper = bootstrap_complete_window_interval(
            work,
            predictor,
            "CrossOutgoing",
            config.bootstrap_repetitions,
            config.random_seed + 100 + predictor_index,
        )
        partial_correlation, partial_p, partial_n = partial_spearman(
            work[predictor],
            work["CrossOutgoing"],
            [work["WithinOutgoing"]],
        )
        cross_share_correlation, cross_share_p, cross_share_n = safe_spearman(
            work[predictor],
            work["CrossShare_NoSelf"],
        )

        cross_rows.append(
            {
                "predictor": predictor,
                "pooled_rho": pooled_correlation,
                "pooled_p": pooled_p,
                "pooled_n": pooled_n,
                "bootstrap_ci_low": lower,
                "bootstrap_median": median,
                "bootstrap_ci_high": upper,
                "partial_rho_given_within": partial_correlation,
                "partial_p_given_within": partial_p,
                "partial_n": partial_n,
                "crossshare_rho": cross_share_correlation,
                "crossshare_p": cross_share_p,
                "crossshare_n": cross_share_n,
            }
        )

    cross_market_diagnostics = pd.DataFrame(cross_rows)
    cross_market_diagnostics.to_csv(
        output_directory / f"{file_prefix}cross_market_diagnostics.csv",
        index=False,
    )

    panel_detail_rows: List[Dict[str, Any]] = []
    panel_summary_rows: List[Dict[str, Any]] = []
    relationship_index = 0

    for predictor in all_predictors:
        for outcome in config.propagation_outcomes:
            panel_correlations: List[float] = []
            panel_sizes: List[int] = []
            panel_subsets: List[pd.DataFrame] = []

            for (year, market), subset in work.groupby(
                ["window", "market"],
                sort=True,
            ):
                correlation, p_value, n_communities = safe_spearman(
                    subset[predictor],
                    subset[outcome],
                )
                panel_correlations.append(correlation)
                panel_sizes.append(n_communities)
                panel_subsets.append(subset)
                panel_detail_rows.append(
                    {
                        "predictor": predictor,
                        "outcome": outcome,
                        "window": int(year),
                        "market": market,
                        "rho": correlation,
                        "p": p_value,
                        "n_communities": n_communities,
                    }
                )

            meta_correlation, normal_p, sum_weight, n_panels = fisher_z_combine(
                panel_correlations,
                panel_sizes,
            )

            if predictor in config.primary_structural_predictors:
                permutable_panels: List[pd.DataFrame] = []

                for subset in panel_subsets:
                    panel_label = (
                        f"{int(subset['window'].iloc[0])}_"
                        f"{subset['market'].iloc[0]}"
                    )
                    permutable_panels.append(
                        subset.assign(_panel=panel_label)
                    )

                permutation_frame = pd.concat(
                    permutable_panels,
                    ignore_index=True,
                ).rename(columns={"window": "_year", "_panel": "window"})

                permutation_p = permutation_meta_p_value(
                    permutation_frame,
                    predictor,
                    outcome,
                    meta_correlation,
                    config.permutations,
                    config.random_seed + 1000 + relationship_index,
                )
            else:
                permutation_p = np.nan

            valid_panel = np.asarray(
                [value for value in panel_correlations if np.isfinite(value)]
            )

            panel_summary_rows.append(
                {
                    "predictor": predictor,
                    "outcome": outcome,
                    "median_panel_rho": float(np.nanmedian(valid_panel)),
                    "frac_positive": float(np.mean(valid_panel > 0)),
                    "meta_rho": meta_correlation,
                    "p_normal": normal_p,
                    "p_perm": permutation_p,
                    "sum_weight": sum_weight,
                    "n_panels_estimable": n_panels,
                }
            )

            relationship_index += 1

    panel_detail = pd.DataFrame(panel_detail_rows)
    panel_summary = pd.DataFrame(panel_summary_rows)
    panel_inferential = panel_summary["predictor"].isin(
        config.primary_structural_predictors
    )
    panel_summary["q_normal"] = np.nan
    panel_summary["q_perm"] = np.nan
    panel_summary.loc[panel_inferential, "q_normal"] = (
        benjamini_hochberg_keep_nan(
            panel_summary.loc[panel_inferential, "p_normal"]
        )
    )
    panel_summary.loc[panel_inferential, "q_perm"] = (
        benjamini_hochberg_keep_nan(
            panel_summary.loc[panel_inferential, "p_perm"]
        )
    )

    panel_detail.to_csv(
        output_directory / f"{file_prefix}market_window_panel_correlations.csv",
        index=False,
    )
    panel_summary.to_csv(
        output_directory / f"{file_prefix}market_window_panel_summary.csv",
        index=False,
    )

    partial_detail_rows: List[Dict[str, Any]] = []
    partial_summary_rows: List[Dict[str, Any]] = []
    relationship_index = 0

    for predictor in all_predictors:
        for outcome in config.propagation_outcomes:
            annual_partial_correlations: List[float] = []
            annual_sizes: List[int] = []

            for year, subset in work.groupby("window", sort=True):
                market_control = subset["market"].eq("CR").astype(float).to_numpy()
                correlation, p_value, n_communities = partial_spearman(
                    subset[predictor],
                    subset[outcome],
                    [market_control],
                )
                annual_partial_correlations.append(correlation)
                annual_sizes.append(n_communities)
                partial_detail_rows.append(
                    {
                        "predictor": predictor,
                        "outcome": outcome,
                        "window": int(year),
                        "partial_rho_market": correlation,
                        "p_approx": p_value,
                        "n_communities": n_communities,
                    }
                )

            meta_correlation, normal_p, sum_weight, n_windows = fisher_z_combine(
                annual_partial_correlations,
                annual_sizes,
                number_of_controls=1,
            )

            permutation_p = (
                permutation_partial_market_p_value(
                    work,
                    predictor,
                    outcome,
                    meta_correlation,
                    config.permutations,
                    config.random_seed + 2000 + relationship_index,
                )
                if predictor in config.primary_structural_predictors
                else np.nan
            )

            valid_partial = np.asarray(
                [
                    value
                    for value in annual_partial_correlations
                    if np.isfinite(value)
                ]
            )

            partial_summary_rows.append(
                {
                    "predictor": predictor,
                    "outcome": outcome,
                    "median_partial_rho": float(np.nanmedian(valid_partial)),
                    "frac_positive": float(np.mean(valid_partial > 0)),
                    "meta_partial_rho": meta_correlation,
                    "p_normal": normal_p,
                    "p_perm": permutation_p,
                    "sum_weight": sum_weight,
                    "n_windows": n_windows,
                }
            )

            relationship_index += 1

    partial_detail = pd.DataFrame(partial_detail_rows)
    partial_summary = pd.DataFrame(partial_summary_rows)
    partial_inferential = partial_summary["predictor"].isin(
        config.primary_structural_predictors
    )
    partial_summary["q_normal"] = np.nan
    partial_summary["q_perm"] = np.nan
    partial_summary.loc[partial_inferential, "q_normal"] = (
        benjamini_hochberg_keep_nan(
            partial_summary.loc[partial_inferential, "p_normal"]
        )
    )
    partial_summary.loc[partial_inferential, "q_perm"] = (
        benjamini_hochberg_keep_nan(
            partial_summary.loc[partial_inferential, "p_perm"]
        )
    )

    partial_detail.to_csv(
        output_directory / f"{file_prefix}partial_market_by_window.csv",
        index=False,
    )
    partial_summary.to_csv(
        output_directory / f"{file_prefix}partial_market_summary.csv",
        index=False,
    )

    normalized_rank_rows: List[Dict[str, Any]] = []

    for predictor in all_predictors:
        for outcome in config.propagation_outcomes:
            ranked = work.copy()
            grouped = ranked.groupby(["window", "market"])
            ranked["rank_predictor"] = (
                grouped[predictor].rank() - 0.5
            ) / grouped[predictor].transform("count")
            ranked["rank_outcome"] = (
                grouped[outcome].rank() - 0.5
            ) / grouped[outcome].transform("count")

            correlation, p_value, n_observations = safe_spearman(
                ranked["rank_predictor"],
                ranked["rank_outcome"],
            )

            normalized_rank_rows.append(
                {
                    "predictor": predictor,
                    "outcome": outcome,
                    "rho": correlation,
                    "p": p_value,
                    "n": n_observations,
                }
            )

    normalized_ranks = pd.DataFrame(normalized_rank_rows)
    normalized_ranks.to_csv(
        output_directory / f"{file_prefix}pooled_within_panel_normalised_ranks.csv",
        index=False,
    )

    conditional_rows: List[Dict[str, Any]] = []

    for predictor in config.primary_structural_predictors:
        for include_market in (False, True):
            annual_correlations: List[float] = []
            annual_sizes: List[int] = []

            for _, subset in work.groupby("window", sort=True):
                controls: List[np.ndarray] = [
                    subset["WithinOutgoing"].to_numpy(dtype=float)
                ]

                if include_market:
                    controls.append(
                        subset["market"].eq("CR").astype(float).to_numpy()
                    )

                correlation, _, n_communities = partial_spearman(
                    subset[predictor],
                    subset["CrossOutgoing"],
                    controls,
                )
                annual_correlations.append(correlation)
                annual_sizes.append(n_communities)

            meta_correlation, p_value, sum_weight, n_windows = fisher_z_combine(
                annual_correlations,
                annual_sizes,
                number_of_controls=2 if include_market else 1,
            )

            conditional_rows.append(
                {
                    "predictor": predictor,
                    "controls": (
                        "WithinOutgoing+Market"
                        if include_market
                        else "WithinOutgoing"
                    ),
                    "partial_meta_rho": meta_correlation,
                    "partial_meta_p": p_value,
                    "sum_weight": sum_weight,
                    "n_windows": n_windows,
                }
            )

    conditional = pd.DataFrame(conditional_rows)
    conditional["q_bh_three"] = np.nan

    for _, indices in conditional.groupby("controls").groups.items():
        conditional.loc[indices, "q_bh_three"] = benjamini_hochberg_keep_nan(
            conditional.loc[indices, "partial_meta_p"]
        )

    conditional.to_csv(
        output_directory / f"{file_prefix}cross_conditional_summary.csv",
        index=False,
    )

    if save_figure:
        plot_crosslayer_six_panel(
            work,
            per_window,
            output_directory,
            config,
        )

    return {
        "per_window": per_window,
        "aggregate": aggregate,
        "predictor_correlation": predictor_correlation,
        "cross_market_diagnostics": cross_market_diagnostics,
        "market_window_detail": panel_detail,
        "market_window_summary": panel_summary,
        "partial_market_detail": partial_detail,
        "partial_market_summary": partial_summary,
        "normalized_ranks": normalized_ranks,
        "cross_conditional": conditional,
    }


#----------------------ROBUSTNESS TESTS-----------------------

def crosslayer_meta_normal(
    frame: pd.DataFrame,
    predictors: Sequence[str],
    outcomes: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for predictor in predictors:
        for outcome in outcomes:
            annual_correlations: List[float] = []
            annual_sizes: List[int] = []

            for _, subset in frame.groupby("window", sort=True):
                correlation, _, n_communities = safe_spearman(
                    subset[predictor],
                    subset[outcome],
                )
                annual_correlations.append(correlation)
                annual_sizes.append(n_communities)

            meta_correlation, meta_p, _, n_windows = fisher_z_combine(
                annual_correlations,
                annual_sizes,
            )
            valid = np.asarray(
                [value for value in annual_correlations if np.isfinite(value)]
            )

            rows.append(
                {
                    "predictor": predictor,
                    "outcome": outcome,
                    "meta_rho": meta_correlation,
                    "meta_p": meta_p,
                    "n_windows": n_windows,
                    "n_pos": int(np.sum(valid > 0)),
                    "n_neg": int(np.sum(valid < 0)),
                    "median_rho": (
                        float(np.median(valid))
                        if len(valid) > 0
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


def partial_cross_meta_normal(
    frame: pd.DataFrame,
    predictor: str,
    given: str = "WithinOutgoing",
    outcome: str = "CrossOutgoing",
) -> Tuple[float, float, int]:
    annual_correlations: List[float] = []
    annual_sizes: List[int] = []

    for _, subset in frame.groupby("window", sort=True):
        correlation, _, n_communities = partial_spearman(
            subset[predictor],
            subset[outcome],
            [subset[given]],
        )
        annual_correlations.append(correlation)
        annual_sizes.append(n_communities)

    meta_correlation, meta_p, _, _ = fisher_z_combine(
        annual_correlations,
        annual_sizes,
        number_of_controls=1,
    )

    return (
        meta_correlation,
        meta_p,
        int(np.sum(np.isfinite(annual_correlations))),
    )


def structural_on_matrix(
    structural_matrix: np.ndarray,
    named_communities: pd.Series,
    config: AnalysisConfig,
) -> pd.DataFrame:
    summary = community_structural_summaries(
        structural_matrix,
        named_communities,
    )
    summary["CommBetweenness"] = community_betweenness(
        structural_matrix,
        named_communities,
        config.graph_distance_epsilon,
    )
    return summary


def market_mode_retained_matrix(
    returns: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (
        _,
        residual,
        eigenvalues,
        eigenvectors,
        informative,
        _,
        _,
    ) = mg3_filter(returns)

    retained = residual + eigenvalues[0] * np.outer(
        eigenvectors[:, 0],
        eigenvectors[:, 0],
    )
    np.fill_diagonal(retained, 0.0)

    return retained, eigenvalues, eigenvectors, informative


def market_mode_retained_embedding(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    informative: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [np.sqrt(max(float(eigenvalues[0]), 0.0)) * eigenvectors[:, 0]]
        + [
            np.sqrt(max(float(eigenvalues[index]), 0.0))
            * eigenvectors[:, index]
            for index in np.where(informative)[0]
        ]
    )


def regularized_partial_correlation_matrix(
    returns: pd.DataFrame,
    tag: str,
    diagnostic_rows: List[Dict[str, Any]],
) -> np.ndarray:
    data = returns.to_numpy(dtype=float)
    data = data - data.mean(axis=0, keepdims=True)
    standard_deviation = data.std(axis=0, ddof=0, keepdims=True)
    standard_deviation[standard_deviation == 0] = 1.0
    standardized = data / standard_deviation

    estimator = LedoitWolf().fit(standardized)

    try:
        precision = np.linalg.inv(estimator.covariance_)
        inverse_method = "inverse"
    except np.linalg.LinAlgError:
        precision = np.linalg.pinv(estimator.covariance_)
        inverse_method = "pseudoinverse"

    diagonal = np.sqrt(np.clip(np.diag(precision), 1e-12, None))
    partial_correlation = -precision / np.outer(diagonal, diagonal)
    np.fill_diagonal(partial_correlation, 0.0)
    partial_correlation = 0.5 * (
        partial_correlation + partial_correlation.T
    )

    diagnostic_rows.append(
        {
            "tag": tag,
            "inverse_method": inverse_method,
            "lw_shrinkage": float(estimator.shrinkage_),
            "cov_condition_number": float(np.linalg.cond(estimator.covariance_)),
        }
    )

    return partial_correlation


def detect_from_embedding(
    embedding: np.ndarray,
    asset_names: Sequence[str],
    market_prefix: str,
    number_of_communities: int,
    seed: int,
    config: AnalysisConfig,
) -> pd.Series:
    labels = KMeans(
        n_clusters=number_of_communities,
        n_init=config.kmeans_n_init,
        random_state=seed,
    ).fit_predict(embedding)

    return pd.Series(
        [f"{market_prefix}_C{label}" for label in labels],
        index=list(asset_names),
        name="Community",
    )


def detect_partial_correlation_communities(
    partial_correlation: np.ndarray,
    asset_names: Sequence[str],
    market_prefix: str,
    number_of_communities: int,
    seed: int,
    config: AnalysisConfig,
) -> Tuple[pd.Series, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(partial_correlation)
    selected = np.argsort(np.abs(eigenvalues))[::-1][:number_of_communities]
    embedding = np.column_stack(
        [
            np.sqrt(abs(float(eigenvalues[index]))) * eigenvectors[:, index]
            for index in selected
        ]
    )

    named = detect_from_embedding(
        embedding,
        asset_names,
        market_prefix,
        number_of_communities,
        seed,
        config,
    )

    return named, embedding


def fit_propagation_variant(
    baseline_window: Dict[str, Any],
    alpha: float,
    max_lag: int,
    use_interval_baselines: bool,
    config: AnalysisConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    equity_events = shock_events(
        standardize_equity(baseline_window["ret_eq"]),
        alpha,
    )
    crypto_events = shock_events(
        standardize_crypto(
            baseline_window["ret_cr"],
            baseline_window["interval"]["ExtendedInterval"],
        ),
        alpha,
    )

    groups, group_counts, group_proportions = compute_community_event_matrices(
        equity_events,
        crypto_events,
        baseline_window["named_eq"],
        baseline_window["named_cr"],
    )

    n_observations_after_lag = len(group_counts) - max_lag
    adjusted_train0 = max(
        min(
            config.train0,
            n_observations_after_lag
            - config.n_folds * config.validation_block,
        ),
        5 * max_lag,
    )

    excitation, _, _, fit_info = fit_community_layer(
        group_counts=group_counts,
        group_proportions=group_proportions,
        groups=groups,
        max_lag=max_lag,
        eta_grid=config.eta_grid,
        train0=adjusted_train0,
        validation_block=config.validation_block,
        n_folds=config.n_folds,
        maxiter_cv=config.maxiter_cv,
        maxiter_final=config.maxiter_final,
        extended_interval=baseline_window["interval"]["ExtendedInterval"],
        use_interval_baselines=use_interval_baselines,
    )

    summary = baseline_window["struct"].join(
        community_contagion_summaries(excitation)
    )
    summary["window"] = baseline_window["year"]
    summary["market"] = [market_of(index) for index in summary.index]

    return summary, fit_info.assign(window=baseline_window["year"]), excitation


def run_tier1_robustness(
    baseline_results: Mapping[int, Dict[str, Any]],
    output_directory: Path,
    config: AnalysisConfig,
) -> Dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)

    specifications = [
        ("baseline", config.shock_alpha, config.max_lag, True),
        ("no_interval_control", config.shock_alpha, config.max_lag, False),
        ("alpha_001", 0.01, config.max_lag, True),
        ("alpha_010", 0.10, config.max_lag, True),
        ("L3", config.shock_alpha, 3, True),
        ("L10", config.shock_alpha, 10, True),
    ]

    summary_frames: Dict[str, pd.DataFrame] = {
        "baseline": pd.concat(
            [
                baseline_results[year]["summary"].set_index("Community")
                for year in config.years
            ]
        )
    }

    fit_tables: List[pd.DataFrame] = [
        baseline_results[year]["fit_info"].assign(spec="baseline")
        for year in config.years
    ]

    mass_rows: List[Dict[str, Any]] = []

    for year in config.years:
        mass = excitation_partition_single(baseline_results[year]["A"])
        mass.update({"spec": "baseline", "window": int(year)})
        mass_rows.append(mass)

    for specification, alpha, max_lag, use_interval in specifications[1:]:
        logger.info("Tier 1 robustness: %s", specification)
        annual_summaries: List[pd.DataFrame] = []

        for year in config.years:
            summary, fit_info, excitation = fit_propagation_variant(
                baseline_results[year],
                alpha,
                max_lag,
                use_interval,
                config,
            )
            annual_summaries.append(summary)
            fit_tables.append(fit_info.assign(spec=specification))

            mass = excitation_partition_single(excitation)
            mass.update({"spec": specification, "window": int(year)})
            mass_rows.append(mass)

        summary_frames[specification] = pd.concat(annual_summaries)

    ordinary_tables: List[pd.DataFrame] = []
    conditional_tables: List[pd.DataFrame] = []

    for specification, _, _, _ in specifications:
        frame = summary_frames[specification]
        ordinary = crosslayer_meta_normal(
            frame,
            config.primary_structural_predictors,
            config.propagation_outcomes,
        )
        ordinary["q_bh_six"] = benjamini_hochberg_keep_nan(
            ordinary["meta_p"]
        )
        ordinary.insert(0, "spec", specification)
        ordinary_tables.append(ordinary)

        conditional_rows: List[Dict[str, Any]] = []

        for predictor in config.primary_structural_predictors:
            correlation, p_value, n_windows = partial_cross_meta_normal(
                frame,
                predictor,
            )
            conditional_rows.append(
                {
                    "spec": specification,
                    "predictor": predictor,
                    "partial_meta_rho": correlation,
                    "partial_meta_p": p_value,
                    "n_windows": n_windows,
                }
            )

        conditional = pd.DataFrame(conditional_rows)
        conditional["q_bh_three"] = benjamini_hochberg_keep_nan(
            conditional["partial_meta_p"]
        )
        conditional_tables.append(conditional)

        frame.reset_index().to_csv(
            output_directory / f"tier1_summary_{specification}.csv",
            index=False,
        )

    ordinary_table = pd.concat(ordinary_tables, ignore_index=True)
    conditional_table = pd.concat(conditional_tables, ignore_index=True)
    fit_table = pd.concat(fit_tables, ignore_index=True)
    mass_table = pd.DataFrame(mass_rows)

    ordinary_table.to_csv(output_directory / "tier1_ordinary.csv", index=False)
    conditional_table.to_csv(
        output_directory / "tier1_conditional.csv",
        index=False,
    )
    fit_table.to_csv(
        output_directory / "tier1_fit_diagnostics.csv",
        index=False,
    )
    mass_table.to_csv(
        output_directory / "tier1_excitation_mass.csv",
        index=False,
    )

    return {
        "summary_frames": summary_frames,
        "ordinary": ordinary_table,
        "conditional": conditional_table,
        "fit_diagnostics": fit_table,
        "excitation_mass": mass_table,
    }


def run_tier2_robustness(
    baseline_results: Mapping[int, Dict[str, Any]],
    output_directory: Path,
    config: AnalysisConfig,
) -> Dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)

    partial_correlation_diagnostics: List[Dict[str, Any]] = []
    concordance_rows: List[Dict[str, Any]] = []

    baseline_frame = pd.concat(
        [
            baseline_results[year]["summary"].set_index("Community")
            for year in config.years
        ]
    )

    summary_frames: Dict[str, pd.DataFrame] = {
        "MG3_baseline": baseline_frame
    }

    # Fixed-community structural comparisons
    for network in ("MM", "PC"):
        annual_summaries: List[pd.DataFrame] = []

        for year in config.years:
            baseline_window = baseline_results[year]
            structural_parts: List[pd.DataFrame] = []

            for market in ("EQ", "CR"):
                lower = market.lower()
                returns = baseline_window[f"ret_{lower}"]
                named = baseline_window[f"named_{lower}"]

                if network == "MM":
                    alternative_matrix = market_mode_retained_matrix(returns)[0]
                else:
                    alternative_matrix = regularized_partial_correlation_matrix(
                        returns,
                        f"PC_fixed_{market}_{year}",
                        partial_correlation_diagnostics,
                    )

                alternative_structure = structural_on_matrix(
                    alternative_matrix,
                    named,
                    config,
                )
                structural_parts.append(alternative_structure)

                baseline_structure = baseline_window["struct"].loc[
                    alternative_structure.index
                ]

                for predictor in config.primary_structural_predictors:
                    correlation, _, n_communities = safe_spearman(
                        baseline_structure[predictor],
                        alternative_structure[predictor],
                    )
                    concordance_rows.append(
                        {
                            "network": network,
                            "window": int(year),
                            "market": market,
                            "predictor": predictor,
                            "rho": correlation,
                            "n_communities": n_communities,
                        }
                    )

            alternative_structure = pd.concat(structural_parts)
            summary = alternative_structure.join(
                community_contagion_summaries(baseline_window["A"])
            )
            summary["window"] = int(year)
            summary["market"] = [market_of(index) for index in summary.index]
            annual_summaries.append(summary)

        summary_frames[f"{network}_fixed"] = pd.concat(annual_summaries)

    # Full reconstruction under alternative dependence representations
    partition_rows: List[Dict[str, Any]] = []
    excitation_mass_rows: List[Dict[str, Any]] = []
    fit_tables: List[pd.DataFrame] = []
    community_size_rows: List[Dict[str, Any]] = []

    for network in ("MM", "PC"):
        logger.info("Tier 2 full reconstruction: %s", network)
        annual_summaries = []

        for year in config.years:
            baseline_window = baseline_results[year]
            alternative_named: Dict[str, pd.Series] = {}
            alternative_matrices: Dict[str, np.ndarray] = {}

            for market in ("EQ", "CR"):
                lower = market.lower()
                returns = baseline_window[f"ret_{lower}"]
                baseline_named = baseline_window[f"named_{lower}"]
                number_of_communities = int(baseline_named.nunique())

                if network == "MM":
                    (
                        alternative_matrix,
                        eigenvalues,
                        eigenvectors,
                        informative,
                    ) = market_mode_retained_matrix(returns)
                    embedding = market_mode_retained_embedding(
                        eigenvalues,
                        eigenvectors,
                        informative,
                    )
                    named = detect_from_embedding(
                        embedding,
                        returns.columns.tolist(),
                        market,
                        number_of_communities,
                        config.kmeans_seed,
                        config,
                    )
                else:
                    alternative_matrix = regularized_partial_correlation_matrix(
                        returns,
                        f"PC_full_{market}_{year}",
                        partial_correlation_diagnostics,
                    )
                    named, _ = detect_partial_correlation_communities(
                        alternative_matrix,
                        returns.columns.tolist(),
                        market,
                        number_of_communities,
                        config.kmeans_seed,
                        config,
                    )

                alternative_named[market] = named
                alternative_matrices[market] = alternative_matrix

                partition_rows.append(
                    {
                        "network": network,
                        "window": int(year),
                        "market": market,
                        "ARI": adjusted_rand_score(
                            baseline_named.values,
                            named.values,
                        ),
                        "NMI": normalized_mutual_info_score(
                            baseline_named.values,
                            named.values,
                        ),
                        "K": number_of_communities,
                        "largest_share": float(
                            named.value_counts().max() / len(named)
                        ),
                        "median_size": float(named.value_counts().median()),
                    }
                )

                for community, size in named.value_counts().items():
                    community_size_rows.append(
                        {
                            "network": network,
                            "window": int(year),
                            "market": market,
                            "community": community,
                            "size": int(size),
                            "share": float(size / len(named)),
                        }
                    )

            groups, group_counts, group_proportions = compute_community_event_matrices(
                baseline_window["E_eq"],
                baseline_window["E_cr"],
                alternative_named["EQ"],
                alternative_named["CR"],
            )

            excitation, _, _, fit_info = fit_community_layer(
                group_counts=group_counts,
                group_proportions=group_proportions,
                groups=groups,
                max_lag=config.max_lag,
                eta_grid=config.eta_grid,
                train0=config.train0,
                validation_block=config.validation_block,
                n_folds=config.n_folds,
                maxiter_cv=config.maxiter_cv,
                maxiter_final=config.maxiter_final,
                extended_interval=baseline_window["interval"]["ExtendedInterval"],
                use_interval_baselines=True,
            )

            fit_tables.append(
                fit_info.assign(network=network, window=int(year))
            )

            mass = excitation_partition_single(excitation)
            mass.update({"network": network, "window": int(year)})
            excitation_mass_rows.append(mass)

            alternative_structure = pd.concat(
                [
                    structural_on_matrix(
                        alternative_matrices["EQ"],
                        alternative_named["EQ"],
                        config,
                    ),
                    structural_on_matrix(
                        alternative_matrices["CR"],
                        alternative_named["CR"],
                        config,
                    ),
                ]
            )

            summary = alternative_structure.join(
                community_contagion_summaries(excitation)
            )
            summary["window"] = int(year)
            summary["market"] = [market_of(index) for index in summary.index]
            annual_summaries.append(summary)

        summary_frames[f"{network}_full"] = pd.concat(annual_summaries)

    strength_tables: List[pd.DataFrame] = []
    betweenness_tables: List[pd.DataFrame] = []
    conditional_tables: List[pd.DataFrame] = []

    for network_name, frame in summary_frames.items():
        strength = crosslayer_meta_normal(
            frame,
            config.strength_predictors,
            config.propagation_outcomes,
        )
        strength["q_bh_four"] = benjamini_hochberg_keep_nan(strength["meta_p"])
        strength.insert(0, "network", network_name)
        strength_tables.append(strength)

        betweenness = crosslayer_meta_normal(
            frame,
            ["CommBetweenness"],
            config.propagation_outcomes,
        )
        betweenness.insert(0, "network", network_name)
        betweenness_tables.append(betweenness)

        conditional_rows: List[Dict[str, Any]] = []

        for predictor in config.primary_structural_predictors:
            correlation, p_value, n_windows = partial_cross_meta_normal(
                frame,
                predictor,
            )
            conditional_rows.append(
                {
                    "network": network_name,
                    "predictor": predictor,
                    "partial_meta_rho": correlation,
                    "partial_meta_p": p_value,
                    "n_windows": n_windows,
                }
            )

        conditional = pd.DataFrame(conditional_rows)
        conditional["q_bh_three"] = benjamini_hochberg_keep_nan(
            conditional["partial_meta_p"]
        )
        conditional_tables.append(conditional)

        frame.reset_index().to_csv(
            output_directory / f"tier2_summary_{network_name}.csv",
            index=False,
        )

    strength_table = pd.concat(strength_tables, ignore_index=True)
    betweenness_table = pd.concat(betweenness_tables, ignore_index=True)
    conditional_table = pd.concat(conditional_tables, ignore_index=True)
    concordance_table = pd.DataFrame(concordance_rows)
    partition_table = pd.DataFrame(partition_rows)
    excitation_mass_table = pd.DataFrame(excitation_mass_rows)
    fit_table = pd.concat(fit_tables, ignore_index=True)
    community_size_table = pd.DataFrame(community_size_rows)
    partial_diagnostic_table = pd.DataFrame(partial_correlation_diagnostics)

    strength_table.to_csv(output_directory / "tier2_strength.csv", index=False)
    betweenness_table.to_csv(
        output_directory / "tier2_commbet_diag.csv",
        index=False,
    )
    conditional_table.to_csv(
        output_directory / "tier2_conditional.csv",
        index=False,
    )
    concordance_table.to_csv(
        output_directory / "tier2_predictor_concordance.csv",
        index=False,
    )
    partition_table.to_csv(
        output_directory / "tier2_partition_agreement.csv",
        index=False,
    )
    excitation_mass_table.to_csv(
        output_directory / "tier2_excitation_mass.csv",
        index=False,
    )
    fit_table.to_csv(
        output_directory / "tier2_fit_diagnostics.csv",
        index=False,
    )
    community_size_table.to_csv(
        output_directory / "tier2_community_sizes.csv",
        index=False,
    )
    partial_diagnostic_table.to_csv(
        output_directory / "tier2_pc_diagnostics.csv",
        index=False,
    )

    return {
        "summary_frames": summary_frames,
        "strength": strength_table,
        "betweenness": betweenness_table,
        "conditional": conditional_table,
        "predictor_concordance": concordance_table,
        "partition_agreement": partition_table,
        "excitation_mass": excitation_mass_table,
        "fit_diagnostics": fit_table,
        "community_sizes": community_size_table,
        "partial_correlation_diagnostics": partial_diagnostic_table,
    }


def run_seed_stability_analysis(
    baseline_results: Mapping[int, Dict[str, Any]],
    output_directory: Path,
    config: AnalysisConfig,
) -> Dict[str, pd.DataFrame]:
    output_directory.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for year in config.years:
        for market in ("EQ", "CR"):
            lower = market.lower()
            baseline_window = baseline_results[year]
            residual = baseline_window[f"B_{lower}"]
            embedding = baseline_window[f"Phi_{lower}"]
            baseline_named = baseline_window[f"named_{lower}"]
            asset_names = baseline_named.index.tolist()
            number_of_communities = int(baseline_named.nunique())
            baseline_labels = (
                baseline_named.str.replace(f"{market}_C", "", regex=False)
                .astype(int)
                .to_numpy()
            )
            baseline_structure = community_structural_summaries(
                residual,
                baseline_named,
            )

            for seed in range(config.seed_stability_repetitions):
                labels = KMeans(
                    n_clusters=number_of_communities,
                    n_init=config.kmeans_n_init,
                    random_state=seed,
                ).fit_predict(embedding)

                named = pd.Series(
                    [f"{market}_C{label}" for label in labels],
                    index=asset_names,
                    name="Community",
                )

                seed_structure = community_structural_summaries(residual, named)

                seed_within = named.map(seed_structure["WithinMean"])
                seed_between = named.map(seed_structure["BetweenMean"])
                baseline_within = baseline_named.map(
                    baseline_structure["WithinMean"]
                )
                baseline_between = baseline_named.map(
                    baseline_structure["BetweenMean"]
                )

                within_correlation, _, _ = safe_spearman(
                    baseline_within,
                    seed_within,
                )
                between_correlation, _, _ = safe_spearman(
                    baseline_between,
                    seed_between,
                )

                rows.append(
                    {
                        "window": int(year),
                        "market": market,
                        "seed": seed,
                        "ARI": adjusted_rand_score(baseline_labels, labels),
                        "NMI": normalized_mutual_info_score(
                            baseline_labels,
                            labels,
                        ),
                        "rho_WithinMean": within_correlation,
                        "rho_BetweenMean": between_correlation,
                    }
                )

    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby(["window", "market"])
        .agg(
            ARI_min=("ARI", "min"),
            ARI_med=("ARI", "median"),
            ARI_max=("ARI", "max"),
            NMI_min=("NMI", "min"),
            NMI_med=("NMI", "median"),
            NMI_max=("NMI", "max"),
            rWM_min=("rho_WithinMean", "min"),
            rWM_med=("rho_WithinMean", "median"),
            rWM_max=("rho_WithinMean", "max"),
            rBM_min=("rho_BetweenMean", "min"),
            rBM_med=("rho_BetweenMean", "median"),
            rBM_max=("rho_BetweenMean", "max"),
        )
        .reset_index()
    )

    raw.to_csv(output_directory / "seed_stability_raw.csv", index=False)
    summary.to_csv(output_directory / "seed_stability_summary.csv", index=False)

    return {"raw": raw, "summary": summary}


def plot_robustness_summary(
    tier1_table: pd.DataFrame,
    tier2_table: pd.DataFrame,
    output_directory: Path,
    config: AnalysisConfig,
) -> None:

    output_directory.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.0, 4.8),
        constrained_layout=True,
    )

    axis_a, axis_b = axes

    tier2_specifications = [
        "MG3_baseline",
        "MM_fixed",
        "PC_fixed",
        "MM_full",
        "PC_full",
    ]

    tier2_labels = [
        "MG3",
        "Market mode\nfixed communities",
        "Partial corr.\nfixed communities",
        "Market mode\nreconstructed",
        "Partial corr.\nreconstructed",
    ]

    predictor_labels = {
        "WithinMean": r"$S_g^{\mathrm{in}}$",
        "BetweenMean": r"$S_g^{\mathrm{out}}$",
    }

    positions_a = np.arange(len(tier2_specifications))
    horizontal_offset = 0.10

    for predictor_index, predictor in enumerate(config.strength_predictors):
        values = []
        for specification in tier2_specifications:
            selected = tier2_table.loc[
                (tier2_table["network"] == specification)
                & (tier2_table["predictor"] == predictor)
                & (tier2_table["outcome"] == "WithinOutgoing"),
                "meta_rho",
            ]
            if len(selected) != 1:
                raise ValueError(
                    "Expected exactly one Tier 2 result for "
                    f"{specification}, {predictor}, WithinOutgoing."
                )
            values.append(float(selected.iloc[0]))

        offset = -horizontal_offset if predictor_index == 0 else horizontal_offset
        axis_a.scatter(
            positions_a + offset,
            values,
            s=65,
            label=predictor_labels[predictor],
            zorder=3,
        )

    axis_a.axhline(0.0, color="grey", linestyle="--", linewidth=0.9)
    axis_a.set_xticks(positions_a)
    axis_a.set_xticklabels(tier2_labels, rotation=25, ha="right")
    axis_a.set_ylabel(r"Meta Spearman correlation $\rho_{\mathrm{meta}}$")
    axis_a.text(0.02, 0.02, "A", transform=axis_a.transAxes, ha="left", va="bottom", fontweight="bold")
    axis_a.legend(loc="upper right", frameon=False)

    tier1_specifications = [
        "baseline",
        "no_interval_control",
        "alpha_001",
        "alpha_010",
        "L3",
        "L10",
    ]

    tier1_labels = [
        "Baseline",
        "No interval\ncontrol",
        r"$\alpha=0.01$",
        r"$\alpha=0.10$",
        r"$L=3$",
        r"$L=10$",
    ]

    positions_b = np.arange(len(tier1_specifications))

    for predictor_index, predictor in enumerate(config.strength_predictors):
        values = []
        for specification in tier1_specifications:
            selected = tier1_table.loc[
                (tier1_table["spec"] == specification)
                & (tier1_table["predictor"] == predictor)
                & (tier1_table["outcome"] == "WithinOutgoing"),
                "meta_rho",
            ]
            if len(selected) != 1:
                raise ValueError(
                    "Expected exactly one Tier 1 result for "
                    f"{specification}, {predictor}, WithinOutgoing."
                )
            values.append(float(selected.iloc[0]))

        offset = -horizontal_offset if predictor_index == 0 else horizontal_offset
        axis_b.scatter(
            positions_b + offset,
            values,
            s=65,
            zorder=3,
        )

    axis_b.axhline(0.0, color="grey", linestyle="--", linewidth=0.9)
    axis_b.set_xticks(positions_b)
    axis_b.set_xticklabels(tier1_labels, rotation=25, ha="right")
    axis_b.set_ylabel(r"Meta Spearman correlation $\rho_{\mathrm{meta}}$")
    axis_b.text(0.02, 0.02, "B", transform=axis_b.transAxes, ha="left", va="bottom", fontweight="bold")

    all_values = np.concatenate(
        [
            tier1_table.loc[
                (tier1_table["predictor"].isin(config.strength_predictors))
                & (tier1_table["outcome"] == "WithinOutgoing"),
                "meta_rho",
            ].to_numpy(dtype=float),
            tier2_table.loc[
                (tier2_table["predictor"].isin(config.strength_predictors))
                & (tier2_table["outcome"] == "WithinOutgoing"),
                "meta_rho",
            ].to_numpy(dtype=float),
        ]
    )

    finite_values = all_values[np.isfinite(all_values)]
    limit = max(0.6, float(np.max(np.abs(finite_values))) + 0.08)
    axis_a.set_ylim(-limit, limit)
    axis_b.set_ylim(-limit, limit)

    figure.savefig(
        output_directory / "Figure4.png",
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        output_directory / "Figure4.pdf",
        bbox_inches="tight",
    )

    if config.show_figures:
        plt.show()

    plt.close(figure)

    
def assemble_and_save_window_results(
    window_results: Mapping[int, Dict[str, Any]],
    output_directory: Path,
    config: AnalysisConfig,
    save_intermediate: bool,
    save_figures: bool = False,
) -> Dict[str, Any]:
     summaries = [window_results[year]["summary"] for year in config.years]
     diagnostics = [window_results[year]["diag"] for year in config.years]
     memberships = [window_results[year]["membership"] for year in config.years]
     fits = [window_results[year]["fit_info"] for year in config.years]

     combined_summary = pd.concat(summaries, ignore_index=True)
     diagnostic_table = pd.DataFrame(diagnostics)
     membership_table = pd.concat(memberships, ignore_index=True)
     fit_table = pd.concat(fits, ignore_index=True)

     if save_intermediate:
         for year in config.years:
             save_annual_intermediate_outputs(
                 window_results[year],
                 output_directory / "annual_windows",
             )

     saved_tables = save_baseline_results(
         window_results,
         combined_summary,
         diagnostic_table,
         membership_table,
         fit_table,
         output_directory,
         config,
     )

     if save_figures:

        figure_directory = (
            output_directory
            / "figures"
        )
    
        figure_directory.mkdir(
            parents=True,
            exist_ok=True,
        )
    
        save_baseline_figures(
            combined_summary,
            window_results,
            figure_directory,
            config,
        )

     return {
         "window_results": dict(window_results),
         "community_summary": combined_summary,
         "window_diagnostics": diagnostic_table,
         "memberships": membership_table,
         "fit_diagnostics": fit_table,
         **saved_tables,
     }
 
    
#----------------------------SUMMARY-------------------------------

def create_key_results_table(
    aggregate: pd.DataFrame,
) -> pd.DataFrame:
    key = aggregate.loc[
        aggregate["predictor"].isin(
            ["WithinMean", "BetweenMean", "CommBetweenness"]
        ),
        [
            "predictor",
            "outcome",
            "median_rho",
            "frac_windows_positive",
            "n_windows",
            "meta_rho",
            "perm_p",
            "perm_q_bh",
            "pooled_rho_descr",
        ],
    ].copy()

    key["conclusion"] = np.where(
        key["perm_q_bh"] < 0.05,
        "FDR-significant",
        "Not FDR-significant",
    )

    return key


def write_key_conclusions(
    key_results: pd.DataFrame,
    excitation_partition: pd.DataFrame,
    window_diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    def row(predictor: str, outcome: str) -> pd.Series:
        return key_results.loc[
            (key_results["predictor"] == predictor)
            & (key_results["outcome"] == outcome)
        ].iloc[0]

    within_internal = row("WithinMean", "WithinOutgoing")
    within_external = row("BetweenMean", "WithinOutgoing")
    cross_internal = row("WithinMean", "CrossOutgoing")
    cross_external = row("BetweenMean", "CrossOutgoing")
    betweenness_within = row("CommBetweenness", "WithinOutgoing")

    zero_variance_rows = window_diagnostics.loc[
        (window_diagnostics["dropped_zero_or_invalid_eq"].fillna("") != "")
        | (window_diagnostics["dropped_zero_or_invalid_cr"].fillna("") != "")
    ]

    lines = [
        "UPDATED 2020-2025 ANALYSIS SUMMARY",
        "",
        (
            "Internal cohesion and external coupling remain positively associated "
            "with within-market outgoing excitation. Their Fisher-z-combined "
            f"correlations are {within_internal['meta_rho']:+.3f} and "
            f"{within_external['meta_rho']:+.3f}, with permutation BH q-values "
            f"of {within_internal['perm_q_bh']:.4f} and "
            f"{within_external['perm_q_bh']:.4f}."
        ),
        (
            "Community betweenness remains weaker. Its within-market combined "
            f"correlation is {betweenness_within['meta_rho']:+.3f}, with "
            f"permutation BH q={betweenness_within['perm_q_bh']:.4f}."
        ),
        (
            "The MG3 baseline now shows positive cross-market associations for "
            "both strength measures. Their combined correlations are "
            f"{cross_internal['meta_rho']:+.3f} and "
            f"{cross_external['meta_rho']:+.3f}."
        ),
        (
            "The within-market between-community share is the largest excitation "
            "component in every window. It ranges from "
            f"{excitation_partition['WithinShare'].min():.3f} to "
            f"{excitation_partition['WithinShare'].max():.3f}."
        ),
        "",
        "Annual zero-variance or invalid-return removals:",
    ]

    if zero_variance_rows.empty:
        lines.append("None")
    else:
        for _, diagnostic in zero_variance_rows.iterrows():
            lines.append(
                f"{int(diagnostic['window'])}: equity="
                f"{diagnostic['dropped_zero_or_invalid_eq'] or 'none'}, crypto="
                f"{diagnostic['dropped_zero_or_invalid_cr'] or 'none'}"
            )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def apply_excel_header_format(
    writer: pd.ExcelWriter,
    sheet_name: str,
    frame: pd.DataFrame,
) -> None:
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]
    header_format = workbook.add_format(
        {
            "bold": True,
            "font_color": "white",
            "bg_color": "#2F75B5",
            "align": "center",
            "valign": "vcenter",
            "border": 1,
            "text_wrap": True,
        }
    )

    worksheet.set_row(0, 26, header_format)
    worksheet.freeze_panes(1, 0)
    worksheet.hide_gridlines(2)

    if frame.shape[1] > 0:
        worksheet.autofilter(0, 0, len(frame), frame.shape[1] - 1)

    for column_position, column in enumerate(frame.columns):
        if len(frame) > 0:
            quantile_length = int(
                frame[column].astype(str).str.len().quantile(0.95)
            )
        else:
            quantile_length = 0

        width = min(max(len(str(column)) + 2, quantile_length + 2, 12), 38)
        worksheet.set_column(column_position, column_position, width)


def write_summary_workbook(
    output_path: Path,
    input_diagnostics: pd.DataFrame,
    baseline: Dict[str, Any],
    inference: Dict[str, pd.DataFrame],
    tier1: Optional[Dict[str, Any]],
    tier2: Optional[Dict[str, Any]],
    seed_stability: Optional[Dict[str, pd.DataFrame]],
) -> None:
    with pd.ExcelWriter(
        output_path,
        engine="xlsxwriter",
        datetime_format="yyyy-mm-dd",
        date_format="yyyy-mm-dd",
    ) as writer:
        sheet_frames: Dict[str, pd.DataFrame] = {
            "Input Diagnostics": input_diagnostics,
            "Window Diagnostics": baseline["window_diagnostics"],
            "Community Structure": baseline["community_structure"],
            "Excitation Partition": baseline["excitation_partition"],
            "Main Inference": inference["aggregate"],
            "Annual Correlations": inference["per_window"],
            "Cross Diagnostics": inference["cross_market_diagnostics"],
            "Market Split": inference["market_window_summary"],
            "Market Partial": inference["partial_market_summary"],
            "Conditional Cross": inference["cross_conditional"],
            "Fit Diagnostics": baseline["fit_diagnostics"],
        }

        if tier1 is not None:
            sheet_frames["Tier 1"] = tier1["ordinary"]
            sheet_frames["Tier 1 Conditional"] = tier1["conditional"]

        if tier2 is not None:
            sheet_frames["Tier 2"] = tier2["strength"]
            sheet_frames["Tier 2 Betweenness"] = tier2["betweenness"]
            sheet_frames["Tier 2 Conditional"] = tier2["conditional"]

        if seed_stability is not None:
            sheet_frames["Seed Stability"] = seed_stability["summary"]

        for sheet_name, frame in sheet_frames.items():
            safe_name = sheet_name[:31]
            frame.to_excel(writer, sheet_name=safe_name, index=False)
            apply_excel_header_format(writer, safe_name, frame)


def write_latex_tables(
    output_path: Path,
    baseline: Dict[str, Any],
    inference: Dict[str, pd.DataFrame],
    tier1: Optional[Dict[str, Any]],
    tier2: Optional[Dict[str, Any]],
) -> None:
    tables: List[str] = []

    tables.append("% Primary structure-propagation results\n")
    tables.append(
        inference["aggregate"].loc[
            inference["aggregate"]["family"] == "inferential",
            [
                "predictor",
                "outcome",
                "median_rho",
                "frac_windows_positive",
                "meta_rho",
                "perm_p",
                "perm_q_bh",
                "pooled_rho_descr",
            ],
        ].to_latex(index=False, float_format="%.3f", escape=False)
    )

    tables.append("\n% Excitation partition by annual window\n")
    tables.append(
        baseline["excitation_partition"].loc[
            :,
            [
                "window",
                "Total",
                "SelfShare",
                "WithinShare",
                "CrossShare",
                "EQtoCR_share",
                "CRtoEQ_share",
                "MeanLag",
            ],
        ].to_latex(index=False, float_format="%.3f", escape=False)
    )

    tables.append("\n% Community structure by annual window\n")
    tables.append(
        baseline["community_structure"].to_latex(
            index=False,
            float_format="%.3f",
            escape=False,
        )
    )

    if tier1 is not None:
        tables.append("\n% Propagation-specification robustness\n")
        tables.append(
            tier1["ordinary"].to_latex(
                index=False,
                float_format="%.3f",
                escape=False,
            )
        )

    if tier2 is not None:
        tables.append("\n% Dependence-representation robustness\n")
        tables.append(
            tier2["strength"].to_latex(
                index=False,
                float_format="%.3f",
                escape=False,
            )
        )

    output_path.write_text("\n".join(tables), encoding="utf-8")


#---------------------------------------------------------------------------------


def input_diagnostic_table(
    equity_path: Path,
    crypto_path: Path,
    equity_prices: pd.DataFrame,
    crypto_prices: pd.DataFrame,
    interval_controls: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "input": "Equity",
            "file": str(equity_path.resolve()),
            "rows": len(equity_prices),
            "assets": equity_prices.shape[1],
            "first_date": equity_prices.index.min(),
            "last_date": equity_prices.index.max(),
            "missing_cells": int(equity_prices.isna().sum().sum()),
            "nonpositive_cells": int((equity_prices <= 0).sum().sum()),
            "weekend_rows": int((equity_prices.index.dayofweek >= 5).sum()),
            "extended_intervals": int(interval_controls["ExtendedInterval"].sum()),
        },
        {
            "input": "Cryptocurrency",
            "file": str(crypto_path.resolve()),
            "rows": len(crypto_prices),
            "assets": crypto_prices.shape[1],
            "first_date": crypto_prices.index.min(),
            "last_date": crypto_prices.index.max(),
            "missing_cells": int(crypto_prices.isna().sum().sum()),
            "nonpositive_cells": int((crypto_prices <= 0).sum().sum()),
            "weekend_rows": int((crypto_prices.index.dayofweek >= 5).sum()),
            "extended_intervals": int(interval_controls["ExtendedInterval"].sum()),
        },
    ]

    return pd.DataFrame(rows)


def copy_reproducibility_files(
    equity_path: Path,
    crypto_path: Path,
) -> Dict[str, str]:
    input_directory = OUTPUT_DIR / "inputs"
    code_directory = OUTPUT_DIR / "code"
    input_directory.mkdir(parents=True, exist_ok=True)
    code_directory.mkdir(parents=True, exist_ok=True)

    normalized_equity_path = input_directory / "sp500_top126_close_usd_aligned_final.csv"
    normalized_crypto_path = input_directory / "crypto_top126_close_usd_aligned_final.csv"

    equity_frame = read_price_matrix(equity_path)
    crypto_frame = read_price_matrix(crypto_path)
    equity_frame.to_csv(normalized_equity_path, date_format="%Y-%m-%d")
    crypto_frame.to_csv(normalized_crypto_path, date_format="%Y-%m-%d")

    script_destination = code_directory / Path(__file__).name
    script_destination.write_bytes(Path(__file__).read_bytes())

    requirements = "\n".join(
        [
            "numpy>=1.24",
            "pandas>=2.0",
            "scipy>=1.10",
            "scikit-learn>=1.3",
            "networkx>=3.1",
            "matplotlib>=3.7",
            "xlsxwriter>=3.1",
            "openpyxl>=3.1",
            "jinja2>=3.1",
        ]
    ) + "\n"
    requirements_path = code_directory / "requirements.txt"
    requirements_path.write_text(requirements, encoding="utf-8")

    return {
        "equity_input_copy": str(normalized_equity_path),
        "crypto_input_copy": str(normalized_crypto_path),
        "script_copy": str(script_destination),
        "requirements": str(requirements_path),
    }


def create_output_zip(output_directory: Path) -> Path:
    import zipfile

    zip_path = output_directory / "complete_analysis_outputs.zip"
    zip_path.unlink(missing_ok=True)

    with zipfile.ZipFile(
        zip_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in sorted(output_directory.rglob("*")):
            if not path.is_file() or path == zip_path:
                continue
            archive.write(path, arcname=str(path.relative_to(output_directory)))

    with zipfile.ZipFile(zip_path, "r") as archive:
        corrupted_member = archive.testzip()

    if corrupted_member is not None:
        raise RuntimeError(f"Corrupted ZIP member: {corrupted_member}")

    return zip_path


def run_complete_analysis() -> Dict[str, Any]:
    start_time = time.time()

    equity_path = resolve_input_path(
        EQUITY_FILE,
        "sp500_top126_close_usd_aligned_final*.csv",
    )
    crypto_path = resolve_input_path(
        CRYPTO_FILE,
        "crypto_top126_close_usd_aligned_final*.csv",
    )

    logger.info("Reading final aligned inputs")
    logger.info("Equity file: %s", equity_path)
    logger.info("Crypto file: %s", crypto_path)

    equity_prices, crypto_prices, interval_controls = load_and_validate_aligned_prices(
        equity_path,
        crypto_path,
        CONFIG,
    )

    input_diagnostics = input_diagnostic_table(
        equity_path,
        crypto_path,
        equity_prices,
        crypto_prices,
        interval_controls,
    )
    input_diagnostics.to_csv(OUTPUT_DIR / "input_diagnostics.csv", index=False)
    interval_controls.to_csv(OUTPUT_DIR / "session_interval_controls.csv")

    reproducibility_files = copy_reproducibility_files(
    equity_path,
    crypto_path,
    )

    logger.info(
        "Inputs validated: %s sessions, %s equities, %s cryptocurrencies",
        len(equity_prices),
        equity_prices.shape[1],
        crypto_prices.shape[1],
    )

    baseline = run_baseline_analysis(
        equity_prices,
        crypto_prices,
        BASELINE_DIR,
        CONFIG,
        save_intermediate=CONFIG.save_intermediate,
    )

    if CONFIG.run_primary_inference:
        logger.info("Running primary inference with %s permutations", CONFIG.permutations)
        inference = run_primary_inference_and_diagnostics(
            baseline["community_summary"],
            BASELINE_DIR,
            CONFIG,
            save_figure=True,
        )
    else:
        inference = {}

    tier1: Optional[Dict[str, Any]] = None
    tier2: Optional[Dict[str, Any]] = None
    seed_stability: Optional[Dict[str, pd.DataFrame]] = None

    if CONFIG.run_tier1:
        logger.info("Running Tier 1 propagation-specification robustness")
        tier1 = run_tier1_robustness(
            baseline["window_results"],
            TIER1_DIR,
            CONFIG,
        )

    if CONFIG.run_tier2:
        logger.info("Running Tier 2 dependence-representation robustness")
        tier2 = run_tier2_robustness(
            baseline["window_results"],
            TIER2_DIR,
            CONFIG,
        )

    if (
        tier1 is not None
        and tier2 is not None
    ):
    
        logger.info(
            "Generating Figure 4 robustness summary"
        )
    
        plot_robustness_summary(
            tier1_table=tier1["ordinary"],
            tier2_table=tier2["strength"],
            output_directory=ROBUSTNESS_FIGURE_DIR,
            config=CONFIG,
        )    


    if CONFIG.run_seed_stability:
        logger.info(
            "Running community seed stability with %s seeds",
            CONFIG.seed_stability_repetitions,
        )
        seed_stability = run_seed_stability_analysis(
            baseline["window_results"],
            SEED_DIR,
            CONFIG,
        )


    if inference:
        key_results = create_key_results_table(inference["aggregate"])
        key_results.to_csv(SUMMARY_DIR / "key_results.csv", index=False)

        write_key_conclusions(
            key_results,
            baseline["excitation_partition"],
            baseline["window_diagnostics"],
            SUMMARY_DIR / "key_conclusions.txt",
        )
    else:
        key_results = pd.DataFrame()

    if CONFIG.save_summary_workbook and inference:
        write_summary_workbook(
            SUMMARY_DIR / "updated_2020_2025_analysis_results.xlsx",
            input_diagnostics,
            baseline,
            inference,
            tier1,
            tier2,
            seed_stability,
        )

    if CONFIG.save_latex_tables and inference:
        write_latex_tables(
            SUMMARY_DIR / "updated_2020_2025_tables.tex",
            baseline,
            inference,
            tier1,
            tier2,
        )

    elapsed_seconds = time.time() - start_time

    manifest = {
        "started_from": {
            "equity_file": str(equity_path.resolve()),
            "crypto_file": str(crypto_path.resolve()),
        },
        "configuration": asdict(CONFIG),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": stats.__version__ if hasattr(stats, "__version__") else None,
        },
        "sample": {
            "aligned_price_sessions": int(len(equity_prices)),
            "equity_assets_input": int(equity_prices.shape[1]),
            "crypto_assets_input": int(crypto_prices.shape[1]),
            "annual_effective_assets": baseline["window_diagnostics"][
                ["window", "eq_assets", "cr_assets", "dropped_zero_or_invalid_eq", "dropped_zero_or_invalid_cr"]
            ].to_dict(orient="records"),
        },
        "blocks_run": {
            "primary_inference": bool(inference),
            "tier1": tier1 is not None,
            "tier2": tier2 is not None,
            "seed_stability": seed_stability is not None,
        },
        "reproducibility_files": reproducibility_files,
        "elapsed_seconds": elapsed_seconds,
        "completed_utc": pd.Timestamp.utcnow().isoformat(),
    }

    manifest_path = OUTPUT_DIR / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=json_ready),
        encoding="utf-8",
    )

    output_zip = create_output_zip(OUTPUT_DIR)

    logger.info("Complete analysis finished in %.1f minutes", elapsed_seconds / 60.0)
    logger.info("Output directory: %s", OUTPUT_DIR.resolve())
    logger.info("Output ZIP: %s", output_zip.resolve())

    if not key_results.empty:
        logger.info(
            "\n%s",
            key_results.to_string(index=False),
        )

    return {
        "config": CONFIG,
        "equity_prices": equity_prices,
        "crypto_prices": crypto_prices,
        "interval_controls": interval_controls,
        "input_diagnostics": input_diagnostics,
        "baseline": baseline,
        "inference": inference,
        "tier1": tier1,
        "tier2": tier2,
        "seed_stability": seed_stability,
        "key_results": key_results,
        "manifest": manifest,
        "output_zip": output_zip,
    }


# ------------------------ RUN -----------------------------------------------------

if __name__ == "__main__":
    analysis_outputs = run_complete_analysis()

    equity_prices = analysis_outputs["equity_prices"]
    crypto_prices = analysis_outputs["crypto_prices"]
    interval_controls = analysis_outputs["interval_controls"]
    baseline_results = analysis_outputs["baseline"]
    primary_inference_results = analysis_outputs["inference"]
    tier1_results = analysis_outputs["tier1"]
    tier2_results = analysis_outputs["tier2"]
    seed_stability_results = analysis_outputs["seed_stability"]
    key_results = analysis_outputs["key_results"]