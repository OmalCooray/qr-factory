#!/usr/bin/env python
"""Regime Detection Comparison Study.

Evaluates 8 regime detectors on synthetic (ground truth) and real XAUUSD data.
Outputs CSV tables and plots to runs/regime_comparison_<timestamp>/.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.regime.protocol import RegimeLabel
from src.regime.synthetic import generate_synthetic_data
from src.regime.adapter_hmm import HMMRegimeDetector
from src.regime.choppiness import ChoppinessDetector
from src.regime.hurst import HurstDetector
from src.regime.bbw_detector import BBWidthDetector
from src.regime.gmm_detector import GMMDetector
from src.regime.kmeans_detector import KMeansDetector
from src.regime.markov_switching import MarkovSwitchingDetector
from src.regime.pelt_detector import PELTDetector
from src.regime.features import compute_regime_features


# ── Helpers ────────────────────────────────────────────────────────


def _run_detector(detector, closes, highs, lows, n_bars) -> list[RegimeLabel]:
    results = []
    for i in range(n_bars):
        label = detector.update(closes[i], highs[i], lows[i], i)
        results.append(label)
    return results


def _compute_er_series(closes: np.ndarray, k: int = 20) -> np.ndarray:
    """Compute efficiency ratio for each bar."""
    n = len(closes)
    er = np.zeros(n)
    for i in range(k, n):
        seg = closes[i - k: i + 1]
        er[i], _, _ = compute_regime_features(seg, k)
    return er


def _regime_runs(labels: list[str]) -> list[int]:
    """Return lengths of consecutive same-regime runs."""
    if not labels:
        return []
    runs = []
    current = labels[0]
    count = 1
    for lab in labels[1:]:
        if lab == current:
            count += 1
        else:
            runs.append(count)
            current = lab
            count = 1
    runs.append(count)
    return runs


# ── Synthetic evaluation ──────────────────────────────────────────


def evaluate_synthetic(
    detectors: dict[str, object],
    closes, highs, lows, labels: list[str],
    n_bars: int,
    max_warmup: int,
) -> pd.DataFrame:
    """Compute accuracy, precision, recall, F1 on synthetic data."""
    rows = []
    for name, det in detectors.items():
        results = _run_detector(det, closes, highs, lows, n_bars)
        pred = [r.regime for r in results]

        # Evaluate only after max warmup
        y_true = labels[max_warmup:]
        y_pred = pred[max_warmup:]

        tp = sum(1 for t, p in zip(y_true, y_pred) if t == "TREND" and p == "TREND")
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == "RANGE" and p == "TREND")
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == "TREND" and p == "RANGE")
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == "RANGE" and p == "RANGE")

        total = tp + fp + fn + tn
        acc = (tp + tn) / total if total > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        rows.append({
            "detector": name,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        })
        print(f"  {name:15s}  acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  F1={f1:.3f}")

    return pd.DataFrame(rows)


# ── Real-data evaluation ──────────────────────────────────────────


def evaluate_real(
    detectors: dict[str, object],
    closes, highs, lows,
    n_bars: int,
    max_warmup: int,
    er_series: np.ndarray,
) -> pd.DataFrame:
    """Compute proxy quality metrics on real data."""
    log_returns = np.zeros(n_bars)
    log_returns[1:] = np.diff(np.log(closes))

    rows = []
    all_pred: dict[str, list[str]] = {}

    for name, det in detectors.items():
        results = _run_detector(det, closes, highs, lows, n_bars)
        pred = [r.regime for r in results]
        all_pred[name] = pred

        # Evaluate only after max warmup
        pred_eval = pred[max_warmup:]
        er_eval = er_series[max_warmup:]
        ret_eval = np.abs(log_returns[max_warmup:])

        trend_mask = np.array([p == "TREND" for p in pred_eval])
        range_mask = ~trend_mask

        n_trend = trend_mask.sum()
        n_range = range_mask.sum()

        # ER separation
        er_trend = float(er_eval[trend_mask].mean()) if n_trend > 0 else 0
        er_range = float(er_eval[range_mask].mean()) if n_range > 0 else 0
        er_sep = er_trend - er_range

        # Return separation
        ret_trend = float(ret_eval[trend_mask].mean()) if n_trend > 0 else 0
        ret_range = float(ret_eval[range_mask].mean()) if n_range > 0 else 0
        ret_sep = ret_trend - ret_range

        # Regime duration
        runs = _regime_runs(pred_eval)
        mean_dur = float(np.mean(runs)) if runs else 0
        median_dur = float(np.median(runs)) if runs else 0

        # Transition frequency per 1000 bars
        transitions = sum(1 for i in range(1, len(pred_eval)) if pred_eval[i] != pred_eval[i - 1])
        trans_per_1k = transitions / len(pred_eval) * 1000

        # % TREND
        pct_trend = n_trend / len(pred_eval) * 100

        rows.append({
            "detector": name,
            "er_separation": round(er_sep, 4),
            "return_separation": round(ret_sep, 6),
            "mean_duration": round(mean_dur, 1),
            "median_duration": round(median_dur, 1),
            "transitions_per_1k": round(trans_per_1k, 1),
            "pct_trend": round(pct_trend, 1),
        })
        print(
            f"  {name:15s}  ER_sep={er_sep:+.4f}  ret_sep={ret_sep:+.6f}  "
            f"dur={mean_dur:.0f}  trans/1k={trans_per_1k:.1f}  %TREND={pct_trend:.1f}"
        )

    # Agreement score: % bars matching majority vote
    n_eval = n_bars - max_warmup
    det_names = list(all_pred.keys())
    for name in det_names:
        agree_count = 0
        for i in range(max_warmup, n_bars):
            votes = [all_pred[d][i] for d in det_names]
            majority = "TREND" if votes.count("TREND") > len(votes) / 2 else "RANGE"
            if all_pred[name][i] == majority:
                agree_count += 1
        rows_idx = next(j for j, r in enumerate(rows) if r["detector"] == name)
        rows[rows_idx]["agreement_pct"] = round(agree_count / n_eval * 100, 1)

    return pd.DataFrame(rows)


# ── Plotting ──────────────────────────────────────────────────────


def plot_synthetic_f1(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(df["detector"], df["f1"], color="steelblue")
    ax.set_xlabel("F1 Score (TREND class)")
    ax.set_title("Synthetic Data: F1 Scores by Detector")
    ax.set_xlim(0, 1)
    for bar, val in zip(bars, df["f1"]):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center")
    plt.tight_layout()
    plt.savefig(out_dir / "synthetic_accuracy.png", dpi=150)
    plt.close()


def plot_er_separation(df: pd.DataFrame, out_dir: Path) -> None:
    df_sorted = df.sort_values("er_separation")
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["forestgreen" if v > 0 else "firebrick" for v in df_sorted["er_separation"]]
    bars = ax.barh(df_sorted["detector"], df_sorted["er_separation"], color=colors)
    ax.set_xlabel("ER Separation (TREND avg - RANGE avg)")
    ax.set_title("Real Data: Efficiency Ratio Separation by Detector")
    ax.axvline(0, color="black", linewidth=0.5)
    for bar, val in zip(bars, df_sorted["er_separation"]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:+.4f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "er_separation_ranking.png", dpi=150)
    plt.close()


def plot_regime_overlay(
    closes: np.ndarray,
    detectors: dict[str, object],
    highs: np.ndarray,
    lows: np.ndarray,
    out_dir: Path,
    start: int = 0,
    length: int = 10_000,
) -> None:
    """Price chart with regime coloring for each detector (subset)."""
    end = min(start + length, len(closes))
    x = np.arange(start, end)
    price = closes[start:end]

    det_names = list(detectors.keys())
    n_det = len(det_names)

    fig, axes = plt.subplots(n_det + 1, 1, figsize=(16, 2.5 * (n_det + 1)), sharex=True)

    # Price on top
    axes[0].plot(x, price, linewidth=0.5, color="black")
    axes[0].set_ylabel("Close")
    axes[0].set_title("Price + Regime Detection Overlay")

    for idx, name in enumerate(det_names):
        det = detectors[name]
        # Re-run detector on subset (need full run for state)
        results = _run_detector(det, closes[:end], highs[:end], lows[:end], end)
        pred = [r.regime for r in results[start:end]]

        ax = axes[idx + 1]
        ax.plot(x, price, linewidth=0.3, color="gray")

        # Color background
        for i in range(len(pred)):
            color = "lightgreen" if pred[i] == "TREND" else "lightsalmon"
            ax.axvspan(x[i] - 0.5, x[i] + 0.5, alpha=0.3, color=color, linewidth=0)

        ax.set_ylabel(name, fontsize=8)
        ax.tick_params(labelsize=7)

    axes[-1].set_xlabel("Bar Index")
    plt.tight_layout()
    plt.savefig(out_dir / "regime_overlay.png", dpi=120)
    plt.close()


def plot_duration_dist(
    detectors: dict[str, object],
    closes, highs, lows,
    n_bars: int,
    max_warmup: int,
    out_dir: Path,
) -> None:
    """Box plots of regime duration distribution."""
    all_runs: dict[str, list[int]] = {}
    for name, det in detectors.items():
        results = _run_detector(det, closes, highs, lows, n_bars)
        pred = [r.regime for r in results[max_warmup:]]
        all_runs[name] = _regime_runs(pred)

    fig, ax = plt.subplots(figsize=(10, 5))
    data = [all_runs[n] for n in all_runs]
    labels = list(all_runs.keys())
    ax.boxplot(data, labels=labels, vert=True, showfliers=False)
    ax.set_ylabel("Regime Duration (bars)")
    ax.set_title("Distribution of Regime Segment Lengths")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "regime_duration_dist.png", dpi=150)
    plt.close()


def plot_agreement_heatmap(
    detectors: dict[str, object],
    closes, highs, lows,
    n_bars: int,
    max_warmup: int,
    out_dir: Path,
) -> None:
    """Pairwise agreement heatmap between detectors."""
    det_names = list(detectors.keys())
    n = len(det_names)
    all_pred: dict[str, list[str]] = {}

    for name, det in detectors.items():
        results = _run_detector(det, closes, highs, lows, n_bars)
        all_pred[name] = [r.regime for r in results[max_warmup:]]

    n_eval = n_bars - max_warmup
    agreement = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            agree = sum(
                1 for k in range(n_eval)
                if all_pred[det_names[i]][k] == all_pred[det_names[j]][k]
            )
            agreement[i, j] = agree / n_eval * 100

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(agreement, cmap="YlGnBu", vmin=30, vmax=100)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(det_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(det_names, fontsize=8)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{agreement[i, j]:.0f}%",
                    ha="center", va="center", fontsize=7)

    ax.set_title("Pairwise Detector Agreement (%)")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(out_dir / "detector_agreement.png", dpi=150)
    plt.close()


# ── Data loading ──────────────────────────────────────────────────


def load_real_data() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load XAUUSD M5 data from the most recent data snapshot."""
    data_dir = Path("data")
    if not data_dir.exists():
        return None

    snapshots = sorted(data_dir.iterdir(), reverse=True)
    if not snapshots:
        return None

    # Find CSVs in latest snapshot
    latest = snapshots[0]
    csv_files = sorted(latest.glob("XAUUSD_M5_*.csv"))
    if not csv_files:
        return None

    frames = []
    for f in csv_files:
        df = pd.read_csv(f)
        frames.append(df)

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    df.sort_values("time", inplace=True)

    return (
        df["close"].values.astype(np.float64),
        df["high"].values.astype(np.float64),
        df["low"].values.astype(np.float64),
    )


# ── Main ──────────────────────────────────────────────────────────


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / f"regime_comparison_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print(f"Output directory: {out_dir}")
    print()

    # ── 1. Synthetic evaluation ────────────────────────────────────
    print("=" * 60)
    print("SYNTHETIC DATA EVALUATION (ground truth)")
    print("=" * 60)

    syn_c, syn_h, syn_l, syn_idx, syn_labels = generate_synthetic_data(
        n_bars=5000, segment_length=250
    )

    # Build streaming detectors for synthetic
    syn_streaming = {
        "Choppiness": ChoppinessDetector(period=14),
        "Hurst": HurstDetector(window=100, compute_every=5),
        "BBWidth": BBWidthDetector(bb_period=20, rank_window=100),
        "GMM": GMMDetector(k=10, window=500, refit_every=200, predict_every=5),
        "KMeans": KMeansDetector(k=10, window=500, refit_every=200, predict_every=5),
    }

    # Build batch detectors for synthetic
    pelt_syn = PELTDetector(k=20, min_size=50, pen=5.0)
    pelt_syn.fit_batch(syn_c)
    syn_streaming["PELT"] = pelt_syn

    msw_syn = MarkovSwitchingDetector(max_fit_bars=5000)
    print("  Fitting Markov Switching on synthetic data...")
    msw_syn.fit_batch(syn_c)
    syn_streaming["MarkovSW"] = msw_syn

    # HMM with small window for synthetic
    syn_streaming["HMM"] = HMMRegimeDetector(
        k=10, window_W=500, refit_every=200, predict_every=5
    )

    # Max warmup across all detectors
    max_warmup_syn = max(
        getattr(d, "warmup_bars", 0) for d in syn_streaming.values()
    )
    max_warmup_syn = max(max_warmup_syn, 520)  # k + window for GMM/KMeans/HMM
    print(f"  Max warmup: {max_warmup_syn} bars")
    print()

    syn_results = evaluate_synthetic(
        syn_streaming, syn_c, syn_h, syn_l, syn_labels, 5000, max_warmup_syn
    )
    syn_results.to_csv(out_dir / "synthetic_results.csv", index=False)
    print()

    # Synthetic F1 plot
    plot_synthetic_f1(syn_results, plots_dir)
    print("  Saved synthetic_accuracy.png")

    # ── 2. Real data evaluation ────────────────────────────────────
    print()
    print("=" * 60)
    print("REAL XAUUSD DATA EVALUATION (proxy metrics)")
    print("=" * 60)

    real_data = load_real_data()
    if real_data is None:
        print("  No real XAUUSD data found in data/. Skipping real data evaluation.")
        print("  Run `uv run loader` first to fetch data from MT5.")

        # Write summary with only synthetic
        syn_results.to_csv(out_dir / "summary_table.csv", index=False)
        print(f"\nResults saved to {out_dir}")
        return

    real_c, real_h, real_l = real_data
    n_real = len(real_c)
    print(f"  Loaded {n_real:,} bars")

    # Precompute ER series
    print("  Computing ER series...")
    er_series = _compute_er_series(real_c, k=20)

    # Build detectors for real data
    real_detectors: dict[str, object] = {
        "Choppiness": ChoppinessDetector(period=14),
        "Hurst": HurstDetector(window=200, compute_every=10),
        "BBWidth": BBWidthDetector(bb_period=20, rank_window=200),
    }

    # GMM and KMeans with full-scale params
    real_detectors["GMM"] = GMMDetector(
        k=20, window=5000, refit_every=500, predict_every=10
    )
    real_detectors["KMeans"] = KMeansDetector(
        k=20, window=5000, refit_every=500, predict_every=10
    )

    # HMM
    real_detectors["HMM"] = HMMRegimeDetector(
        k=20, window_W=5000, refit_every=500, predict_every=10
    )

    # PELT (batch)
    print("  Fitting PELT on real data...")
    t0 = time.time()
    pelt_real = PELTDetector(k=20, min_size=100, pen=10.0)
    pelt_real.fit_batch(real_c)
    print(f"    PELT fit took {time.time() - t0:.1f}s")
    real_detectors["PELT"] = pelt_real

    # Markov Switching (batch, last 50k bars)
    print("  Fitting Markov Switching on real data (last 50k bars)...")
    t0 = time.time()
    msw_real = MarkovSwitchingDetector(max_fit_bars=50_000)
    msw_real.fit_batch(real_c)
    print(f"    MarkovSW fit took {time.time() - t0:.1f}s")
    real_detectors["MarkovSW"] = msw_real

    max_warmup_real = max(
        getattr(d, "warmup_bars", 0) for d in real_detectors.values()
    )
    max_warmup_real = max(max_warmup_real, 5020)
    print(f"  Max warmup: {max_warmup_real} bars")
    print()

    real_results = evaluate_real(
        real_detectors, real_c, real_h, real_l, n_real, max_warmup_real, er_series
    )
    real_results.to_csv(out_dir / "real_data_results.csv", index=False)
    print()

    # Plots
    print("  Generating plots...")

    plot_er_separation(real_results, plots_dir)
    print("    er_separation_ranking.png")

    # Regime overlay (10k bar subset from middle of data)
    overlay_start = max(max_warmup_real, n_real // 2 - 5000)
    # Need fresh detectors for overlay since state is consumed
    overlay_detectors: dict[str, object] = {
        "Choppiness": ChoppinessDetector(period=14),
        "Hurst": HurstDetector(window=200, compute_every=10),
        "BBWidth": BBWidthDetector(bb_period=20, rank_window=200),
    }
    pelt_overlay = PELTDetector(k=20, min_size=100, pen=10.0)
    pelt_overlay.fit_batch(real_c)
    overlay_detectors["PELT"] = pelt_overlay
    plot_regime_overlay(
        real_c, overlay_detectors, real_h, real_l,
        plots_dir, start=overlay_start, length=10_000
    )
    print("    regime_overlay.png")

    # Duration distribution — need fresh detectors
    dur_detectors: dict[str, object] = {
        "Choppiness": ChoppinessDetector(period=14),
        "Hurst": HurstDetector(window=200, compute_every=10),
        "BBWidth": BBWidthDetector(bb_period=20, rank_window=200),
        "GMM": GMMDetector(k=20, window=5000, refit_every=500, predict_every=10),
        "KMeans": KMeansDetector(k=20, window=5000, refit_every=500, predict_every=10),
        "HMM": HMMRegimeDetector(k=20, window_W=5000, refit_every=500, predict_every=10),
    }
    pelt_dur = PELTDetector(k=20, min_size=100, pen=10.0)
    pelt_dur.fit_batch(real_c)
    dur_detectors["PELT"] = pelt_dur
    msw_dur = MarkovSwitchingDetector(max_fit_bars=50_000)
    msw_dur.fit_batch(real_c)
    dur_detectors["MarkovSW"] = msw_dur
    plot_duration_dist(
        dur_detectors, real_c, real_h, real_l, n_real, max_warmup_real, plots_dir
    )
    print("    regime_duration_dist.png")

    # Agreement heatmap — need fresh detectors
    agree_detectors: dict[str, object] = {
        "Choppiness": ChoppinessDetector(period=14),
        "Hurst": HurstDetector(window=200, compute_every=10),
        "BBWidth": BBWidthDetector(bb_period=20, rank_window=200),
        "GMM": GMMDetector(k=20, window=5000, refit_every=500, predict_every=10),
        "KMeans": KMeansDetector(k=20, window=5000, refit_every=500, predict_every=10),
        "HMM": HMMRegimeDetector(k=20, window_W=5000, refit_every=500, predict_every=10),
    }
    pelt_agree = PELTDetector(k=20, min_size=100, pen=10.0)
    pelt_agree.fit_batch(real_c)
    agree_detectors["PELT"] = pelt_agree
    msw_agree = MarkovSwitchingDetector(max_fit_bars=50_000)
    msw_agree.fit_batch(real_c)
    agree_detectors["MarkovSW"] = msw_agree
    plot_agreement_heatmap(
        agree_detectors, real_c, real_h, real_l, n_real, max_warmup_real, plots_dir
    )
    print("    detector_agreement.png")

    # ── 3. Summary table ──────────────────────────────────────────
    summary = syn_results.merge(real_results, on="detector", how="outer")
    summary.to_csv(out_dir / "summary_table.csv", index=False)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(summary.to_string(index=False))
    print()
    print(f"All results saved to {out_dir}")


if __name__ == "__main__":
    main()
