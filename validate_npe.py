#!/usr/bin/env python3
"""
validate_npe.py - Posterior calibration check via Simulation-Based Calibration (SBC).

For each held-out simulation:
  - its CSFS is used as a pseudo-observed target (true parameters are known)
  - the trained posterior is sampled
  - the normalized rank of the true parameter among posterior samples is recorded

A well-calibrated posterior produces:
  - uniform rank histograms (flat bars)
  - a coverage curve lying on the diagonal (expected == actual coverage)

Usage:
    python validate_npe.py --pkl <posterior.pkl> --simTAG <tag>
                           [--n_held_out 500] [--n_posterior_samples 500]
                           [--threepara] [--seed 42] [--out_prefix validation]
                           [--normalize none|median|median_pre_cut|sum|sum_pre_cut] [--cut 10]
"""

import argparse
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from npe_utils import NORMALIZE_CHOICES, make_normalizer, load_posterior, kde_mode


PARAM_NAMES_DEFAULT = ["alpha", "T_merge", "T_split", "Ghost_Ne"]
COVERAGE_LEVELS     = np.round(np.linspace(0.05, 0.95, 19), 3)


# ── I/O ───────────────────────────────────────────────────────────────────────


def load_simulations(simTAG, threepara, normalize_fn):
    parfile  = f"{simTAG}.par.txt_DEN"
    csfsfile = f"{simTAG}.sim.txt_DEN"
    print(f"Loading {parfile}")
    params = np.loadtxt(parfile, dtype=np.float32)
    print(f"Loading {csfsfile}")
    csfs   = np.loadtxt(csfsfile, dtype=np.float32)
    if threepara:
        params = params[:, :3]
    csfs = np.apply_along_axis(normalize_fn, 1, csfs)
    return params, csfs


# ── Core SBC loop ─────────────────────────────────────────────────────────────

def compute_ranks(posterior, true_log_params, x_sims, n_samples):
    """
    Returns:
      ranks     : (n_held, p)  normalised rank of true value among posterior samples.
      post_mode : (n_held, p)  posterior mode in original scale, estimated via KDE
                               in log-space (matching the flow's training space) then
                               back-transformed. Uses direct flow sampling for speed —
                               no prior rejection applied here, consistent with SBC intent.

    rank[i, j] = fraction of posterior samples < true_log_params[i, j].
    A uniform distribution of ranks indicates a calibrated posterior.
    """
    n_held, p = true_log_params.shape

    print(f"  Running batched forward pass ({n_held} sims × {n_samples} samples)...")
    x_batch = torch.from_numpy(x_sims)          # (n_held, D)

    with torch.no_grad():
        # One batched call: returns (n_samples, n_held, p)
        samples = posterior.posterior_estimator.sample(
            (n_samples,), condition=x_batch
        ).numpy()

    # Transpose to (n_held, n_samples, p)
    samples = samples.transpose(1, 0, 2)

    # rank[i, j] = fraction of posterior samples < true value
    ranks = (samples < true_log_params[:, None, :]).mean(axis=1)   # (n_held, p)

    # posterior mode via KDE in log-space, then back-transform to original scale
    post_mode = np.array([
        [np.exp(kde_mode(samples[i, :, j])) for j in range(p)]
        for i in range(n_held)
    ])                                                               # (n_held, p)

    print(f"  Done.")
    return ranks, post_mode


def compute_coverage(ranks, levels=COVERAGE_LEVELS):
    """
    coverage[l, j] = fraction of sims where true param falls inside the
    (levels[l])-credible interval, derived from rank statistics.
    """
    n_held, p  = ranks.shape
    coverage   = np.zeros((len(levels), p))
    for li, level in enumerate(levels):
        lo = (1 - level) / 2
        hi = 1 - lo
        coverage[li] = ((ranks >= lo) & (ranks <= hi)).mean(axis=0)
    return coverage


# ── Prediction error summary ──────────────────────────────────────────────────

def print_pred_error_summary(true_params, post_mode, names, n_held_out):
    """
    Mirrors the Default / Robust summary printed by summary.cv4abc() in the
    R abc package (normal mode).

    Default summary  — normalised MSE per parameter:
        prederr[j] = sum((post_mode[:,j] - true[:,j])^2) / (var(true[:,j]) * n)

    Robust summary   — statistics of the per-sample normalised errors:
        err[i,j] = (post_mode[i,j] - true[i,j])^2 / var(true[:,j])
        rows: mean, median, trimmed (10% each tail), p90
    """
    p = len(names)
    true_var = true_params.var(axis=0) * n_held_out          # var * n, matching R

    # per-sample normalised squared errors: (n_held, p)
    per_sample = (post_mode - true_params) ** 2 / true_params.var(axis=0)

    default_err = per_sample.mean(axis=0)                     # == sum / (var*n)

    trim_k   = max(1, int(round(0.10 * n_held_out)))          # 10% trim each tail
    trimmed  = np.array([
        np.mean(np.sort(per_sample[:, j])[trim_k:-trim_k])
        for j in range(p)
    ])
    p90      = np.percentile(per_sample, 90, axis=0)

    col_w = max(12, max(len(n) for n in names) + 2)
    header = " " * 10 + "".join(f"{n:>{col_w}}" for n in names)
    sep    = "─" * (10 + col_w * p)

    print(f"\nDefault summary:")
    print(f"Prediction error based on a cross-validation sample of {n_held_out}\n")
    print(header)
    row = f"{'NPE':<10}" + "".join(f"{default_err[j]:>{col_w}.7f}" for j in range(p))
    print(row)

    print(f"\nRobust summary:")
    print(header)
    stats = [
        ("mean",    per_sample.mean(axis=0)),
        ("median",  np.median(per_sample, axis=0)),
        ("trimmed", trimmed),
        ("p90",     p90),
    ]
    for label, vals in stats:
        print(f"{label:<10}" + "".join(f"{vals[j]:>{col_w}.7f}" for j in range(p)))
    print()


# ── Text output ───────────────────────────────────────────────────────────────

def print_coverage_table(coverage, names, levels=COVERAGE_LEVELS):
    sep = "─" * (10 + 10 * len(names))
    print(f"\n{'Expected Coverage vs Actual Coverage':^{len(sep)}}")
    print(sep)
    header = f"  {'Level':>6}  " + "".join(f"{n:>10}" for n in names)
    print(header)
    print(sep)
    for li, level in enumerate(levels):
        row = f"  {level:>6.0%}  " + "".join(f"{coverage[li, j]:>10.3f}" for j in range(len(names)))
        print(row)
    print(sep)

    # Mean absolute error from diagonal per parameter
    expected = levels[:, None]
    mae = np.abs(coverage - expected).mean(axis=0)
    print(f"\n  Mean |actual - expected| coverage (lower = better calibrated):")
    for j, name in enumerate(names):
        flag = "  <-- well calibrated" if mae[j] < 0.05 else ("  <-- moderate" if mae[j] < 0.10 else "  <-- POOR")
        print(f"    {name:<14} {mae[j]:.4f}{flag}")
    print()


def save_coverage_txt(coverage, names, levels, out_prefix):
    path = f"{out_prefix}_coverage.txt"
    lines = ["Expected Coverage vs Actual Coverage", ""]
    lines.append("  Level    " + "  ".join(f"{n:>10}" for n in names))
    for li, level in enumerate(levels):
        lines.append(f"  {level:.3f}    " + "  ".join(f"{coverage[li,j]:>10.4f}" for j in range(len(names))))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {path}")


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_coverage(coverage, names, out_prefix, levels=COVERAGE_LEVELS):
    p   = len(names)
    fig, ax = plt.subplots(figsize=(6, 5))

    colors = plt.cm.tab10(np.linspace(0, 0.9, p))
    for j, (name, col) in enumerate(zip(names, colors)):
        ax.plot(levels, coverage[:, j], marker="o", ms=4, lw=1.5,
                color=col, label=name)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    ax.fill_between([0, 1], [0, 1], [0, 1], alpha=0)   # anchor limits
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Expected coverage"); ax.set_ylabel("Actual coverage")
    ax.set_title("SBC Coverage Curve\n(closer to diagonal = better calibrated)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = f"{out_prefix}_coverage.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_ranks(ranks, names, out_prefix, bins=20):
    p   = ranks.shape[1]
    fig, axes = plt.subplots(1, p, figsize=(4 * p, 3.5))
    if p == 1:
        axes = [axes]

    expected_height = len(ranks) / bins    # flat uniform bar height

    for j, (ax, name) in enumerate(zip(axes, names)):
        ax.hist(ranks[:, j], bins=bins, range=(0, 1),
                color="steelblue", alpha=0.75, edgecolor="white")
        ax.axhline(expected_height, color="crimson", lw=1.5,
                   ls="--", label="uniform")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("normalized rank")
        if j == 0:
            ax.set_ylabel("count")
        if j == p - 1:
            ax.legend(fontsize=8)

    fig.suptitle("SBC Rank Histograms\n(flat = well calibrated)", fontsize=12)
    plt.tight_layout()
    path = f"{out_prefix}_ranks.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_predictions(true_params, post_mode, names, out_prefix):
    """
    Scatter plot of true parameter value vs posterior median, one panel per
    parameter — mirrors the default cv4abc plot.
    true_params : (n_held, p) in original scale
    post_mode    : (n_held, p) in original scale
    """
    p   = len(names)
    fig, axes = plt.subplots(1, p, figsize=(4 * p, 4))
    if p == 1:
        axes = [axes]

    for j, (ax, name) in enumerate(zip(axes, names)):
        ax.scatter(true_params[:, j], post_mode[:, j],
                   s=10, alpha=0.5, color="steelblue", edgecolors="none")
        lims = [min(true_params[:, j].min(), post_mode[:, j].min()),
                max(true_params[:, j].max(), post_mode[:, j].max())]
        ax.plot(lims, lims, "k--", lw=1, label="ideal")
        ax.set_xlabel("True value"); ax.set_ylabel("Posterior median")
        ax.set_title(name, fontsize=10)

    fig.suptitle("True vs Posterior Median", fontsize=12)
    plt.tight_layout()
    path = f"{out_prefix}_predictions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SBC validation for NPE posterior")
    parser.add_argument("--pkl",     required=True, help="Trained posterior .pkl")
    parser.add_argument("--simTAG", required=True, help="Simulation tag (same as npe.py)")
    parser.add_argument("--n_held_out",          type=int, default=500,
                        help="Number of held-out simulations to test (default: 500)")
    parser.add_argument("--n_posterior_samples", type=int, default=500,
                        help="Posterior samples per held-out sim (default: 500)")
    parser.add_argument("--threepara", action="store_true", default=False)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--out_prefix", default="npe_validation")
    parser.add_argument("--normalize", type=str, default="none",
                        choices=NORMALIZE_CHOICES,
                        help="Normalization applied to CSFS — must match what was used during training")
    parser.add_argument("--cut", type=int, default=0,
                        help="Elements to trim from each tail of each half (default 0; abc_new.R uses 10)")
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"Loading posterior: {args.pkl}")
    expected_meta = {"normalize": args.normalize, "cut": args.cut}
    posterior, _ = load_posterior(args.pkl, expected_meta)

    normalize_fn = make_normalizer(args.normalize, args.cut)
    params, csfs = load_simulations(args.simTAG, args.threepara, normalize_fn)
    n_total, p   = params.shape

    names = PARAM_NAMES_DEFAULT[:p]

    if args.n_held_out > n_total:
        sys.exit(f"ERROR: --n_held_out ({args.n_held_out}) > total sims ({n_total})")

    # Random held-out subset
    rng     = np.random.default_rng(args.seed)
    idx     = rng.choice(n_total, size=args.n_held_out, replace=False)
    held_params = params[idx]           # original scale
    held_csfs   = csfs[idx]

    log_params  = np.log(held_params).astype(np.float32)

    print(f"\nHeld-out: {args.n_held_out} simulations  |  "
          f"Posterior samples per sim: {args.n_posterior_samples}")
    print(f"Parameters: {names}\n")

    # ── SBC ───────────────────────────────────────────────────────────────────
    ranks, post_mode = compute_ranks(posterior, log_params, held_csfs, args.n_posterior_samples)
    coverage = compute_coverage(ranks)

    # ── Report ────────────────────────────────────────────────────────────────
    print_pred_error_summary(held_params, post_mode, names, args.n_held_out)
    print_coverage_table(coverage, names)

    print("Saving outputs...")
    save_coverage_txt(coverage, names, COVERAGE_LEVELS, args.out_prefix)
    plot_coverage(coverage, names, args.out_prefix)
    plot_ranks(ranks, names, args.out_prefix)
    plot_predictions(held_params, post_mode, names, args.out_prefix)

    # Save raw ranks for further analysis
    np.save(f"{args.out_prefix}_ranks.npy", ranks)
    print(f"  Saved: {args.out_prefix}_ranks.npy")

    print("\nDone.")
    print("Interpretation:")
    print("  Coverage curve on diagonal → posterior is well calibrated")
    print("  Curve below diagonal       → posterior is overconfident (CIs too narrow)")
    print("  Curve above diagonal       → posterior is underconfident (CIs too wide)")
    print("  Rank histograms flat       → well calibrated")
    print("  Rank histograms U-shaped   → overconfident")
    print("  Rank histograms ∩-shaped   → underconfident")


if __name__ == "__main__":
    main()
