#!/usr/bin/env python3
"""
summarize_npe.py - Summarize NPE posterior results from .npz and .pkl files.

Usage:
    python summarize_npe.py --npz <results.npz> [--pkl <posterior.pkl>]
                            [--names rate T_split N1 N2]
                            [--extra_samples 0] [--out_prefix summary]
"""

import argparse
import pickle
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from npe_utils import kde_mode


PARAM_NAMES_DEFAULT = ["alpha", "T_merge", "T_split", "Ghost_Ne"]
QUANTILE_LEVELS = [0.025, 0.25, 0.50, 0.75, 0.975]
QUANTILE_LABELS = ["2.5%", "25%", "50% (median)", "75%", "97.5%"]


# ── helpers ──────────────────────────────────────────────────────────────────

def load_npz(path):
    data = np.load(path)
    required = {"samples", "param_means", "param_stds", "param_quantiles", "x_obs"}
    missing = required - set(data.files)
    if missing:
        sys.exit(f"ERROR: NPZ is missing keys: {missing}")
    return data


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def draw_extra_samples(posterior, x_obs_np, n):
    import torch
    x_obs = torch.from_numpy(x_obs_np.astype("float32"))
    samples_log = posterior.sample((n,), x=x_obs)
    return np.expm1(samples_log.numpy()) if False else np.exp(samples_log.numpy())


# ── text summary ─────────────────────────────────────────────────────────────

def print_summary(samples, names):
    n, p = samples.shape
    sep = "─" * 74

    print(f"\n{'NPE Posterior Summary':^74}")
    print(sep)
    print(f"  Posterior samples : {n:,}")
    print(f"  Parameters        : {p}")
    print(sep)

    header = f"  {'Parameter':<14}" + "".join(f"{lbl:>11}" for lbl in
             ["Mode", "Mean", "Std", "2.5%", "25%", "50%", "75%", "97.5%"])
    print(header)
    print(sep)

    quantiles = np.quantile(samples, QUANTILE_LEVELS, axis=0)
    for i, name in enumerate(names):
        mode = kde_mode(np.log(samples[:, i]))
        mode = np.exp(mode)
        mean = samples[:, i].mean()
        std  = samples[:, i].std()
        row  = f"  {name:<14}" + f"{mode:>11.4g}" + f"{mean:>11.4g}" + f"{std:>11.4g}"
        row += "".join(f"{quantiles[q, i]:>11.4g}" for q in range(5))
        print(row)

    print(sep)

    # Flag parameters whose 95% CI sits outside the original prior range shown
    # in the samples themselves (rough diagnostic)
    sample_min = samples.min(axis=0)
    sample_max = samples.max(axis=0)
    print("\n  Sample range (min / max):")
    for i, name in enumerate(names):
        print(f"    {name:<14}  {sample_min[i]:.4g}  –  {sample_max[i]:.4g}")

    print()


# ── text marginals ───────────────────────────────────────────────────────────

def text_histogram(values, bins=20, width=40):
    """Return a list of lines forming an ASCII histogram for a 1-D array."""
    counts, edges = np.histogram(values, bins=bins)
    bar_scale = width / counts.max()
    lines = []
    for i, c in enumerate(counts):
        lo, hi = edges[i], edges[i + 1]
        bar = "█" * int(round(c * bar_scale))
        lines.append(f"  {lo:>12.4g} – {hi:<12.4g} │{bar:<{width}}│ {c:>6,}")
    return lines


def print_marginals(samples, names, out_prefix, bins=20, width=40):
    sep = "─" * (28 + width + 8)
    path = f"{out_prefix}_marginals.txt"
    all_lines = []

    for i, name in enumerate(names):
        col  = samples[:, i]
        q    = np.quantile(col, QUANTILE_LEVELS)
        mode = np.exp(kde_mode(np.log(col)))
        header_lines = [
            "",
            f"  {'─'*60}",
            f"  Parameter : {name}",
            f"  Mode      : {mode:.6g}",
            f"  Mean      : {col.mean():.6g}",
            f"  Std       : {col.std():.6g}",
            f"  2.5%      : {q[0]:.6g}",
            f"  25%       : {q[1]:.6g}",
            f"  Median    : {q[2]:.6g}",
            f"  75%       : {q[3]:.6g}",
            f"  97.5%     : {q[4]:.6g}",
            f"  {'─'*60}",
            f"  {'value range':^26} │{'histogram (n samples)':^{width}}│ {'count':>6}",
            sep,
        ]
        hist_lines  = text_histogram(col, bins=bins, width=width)
        footer      = [sep, ""]
        all_lines  += header_lines + hist_lines + footer

    # print to stdout
    for line in all_lines:
        print(line)

    # write to file
    with open(path, "w") as f:
        f.write("\n".join(all_lines) + "\n")
    print(f"  Saved: {path}")


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_marginals(samples, names, out_prefix):
    p = samples.shape[1]
    fig, axes = plt.subplots(1, p, figsize=(4 * p, 4))
    if p == 1:
        axes = [axes]

    quantiles = np.quantile(samples, QUANTILE_LEVELS, axis=0)
    modes = np.array([np.exp(kde_mode(np.log(samples[:, i]))) for i in range(p)])

    for i, ax in enumerate(axes):
        ax.hist(samples[:, i], bins=60, density=True, color="steelblue",
                alpha=0.75, edgecolor="none")
        ax.axvline(modes[i],          color="black",   lw=1.8, label="mode")
        ax.axvline(quantiles[2, i],   color="dimgray", lw=1.2, ls=":",  label="median")
        ax.axvline(quantiles[0, i],   color="crimson", lw=1.2, ls="--", label="95% CI")
        ax.axvline(quantiles[4, i],   color="crimson", lw=1.2, ls="--")
        ax.set_title(names[i], fontsize=11)
        ax.set_xlabel("value")
        if i == 0:
            ax.set_ylabel("density")
        if i == p - 1:
            ax.legend(fontsize=8)

    fig.suptitle("NPE Posterior Marginals", fontsize=13, y=1.02)
    plt.tight_layout()
    path = f"{out_prefix}_marginals.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_pairs(samples, names, out_prefix):
    p = samples.shape[1]
    fig, axes = plt.subplots(p, p, figsize=(3 * p, 3 * p))

    for i in range(p):
        for j in range(p):
            ax = axes[i, j]
            if i == j:
                ax.hist(samples[:, i], bins=40, color="steelblue",
                        alpha=0.75, edgecolor="none", density=True)
                ax.set_ylabel("density" if j == 0 else "")
            else:
                ax.scatter(samples[:, j], samples[:, i],
                           alpha=0.05, s=2, color="steelblue", rasterized=True)
            if i == p - 1:
                ax.set_xlabel(names[j], fontsize=9)
            if j == 0:
                ax.set_ylabel(names[i], fontsize=9)

    fig.suptitle("NPE Posterior Pair Plot", fontsize=13)
    plt.tight_layout()
    path = f"{out_prefix}_pairs.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_csfs(x_obs, out_prefix):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    half = len(x_obs) // 2
    for ax, chunk, title in zip(axes,
                                 [x_obs[:half], x_obs[half:]],
                                 ["CSFS — first half", "CSFS — second half"]):
        ax.bar(range(len(chunk)), chunk, color="steelblue", alpha=0.8)
        ax.set_xlabel("index")
        ax.set_ylabel("count")
        ax.set_title(title)
    fig.suptitle("Observed CSFS (x_obs)", fontsize=13)
    plt.tight_layout()
    path = f"{out_prefix}_csfs.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Summarize NPE results")
    parser.add_argument("--npz",  required=True, help="Path to _results.npz")
    parser.add_argument("--pkl",  default=None,  help="Path to _posterior.pkl (optional)")
    parser.add_argument("--names", nargs="+", default=None,
                        help="Parameter names (space-separated, must match param count)")
    parser.add_argument("--extra_samples", type=int, default=0,
                        help="Draw N additional samples from pkl posterior (requires --pkl)")
    parser.add_argument("--out_prefix", default="npe_summary",
                        help="Output file prefix for plots (default: npe_summary)")
    parser.add_argument("--no_plots", action="store_true",
                        help="Skip plot generation")
    args = parser.parse_args()

    # ── load ─────────────────────────────────────────────────────────────────
    print(f"\nLoading NPZ: {args.npz}")
    data = load_npz(args.npz)
    samples = data["samples"]          # (N, p) original scale
    x_obs   = data["x_obs"]            # (D,)
    n, p    = samples.shape

    names = args.names if args.names else PARAM_NAMES_DEFAULT[:p]
    if len(names) != p:
        sys.exit(f"ERROR: --names has {len(names)} entries but data has {p} parameters.")

    # ── optionally draw extra samples from pkl ────────────────────────────────
    if args.extra_samples > 0:
        if args.pkl is None:
            sys.exit("ERROR: --extra_samples requires --pkl to be provided.")
        print(f"Loading PKL: {args.pkl}")
        posterior = load_pkl(args.pkl)
        print(f"Drawing {args.extra_samples:,} extra samples from posterior...")
        extra = draw_extra_samples(posterior, x_obs, args.extra_samples)
        samples = np.concatenate([samples, extra], axis=0)
        print(f"Total samples after augmentation: {len(samples):,}")

    # ── text summary ──────────────────────────────────────────────────────────
    print_summary(samples, names)

    # ── correlation table ─────────────────────────────────────────────────────
    print("  Correlation matrix:")
    corr = np.corrcoef(samples.T)
    header = f"  {'':>14}" + "".join(f"{n:>10}" for n in names)
    print(header)
    for i, name in enumerate(names):
        row = f"  {name:<14}" + "".join(f"{corr[i,j]:>10.3f}" for j in range(p))
        print(row)
    print()

    # ── text marginals ────────────────────────────────────────────────────────
    print_marginals(samples, names, args.out_prefix)

    # ── plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots:
        print("Generating plots...")
        plot_marginals(samples, names, args.out_prefix)
        if p > 1:
            plot_pairs(samples, names, args.out_prefix)
        plot_csfs(x_obs, args.out_prefix)

    print("\nDone.")


if __name__ == "__main__":
    main()
