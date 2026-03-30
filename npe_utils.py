"""
npe_utils.py - Shared utilities for the NPE pipeline.
"""

import pickle
import sys
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


NORMALIZE_CHOICES = ["none", "median", "median_pre_cut", "sum", "sum_pre_cut"]
SAMPLER_CHOICES   = ["rejection", "flow", "mcmc"]


def make_normalizer(method, cut=0):
    """Return a function that normalizes a 1-D CSFS array of length 198.

    Normalization methods (each operates per half: indices 0-98 and 99-197):
        none           : no transformation
        median         : cut tails, normalize inner region by (x - median) / MAD
        median_pre_cut : normalize full half by (x - median) / MAD, then cut tails
        sum            : cut tails, divide inner region by its sum
        sum_pre_cut    : divide full half by its sum, then cut tails

    cut > 0 removes that many elements from each end of each half, matching
    abc_new.R's CUT parameter (default there: 10).
    Output length = (99 - 2*cut) * 2 when cut > 0, else 198.
    """
    if method not in NORMALIZE_CHOICES:
        sys.exit(f"ERROR: unknown --normalize '{method}'. "
                 f"Choose: {', '.join(NORMALIZE_CHOICES)}.")

    if method == "none":
        return lambda x: x.copy().astype(np.float32)

    inner1 = slice(cut, 99 - cut)
    inner2 = slice(99 + cut, 198 - cut)

    def _mad(v):
        return 1.4826 * np.median(np.abs(v - np.median(v)))

    def norm(x):
        out = x.copy().astype(np.float64)

        if method == "median":
            for sl in (inner1, inner2):
                h = out[sl]
                out[sl] = (h - np.median(h)) / _mad(h)

        elif method == "median_pre_cut":
            for sl in (slice(0, 99), slice(99, 198)):
                h = out[sl]
                out[sl] = (h - np.median(h)) / _mad(h)

        elif method == "sum":
            for sl in (inner1, inner2):
                h = out[sl]
                out[sl] = h / h.sum()

        elif method == "sum_pre_cut":
            for sl in (slice(0, 99), slice(99, 198)):
                h = out[sl]
                out[sl] = h / h.sum()

        return np.concatenate([out[inner1], out[inner2]]).astype(np.float32)

    return norm


def save_posterior(path, posterior, meta):
    """Pickle posterior together with preprocessing metadata.

    meta should contain at minimum: normalize, cut, threepara.
    Stored as {"posterior": ..., "meta": {...}} so nothing can get separated.
    """
    with open(path, "wb") as f:
        pickle.dump({"posterior": posterior, "meta": meta}, f)


def load_posterior(path, expected_meta=None):
    """Load a posterior .pkl and optionally validate preprocessing metadata.

    Returns (posterior, meta). meta is None for legacy pkls (raw posterior object).

    If expected_meta is provided, hard-errors on any mismatch in 'normalize' or 'cut'
    so silently wrong results are impossible.
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and "posterior" in obj:
        posterior = obj["posterior"]
        meta = obj.get("meta", {})
        if expected_meta is not None:
            _validate_meta(meta, expected_meta, path)
        return posterior, meta
    else:
        # Legacy pkl — raw posterior with no metadata
        print(f"WARNING: {path} is a legacy pkl with no embedded metadata. "
              f"Cannot validate --normalize / --cut. Ensure they match training.")
        return obj, None


def _validate_meta(meta, expected, path):
    """Hard-error if normalize or cut in the pkl don't match what was requested."""
    mismatches = []
    for key in ("normalize", "cut"):
        if key in meta and meta[key] != expected.get(key):
            mismatches.append(
                f"  {key}: pkl has '{meta[key]}', requested '{expected[key]}'"
            )
    if mismatches:
        sys.exit(
            f"ERROR: preprocessing mismatch between {path} and current arguments:\n"
            + "\n".join(mismatches)
            + "\nUse the same --normalize and --cut that were used during training."
        )


def kde_mode(x, n_grid=512):
    """Estimate the mode of a 1-D sample via KDE (Silverman bandwidth, matching R's density()).

    x should be in the space where you want the mode (log-space or original scale).
    """
    kde  = gaussian_kde(x)
    grid = np.linspace(x.min(), x.max(), n_grid)
    return grid[np.argmax(kde(grid))]


def load_target(path, normalize_fn):
    """Load a CSFS file, sum rows, and return (normalized, raw) as float32."""
    raw = pd.read_csv(path, sep=r'\s+', header=None).values.sum(axis=0).astype(np.float32)
    return normalize_fn(raw), raw


def save_results(outbase, samples, samples_log, x_obs_np):
    """Save posterior samples and summary statistics to a .npz archive.

    Modes are computed via KDE in log-space (samples_log) then back-transformed,
    matching the space in which the flow was trained.
    """
    param_modes = np.array([
        np.exp(kde_mode(samples_log[:, j]))
        for j in range(samples_log.shape[1])
    ])
    np.savez(
        outbase + "_results.npz",
        samples=samples,
        samples_log=samples_log,
        x_obs=x_obs_np,
        param_means=samples.mean(axis=0),
        param_stds=samples.std(axis=0),
        param_modes=param_modes,
        param_quantiles=np.quantile(samples, [0.025, 0.25, 0.5, 0.75, 0.975], axis=0),
    )
    print(f"  Results archive : {outbase}_results.npz")
    print(f"Posterior samples shape: {samples.shape}")
    print(f"Parameter modes (KDE): {param_modes}")
    print(f"Parameter means:       {samples.mean(axis=0)}")
    print(f"Parameter stds:        {samples.std(axis=0)}")
    print(f"95% CI:\n{np.quantile(samples, [0.025, 0.975], axis=0)}")
