#!/usr/bin/env python3
"""
npe2.py - NPE with MCMC sampling.

In --load_posterior mode, samples via MCMC (NUTS by default), which properly
respects the prior and handles cases where the flow's proposal has low
acceptance under rejection sampling.

Modes:
    Train mode  (--simTAG):           train on simulations, save .npz + .pkl
    Resample mode (--load_posterior): skip training, query existing .pkl for a new target

Normalization (--normalize):
    none   : raw CSFS (default)
    median : each half normalized by (x - median) / MAD  (MAD with 1.4826 scale, matching R)
    sum    : each half divided by its sum

    NOTE: train and load_posterior must use the same normalization.

Usage:
    python npe2.py --target <csfs_file> --simTAG <tag> [--threepara] [--normalize median|sum]
    python npe2.py --target <csfs_file> --load_posterior <posterior.pkl> [--normalize median|sum] [--mcmc_method nuts|slice|hmc]
"""

import argparse
import pickle
import sys
import numpy as np
import pandas as pd
import torch
from sbi.inference import SNPE
from sbi.utils import BoxUniform


def make_normalizer(method):
    """Return a function that normalizes a 1-D CSFS array of length 198."""
    if method == "none":
        return lambda x: x.copy()

    if method == "median":
        def _mad(v):
            return 1.4826 * np.median(np.abs(v - np.median(v)))
        def norm(x):
            out = x.copy().astype(np.float64)
            for sl in (slice(0, 99), slice(99, 198)):
                h = out[sl]
                out[sl] = (h - np.median(h)) / _mad(h)
            return out.astype(np.float32)

    elif method == "sum":
        def norm(x):
            out = x.copy().astype(np.float64)
            for sl in (slice(0, 99), slice(99, 198)):
                h = out[sl]
                out[sl] = h / h.sum()
            return out.astype(np.float32)

    else:
        sys.exit(f"ERROR: unknown --normalize '{method}'. Choose: none, median, sum.")

    return norm


def load_target(path, normalize_fn):
    target_raw = pd.read_csv(path, sep=r'\s+', header=None)
    raw = target_raw.values.sum(axis=0).astype(np.float32)
    return normalize_fn(raw), raw


def save_results(outbase, samples, samples_log, x_obs_np):
    np.savez(
        outbase + "_results.npz",
        samples=samples,
        samples_log=samples_log,
        x_obs=x_obs_np,
        param_means=samples.mean(axis=0),
        param_stds=samples.std(axis=0),
        param_quantiles=np.quantile(samples, [0.025, 0.25, 0.5, 0.75, 0.975], axis=0),
    )
    print(f"  Results archive : {outbase}_results.npz")
    print(f"Posterior samples shape: {samples.shape}")
    print(f"Parameter means: {samples.mean(axis=0)}")
    print(f"Parameter stds:  {samples.std(axis=0)}")
    print(f"95% CI:\n{np.quantile(samples, [0.025, 0.975], axis=0)}")


def main():
    parser = argparse.ArgumentParser(description="NPE — MCMC sampling")
    parser.add_argument("--target", type=str, required=True,
                        help="Observed CSFS file")
    parser.add_argument("--simTAG", type=str, default=None,
                        help="Simulation tag — triggers training mode")
    parser.add_argument("--load_posterior", type=str, default=None,
                        help="Path to existing .pkl — skips training, resamples for new target")
    parser.add_argument("--threepara", action="store_true", default=False,
                        help="Use only first 3 parameters (train mode only)")
    parser.add_argument("--normalize", type=str, default="none",
                        choices=["none", "median", "sum"],
                        help="Normalization applied to CSFS (must match between train and resample)")
    parser.add_argument("--num_posterior_samples", type=int, default=10000,
                        help="Posterior samples to draw")
    parser.add_argument("--mcmc_method", type=str, default="nuts",
                        choices=["nuts", "slice", "hmc"],
                        help="MCMC method (default: nuts)")
    args = parser.parse_args()

    if args.simTAG is None and args.load_posterior is None:
        sys.exit("ERROR: provide either --simTAG (train) or --load_posterior (resample).")
    if args.simTAG and args.load_posterior:
        sys.exit("ERROR: --simTAG and --load_posterior are mutually exclusive.")

    normalize_fn = make_normalizer(args.normalize)

    # ── Load observed target ──────────────────────────────────────────────────
    print(f"Loading observed target from {args.target}")
    x_obs_np, x_obs_raw = load_target(args.target, normalize_fn)
    x_obs = torch.from_numpy(x_obs_np)

    # ── Resample mode ─────────────────────────────────────────────────────────
    if args.load_posterior:
        print(f"Loading posterior from {args.load_posterior}")
        with open(args.load_posterior, "rb") as f:
            posterior = pickle.load(f)

        for nm in ("median", "sum"):
            if f"_{nm}_npe" in args.load_posterior and args.normalize != nm:
                print(f"WARNING: posterior filename suggests --normalize {nm} "
                      f"but got '{args.normalize}'. Results may be wrong.")

        print(f"Sampling {args.num_posterior_samples} posterior samples via MCMC ({args.mcmc_method})...")
        print("Note: MCMC is slower than direct sampling but respects the prior.")
        samples_log = posterior.sample(
            (args.num_posterior_samples,),
            x=x_obs,
            sample_with="mcmc",
            mcmc_method=args.mcmc_method,
        )
        samples = torch.exp(samples_log.numpy())

        outbase = f"{args.target}_{args.normalize}_npe2"
        save_results(outbase, samples, samples_log.numpy(), x_obs_raw)
        print(f"\nDone.")
        return

    # ── Train mode ────────────────────────────────────────────────────────────
    TAG      = args.simTAG
    parfile  = f"{TAG}.par.txt_DEN"
    csfsfile = f"{TAG}.sim.txt_DEN"

    print(f"Loading parameters from {parfile}")
    parsim = pd.read_csv(parfile, sep=r'\s+', header=None)

    print(f"Loading simulated CSFS from {csfsfile}")
    csfs_sim = pd.read_csv(csfsfile, sep=r'\s+', header=None)

    assert len(parsim) == len(csfs_sim), (
        f"Row count mismatch: {len(parsim)} params vs {len(csfs_sim)} simulations"
    )

    if args.threepara:
        params = parsim.iloc[:, :3].values
    else:
        params = parsim.values

    log_params = np.log(params).astype(np.float32)
    x_sim_np   = np.apply_along_axis(normalize_fn, 1, csfs_sim.values.astype(np.float32))

    theta = torch.from_numpy(log_params)
    x_sim = torch.from_numpy(x_sim_np)

    low    = torch.tensor(log_params.min(axis=0), dtype=torch.float32)
    high   = torch.tensor(log_params.max(axis=0), dtype=torch.float32)
    prior  = BoxUniform(low=low, high=high)

    inference = SNPE(prior=prior, show_progress_bars=True)
    inference.append_simulations(theta, x_sim)

    print("Training neural density estimator...")
    density_estimator = inference.train(show_train_summary=True)

    print("Building posterior...")
    posterior = inference.build_posterior(density_estimator, sample_with="mcmc",
                                          mcmc_method=args.mcmc_method)

    print(f"Sampling {args.num_posterior_samples} posterior samples via MCMC ({args.mcmc_method})...")
    samples_log = posterior.sample(
        (args.num_posterior_samples,), x=x_obs
    )
    samples = torch.exp(samples_log.numpy())

    tag_suffix = "_3para" if args.threepara else "_4para"
    outbase    = f"{args.target}{TAG}{tag_suffix}_{args.normalize}_npe2"

    save_results(outbase, samples, samples_log.numpy(), x_obs_raw)

    with open(outbase + "_posterior.pkl", "wb") as f:
        pickle.dump(posterior, f)

    print(f"\nDone.")
    print(f"  Posterior object: {outbase}_posterior.pkl")


if __name__ == "__main__":
    main()
