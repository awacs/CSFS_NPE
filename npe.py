#!/usr/bin/env python3
"""
npe.py - Neural Posterior Estimation replicating abc_new.R with python-sbi.

Modes:
    Train mode  (--simTAG):           train on simulations, save .npz + .pkl
    Resample mode (--load_posterior): skip training, query existing .pkl for a new target

Sampler (--sampler):
    rejection : SBI default — draw proposals from the flow, reject outside prior (default)
    flow      : sample directly from the normalizing flow, no prior rejection
    mcmc      : MCMC sampling (NUTS/slice/HMC via --mcmc_method); slower but prior-respecting

Normalization (--normalize, --cut):
    none           : raw CSFS (default)
    median         : cut tails, then normalize inner region by (x - median) / MAD
    median_pre_cut : normalize full half by (x - median) / MAD, then cut tails
    sum            : cut tails, then divide inner region by its sum
    sum_pre_cut    : divide full half by its sum, then cut tails

    --cut N (default 0): elements to trim from each tail of each half.
        N=10 matches abc_new.R default (output length = (99 - 2*N) * 2).
        N=0 keeps all 198 elements.

    NOTE: --normalize and --cut must match between train and resample.

Usage:
    python npe.py --target <csfs_file> --simTAG <tag> [options]
    python npe.py --target <csfs_file> --load_posterior <posterior.pkl> [options]
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from sbi.inference import SNPE
from sbi.utils import BoxUniform

from npe_utils import (NORMALIZE_CHOICES, SAMPLER_CHOICES, make_normalizer,
                       load_target, save_results, save_posterior, load_posterior)


def draw_samples(posterior, x_obs, n, sampler, mcmc_method):
    """Sample from the posterior using the specified sampler."""
    if sampler == "rejection":
        with torch.no_grad():
            return posterior.sample((n,), x=x_obs)

    if sampler == "flow":
        with torch.no_grad():
            return posterior.posterior_estimator.sample(
                (n,), condition=x_obs.unsqueeze(0)
            ).reshape(n, -1)

    if sampler == "mcmc":
        return posterior.sample(
            (n,), x=x_obs, sample_with="mcmc", mcmc_method=mcmc_method
        )

    raise ValueError(f"Unknown sampler '{sampler}'. Choose: {', '.join(SAMPLER_CHOICES)}.")


def main():
    parser = argparse.ArgumentParser(description="NPE equivalent of abc_new.R")
    parser.add_argument("--target", type=str, required=True,
                        help="Observed CSFS file")
    parser.add_argument("--simTAG", type=str, default=None,
                        help="Simulation tag — triggers training mode; "
                             "expects <tag>.par.txt<suffix> and <tag>.sim.txt<suffix>")
    parser.add_argument("--suffix", type=str, default="_DEN",
                        help="Filename suffix on the .par.txt/.sim.txt inputs "
                             "(default: _DEN). Also appended to output filenames "
                             "when non-default.")
    parser.add_argument("--load_posterior", type=str, default=None,
                        help="Path to existing .pkl — skips training, resamples for new target")
    parser.add_argument("--threepara", action="store_true", default=False,
                        help="Use only first 3 parameters (train mode only)")
    parser.add_argument("--normalize", type=str, default="none",
                        choices=NORMALIZE_CHOICES,
                        help="Normalization applied to CSFS (must match between train and resample)")
    parser.add_argument("--cut", type=int, default=0,
                        help="Elements to trim from each tail of each half "
                             "(default 0; abc_new.R uses 10)")
    parser.add_argument("--sampler", type=str, default="rejection",
                        choices=SAMPLER_CHOICES,
                        help="Posterior sampling method (default: rejection)")
    parser.add_argument("--mcmc_method", type=str, default="nuts",
                        choices=["nuts", "slice", "hmc"],
                        help="MCMC algorithm — only used when --sampler mcmc (default: nuts)")
    parser.add_argument("--num_posterior_samples", type=int, default=10000,
                        help="Posterior samples to draw")
    parser.add_argument("--out_dir", type=str, default=".",
                        help="Directory for output files (default: current working directory)")
    args = parser.parse_args()

    if args.simTAG is None and args.load_posterior is None:
        sys.exit("ERROR: provide either --simTAG (train) or --load_posterior (resample).")
    if args.simTAG and args.load_posterior:
        sys.exit("ERROR: --simTAG and --load_posterior are mutually exclusive.")

    normalize_fn = make_normalizer(args.normalize, args.cut)

    # suffix encodes sampler only when non-default, to keep filenames clean
    sampler_tag = "" if args.sampler == "rejection" else f"_{args.sampler}"
    cut_tag     = f"_cut{args.cut}" if args.cut > 0 else ""
    suffix_tag  = "" if args.suffix == "_DEN" else args.suffix

    # ── Load observed target ──────────────────────────────────────────────────
    print(f"Loading observed target from {args.target}")
    x_obs_np, x_obs_raw = load_target(args.target, normalize_fn)
    x_obs = torch.from_numpy(x_obs_np)

    # ── Resample mode ─────────────────────────────────────────────────────────
    if args.load_posterior:
        print(f"Loading posterior from {args.load_posterior}")
        expected_meta = {"normalize": args.normalize, "cut": args.cut}
        posterior, _ = load_posterior(args.load_posterior, expected_meta)

        print(f"Sampling {args.num_posterior_samples} posterior samples "
              f"({args.sampler})...")
        samples_log = draw_samples(
            posterior, x_obs, args.num_posterior_samples,
            args.sampler, args.mcmc_method
        )
        samples = torch.exp(samples_log).numpy()

        target_base = os.path.basename(args.target)
        os.makedirs(args.out_dir, exist_ok=True)
        outbase = os.path.join(args.out_dir,
                               f"{target_base}_{args.normalize}{cut_tag}_npe{sampler_tag}")
        save_results(outbase, samples, samples_log.numpy(), x_obs_raw)
        print("\nDone.")
        return

    # ── Train mode ────────────────────────────────────────────────────────────
    TAG      = args.simTAG
    parfile  = f"{TAG}.par.txt{args.suffix}"
    csfsfile = f"{TAG}.sim.txt{args.suffix}"

    print(f"Loading parameters from {parfile}")
    parsim = pd.read_csv(parfile, sep=r'\s+', header=None)

    print(f"Loading simulated CSFS from {csfsfile}")
    csfs_sim = pd.read_csv(csfsfile, sep=r'\s+', header=None)

    assert len(parsim) == len(csfs_sim), (
        f"Row count mismatch: {len(parsim)} params vs {len(csfs_sim)} simulations"
    )

    params = parsim.iloc[:, :3].values if args.threepara else parsim.values

    log_params = np.log(params).astype(np.float32)
    x_sim_np   = np.apply_along_axis(normalize_fn, 1, csfs_sim.values.astype(np.float32))

    theta  = torch.from_numpy(log_params)
    x_sim  = torch.from_numpy(x_sim_np)
    prior  = BoxUniform(
        low=torch.tensor(log_params.min(axis=0), dtype=torch.float32),
        high=torch.tensor(log_params.max(axis=0), dtype=torch.float32),
    )

    inference = SNPE(prior=prior, show_progress_bars=True)
    inference.append_simulations(theta, x_sim)

    print("Training neural density estimator...")
    density_estimator = inference.train(show_train_summary=True)

    print("Building posterior...")
    if args.sampler == "mcmc":
        posterior = inference.build_posterior(
            density_estimator, sample_with="mcmc", mcmc_method=args.mcmc_method
        )
    else:
        posterior = inference.build_posterior(density_estimator)

    print(f"Sampling {args.num_posterior_samples} posterior samples "
          f"({args.sampler})...")
    samples_log = draw_samples(
        posterior, x_obs, args.num_posterior_samples,
        args.sampler, args.mcmc_method
    )
    samples = torch.exp(samples_log).numpy()

    tag_suffix  = "_3para" if args.threepara else "_4para"
    target_base = os.path.basename(args.target)
    tag_base    = os.path.basename(TAG)
    os.makedirs(args.out_dir, exist_ok=True)
    outbase     = os.path.join(args.out_dir,
                               f"{target_base}{tag_base}{tag_suffix}_{args.normalize}{cut_tag}{suffix_tag}_npe{sampler_tag}")

    save_results(outbase, samples, samples_log.numpy(), x_obs_raw)

    meta = {
        "normalize": args.normalize,
        "cut":       args.cut,
        "threepara": args.threepara,
        "sampler":   args.sampler,
        "suffix":    args.suffix,
    }
    save_posterior(outbase + "_posterior.pkl", posterior, meta)

    print("\nDone.")
    print(f"  Posterior object: {outbase}_posterior.pkl")


if __name__ == "__main__":
    main()
