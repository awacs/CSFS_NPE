# NPE Pipeline for CSFS-based Demographic Inference

Neural Posterior Estimation (NPE) pipeline using the `sbi` library.
Replaces the ABC approach in `abc_new.R` with a trained normalizing flow
that directly learns the mapping from CSFS summary statistics to a
posterior distribution over demographic parameters.

The four inferred parameters are:

| # | Parameter | Description |
|---|-----------|-------------|
| 1 | alpha     | Admixture proportion |
| 2 | T_merge   | Merge time |
| 3 | T_split   | Split/divergence time |
| 4 | Ghost_Ne  | Ghost population effective size |

---

## Dependencies

```
sbi
torch
numpy
pandas
scipy
matplotlib
```

---

## Scripts

### `npe.py` — Train and query the posterior

**Train mode** — loads simulations, trains the neural network, saves a `.pkl`
posterior and a `.npz` results archive:

```bash
python npe.py \
  --target <observed.csfs> \
  --simTAG <simulation_tag> \
  [--threepara] \
  [--normalize none|median|sum] \
  [--num_posterior_samples 10000]
```

**Resample mode** — loads an existing trained posterior and queries it for a
new target. No retraining required:

```bash
python npe.py \
  --target <new_observed.csfs> \
  --load_posterior <posterior.pkl> \
  [--normalize none|median|sum] \
  [--num_posterior_samples 10000]
```

**Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--target` | required | Observed CSFS file (colSums applied internally) |
| `--simTAG` | — | Simulation tag; expects `<tag>.par.txt_DEN` and `<tag>.sim.txt_DEN`. Triggers train mode |
| `--load_posterior` | — | Path to existing `.pkl`. Triggers resample mode. Mutually exclusive with `--simTAG` |
| `--threepara` | False | Use only first 3 parameters (train mode only) |
| `--normalize` | `none` | CSFS normalization: `none`, `median`, or `sum`. Must match between train and resample |
| `--num_posterior_samples` | 10000 | Number of posterior samples to draw |

**Normalization methods**

Both methods operate independently on each half of the CSFS (indices 0–98 and 99–197):

- `none` — raw CSFS, no transformation
- `median` — `(x - median) / MAD` per half (MAD with 1.4826 scale factor, matching R's `mad()`)
- `sum` — `x / sum(x)` per half

> The same normalization must be used at train time and resample time.
> The normalization is included in the output filename to make this explicit.

**Outputs (train mode)**

- `<target><simTAG><para_suffix>_<normalize>_npe_results.npz` — results archive
- `<target><simTAG><para_suffix>_<normalize>_npe_posterior.pkl` — trained posterior object

**Outputs (resample mode)**

- `<target>_<normalize>_npe_results.npz` — results archive

**NPZ contents**

| Key | Shape | Description |
|-----|-------|-------------|
| `samples` | (N, p) | Posterior samples, original parameter scale |
| `samples_log` | (N, p) | Posterior samples, log scale |
| `x_obs` | (198,) | Raw observed CSFS (always un-normalized) |
| `param_means` | (p,) | Posterior means |
| `param_stds` | (p,) | Posterior standard deviations |
| `param_quantiles` | (5, p) | 2.5 / 25 / 50 / 75 / 97.5 percentiles |

---

### `summarize_npe.py` — Summarize posterior results

Prints a summary table, ASCII marginal histograms, and saves PNG plots.

```bash
python summarize_npe.py \
  --npz <results.npz> \
  [--pkl <posterior.pkl>] \
  [--names alpha T_merge T_split Ghost_Ne] \
  [--extra_samples 0] \
  [--out_prefix npe_summary] \
  [--no_plots]
```

**Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--npz` | required | Path to `_results.npz` |
| `--pkl` | — | Path to `_posterior.pkl` (only needed with `--extra_samples`) |
| `--names` | `alpha T_merge T_split Ghost_Ne` | Parameter names |
| `--extra_samples` | 0 | Draw additional samples from the PKL and pool with NPZ samples |
| `--out_prefix` | `npe_summary` | Prefix for all output files |
| `--no_plots` | False | Skip PNG plot generation |

**Outputs**

| File | Description |
|------|-------------|
| `<prefix>_marginals.txt` | ASCII histograms with per-parameter stats |
| `<prefix>_marginals.png` | Marginal posterior histograms with median and 95% CI |
| `<prefix>_pairs.png` | Pairwise scatter plot of all parameters |
| `<prefix>_csfs.png` | Observed CSFS plotted in two halves |

---

### `validate_npe.py` — Calibration check via Simulation-Based Calibration (SBC)

Tests whether the posterior is well-calibrated by running it on held-out
simulations where the true parameters are known.

For each held-out simulation, its CSFS is treated as a pseudo-observed target.
The trained posterior is queried and the **rank** of the true parameter among
posterior samples is recorded. A well-calibrated posterior produces uniform
rank distributions.

```bash
python validate_npe.py \
  --pkl <posterior.pkl> \
  --simTAG <simulation_tag> \
  [--n_held_out 500] \
  [--n_posterior_samples 500] \
  [--threepara] \
  [--seed 42] \
  [--out_prefix npe_validation]
```

**Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--pkl` | required | Trained posterior `.pkl` |
| `--simTAG` | required | Simulation tag (same as `npe.py`) |
| `--n_held_out` | 500 | Number of held-out simulations to test |
| `--n_posterior_samples` | 500 | Posterior samples drawn per held-out simulation |
| `--threepara` | False | Use only first 3 parameters |
| `--seed` | 42 | Random seed for held-out selection |
| `--out_prefix` | `npe_validation` | Prefix for all output files |

**Outputs**

| File | Description |
|------|-------------|
| `<prefix>_coverage.txt` | Expected vs actual coverage table |
| `<prefix>_coverage.png` | Coverage curve (should lie on diagonal) |
| `<prefix>_ranks.png` | Rank histograms (should be flat/uniform) |
| `<prefix>_ranks.npy` | Raw rank data for further analysis |

**Interpreting results**

| Observation | Meaning |
|-------------|---------|
| Coverage curve on diagonal / flat rank histograms | Posterior is well calibrated |
| Coverage curve below diagonal / U-shaped rank histograms | Posterior is overconfident — credible intervals too narrow |
| Coverage curve above diagonal / hill-shaped rank histograms | Posterior is underconfident — credible intervals too wide |
| MAE < 0.05 | Well calibrated |
| MAE > 0.10 | Poor calibration — consider more simulations or a larger network |

---

## Typical workflow

```bash
# 1. Train on simulations
python npe.py \
  --target observed.csfs \
  --simTAG wideprior.csv100000 \
  --normalize median \
  --num_posterior_samples 10000

# 2. Validate calibration
python validate_npe.py \
  --pkl observed.csfswideprior.csv100000_4para_median_npe_posterior.pkl \
  --simTAG wideprior.csv100000 \
  --n_held_out 500 --n_posterior_samples 500

# 3. Summarize results
python summarize_npe.py \
  --npz observed.csfswideprior.csv100000_4para_median_npe_results.npz \
  --out_prefix results_summary

# 4. Apply to a new target without retraining
python npe.py \
  --target new_target.csfs \
  --load_posterior observed.csfswideprior.csv100000_4para_median_npe_posterior.pkl \
  --normalize median
```
