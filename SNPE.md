# Sequential Neural Posterior Estimation (SNPE)

## What Problem Does It Solve?

In many scientific models (e.g. population genetics, physics simulations), you can **simulate** data easily given parameters, but you cannot write down the likelihood `p(x | θ)` analytically. This makes classical Bayesian inference (MCMC, etc.) impossible or intractable.

**Simulation-Based Inference (SBI)** sidesteps this by learning the posterior directly from simulated `(θ, x)` pairs.

SNPE is one such method. It trains a **neural density estimator** — specifically a normalizing flow — to approximate the true posterior `p(θ | x)`.

---

## Core Idea

Given:
- A simulator: `x ~ sim(θ)` — produces data given parameters
- A prior: `p(θ)` — your belief about parameters before seeing data
- An observed dataset: `x_obs`

Goal: approximate `p(θ | x_obs)` — the posterior over parameters given the real observation.

SNPE trains a neural network `q_φ(θ | x)` such that:

```
q_φ(θ | x)  ≈  p(θ | x)
```

The network is a **normalizing flow**: a flexible, invertible transformation that can represent complex, multimodal distributions and allows exact density evaluation.

---

## The Training Objective

SNPE minimizes the **negative log-likelihood** of the true parameters θ under the learned conditional density, averaged over simulated pairs:

```
L(φ) = - E_{(θ,x) ~ p(θ)·sim(x|θ)} [ log q_φ(θ | x) ]
```

Intuitively: for each simulated `(θ, x)` pair, the network should assign high probability to the correct θ given x.

### Why log-space parameters?

Parameters like population sizes span orders of magnitude. Taking `log(θ)` makes the space more uniform and better suited for the flow to learn over.

---

## What is a Normalizing Flow?

A normalizing flow is a neural network that transforms a simple base distribution (e.g. standard Gaussian) into a complex target distribution through a series of invertible, differentiable transformations.

```
z ~ N(0, I)          (simple base)
θ = f(z | x)         (invertible transform conditioned on observation x)
```

Because the transform is invertible, we can compute exact densities via the change-of-variables formula:

```
log q(θ | x) = log p_z(f⁻¹(θ | x)) + log |det J_{f⁻¹}|
```

**MAF (Masked Autoregressive Flow)** — the default in SBI — stacks several autoregressive transformations. Each dimension of θ is transformed conditioned on previous dimensions and on x, giving it expressive power while keeping the Jacobian tractable.

### SBI Default MAF Architecture

| Parameter | Default | Description |
|---|---|---|
| `num_transforms` | 5 | Number of stacked autoregressive flow steps |
| `hidden_features` | 50 | Width of the MLP inside each transform |

The MLP inside each transform takes the full observation `x` (e.g. 198-dim CSFS) as input and outputs scale/shift values that steer the `z → θ` transformation. The `hidden_features` width controls how much capacity that MLP has to extract useful information from `x`.

These are fixed defaults — SBI does not adapt the architecture to your data. To override:

```python
from sbi.neural_nets import posterior_nn
density_estimator = posterior_nn("maf", hidden_features=128, num_transforms=8)
inference = SNPE(prior=prior, density_estimator=density_estimator)
```

For a 198-dim CSFS input, increasing `hidden_features` to 128–256 is worth trying if posterior samples look poorly concentrated.

---

## SNPE Variants

| Variant | Key idea |
|---|---|
| SNPE-A | Importance weighting to correct for proposal/prior mismatch |
| SNPE-B | Direct posterior targeting via ratio correction |
| **SNPE-C / APT** | Atomic proposal + cross-entropy loss; most stable and widely used — **this is what `sbi.SNPE` implements** |

APT (Automatic Posterior Transformation) avoids mode-seeking collapse by training with a contrastive loss over "atomic" proposals, making it robust even in sequential rounds.

---

## Sequential vs. Single-Round

SNPE can be run in two modes:

- **Single-round**: sample θ from prior, simulate x, train once. Simple but may waste simulations on unlikely parameter regions.
- **Sequential (multi-round)**: after each round, focus new simulations around the current posterior estimate, converging with fewer total simulations. Requires care to avoid proposal/posterior mismatch (handled by APT's correction).

The scripts `npe.py` / `npe1.py` use **single-round SNPE** (one call to `inference.train()`).

---

## Intuitive Pseudocode

```
# === TRAINING ===

# 1. Define prior over parameters
prior = Uniform(low, high)   # in log-space

# 2. Draw parameters and simulate data
for i in 1..N:
    θ_i  ~ prior
    x_i  ~ simulator(θ_i)    # e.g. run population genetics forward model
    dataset.append((θ_i, x_i))

# 3. Train a normalizing flow to learn p(θ | x)
flow = MaskedAutoregressiveFlow(input_dim=|θ|, context_dim=|x|)

for each training batch (θ_batch, x_batch):
    loss = -mean( flow.log_prob(θ_batch | context=x_batch) )
    backprop(loss)
    update(flow.parameters)

# flow now encodes q_φ(θ | x) ≈ p(θ | x)


# === INFERENCE (at test time) ===

x_obs = load_real_observation()   # e.g. real CSFS from data

# Option A — direct flow sampling (npe1.py):
samples = flow.sample(n=10000, condition=x_obs)
# No rejection; samples may fall outside prior range

# Option B — posterior wrapper sampling (npe.py):
samples = posterior.sample(n=10000, x=x_obs)
# SBI wrapper applies prior rejection: discards samples outside prior support


# === OUTPUT ===
params = exp(samples)   # back-transform from log-space
# samples now represent draws from the approximate posterior p(θ | x_obs)
```

---

## Summary

| Component | Role |
|---|---|
| Prior `p(θ)` | Initial beliefs over parameters |
| Simulator `sim(x | θ)` | Generates synthetic data — replaces the likelihood |
| Normalizing flow `q_φ(θ | x)` | The neural network being trained |
| Training | Minimize `-log q_φ(θ | x)` over simulated pairs |
| Inference | Condition the trained flow on `x_obs` and sample |

SNPE is powerful because once trained, querying the posterior for a new observation `x_obs` is essentially free — just a forward pass through the flow. No new simulations needed.
