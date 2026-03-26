"""
simulate_csfs_demes.py — demes-based replacement for simulate_csfs.py

Demography is fully specified by a demes YAML file (e.g. prufer.yaml).
Samples are specified by a JSON file (e.g. prufer_samples.json).
Population names for the target, outgroup, and archaic reference are passed
as arguments and looked up by name in the tree sequence — no hardcoded indices.

All ABC parameters (alpha, ta, ts, ne) are baked into the YAML; this script
just runs simulations and computes the CSFS.

Usage:
    python simulate_csfs_demes.py \\
        -t out \\
        -d prufer.yaml \\
        -s prufer_samples.json \\
        --admix-pop  Eurasia \\
        --outgroup-pop Africa \\
        --ref Neanderthal \\
        -ns 1000000 -r 300

    # Denisovan-conditioned CSFS:
    python simulate_csfs_demes.py ... --ref Denisovan
"""

import argparse
import json

import demes
import msprime
import numpy as np
import scipy.sparse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def findpop(name: str, ts) -> int:
    """Return the integer population id whose metadata name matches *name*."""
    for pop in ts.populations():
        if pop.metadata["name"] == name:
            return pop.id
    raise ValueError(f"Population '{name}' not found in tree sequence. "
                     f"Available: {[p.metadata['name'] for p in ts.populations()]}")


def combine_segs(segs, get_segs=False):
    """Merge overlapping genomic segments (adapted from Skov et al. 2018)."""
    merged = np.empty([0, 2])
    if len(segs) == 0:
        return [] if get_segs else 0
    sorted_segs = segs[np.argsort(segs[:, 0]), :]
    for higher in sorted_segs:
        if len(merged) == 0:
            merged = np.vstack([merged, higher])
        else:
            lower = merged[-1, :]
            if higher[0] <= lower[1]:
                merged[-1, :] = (lower[0], max(lower[1], higher[1]))
            else:
                merged = np.vstack([merged, higher])
    if get_segs:
        return merged
    return np.sum(merged[:, 1] - merged[:, 0])


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(deme_file: str, sample_file: str, nsite: int,
             seed=None, nrep: int = 1,
             rec_rate: float = 1.3e-8, mu: float = 1.2e-8):
    """
    Load a demes YAML and a samples JSON, run msprime, and yield mutated tree
    sequences one at a time (same generator pattern as archiesim.py).

    Args:
        deme_file:   Path to demes YAML (e.g. prufer.yaml).
        sample_file: Path to samples JSON (e.g. prufer_samples.json).
                     Format: list of {population_name: n_haplotypes} dicts.
        nsite:       Sequence length in base pairs.
        seed:        Random seed (passed to both sim_ancestry and sim_mutations).
        nrep:        Number of replicate tree sequences.
        rec_rate:    Recombination rate per base per generation.
        mu:          Mutation rate per base per generation.

    Yields:
        msprime.TreeSequence (mutated)
    """
    graph = demes.load(deme_file)
    demography = msprime.Demography.from_demes(graph)

    with open(sample_file) as fh:
        samples = json.load(fh)

    ancestry_reps = msprime.sim_ancestry(
        samples=samples, #ploidy = 2 in default
        demography=demography,
        sequence_length=nsite,
        recombination_rate=rec_rate,
        num_replicates=nrep,
        record_migrations=True,
        random_seed=seed,
    )
    for ts in ancestry_reps:
        yield msprime.sim_mutations(ts, rate=mu, random_seed=seed)


# ---------------------------------------------------------------------------
# CSFS computation
# ---------------------------------------------------------------------------

def N_ton(genotypes, n_samples, r):
    """
    Aggregate N-ton allele-frequency vector conditioned on sites where the
    archaic reference panel *r* carries derived alleles.

    Sites where r has exactly 1 derived allele are counted once.
    Sites where r has exactly 2 derived alleles are counted twice (added again
    via the dH term), matching the original c1.py / c1_denisova.py behaviour.

    Args:
        genotypes (csr_matrix): sites × haplotypes for the target population.
        n_samples (int):        number of target haplotypes.
        r (csr_matrix):         sites × haplotypes for the archaic reference.

    Returns:
        numpy.ndarray: 1-D count vector of length n_samples - 1.
    """
    if genotypes.shape[0] != r.shape[0]:
        raise ValueError("genotypes and r must have the same number of rows.")

    valid_rows  = r.getnnz(axis=1) > 1   # ref carries ≥ 2 derived alleles
    double_rows = r.getnnz(axis=1) == 2  # ref carries exactly 2

    s_genotypes  = genotypes[valid_rows]
    db_genotypes = genotypes[double_rows]

    v = s_genotypes.toarray().sum(axis=1)
    u = db_genotypes.toarray().sum(axis=1)

    H,  _ = np.histogram(v, bins=np.arange(n_samples + 2))
    dH, _ = np.histogram(u, bins=np.arange(n_samples + 2))
    return np.add(H[1:-1], dH[1:-1])


def compute_csfs(ts, admix_pop: str, outgroup_pop: str, ref_pop: str):
    """
    Extract genotype matrices for the three populations and compute the
    archaic-reference-conditioned CSFS for both the admixed and outgroup pops.

    Args:
        ts:            Mutated msprime TreeSequence.
        admix_pop:     Name of the admixed / target population (e.g. 'Eurasia').
        outgroup_pop:  Name of the outgroup population (e.g. 'Africa').
        ref_pop:       Name of the archaic reference population
                       (e.g. 'Neanderthal' or 'Denisovan').

    Returns:
        (csfs_outgroup, csfs_admix): tuple of 1-D numpy arrays, each of length
        n_admix - 1, where n_admix is the number of admixed haplotypes.
    """
    admix_ids    = ts.samples(population=findpop(admix_pop,   ts))
    outgroup_ids = ts.samples(population=findpop(outgroup_pop, ts))
    ref_ids      = ts.samples(population=findpop(ref_pop,      ts))

    n_admix = len(admix_ids)

    ADMIX, OUTGROUP, REF = [], [], []
    for variant in ts.variants():
        g = variant.genotypes
        ADMIX.append(g[admix_ids].tolist())
        OUTGROUP.append(g[outgroup_ids].tolist())
        REF.append(g[ref_ids].tolist())

    ADMIX    = scipy.sparse.csr_matrix(ADMIX)
    OUTGROUP = scipy.sparse.csr_matrix(OUTGROUP)
    REF      = scipy.sparse.csr_matrix(REF)

    csfs_outgroup = N_ton(OUTGROUP, n_admix, REF)
    csfs_admix    = N_ton(ADMIX,    n_admix, REF)
    return np.array(csfs_outgroup), np.array(csfs_admix)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args):
    csfs_outgroup_total = 0
    csfs_admix_total    = 0

    for i, ts in enumerate(simulate(
            args.demes, args.samples, args.nsites,
            seed=args.seed, nrep=args.rep,
            rec_rate=args.rec_rate, mu=args.mut_rate)):

        if i % 100 == 0:
            print(f"Replicate {i}", flush=True)

        csfs_out, csfs_adm = compute_csfs(
            ts, args.admix_pop, args.outgroup_pop, args.ref)

        csfs_outgroup_total = csfs_out + csfs_outgroup_total
        csfs_admix_total    = csfs_adm + csfs_admix_total

    joined = np.concatenate(
        (csfs_outgroup_total.reshape(1, -1),
         csfs_admix_total.reshape(1, -1)), axis=1)
    np.savetxt(args.tag + ".csfs", joined, delimiter="\t")
    print(f"Written: {args.tag}.csfs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Simulate the archaic-conditioned CSFS using a demes YAML for "
            "demography and a JSON file for samples. "
            "All demographic parameters (alpha, ta, ts, ne) are specified in "
            "the YAML — not as command-line arguments."
        )
    )
    parser.add_argument("-t",  "--tag",          type=str, default="out",
                        help="Output file prefix  [default: out]")
    parser.add_argument("-d",  "--demes",        type=str, required=True,
                        help="Path to demes YAML file  (e.g. prufer.yaml)")
    parser.add_argument("-s",  "--samples",      type=str, required=True,
                        help="Path to samples JSON file  (e.g. prufer_samples.json). "
                             "Format: [{\"PopName\": n_haplotypes}, ...]")
    parser.add_argument("--admix-pop",           type=str, default="Eurasia",
                        help="Name of the admixed/target population in the YAML  "
                             "[default: Eurasia]")
    parser.add_argument("--outgroup-pop",        type=str, default="Africa",
                        help="Name of the outgroup population in the YAML  "
                             "[default: Africa]")
    parser.add_argument("--ref",                 type=str, default="Neanderthal",
                        help="Name of the archaic reference population used to "
                             "condition the CSFS  [default: Neanderthal]")
    parser.add_argument("-ns", "--nsites",       type=int,   default=1_000_000,
                        help="Sequence length to simulate (bp)  [default: 1000000]")
    parser.add_argument("-r",  "--rep",          type=int,   default=10,
                        help="Number of replicate tree sequences  [default: 10]")
    parser.add_argument("-x",  "--seed",         type=int,   default=None,
                        help="Random seed")
    parser.add_argument("--rec-rate",            type=float, default=1.3e-8,
                        help="Recombination rate per bp per generation  "
                             "[default: 1.3e-8]")
    parser.add_argument("--mut-rate",            type=float, default=1.2e-8,
                        help="Mutation rate per bp per generation  "
                             "[default: 1.2e-8]")
    main(parser.parse_args())
