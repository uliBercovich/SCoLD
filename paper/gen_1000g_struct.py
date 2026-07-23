"""Extract the Bercovich2025 human dataset: 50 CEU + 50 YRI + 50 ASW, chr22, MAF>5%.

ASW is admixed between European and African ancestry, so the sample contains two
ancestral populations (k=2) with one admixed group -- the structured setting used in
Bercovich et al. (2025).

Writes paper/data/1000G_struct_chr22.npz with G (n x m, 0/1/2), pos, pop labels.
Run:  python paper/gen_1000g_struct.py
"""
import os
import numpy as np
import pandas as pd
import allel

HERE = os.path.dirname(os.path.abspath(__file__))
# Directory holding the raw 1000 Genomes phase-3 chr22 VCF and panel file.
# Download both from the 1000 Genomes FTP and point G1000_DIR at them, e.g.
#   export G1000_DIR=/path/to/1000G
# Defaults to paper/data/raw/.
SRC = os.environ.get("G1000_DIR", os.path.join(HERE, "data", "raw"))
VCF = os.path.join(SRC, "ALL_chr22_phase3_genotypes.vcf.gz")
PANEL = os.path.join(SRC, "1000G_panel.txt")
OUT = os.path.join(HERE, "data", "1000G_struct_chr22.npz")

POPS = ["CEU", "YRI", "ASW"]
N_PER_POP = 50
MAF = 0.05
SEED = 42


def main():
    rng = np.random.default_rng(SEED)
    panel = pd.read_csv(PANEL, sep="\t")
    panel.columns = [c.strip() for c in panel.columns]

    chosen, labels = [], []
    for p in POPS:
        ids = panel.loc[panel["pop"] == p, "sample"].tolist()
        pick = rng.choice(ids, size=N_PER_POP, replace=False)
        chosen += list(pick)
        labels += [p] * N_PER_POP
    print(f"selected {len(chosen)} individuals: " +
          ", ".join(f"{p}={N_PER_POP}" for p in POPS), flush=True)

    print("reading VCF (subsetting samples on read) ...", flush=True)
    cs = allel.read_vcf(VCF, samples=chosen,
                        fields=["samples", "calldata/GT", "variants/POS", "variants/ALT"],
                        alt_number=1)

    # reorder to our population order
    vcf_samples = list(cs["samples"])
    order = [vcf_samples.index(s) for s in chosen]

    gt = allel.GenotypeArray(cs["calldata/GT"])[:, order, :]
    G = gt.to_n_alt(fill=-1).T.astype(float)          # n x m, count of ALT
    pos = np.asarray(cs["variants/POS"])
    print(f"raw: {G.shape[0]} individuals x {G.shape[1]} variants", flush=True)

    # drop variants with any missing call, then MAF filter on the pooled sample
    keep = ~(G < 0).any(axis=0)
    G, pos = G[:, keep], pos[keep]

    ac = G.sum(axis=0)
    nchrom = 2 * G.shape[0]
    af = ac / nchrom
    maf = np.minimum(af, 1 - af)
    keep = (maf >= MAF) & (ac > 0) & (ac < nchrom)
    G, pos = G[:, keep], pos[keep]
    print(f"after missing + MAF{MAF} filter: {G.shape[0]} x {G.shape[1]}", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, G=G.astype(np.int8), pos=pos,
                        pop=np.array(labels), samples=np.array(chosen))
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
