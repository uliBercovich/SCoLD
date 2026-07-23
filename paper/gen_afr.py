"""Generate the AFR dataset (stdpopsim Africa_1T12, chr22 20-30Mb) and cache to disk.
Run with the pythonSample env (has stdpopsim/msprime). Matches SampCorrLD/plots.ipynb.
"""
import os
import numpy as np
import stdpopsim

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "AFR_chr22.npz")


def maf_filter(Gd, positions, thr=0.05):
    alt = Gd.sum(0)
    nch = 2 * Gd.shape[0]
    af = alt / nch
    maf = np.minimum(af, 1 - af)
    mask = (maf >= thr) & (alt > 0) & (alt < nch)
    return Gd[:, mask], positions[mask]


species = stdpopsim.get_species("HomSap")
model = species.get_demographic_model("Africa_1T12")
contig = species.get_contig("chr22", left=20_000_000, right=30_000_000)
engine = stdpopsim.get_engine("msprime")
print("simulating AFR ...", flush=True)
ts = engine.simulate(model, contig, {"AFR": 400}, seed=42)

G = ts.genotype_matrix().T
G = np.where(G >= 2, 1, G)
G = np.array([G[2 * i, :] + G[2 * i + 1, :] for i in range(G.shape[0] // 2)])
pos = np.asarray(ts.sites_position)
G, pos = maf_filter(G, pos, 0.05)
G[:, G.mean(0) > 1] = 2 - G[:, G.mean(0) > 1]

os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez_compressed(OUT, G=G.astype(np.int8), pos=pos)
print(f"saved {OUT}  G={G.shape}  pos={pos.shape}", flush=True)
