"""Draws Figure 2 of the article from simulation_results.csv (run simulate_protocol.py first)."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1] / "results"
RHOS = (0.0, 0.5, 0.7)
RULES = [("pick_best", "Best of one seed", "#b03a2e", "o"),
         ("tie_1pt", "One seed, fixed 1-point tie band", "#d68910", "s"),
         ("protocol", "Calibrated two-stage protocol", "#1f5f8b", "D")]

with open(HERE / "simulation_results.csv") as f:
    res = {(r["rule"], float(r["rho"]), float(r["sigma"]), float(r["effect"])): r for r in csv.DictReader(f)}
sigmas = sorted({k[2] for k in res})
effects = [0.5, 1.0, 1.5, 2.0]


def band(rule, field, pts):
    vals = [[float(res[(rule, rho, s, e)][field]) for rho in RHOS] for s, e in pts]
    return [v[1] for v in vals], [min(v) for v in vals], [max(v) for v in vals]


fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
for rule, lab, col, mk in RULES:
    mid, lo, hi = band(rule, "p_adopt_null", [(s, 0.0) for s in sigmas])
    ax[0].fill_between(sigmas, lo, hi, color=col, alpha=0.15, lw=0)
    ax[0].plot(sigmas, mid, color=col, marker=mk, label=lab)
    mid, lo, hi = band(rule, "p_adopt_real", [(0.8, e) for e in effects])
    ax[1].fill_between(effects, lo, hi, color=col, alpha=0.15, lw=0)
    ax[1].plot(effects, mid, color=col, marker=mk, label=lab)

ax[0].set_title("(a) No candidate is better than the incumbent", fontsize=10)
ax[0].set_xlabel("Seed noise $\\sigma$ (Macro-F1 points)")
ax[0].set_ylabel("Probability that a useless design is adopted")
ax[1].set_title("(b) One candidate is truly better ($\\sigma$ = 0.8)", fontsize=10)
ax[1].set_xlabel("True gain of that candidate (Macro-F1 points)")
ax[1].set_ylabel("Probability that it is adopted")
for a in ax:
    a.set_ylim(0, 1)
    a.grid(alpha=0.3)
ax[0].legend(fontsize=8, loc="center right")
fig.tight_layout()
fig.savefig(HERE / "Figure_2.pdf")
fig.savefig(HERE / "Figure_2.png", dpi=300)
