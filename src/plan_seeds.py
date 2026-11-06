"""Planning simulator: choose the number of screening and confirmation seeds before development (Method validation, Part C).

Set the values below, run the script, and read the recommendation. It simulates every combination of
screening seeds, screening threshold, and confirmation seeds, and writes seed_plan.csv and seed_plan.png
to ../results.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SIGMA = 0.8          # measured or expected seed noise (Macro-F1 points)
K = 16               # number of candidate designs
RHO = 0.5            # correlation between two designs trained with the same seed
EFFECT = 1.0         # size of the real gain worth finding
TARGET_FALSE = 0.10  # highest acceptable chance of adopting a useless design
TARGET_REAL = 0.60   # lowest acceptable chance of adopting the real gain
SCREEN_SEEDS = [1, 2]
SCREEN_FACTOR = [1.0, 0.5]
CONF_SEEDS = [2, 3, 4, 5]
CAL_SEEDS = 3
FLOOR = 0.5
N_SIM = 20000
SEED = 20261029
OUT = Path(__file__).resolve().parents[1] / "results"


def paired(mu, s, rng, n, k, m):
    # (n, k, m) paired differences; each of the m seeds has its own incumbent run shared by all k candidates.
    return mu[None, :, None] + s * (rng.standard_normal((n, k, m)) - rng.standard_normal((n, 1, m)))


def run_protocol(effect, n_scr, factor, n_conf, rng, sigma=SIGMA, k=K, rho=RHO,
                 n_cal=CAL_SEEDS, floor=FLOOR, n=N_SIM):
    s = sigma * np.sqrt(1.0 - rho)
    mu = np.zeros(k)
    mu[0] = effect
    dof = 2 * (n_cal - 1)
    delta = np.maximum(floor, np.round(sigma * np.sqrt(rng.chisquare(dof, n) / dof), 1))

    d_scr = paired(mu, s, rng, n, k, n_scr).mean(2)
    d_conf = paired(mu, s, rng, n, k, n_conf)
    need_pos = int(np.ceil(2 * n_conf / 3))

    cand = d_scr >= factor * delta[:, None]
    m_conf = d_conf.mean(2)
    ok = cand & (m_conf >= delta[:, None]) & ((d_conf > 0).sum(2) >= need_pos)
    adopt = ok.any(1)
    choice = np.where(ok, m_conf, -np.inf).argmax(1)
    est = m_conf[np.arange(n), choice]

    # Incumbent calibration runs are reused for screening and confirmation where the seeds overlap.
    reuse = max(0, n_cal - n_scr)
    runs = 2 * n_cal + max(0, n_scr - n_cal) + k * n_scr + max(0, n_conf - reuse) + n_conf * cand.sum(1)
    real = adopt & (choice == 0) & (effect > 0)
    false = adopt & ((choice != 0) | (effect == 0))
    bias = float((est - mu[choice])[adopt].mean()) if adopt.any() else float("nan")
    return {"p_false": float(false.mean()), "p_real": float(real.mean()),
            "runs": float(runs.mean()), "bias": bias, "need_pos": need_pos}


def run_single_seed(effect, rng, sigma=SIGMA, k=K, rho=RHO, n=N_SIM):
    s = sigma * np.sqrt(1.0 - rho)
    mu = np.zeros(k)
    mu[0] = effect
    d = paired(mu, s, rng, n, k, 1)[:, :, 0]
    best = d.argmax(1)
    adopt = d.max(1) > 0
    real = adopt & (best == 0) & (effect > 0)
    false = adopt & ((best != 0) | (effect == 0))
    return {"p_false": float(false.mean()), "p_real": float(real.mean()), "runs": float(1 + k)}


def plot(prot, b0, b1):
    styles = {(1, 1.0): ("#1f5f8b", "D", "-"), (1, 0.5): ("#1f5f8b", "o", "--"),
              (2, 1.0): ("#2e8b57", "D", "-"), (2, 0.5): ("#2e8b57", "o", "--")}
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for (n_scr, fac), (col, mk, ls) in styles.items():
        sel = [r for r in prot if r["screening_seeds"] == n_scr and r["screening_threshold"] == f"{fac:g} x delta"]
        x = [r["confirmation_seeds"] for r in sel]
        lab = f"screen {n_scr} seed{'s' if n_scr > 1 else ''}, {fac:g} x delta"
        for a, key in zip(ax, ("p_false_no_effect", "p_real_with_effect", "expected_runs")):
            a.plot(x, [r[key] for r in sel], color=col, marker=mk, ls=ls, label=lab)
    ax[0].axhline(b0["p_false"], color="#c0392b", ls=":", label="best of one seed")
    ax[1].axhline(b1["p_real"], color="#c0392b", ls=":")
    ax[2].axhline(b0["runs"], color="#c0392b", ls=":")
    ax[0].axhline(TARGET_FALSE, color="gray", lw=0.8)
    ax[1].axhline(TARGET_REAL, color="gray", lw=0.8)
    ax[0].set_ylabel("P(a useless design is adopted)")
    ax[1].set_ylabel(f"P(a real +{EFFECT:g} gain is adopted)")
    ax[2].set_ylabel("Expected training runs")
    for a in ax[:2]:
        a.set_ylim(0, 1)
    for a in ax:
        a.set_xlabel("Confirmation seeds")
        a.set_xticks(CONF_SEEDS)
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.suptitle(f"sigma = {SIGMA}, {K} candidates, rho = {RHO}", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "seed_plan.png", dpi=200)


def main():
    rng = np.random.default_rng(SEED)
    b0, b1 = run_single_seed(0.0, rng), run_single_seed(EFFECT, rng)
    rows = [{"option": "best of one seed", "screening_seeds": 1, "screening_threshold": "-",
             "confirmation_seeds": 0, "positive_needed": "-",
             "p_false_no_effect": round(b0["p_false"], 3), "p_real_with_effect": round(b1["p_real"], 3),
             "expected_runs": round(b0["runs"], 1), "overstatement": "-"}]
    for n_scr in SCREEN_SEEDS:
        for fac in SCREEN_FACTOR:
            for c in CONF_SEEDS:
                r0 = run_protocol(0.0, n_scr, fac, c, rng)
                r1 = run_protocol(EFFECT, n_scr, fac, c, rng)
                rows.append({"option": f"screen {n_scr} seed(s) at {fac:g} x delta, confirm {c}",
                             "screening_seeds": n_scr, "screening_threshold": f"{fac:g} x delta",
                             "confirmation_seeds": c, "positive_needed": f"{r0['need_pos']}/{c}",
                             "p_false_no_effect": round(r0["p_false"], 3),
                             "p_real_with_effect": round(r1["p_real"], 3),
                             "expected_runs": round(0.5 * (r0["runs"] + r1["runs"]), 1),
                             "overstatement": round(r1["bias"], 2)})

    prot = rows[1:]
    both = [r for r in prot if r["p_false_no_effect"] <= TARGET_FALSE and r["p_real_with_effect"] >= TARGET_REAL]
    if both:
        rec, why = min(both, key=lambda r: r["expected_runs"]), "cheapest option meeting both targets"
    else:
        ok = [r for r in prot if r["p_false_no_effect"] <= TARGET_FALSE]
        rec = max(ok, key=lambda r: (r["p_real_with_effect"], -r["expected_runs"])) if ok else None
        why = "no option meets the power target; highest power within the false-adoption target"

    with open(OUT / "seed_plan.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"sigma={SIGMA}, candidates={K}, rho={RHO}, effect={EFFECT}, "
          f"targets: false<={TARGET_FALSE}, real>={TARGET_REAL}")
    for r in rows:
        print(f"{r['option']:<45} false={r['p_false_no_effect']:.3f} real={r['p_real_with_effect']:.3f} "
              f"runs={r['expected_runs']}")
    print("Recommended:", rec["option"] if rec else "none", f"({why})")
    plot(prot, b0, b1)


if __name__ == "__main__":
    main()
