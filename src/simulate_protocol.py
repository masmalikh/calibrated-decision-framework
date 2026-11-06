"""Monte Carlo comparison of three design-selection rules under seed noise (Method validation, Part B).

K candidates are compared with one incumbent on one validation fold:
  pick_best  adopt the best candidate on one seed if it beats the incumbent
  tie_1pt    adopt the best candidate on one seed if it wins by at least 1 point
  protocol   calibrate the noise, screen with one seed, confirm with three new paired seeds
Writes simulation_results.csv and observed_check.csv to ../results.
"""
import csv
from pathlib import Path

import numpy as np

K = 16
N_SIM = 20000
SIGMAS = [0.5, 0.8, 1.2, 1.6]
RHOS = [0.0, 0.5, 0.7]
EFFECTS = [0.0, 0.5, 1.0, 1.5, 2.0]
SEED = 20261027
OUT = Path(__file__).resolve().parents[1] / "results"


def simulate(sigma, rho, effect, rng, n=N_SIM, k=K):
    # Two runs with the same seed share a noise part with correlation rho; only the rest enters a paired difference.
    s = sigma * np.sqrt(1.0 - rho)
    mu = np.zeros(k)
    mu[0] = effect  # candidate 0 is the only one that can be truly better

    # Pooled SD over 2 designs x 3 seeds has 4 degrees of freedom.
    sigma_hat = sigma * np.sqrt(rng.chisquare(4, n) / 4.0)
    delta = np.maximum(0.5, np.round(sigma_hat, 1))

    # d_scr: (n, k); every candidate is screened against the same incumbent run.
    # d_conf: (n, k, 3); each confirmation seed has its own incumbent run.
    d_scr = mu + s * (rng.standard_normal((n, k)) - rng.standard_normal((n, 1)))
    d_conf = mu[None, :, None] + s * (rng.standard_normal((n, k, 3)) - rng.standard_normal((n, 1, 3)))

    best = d_scr.argmax(1)
    best_val = d_scr.max(1)
    res = {
        "pick_best": (best_val > 0.0, best, best_val, np.zeros(n)),
        "tie_1pt": (best_val >= 1.0, best, best_val, np.zeros(n)),
    }

    cand = d_scr >= delta[:, None]
    m_conf = d_conf.mean(2)
    ok = cand & (m_conf >= delta[:, None]) & ((d_conf > 0).sum(2) >= 2)
    choice = np.where(ok, m_conf, -np.inf).argmax(1)
    res["protocol"] = (ok.any(1), choice, m_conf[np.arange(n), choice], cand.sum(1))

    out = {}
    for rule, (adopt, choice, est, n_cand) in res.items():
        real = adopt & (choice == 0) & (effect > 0)
        null = adopt & ((choice != 0) | (effect == 0))
        bias = (est - mu[choice])[adopt].mean() if adopt.any() else np.nan
        # Protocol: 6 calibration runs (one is the screening incumbent), k screening runs,
        # 1 incumbent run for the seed not covered by calibration, 3 runs per candidate.
        runs = 6 + k + 1 + 3 * n_cand.mean() if rule == "protocol" else 1 + k
        out[rule] = {"p_adopt_any": adopt.mean(), "p_adopt_real": real.mean(),
                     "p_adopt_null": null.mean(), "winner_bias": bias, "runs": runs}
    return out


def observed_check(rng, sigma=0.8, rho=0.5, n=N_SIM, k=K):
    # How often the case-study screening pattern appears when no candidate is better.
    s = sigma * np.sqrt(1.0 - rho)
    d_scr = s * (rng.standard_normal((n, k)) - rng.standard_normal((n, 1)))
    return {
        "p_ge3_candidates_delta08": float(((d_scr >= 0.8).sum(1) >= 3).mean()),
        "p_max_ge_2_07": float((d_scr.max(1) >= 2.07).mean()),
        "median_max": float(np.median(d_scr.max(1))),
    }


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for sigma in SIGMAS:
        for rho in RHOS:
            for effect in EFFECTS:
                for rule, m in simulate(sigma, rho, effect, rng).items():
                    rows.append({"sigma": sigma, "rho": rho, "effect": effect, "rule": rule,
                                 **{key: round(float(v), 4) for key, v in m.items()}})
    write_csv(OUT / "simulation_results.csv", rows)

    checks = [{"sigma": 0.8, "rho": rho, **{key: round(v, 4) for key, v in observed_check(rng, rho=rho).items()}}
              for rho in RHOS]
    write_csv(OUT / "observed_check.csv", checks)


if __name__ == "__main__":
    main()
