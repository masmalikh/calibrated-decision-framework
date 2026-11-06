"""Applies the decision protocol to the recorded validation scores of the case study (Method validation, Part A).

Reads ../case_study/log_eksperimen_faseF_H100.csv (the original run log, Indonesian column names) and reproduces the noise
calibration, screening, shared-setting rule, confirmation, the comparison with two common rules, and the cost.
"""
from pathlib import Path

import numpy as np
import pandas as pd

LOG = Path(__file__).resolve().parents[1] / "case_study" / "log_eksperimen_faseF_H100.csv"
INCUMBENT, REFERENCE = "F0b_V7", "F0a_V1"
CAL_SEEDS = [42, 43, 44]
SCREEN_SEED = 42
CONF_SEEDS = [43, 44, 45]
FLOOR = 0.5
FAMILIES = {
    "Gate input form": ["F1a_fdeg_standar", "F1b_fdeg_linear_standar", "F1c_fdeg_skalar_standar",
                        "F1d_fdeg_film_standar", "F1e_fdeg_kuantil20", "F1f_fdeg_kuantil5"],
    "Context module": ["F2a_konteks_tgca_bias", "F2b_konteks_gerbang_vektor"],
    "Path normalization": ["F3a_layernorm", "F3b_layernorm_fdeg_standar"],
    "Learning rates": ["F4a_V7_lrplm3e-5", "F4b_V7_lrbaru5e-4", "F4c_V7_lrplm3e-5_lrbaru5e-4"],
    "TF-IDF settings": ["F5a_tfidf_10k", "F5b_tfidf_50k", "F5c_tfidf_mindf5"],
}
# Shared-setting rule: a learning-rate candidate on the incumbent is judged by its mean gain over the
# incumbent and the reference variant run with the same setting.
SHARED = {"F4a_V7_lrplm3e-5": "F4d_V1_lrplm3e-5", "F4b_V7_lrbaru5e-4": "F4e_V1_lrbaru5e-4",
          "F4c_V7_lrplm3e-5_lrbaru5e-4": "F4f_V1_lrplm3e-5_lrbaru5e-4"}


def score(df, name, seed):
    return df.loc[(df["nama"] == name) & (df["seed"] == seed), "f1"].item()


def pooled_sd(*groups):
    num = sum((len(g) - 1) * np.var(g, ddof=1) for g in groups)
    return np.sqrt(num / sum(len(g) - 1 for g in groups))


def screen(delta_val, delta):
    if delta_val >= delta:
        return "candidate"
    return "dropped" if delta_val <= -delta else "tie"


def confirm(deltas, delta):
    return np.mean(deltas) >= delta and (np.asarray(deltas) > 0).sum() >= 2


def main():
    df = pd.read_csv(LOG)
    df = df[df["dataset"] == "laporgub"].copy()
    df["f1"] = 100 * df["val_macro_f1"]

    print("Noise calibration (validation Macro-F1, fold 0)")
    cal = {}
    for name in (REFERENCE, INCUMBENT):
        cal[name] = [score(df, name, s) for s in CAL_SEEDS]
        print(f"  {name:<8} " + " ".join(f"{v:6.2f}" for v in cal[name])
              + f"   mean {np.mean(cal[name]):.2f}  SD {np.std(cal[name], ddof=1):.2f}")
    sigma = pooled_sd(cal[REFERENCE], cal[INCUMBENT])
    delta = max(FLOOR, round(sigma, 1))
    print(f"  pooled sigma = {sigma:.2f} (4 df), tie band delta = {delta}")

    base = score(df, INCUMBENT, SCREEN_SEED)
    print(f"\nScreening with seed {SCREEN_SEED} (delta against the incumbent run with the same seed)")
    scr = {}
    for fam, names in FAMILIES.items():
        for name in names:
            scr[name] = score(df, name, SCREEN_SEED) - base
        out = [screen(scr[n], delta) for n in names]
        print(f"  {fam:<20} designs {len(names)}  largest {max(scr[n] for n in names):+.2f}  "
              f"candidates {out.count('candidate')}  ties {out.count('tie')}  dropped {out.count('dropped')}")
    cands = [n for n in scr if screen(scr[n], delta) == "candidate"]
    print("  candidates:", ", ".join(f"{n} ({scr[n]:+.2f})" for n in cands))

    print("\nShared-setting rule")
    ref_base = score(df, REFERENCE, SCREEN_SEED)
    for name in [n for n in cands if n in SHARED]:
        ref_gain = score(df, SHARED[name], SCREEN_SEED) - ref_base
        mean_gain = (scr[name] + ref_gain) / 2
        if mean_gain < delta:
            cands.remove(name)
        print(f"  {name}: incumbent {scr[name]:+.2f}, reference {ref_gain:+.2f}, mean {mean_gain:+.2f} "
              f"-> {'withdrawn' if mean_gain < delta else 'kept'}")

    print(f"\nConfirmation with seeds {CONF_SEEDS}, paired with the incumbent")
    for name in cands:
        d = [score(df, name, s) - score(df, INCUMBENT, s) for s in CONF_SEEDS]
        print(f"  {name:<28} " + " ".join(f"{v:+.2f}" for v in d) + f"   mean {np.mean(d):+.2f}  "
              f"positive {(np.asarray(d) > 0).sum()}/3 -> {'adopted' if confirm(d, delta) else 'rejected'}")

    top = max(scr, key=scr.get)
    print("\nCommon rules on the same screening run")
    print(f"  best of one seed: adopt {top} ({scr[top]:+.2f})")
    print(f"  1-point tie band: adopt {', '.join(n for n in scr if scr[n] >= 1.0) or 'none'}")
    four = cal[INCUMBENT] + [score(df, INCUMBENT, 45)]
    print(f"  incumbent seed {SCREEN_SEED} = {base:.2f}, mean over four seeds = {np.mean(four):.2f}")

    # Extra runs of the protocol over the single-seed rule: calibration seeds beyond the screening seed
    # (the reference run with the screening seed already existed for learning-rate tuning) plus confirmation.
    mins = df.set_index(["nama", "seed"])["durasi_menit"]
    extra = [(n, s) for n in (REFERENCE, INCUMBENT) for s in CAL_SEEDS if s != SCREEN_SEED]
    extra += [(n, s) for n in cands + [INCUMBENT] for s in CONF_SEEDS if (n, s) not in extra]
    screening = [(INCUMBENT, SCREEN_SEED)] + [(n, SCREEN_SEED) for n in scr]
    h_extra = sum(mins[k] for k in extra) / 60
    h_scr = sum(mins[k] for k in screening) / 60
    print(f"\nCost: {len(extra)} extra runs, {h_extra:.2f} GPU hours; screening {len(screening)} runs, "
          f"{h_scr:.2f} GPU hours; ratio {100 * h_extra / h_scr:.0f}%")


if __name__ == "__main__":
    main()
