# Calibrated decision framework: code, case-study logs, and development history

[![DOI](https://zenodo.org/badge/1448022263.svg)](https://doi.org/10.5281/zenodo.24286518)

Code and audit material for the article:

> Hakim, M. M., Priyambodo, T. K., & Mustofa, K. *Separating real gains from seed noise: a calibrated decision framework for fine-tuned transformer classifiers.* Manuscript prepared for MethodsX.

The framework decides whether a candidate design should replace the current (incumbent) design of a fine-tuned transformer classifier when every training run is expensive. It has three components: a calibrated two-stage seed protocol, a planning simulator, and an audit trail. This repository holds the scripts behind every number in the Method validation section of the article, the run log of the case study, and the full development history up to the moment the test data were first opened.

## Contents

| Path | Description |
|---|---|
| `src/simulate_protocol.py` | Monte Carlo comparison of three selection rules (Part B of the article) |
| `src/plan_seeds.py` | Planning simulator: chooses the number of screening and confirmation seeds (Part C) |
| `src/make_figure2.py` | Draws Figure 2 from `results/simulation_results.csv` |
| `src/apply_rules.py` | Applies the protocol to the recorded validation scores of the case study (Part A) |
| `results/` | Output of the four scripts, as reported in the article |
| `case_study/log_eksperimen_faseF_H100.csv` | One row per development run (42 runs on one NVIDIA H100 SXM, 80 GB), with validation scores and the commit of the code each run used |
| `development_history/` | Decision log, its English translation, the git history of the development repository, the test-opening audit file, and the console log (see below) |
| `data/` | LaporGub complaint IDs, labels, and fold assignments (no complaint texts) |

## Running the scripts

Requirements: Python 3.10 or newer with `numpy`, `pandas`, and `matplotlib` (`pip install -r requirements.txt`). No GPU and no complaint text are needed.

```
python src/simulate_protocol.py   # writes results/simulation_results.csv and results/observed_check.csv
python src/make_figure2.py        # writes results/Figure_2.pdf and results/Figure_2.png
python src/plan_seeds.py          # prints the planning table and writes results/seed_plan.csv and results/seed_plan.png
python src/apply_rules.py         # prints the case-study decisions and cost (Tables 2 to 4)
```

Each script runs in a few seconds on a laptop CPU. The simulations use fixed random seeds (20261027 and 20261029), so the CSV files are reproduced exactly; they were identical on two different computers.

`results/seed_plan.png` is a graphical version of the planning table (Table 6 of the article) and does not appear in the article.

### Planning your own study

Open `src/plan_seeds.py` and set the values at the top: the expected seed noise `SIGMA` (in points of your metric), the number of candidate designs `K`, the seed correlation `RHO`, the gain worth finding `EFFECT`, and the two targets `TARGET_FALSE` (highest acceptable chance of adopting a useless design) and `TARGET_REAL` (lowest acceptable chance of finding a real gain). The script prints the expected error rates and run counts of every option and recommends the cheapest option that meets both targets.

## Case study

`src/apply_rules.py` reads `case_study/log_eksperimen_faseF_H100.csv` and reproduces the noise calibration (Table 2), the screening (Table 3), the shared learning-rate rule, the confirmation (Table 4), the comparison with two common rules, and the cost of the protocol.

The run log keeps the original Indonesian column names: `nama` (design), `varian` (variant), `seed`, `hash_kode` (commit of the code), `mulai`/`selesai` (start/end, UTC), `durasi_menit` (minutes), `epoch_terbaik` (best epoch), `val_macro_f1` and `val_g_mean` (validation scores on fold 0), and `status` (`selesai` = finished). Design names follow the decision log (see the abbreviations below), for example `F5a_tfidf_10k` for the TF-IDF vocabulary of 10,000 features.

## Development history and decision log

| File | Content |
|---|---|
| `development_history/riwayat_pengembangan.bundle` | Git bundle of the original private development repository, from its first commit (12 October 2026) to commit `0775e6f` (16 October 2026, end of development). Commit codes are unchanged. |
| `development_history/log_keputusan.csv` | Decision log as written during development (Indonesian, authoritative version). Identical to the file in commit `0775e6f`. |
| `development_history/decision_log_en.csv` | English translation of the decision log, row by row. Decimal commas are written as points. |
| `development_history/log_uji_redacted.csv` | Test-opening audit file: one row each time the prediction script opened a test fold, without the test-score columns (see below). |
| `development_history/log_fase_F.txt` | Console log of the development stage (Stage F), written by the run scheduler. |
| `development_history/log_pengembangan.csv` | Development summary written by the run scheduler, one row per Stage F run. |

**Duplicate rows in `log_pengembangan.csv`.** The file has 61 rows for 42 runs. Nineteen runs of 14 October appear twice, once dated 14 October and once dated 15 October, with identical scores. On 14 October the pod ran under a 7-hour safety timer: the first run started at 09:32 UTC, the nineteenth ended at 16:29 UTC, and the pod stopped before the next run finished. When the scheduler was restarted on 15 October it skipped the finished runs and wrote their summary rows again. The console log shows this at lines 624 to 715 (`sudah selesai, dilewati`, i.e. "already finished, skipped"), and `case_study/log_eksperimen_faseF_H100.csv` holds one row per run. No run was trained twice.

**Test-opening audit file.** The prediction script appends one row to `log_uji.csv` each time it opens a test fold, after its three checks have passed. No such file could be written during the development stage: the run scheduler used there never calls the prediction script in Stage F (`FASE_TANPA_UJI = {"F"}` in `src/jalankan_fase.py` at commits `dab3b52` and `85a5224`, with a built-in test of this behaviour), and the prediction script is the only program that writes `log_uji.csv`. The file released here comes from the final testing stage of the related study (NVIDIA A40) and has 90 rows; the first opening is dated 18 October 2026 at 01:45 UTC, two days after the closing row of the decision log.

The full file also holds the test scores of that study, which has not yet been published. Those seven columns (`uji_macro_f1`, `uji_g_mean`, `uji_akurasi`, `uji_n_recall_kecil`, `acak_lokasi_macro_f1`, `acak_waktu_macro_f1`, `acak_keduanya_macro_f1`) are removed in `log_uji_redacted.csv`. No row was removed or edited, and every value is copied as text from the original. The kept columns include the gate check: `val_macro_f1_latih` (validation Macro-F1 recorded at training) and `val_macro_f1_ulang` (recomputed from the saved model before the test fold was opened), which are equal in all 90 rows, and `selisih_logit_validasi_maks` (largest logit difference between the two, 0 in all rows). The full file will be added after the related study is published. Its SHA-256 checksum is

`899d749c73f75b7a3e83d14333e5cf3ba56a5bb2a050d51ef35443698757f116  log_uji.csv`

so the added file can be checked against this record. The commits recorded in this file (`0def485`, `38b3755`, `efd97cb`) belong to the testing stage and come after the end of the released history (`0775e6f`).

### How to inspect the history

Git is required. The bundle holds one branch, `riwayat-pengembangan`.

```
git clone -b riwayat-pengembangan development_history/riwayat_pengembangan.bundle history
git -C history log --format="%h %ci %s"
git -C history show dab3b52:log_keputusan.csv
git -C history show e6de11f:log_keputusan.csv
git -C history diff dab3b52 e6de11f -- log_keputusan.csv
```

The code that was actually run during development (`history/src/*.py`) is kept exactly as it was, with Indonesian names and comments, because it is part of the evidence. The four scripts in `src/` of this repository were written afterwards, in English, to reproduce the article.

Figures in the decision log were computed during development from scores rounded to two decimals. `apply_rules.py` recomputes them from the unrounded scores in the run log, which gives the values reported in the article: pooled sigma 0.79 instead of 0.80, a learning-rate mean of +0.56 instead of +0.55, and a confirmation mean of -1.01 instead of -1.00 for the per-dimension context gate. Every decision is the same under both, because the tie band is 0.8 in either case.

The bundle contains code, configuration files, and logs only. No complaint text, label file, or model weight was ever committed to the development repository.

### Time zones

Commit times are in Western Indonesia Time (WIB, UTC+7). Times in the run log and in the console log come from the RunPod pod clock, which is in UTC. Every run records the commit it used, and a run cannot use code that did not yet exist, which confirms the UTC reading: the first run (09:32 in the log) uses commit `dab3b52`, committed at 16:11 WIB (09:11 UTC); the first CFPB run (16:39 in the log) uses commit `85a5224`, committed at 23:23 WIB (16:23 UTC).

### Timeline (WIB)

| Event | Time (WIB) | Evidence |
|---|---|---|
| Rules for calibration, tie band, screening, confirmation, shared learning rate, and dataset roles committed | 14 Oct 16:11 | commit `dab3b52`, log rows 1 to 6 |
| First development run starts | 14 Oct 16:32 | run log, row 1 |
| Withdrawal of the learning-rate candidate and third-party grouping rule committed ("Block 2a not yet run") | 15 Oct 14:28 | commit `e6de11f`, log rows 8 and 11 |
| First confirmation run starts | 15 Oct 14:36 | run log |
| Confirmation results and CFPB tuning rule committed | 15 Oct 23:23 | commit `85a5224`, log rows 14 and 15 |
| Last development run ends | 16 Oct 04:36 | run log |
| Closing row: development finished, all designs and hyperparameters locked | 16 Oct 12:46 | commit `b9f1433`, log row 16 |
| Analysis plan for the test data (primary and secondary tests, Holm correction) committed | 16 Oct 17:38 | commit `0775e6f`, log row 19 |
| Test data opened for the first time | 18 Oct | `log_uji_redacted.csv` |

Commit times are recorded by the authors' own computer. The bundle proves the content and order of the commits, and the commit codes recorded by each run tie the runs to that order.

### Abbreviations used in the decision log

| Code | Meaning (term used in the article) |
|---|---|
| Stage (Fase) A to E | Setup, code tests, and a 1-epoch pilot before development |
| Stage F | Development stage described in the article (screening, calibration, confirmation) |
| Stage G to H | Main training on all folds and first opening of the test data |
| Stage I | Reading of the test results with the supervisors |
| Stage J, K | Additional variants of the original study (Stage K was later cancelled) |
| Block 1 | Screening with seed 42 plus noise calibration (seeds 43 and 44) |
| Block 2a | Confirmation on LaporGub with seeds 43, 44, 45 |
| Block 2b | Learning-rate tuning on CFPB |
| F-1 to F-5 | Rule numbers: F-1 noise calibration, F-2 tie band, F-3 screening and mechanism route, F-4 confirmation, F-5 combination |
| V1 | PLM only (reference variant in the noise calibration) |
| V2, V3 | Concatenation without a gate; gate without f_deg |
| V4 | Gate with f_deg |
| V5 | V4 with f_deg measured after word normalization |
| V6 | V4 with location and time concatenated |
| V7 | Full model, the incumbent design in the article |
| V4s | V4 with f_deg computed without CFPB masking tokens |
| F1a to F1f | The six gate input forms (ways of feeding f_deg to the gate) |
| F2a, F2b | Context module designs; F2b is the per-dimension (vector) context gate |
| F3a, F3b | Path normalization designs |
| F4a to F4c | Learning-rate candidates on V7; F4d to F4f are the same settings on V1, used for the averaging rule |
| F5a to F5c | TF-IDF settings (F5a = vocabulary of 10,000 features) |
| M1, M2, K1 | Mechanism criteria: M1 variation of the fusion gate across complaints, M2 within-class correlation between the gate and f_deg, K1 opening of the context gate |
| alpha, a_L, a_D | Fusion gate; location gate; time (date) gate |
| TGCA | The location and time context gate of the incumbent design |
| CGCA | An alternative context design that was considered and not used |
| P2 | Check without a PLM: information on the class carried by location and time alone (log-loss reduction) |
| H1, H2 | Primary test comparisons: V2 vs V1 and V3 vs V2 |
| H4, H4a, H4b | V4 vs V3 (LaporGub); V5 vs V4; V4 vs V3 on CFPB without the Credit reporting class |
| H5, H5a, H6 | V6 vs V4; V7 vs V4 on the least confident fifth of complaints; V7 vs V6 |
| H7 | V7 on CFPB (not run) |
| H8 | Main loss function vs standard cross-entropy |
| Chapter 6 (Bab 6) form | The design as specified in the research plan before development |
| sb | Standard deviation |

## Data

`data/laporgub_ids.csv` lists the 59,251 LaporGub complaints of the case study with their date, regency or city, label, near-duplicate group, fold assignment, role (train, validation, or test) in each of the five folds, and the precomputed subword fragmentation features (`f_deg`, `f_deg_refined`), and `data/label_laporgub.csv` gives the 29 classes of government affairs in Indonesian and English. Both files are identical copies of the files in the repository of the related classification study (checksums in `SHA256SUMS.txt`). The case study uses only the validation data of fold 0.

The complaint texts are not redistributed, because they are written freely by citizens and may contain personal information. Each complaint can be opened on the LaporGub portal of the Central Java Provincial Government at `https://laporgub.jatengprov.go.id/detail/<complaint_id>.html`. Please retrieve texts responsibly and avoid high-frequency automated requests.

## License

- Code (`src/`, and the code inside the git bundle): MIT License (see `LICENSE`).
- Logs, results, IDs, labels, and folds (`case_study/`, `development_history/` except the code in the bundle, `results/`, `data/`): Creative Commons Attribution 4.0 International (see `LICENSE-DATA`).

## Citation

Please cite the article above and the archived version of this repository on Zenodo: https://doi.org/10.5281/zenodo.24286518 (this DOI always resolves to the latest version).
