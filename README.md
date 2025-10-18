

# FAST — Feature Activation-induced Shapelet Transform

**FAST** is a two-stage, CPU-efficient time-series classifier that joins multiROCKET speed with shapelet interpretability.

* **Stage-1 (MR-style kernels):** DC-balanced short kernels (K=9; optionally K=6, K=3) with dilations, bias (threshold) calibration via LDS quantiles, and 4 statistics per bias (optionally +1 max). This reproduces the spirit of multiROCKET and yields a strong, fast baseline.
* **Stage-2 (harvest shapelets):** Use the **arg-max activations** from Stage-1 to cut actual subsequences from the training series (the “activation-induced” shapelets). Keep a candidate only if its features (same stats/biases as its kernel parent) **improve ANOVA η²** over the parent. Features are computed on base and first-difference signals, then concatenated.

---

## Environment

We recommend Python **3.9–3.11**. 

---

## Quick start

Run a single dataset benchmark with all six ablations:

```bash
# from the repo root
python -m run.main --dataset GunPoint --threads 20
```

The output will display:

* Number of features:

  * `kernel_budget_total` (defaults scale as below)
  * `shapelet_budget_total = kernel_budget_total // 2`
* Feature matrix shapes for:

  * **K369** kernels, shapelets, and both
* Final accuracies for:

  * K9 kernels | K9 shapelets | K9 both
  * K369 kernels | K369 shapelets | K369 both

### Useful flags (selected)

* `--dataset NAME` — any UCR/UEA name (downloaded via `sktime.datasets`)
* `--threads N` — Numba threads
* `--include_max_feature {0,1}` — add “max activation” feature
* `--kernel_budget_total INT` — override default kernel budget
* `--shapelet_budget_total INT` — override default shapelet budget


---

## Reproducibility notes

* **Data**: 109 UCR datasets with official splits; 10 resampling using origianl train/test ratio.

or simply:
```bash
# from the repo root
python -m run.run_resample --threads 20 --also_official_split
```



