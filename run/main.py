# FAST/run/main.py
# Example usage:
# Default (ridge heads, EB from a single series, K369 harvesting, 20 threads):
#   python -m run.main --dataset Computers --threads 20
#
# Try logistic regression heads:
#   python -m run.main --dataset GunPoint --threads 20 --clf_type logreg
#
# Tighten EB gate:
#   python -m run.main --dataset ItalyPowerDemand --delta_eta2 0.03 --sigma_optimism 0.

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression

from codes.utils import (
    load_ucr_uea_sktime,
    set_numba_threads_count,
)
from codes.fast_functions import fast


@dataclass
class HeadResult:
    name: str
    acc: float
    fit_s: float
    pred_s: float
    Xtr_shape: Tuple[int, int]
    Xte_shape: Tuple[int, int]


def _train_eval(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    clf_type: str,
    alpha_grid: np.ndarray,
    max_iter_logreg: int,
    C_logreg: float,
    label: str,
) -> HeadResult:
    if Xtr is None or Xtr.shape[1] == 0:
        return HeadResult(label, float("nan"), 0.0, 0.0, (Xtr.shape[0], 0), (Xte.shape[0], 0))

    if clf_type == "ridge":
        clf = RidgeClassifierCV(alphas=alpha_grid)
    else:
        clf = LogisticRegression(penalty="l2", C=C_logreg, solver="lbfgs", max_iter=max_iter_logreg, n_jobs=None)

    pipe = make_pipeline(StandardScaler(with_mean=False), clf)

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(Xtr, ytr)
    fit_s = time.time() - t0

    t1 = time.time()
    yp = pipe.predict(Xte)
    pred_s = time.time() - t1
    acc = accuracy_score(yte, yp)

    return HeadResult(label, acc, fit_s, pred_s, Xtr.shape, Xte.shape)


def _shapelet_family_column_masks(F: fast) -> Dict[str, np.ndarray]:
    """
    Rebuild per-family boolean masks over the concatenated shapelet feature matrix
    from F.kept_shapelets_. Ordering in fast.assemble_features (current version) is:
      for each kept shapelet: [RAW-channel block] appended in sequence.
    Each block contributes (#biases * stats_per_bias) columns.
    """
    kept = F.kept_shapelets_
    n = len(kept)
    if n == 0:
        return {"K9s3": np.zeros((0,), dtype=bool), "K6s2": np.zeros((0,), dtype=bool), "K3s1": np.zeros((0,), dtype=bool)}

    stats_per_bias = 4 + (1 if F.include_max_feature else 0)

    # Build spans
    spans: List[Tuple[str, int, int]] = []
    cursor = 0
    for shp in kept:
        fam = shp.parent.family.name
        c_base = stats_per_bias * int(shp.thr_base.shape[0])  # RAW only
        if c_base > 0:
            spans.append((fam, cursor, cursor + c_base))
            cursor += c_base
    total_cols = cursor

    masks = {
        "K9s3": np.zeros((total_cols,), dtype=bool),
        "K6s2": np.zeros((total_cols,), dtype=bool),
        "K3s1": np.zeros((total_cols,), dtype=bool),
    }
    for fam, s, e in spans:
        masks[fam][s:e] = True
    return masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, help="UCR/UEA dataset name, e.g. GunPoint")
    ap.add_argument("--threads", type=int, default=0, help="Numba threads (0=leave default)")
    ap.add_argument("--include_max_feature", type=int, default=0,
                    help="Add 5th stat 'max' (applies to kernels and shapelets) (0/1)")
    ap.add_argument("--kernel_budget_total", type=int, default=None,
                    help="Total kernel columns; None => base 50k (K9). If K369 enabled: ×1.20; if include_max: ×1.25.")
    ap.add_argument("--shapelet_budget_total", type=int, default=None,
                    help="Total shapelet columns; None => 0.5 * kernel budget")
    ap.add_argument("--delta_eta2", type=float, default=0.02, help="Minimum eta^2 improvement over parent")
    ap.add_argument("--sigma_optimism", type=float, default=0.25, help="Optimism slack at EB")
    ap.add_argument("--shapelet_bias_mode", type=str, default="reuse", choices=["recalibrate_small", "reuse"])
    ap.add_argument("--eb_bias_source", type=str, default="single_series", choices=["single_series", "subseries"])
    ap.add_argument("--keep_best_dilation_only", type=int, default=0)
    ap.add_argument("--shapelet_subsample_frac", type=float, default=0.20)
    ap.add_argument("--shapelet_subsample_min", type=int, default=10)
    ap.add_argument("--shapelet_subsample_max", type=int, default=256)
    ap.add_argument("--clf_type", type=str, default="ridge", choices=["ridge", "logreg"])
    ap.add_argument("--alpha_low", type=float, default=1e-1)
    ap.add_argument("--alpha_high", type=float, default=1e3)
    ap.add_argument("--alpha_points", type=int, default=9)
    args = ap.parse_args()

    if args.threads and args.threads > 0:
        set_numba_threads_count(args.threads)

    print(f">> Dataset: {args.dataset} | threads={args.threads or 'default'} | include_max={args.include_max_feature} | "
          f"kernel_budget_total={args.kernel_budget_total} | shapelet_budget_total={args.shapelet_budget_total} | "
          f"eb_bias_source={args.eb_bias_source}")

    # Load data
    t0 = time.time()
    X_train, y_train, X_test, y_test = load_ucr_uea_sktime(args.dataset)
    load_s = time.time() - t0
    n_tr, n_te, L = X_train.shape[0], X_test.shape[0], X_train.shape[1]
    print(f"Loaded: n_tr={n_tr}, n_te={n_te}, L={L} | {load_s:.2f}s")

    # -------------------------------
    # K369 run (families = K9+K6+K3)
    # -------------------------------
    F369 = fast(
        include_max_feature=bool(args.include_max_feature),
        additional_kernel_families=True,
        kernel_budget_total=args.kernel_budget_total,
        shapelet_budget_total=args.shapelet_budget_total,
        delta_eta2=args.delta_eta2,
        sigma_optimism=args.sigma_optimism,
        shapelet_bias_mode=args.shapelet_bias_mode,
        shapelet_subsample_frac=args.shapelet_subsample_frac,
        shapelet_subsample_min=args.shapelet_subsample_min,
        shapelet_subsample_max=args.shapelet_subsample_max,
        eb_bias_source=args.eb_bias_source,
        keep_best_dilation_only=bool(args.keep_best_dilation_only),
    )

    t1 = time.time()
    Xtr_both_369, Xte_both_369, _ = F369.assemble_features(X_train, y_train, X_test, features_used="both")
    asm369_s = time.time() - t1

    print(f"Resolved budgets[K369]: kernel={F369.kernel_budget_total} | shapelet={F369.shapelet_budget_total}")
    print(f"Features[K369]: kernels={F369._Xtr_kernels.shape} | shapelets={F369._Xtr_shapelets.shape} | both={Xtr_both_369.shape} | assemble {asm369_s:.2f}s")
    print(f"Counts[K369]: n_kernels={len(F369.ranked_parents_)} | n_shapelets={len(F369.kept_shapelets_)}")

    # Family masks for shapelets (RAW-only blocks)
    shp_masks = _shapelet_family_column_masks(F369)
    shp_cols_k9 = shp_masks.get("K9s3", np.zeros((0,), dtype=bool))

    # Shapelet views
    Xtr_shp_369 = F369._Xtr_shapelets
    Xte_shp_369 = F369._Xte_shapelets
    Xtr_shp_k9 = F369._Xtr_shapelets[:, shp_cols_k9] if F369._Xtr_shapelets.shape[1] > 0 else np.zeros((n_tr, 0), dtype=np.float32)
    Xte_shp_k9 = F369._Xte_shapelets[:, shp_cols_k9] if F369._Xte_shapelets.shape[1] > 0 else np.zeros((n_te, 0), dtype=np.float32)

    # Kernel views (K369 => from F369; K9 => run a tiny K9-only Step-1)
    Xtr_ker_369 = F369._Xtr_kernels
    Xte_ker_369 = F369._Xte_kernels

    # Small K9-only kernel run (no shapelets) to get a clean K9 kernel matrix
    F9 = fast(
        include_max_feature=False,               # kernels use 4 stats; parity with MR
        additional_kernel_families=False,        # K9 only
        kernel_budget_total=args.kernel_budget_total,  # None => base 50k
        shapelet_budget_total=0,                 # no shapelets here
    )
    t1b = time.time()
    Xtr_ker_9, Xte_ker_9, _ = F9.assemble_features(X_train, y_train, X_test, features_used="kernels")
    asm9_s = time.time() - t1b
    print(f"Features[K9-only kernels]: {Xtr_ker_9.shape} | assemble {asm9_s:.2f}s")

    # Both views
    Xtr_both_k9 = np.hstack([Xtr_ker_9, Xtr_shp_k9]).astype(np.float32, copy=False)
    Xte_both_k9 = np.hstack([Xte_ker_9, Xte_shp_k9]).astype(np.float32, copy=False)
    Xtr_both_369 = np.hstack([Xtr_ker_369, Xtr_shp_369]).astype(np.float32, copy=False)
    Xte_both_369 = np.hstack([Xte_ker_369, Xte_shp_369]).astype(np.float32, copy=False)

    # Classifier grid
    alphas = np.logspace(np.log10(args.alpha_low), np.log10(args.alpha_high), args.alpha_points)

    # Train 6 heads
    heads: List[HeadResult] = []
    heads.append(_train_eval(Xtr_ker_9,   y_train, Xte_ker_9,   y_test, args.clf_type, alphas, 200, 1.0, "K9 kernels"))
    heads.append(_train_eval(Xtr_shp_k9,  y_train, Xte_shp_k9,  y_test, args.clf_type, alphas, 200, 1.0, "K9 shapelets"))
    heads.append(_train_eval(Xtr_both_k9, y_train, Xte_both_k9, y_test, args.clf_type, alphas, 200, 1.0, "K9 both"))

    heads.append(_train_eval(Xtr_ker_369,   y_train, Xte_ker_369,   y_test, args.clf_type, alphas, 200, 1.0, "K369 kernels"))
    heads.append(_train_eval(Xtr_shp_369,   y_train, Xte_shp_369,   y_test, args.clf_type, alphas, 200, 1.0, "K369 shapelets"))
    heads.append(_train_eval(Xtr_both_369,  y_train, Xte_both_369,  y_test, args.clf_type, alphas, 200, 1.0, "K369 both"))

    # Report
    for h in heads:
        acc_txt = "NA" if np.isnan(h.acc) else f"{h.acc:.4f}"
        print(f"Accuracy[{h.name}]: {acc_txt} | fit {h.fit_s:.2f}s | predict {h.pred_s:.2f}s | Xtr{h.Xtr_shape} Xte{h.Xte_shape}")

    # Headline
    best = max([h for h in heads if not np.isnan(h.acc)], key=lambda z: z.acc, default=None)
    if best:
        print(f"[headline={best.name}] {args.dataset}: {best.acc:.4f} "
              f"(assemble K369 {asm369_s:.2f}s + K9 kernels {asm9_s:.2f}s, load {load_s:.2f}s)")

if __name__ == "__main__":
    main()
