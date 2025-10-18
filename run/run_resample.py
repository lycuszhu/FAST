# FAST/run/run_resample.py
from __future__ import annotations

import argparse
import csv
import os
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression

from codes.utils import load_ucr_uea_sktime, set_numba_threads_count
from codes.fast_functions import fast


# -----------------------
# Dataset groups
# -----------------------
# From your tuning outcome: 17 datasets where logreg generally outperforms ridge
LOGREG_DATASETS = {
    "BeetleFly", "DistalPhalanxOutlineAgeGroup", "DistalPhalanxTW", "EthanolLevel",
    "Ham", "ItalyPowerDemand", "Meat", "MoteStrain", "OliveOil", "Phoneme",
    "PowerCons", "ProximalPhalanxTW", "Rock", "Symbols", "UMD", "Wine",
    "WormsTwoClass"
}

# Default list (same broad list used elsewhere); you can override with --datasets/--datasets_file
DEFAULT_DATASETS = [
    "ACSF1","Adiac","ArrowHead","BME","Beef","BeetleFly","BirdChicken","CBF","Car","Chinatown",
    "ChlorineConcentration","CinCECGTorso","Coffee","Computers","CricketX","CricketY","CricketZ","Crop",
    "DiatomSizeReduction","DistalPhalanxOutlineAgeGroup","DistalPhalanxOutlineCorrect","DistalPhalanxTW",
    "ECG200","ECG5000","ECGFiveDays","EOGHorizontalSignal","EOGVerticalSignal","Earthquakes","ElectricDevices",
    "EthanolLevel","FaceAll","FaceFour","FacesUCR","FiftyWords","Fish","FordA","FordB","FreezerRegularTrain",
    "FreezerSmallTrain","GunPoint","GunPointAgeSpan","GunPointMaleVersusFemale","GunPointOldVersusYoung",
    "Ham","Haptics","Herring","HouseTwenty","InlineSkate","InsectEPGRegularTrain","InsectEPGSmallTrain",
    "InsectWingbeatSound","ItalyPowerDemand","LargeKitchenAppliances","Lightning2","Lightning7","Mallat",
    "Meat","MedicalImages","MiddlePhalanxOutlineAgeGroup","MiddlePhalanxOutlineCorrect","MiddlePhalanxTW",
    "MixedShapesRegularTrain","MixedShapesSmallTrain","MoteStrain","OSULeaf","OliveOil","PhalangesOutlinesCorrect",
    "Phoneme","PigAirwayPressure","PigArtPressure","PigCVP","Plane","PowerCons","ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxOutlineCorrect","ProximalPhalanxTW","RefrigerationDevices","Rock","ScreenType",
    "SemgHandGenderCh2","SemgHandMovementCh2","SemgHandSubjectCh2","ShapeletSim","ShapesAll","SmallKitchenAppliances",
    "SmoothSubspace","SonyAIBORobotSurface1","SonyAIBORobotSurface2","StarLightCurves","Strawberry","SwedishLeaf",
    "Symbols","SyntheticControl","ToeSegmentation1","ToeSegmentation2","Trace","TwoLeadECG","TwoPatterns","UMD",
    "UWaveGestureLibraryAll","UWaveGestureLibraryX","UWaveGestureLibraryY","UWaveGestureLibraryZ","Wafer","Wine",
    "WordSynonyms","Worms","WormsTwoClass","Yoga"
]


# -----------------------
# Helpers
# -----------------------
def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _load_done_keys(csv_path: str) -> set[Tuple]:
    if not os.path.exists(csv_path):
        return set()
    keys = set()
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # resume key: (dataset, resample, include_max)
            try:
                keys.add((row["dataset"], int(row["resample"]), int(row["include_max"])))
            except Exception:
                continue
    return keys


def _append_row(csv_path: str, header: List[str], row: Dict):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow(row)
        f.flush()


def _train_eval(
    Xtr: np.ndarray, ytr: np.ndarray,
    Xte: np.ndarray, yte: np.ndarray,
    clf_type: str, alpha_grid: np.ndarray,
    max_iter_logreg: int, C_logreg: float
) -> Tuple[float, float, float]:
    """Fit & eval; return (acc, fit_time, pred_time)."""
    if Xtr is None or Xtr.shape[1] == 0:
        return (float("nan"), 0.0, 0.0)
    if clf_type == "ridge":
        clf = RidgeClassifierCV(alphas=alpha_grid)
    else:
        clf = LogisticRegression(
            penalty="l2", C=C_logreg, solver="lbfgs", max_iter=max_iter_logreg, n_jobs=None
        )
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
    return (acc, fit_s, pred_s)


def _shapelet_family_column_masks(F) -> Dict[str, np.ndarray]:
    """Boolean masks over concatenated shapelet matrix to select families."""
    kept = getattr(F, "kept_shapelets_", [])
    if not kept:
        return {"K9s3": np.zeros((0,), dtype=bool), "K6s2": np.zeros((0,), dtype=bool), "K3s1": np.zeros((0,), dtype=bool)}
    stats_per_bias = 4 + (1 if F.include_max_feature else 0)
    cursor = 0
    spans: List[Tuple[str, int, int]] = []
    for shp in kept:
        fam = shp.parent.family.name
        cb = stats_per_bias * int(shp.thr_base.shape[0])
        cd = stats_per_bias * int(shp.thr_diff.shape[0])
        if cb > 0:
            spans.append((fam, cursor, cursor + cb))
            cursor += cb
        if cd > 0:
            spans.append((fam, cursor, cursor + cd))
            cursor += cd
    total_cols = cursor
    masks = {"K9s3": np.zeros((total_cols,), dtype=bool),
             "K6s2": np.zeros((total_cols,), dtype=bool),
             "K3s1": np.zeros((total_cols,), dtype=bool)}
    for fam, s, e in spans:
        masks[fam][s:e] = True
    return masks


def _kernel_family_column_mask(F, family_name: str) -> np.ndarray:
    reg = getattr(F, "kernel_feature_registry_", None)
    parents = getattr(F, "ranked_parents_", [])
    if reg is None or len(parents) == 0 or F._Xtr_kernels is None:
        return np.zeros((0,), dtype=bool)
    total_cols = F._Xtr_kernels.shape[1]
    mask = np.zeros((total_cols,), dtype=bool)
    for pk in parents:
        if pk.family.name == family_name:
            s, e = reg[pk]
            mask[s:e] = True
    return mask


def _run_one_case(
    ds: str, include_max: int, clf_type: str,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    args: argparse.Namespace
) -> Dict:
    """Assemble K369 once, optionally run a fair K9 pass, train 6 heads, return row dict."""
    n_tr, n_te, L = len(y_train), len(y_test), X_train.shape[1]

    # Fixed params per your spec
    delta_eta2 = 0.01
    sigma_optimism = 0.25
    keep_best = True
    shapelet_bias_mode = "recalibrate_small"
    lds_offset = 0.0

    # Build K369 once (kernels+shapelets)
    F369 = fast(
        include_max_feature=bool(include_max),
        additional_kernel_families=True,   # K369
        kernel_budget_total=args.kernel_budget_total,    # None → default scaling inside fast()
        shapelet_budget_total=args.shapelet_budget_total,  # None → half of kernel
        delta_eta2=delta_eta2,
        sigma_optimism=sigma_optimism,
        shapelet_bias_mode=shapelet_bias_mode,
        shapelet_subsample_frac=0.20,
        shapelet_subsample_min=10,
        shapelet_subsample_max=256,
        eb_bias_source="single_series",
        keep_best_dilation_only=keep_best,
        lds_offset=lds_offset,
    )
    # Assemble (both)
    tA = time.time()
    Xtr_both_369, Xte_both_369, _ = F369.assemble_features(X_train, y_train, X_test, features_used="both")
    asm_s = round(time.time() - tA, 4)

    # Prepare K369 views
    Xtr_ker_369 = F369._Xtr_kernels
    Xte_ker_369 = F369._Xte_kernels
    Xtr_shp_369 = F369._Xtr_shapelets
    Xte_shp_369 = F369._Xte_shapelets
    Xtr_both_369 = np.hstack([Xtr_ker_369, Xtr_shp_369]).astype(np.float32, copy=False)
    Xte_both_369 = np.hstack([Xte_ker_369, Xte_shp_369]).astype(np.float32, copy=False)

    # Slice K9 shapelets (family K9s3) from K369 shapelet matrix
    shp_masks = _shapelet_family_column_masks(F369)
    mask_k9_shp = shp_masks.get("K9s3", np.zeros((0,), dtype=bool))
    Xtr_shp_k9 = (Xtr_shp_369[:, mask_k9_shp]
                  if Xtr_shp_369 is not None and Xtr_shp_369.shape[1] else np.zeros((n_tr, 0), np.float32))
    Xte_shp_k9 = (Xte_shp_369[:, mask_k9_shp]
                  if Xte_shp_369 is not None and Xte_shp_369.shape[1] else np.zeros((n_te, 0), np.float32))

    # K9 kernels: either approximate (slice) or fair separate K9 pass
    if args.approx_k9_from_k369:
        mask_k9_ker = _kernel_family_column_mask(F369, "K9s3")
        Xtr_ker_9 = (Xtr_ker_369[:, mask_k9_ker]
                     if Xtr_ker_369 is not None and Xtr_ker_369.shape[1] else np.zeros((n_tr, 0), np.float32))
        Xte_ker_9 = (Xte_ker_369[:, mask_k9_ker]
                     if Xte_ker_369 is not None and Xte_ker_369.shape[1] else np.zeros((n_te, 0), np.float32))
    else:
        F9 = fast(
            include_max_feature=bool(include_max),
            additional_kernel_families=False,   # K9 only
            kernel_budget_total=args.kernel_budget_total,
            shapelet_budget_total=0,            # no shapelets in K9-only pass
        )
        try:
            Xtr_ker_9, Xte_ker_9, _ = F9.assemble_features(X_train, y_train, X_test, features_used="kernels")
        except Exception:
            Xtr_ker_9 = np.zeros((n_tr, 0), dtype=np.float32)
            Xte_ker_9 = np.zeros((n_te, 0), dtype=np.float32)

    # K9 both (kernels + its corresponding shapelets sliced from K369)
    Xtr_both_k9 = np.hstack([Xtr_ker_9, Xtr_shp_k9]).astype(np.float32, copy=False)
    Xte_both_k9 = np.hstack([Xte_ker_9, Xte_shp_k9]).astype(np.float32, copy=False)

    # Train heads (single classifier per dataset group)
    agrid = np.logspace(np.log10(args.alpha_low), np.log10(args.alpha_high), args.alpha_points)
    Clog, maxit = args.C_logreg, args.max_iter_logreg
    clf_type_local = clf_type

    acc_k9_ker,  _, _ = _train_eval(Xtr_ker_9,   y_train, Xte_ker_9,   y_test, clf_type_local, agrid, maxit, Clog)
    acc_k9_shp,  _, _ = _train_eval(Xtr_shp_k9,  y_train, Xte_shp_k9,  y_test, clf_type_local, agrid, maxit, Clog)
    acc_k9_both, _, _ = _train_eval(Xtr_both_k9, y_train, Xte_both_k9, y_test, clf_type_local, agrid, maxit, Clog)

    acc_ker369,  _, _ = _train_eval(Xtr_ker_369,   y_train, Xte_ker_369,   y_test, clf_type_local, agrid, maxit, Clog)
    acc_shp369,  _, _ = _train_eval(Xtr_shp_369,   y_train, Xte_shp_369,   y_test, clf_type_local, agrid, maxit, Clog)
    acc_both369, _, _ = _train_eval(Xtr_both_369,  y_train, Xte_both_369,  y_test, clf_type_local, agrid, maxit, Clog)

    n_kernels = len(getattr(F369, "ranked_parents_", []))
    n_shapelets = len(getattr(F369, "kept_shapelets_", []))

    row = {
        "dataset": ds,
        "n_tr": n_tr,
        "n_te": n_te,
        "L": L,
        "include_max": include_max,
        "clf_type": clf_type_local,
        "assemble_time_k369": f"{asm_s:.4f}",
        "n_kernels": n_kernels,
        "n_shapelets": n_shapelets,
        "kernel_budget": F369.kernel_budget_total,
        "shapelet_budget": F369.shapelet_budget_total,
        "acc_k9_kernels": "NA" if np.isnan(acc_k9_ker) else f"{acc_k9_ker:.4f}",
        "acc_k9_shapelets": "NA" if np.isnan(acc_k9_shp) else f"{acc_k9_shp:.4f}",
        "acc_k9_both": "NA" if np.isnan(acc_k9_both) else f"{acc_k9_both:.4f}",
        "acc_k369_kernels": "NA" if np.isnan(acc_ker369) else f"{acc_ker369:.4f}",
        "acc_k369_shapelets": "NA" if np.isnan(acc_shp369) else f"{acc_shp369:.4f}",
        "acc_k369_both": "NA" if np.isnan(acc_both369) else f"{acc_both369:.4f}",
    }
    return row


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=str, default="", help="Comma-separated list, or leave empty to use DEFAULT_DATASETS")
    ap.add_argument("--datasets_file", type=str, default="", help="Optional: path to a txt file with one dataset per line")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--num_resamples", type=int, default=10)
    ap.add_argument("--base_seed", type=int, default=0)
    ap.add_argument("--approx_k9_from_k369", action="store_true", help="Slice K9 kernels from K369 matrix instead of separate K9 pass")
    ap.add_argument("--also_official_split", action="store_true", help="Run the original UCR split once per dataset (include_max=0 and 1) in addition to resamples")
    # Optional budget overrides; defaults are handled inside fast()
    ap.add_argument("--kernel_budget_total", type=int, default=None)
    ap.add_argument("--shapelet_budget_total", type=int, default=None)
    # Classifier grids
    ap.add_argument("--alpha_low", type=float, default=1e-1)
    ap.add_argument("--alpha_high", type=float, default=1e3)
    ap.add_argument("--alpha_points", type=int, default=9)
    ap.add_argument("--C_logreg", type=float, default=1.0)
    ap.add_argument("--max_iter_logreg", type=int, default=200)
    # Output
    ap.add_argument("--out_csv", type=str, default=os.path.join("results", "results_resample.csv"))
    args = ap.parse_args()

    if args.threads and args.threads > 0:
        set_numba_threads_count(args.threads)

    # dataset list
    datasets: List[str] = []
    if args.datasets_file:
        with open(args.datasets_file, "r", encoding="utf-8") as f:
            datasets = [ln.strip() for ln in f if ln.strip()]
    elif args.datasets:
        datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    else:
        datasets = DEFAULT_DATASETS

    # ridge vs logreg split
    ridge_datasets = [d for d in datasets if d not in LOGREG_DATASETS]
    logreg_datasets = [d for d in datasets if d in LOGREG_DATASETS]

    print(f"Datasets: {len(datasets)} (ridge={len(ridge_datasets)}, logreg={len(logreg_datasets)})")
    print(f"Writing results to: {args.out_csv}")

    _ensure_dir(args.out_csv)
    done_keys = _load_done_keys(args.out_csv)

    header = [
        "dataset", "resample", "include_max", "clf_type",
        "assemble_time_k369", "n_kernels", "n_shapelets",
        "kernel_budget", "shapelet_budget",
        "n_tr", "n_te", "L",
        "acc_k9_kernels", "acc_k9_shapelets", "acc_k9_both",
        "acc_k369_kernels", "acc_k369_shapelets", "acc_k369_both",
    ]

    # Iterate datasets
    for ds in datasets:
        clf_type = "logreg" if ds in LOGREG_DATASETS else "ridge"

        # --- Optional official UCR split (resample = -1) ---
        if args.also_official_split:
            for include_max in (0, 1):
                key = (ds, -1, include_max)
                if key not in done_keys:
                    try:
                        X_train, y_train, X_test, y_test = load_ucr_uea_sktime(
                            ds, merge_and_resplit=False, random_state=0, stratify=True
                        )
                        row = _run_one_case(ds, include_max, clf_type, X_train, y_train, X_test, y_test, args)
                        row["dataset"] = ds
                        row["resample"] = -1
                        row["include_max"] = include_max
                        row["clf_type"] = clf_type
                        _append_row(args.out_csv, header, row)
                        done_keys.add(key)
                        print(f"[{ds} official max={include_max} | {clf_type}] K9 both={row['acc_k9_both']} | "
                              f"K369 both={row['acc_k369_both']} | kbud={row['kernel_budget']} sbud={row['shapelet_budget']} "
                              f"| asm={row['assemble_time_k369']}s")
                    except Exception as e:
                        print(f"[{ds} official max={include_max}] failed: {e}")
                        fail = {
                            "dataset": ds, "resample": -1, "include_max": include_max, "clf_type": clf_type,
                            "assemble_time_k369": "FAIL", "n_kernels": 0, "n_shapelets": 0,
                            "kernel_budget": "NA", "shapelet_budget": "NA",
                            "n_tr": 0, "n_te": 0, "L": 0,
                            "acc_k9_kernels": "NA", "acc_k9_shapelets": "NA", "acc_k9_both": "NA",
                            "acc_k369_kernels": "NA", "acc_k369_shapelets": "NA", "acc_k369_both": "NA",
                        }
                        _append_row(args.out_csv, header, fail)
                        done_keys.add(key)

        # --- Resamples (merge & resplit preserving official fractions) ---
        for r in range(args.num_resamples):
            seed = args.base_seed + r
            for include_max in (0, 1):
                key = (ds, r, include_max)
                if key in done_keys:
                    continue
                try:
                    X_train, y_train, X_test, y_test = load_ucr_uea_sktime(
                        ds, merge_and_resplit=True, random_state=seed, stratify=True
                    )
                    row = _run_one_case(ds, include_max, clf_type, X_train, y_train, X_test, y_test, args)
                    row["dataset"] = ds
                    row["resample"] = r
                    row["include_max"] = include_max
                    row["clf_type"] = clf_type
                    _append_row(args.out_csv, header, row)
                    done_keys.add(key)
                    print(f"[{ds} r{r} max={include_max} | {clf_type}] K9 both={row['acc_k9_both']} | "
                          f"K369 both={row['acc_k369_both']} | kbud={row['kernel_budget']} sbud={row['shapelet_budget']} "
                          f"| asm={row['assemble_time_k369']}s")
                except Exception as e:
                    print(f"[{ds} r{r} max={include_max}] failed: {e}")
                    fail = {
                        "dataset": ds, "resample": r, "include_max": include_max, "clf_type": clf_type,
                        "assemble_time_k369": "FAIL", "n_kernels": 0, "n_shapelets": 0,
                        "kernel_budget": "NA", "shapelet_budget": "NA",
                        "n_tr": 0, "n_te": 0, "L": 0,
                        "acc_k9_kernels": "NA", "acc_k9_shapelets": "NA", "acc_k9_both": "NA",
                        "acc_k369_kernels": "NA", "acc_k369_shapelets": "NA", "acc_k369_both": "NA",
                    }
                    _append_row(args.out_csv, header, fail)
                    done_keys.add(key)

    print("\nAll requested runs attempted. Results appended to:", args.out_csv)


if __name__ == "__main__":
    main()
