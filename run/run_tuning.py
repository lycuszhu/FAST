# FAST/run/run_tuning.py
# Run with "python -m run.run_tuning --threads 20"
from __future__ import annotations

import argparse
import csv
import os
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression

from codes.utils import load_ucr_uea_sktime, set_numba_threads_count
from codes.fast_functions import fast


# -----------------------
# Default dataset list
# -----------------------
DEFAULT_DATASETS = [
    #"ACSF1","Adiac","ArrowHead","BME","Beef","BeetleFly","BirdChicken","CBF","Car","Chinatown",
    #"ChlorineConcentration","CinCECGTorso","Coffee","Computers","CricketX","CricketY","CricketZ","Crop",
    #"DiatomSizeReduction","DistalPhalanxOutlineAgeGroup","DistalPhalanxOutlineCorrect","DistalPhalanxTW",
    #"ECG200","ECG5000","ECGFiveDays","EOGHorizontalSignal","EOGVerticalSignal","Earthquakes","ElectricDevices",
    #"EthanolLevel","FaceAll","FaceFour","FacesUCR","FiftyWords","Fish","FordA","FordB","FreezerRegularTrain",
    #"FreezerSmallTrain","GunPoint","GunPointAgeSpan","GunPointMaleVersusFemale","GunPointOldVersusYoung",
    #"Ham","Haptics","Herring","HouseTwenty","InlineSkate","InsectEPGRegularTrain","InsectEPGSmallTrain",
    #"InsectWingbeatSound","ItalyPowerDemand","LargeKitchenAppliances","Lightning2","Lightning7","Mallat",
    #"Meat","MedicalImages","MiddlePhalanxOutlineAgeGroup","MiddlePhalanxOutlineCorrect","MiddlePhalanxTW",
    #"MixedShapesRegularTrain","MixedShapesSmallTrain","MoteStrain","OSULeaf","OliveOil","PhalangesOutlinesCorrect",
    #"Phoneme","PigAirwayPressure","PigArtPressure",
    "PigCVP","Plane","PowerCons","ProximalPhalanxOutlineAgeGroup",
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
@dataclass
class HeadResult:
    acc: float
    fit_s: float
    pred_s: float
    shape_tr: Tuple[int, int]
    shape_te: Tuple[int, int]


def _train_eval(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray,
                clf_type: str, alpha_grid: np.ndarray, max_iter_logreg: int, C_logreg: float) -> HeadResult:
    if Xtr is None or Xtr.shape[1] == 0:
        return HeadResult(float("nan"), 0.0, 0.0, (Xtr.shape[0], 0), (Xte.shape[0], 0))
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
    return HeadResult(acc, fit_s, pred_s, Xtr.shape, Xte.shape)


def _shapelet_family_column_masks(F: fast) -> Dict[str, np.ndarray]:
    """
    Build boolean masks over the concatenated shapelet matrix to select families.
    Ordering in assemble_features: for each kept shapelet, [base-block, diff-block] appended.
    Each block contributes (#biases * stats_per_bias) columns.
    """
    kept = getattr(F, "kept_shapelets_", [])
    if not kept:
        return {"K9s3": np.zeros((0,), dtype=bool), "K6s2": np.zeros((0,), dtype=bool), "K3s1": np.zeros((0,), dtype=bool)}
    stats_per_bias = 4 + (1 if F.include_max_feature else 0)
    total_cols = 0
    spans: List[Tuple[str, int, int]] = []
    cursor = 0
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


def _kernel_family_column_mask(F: fast, family_name: str) -> np.ndarray:
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


def _ensure_results_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _load_done_keys(csv_path: str) -> set[Tuple]:
    if not os.path.exists(csv_path):
        return set()
    keys = set()
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["dataset"], int(row["include_max"]), float(row["delta_eta2"]),
                   float(row["sigma_optimism"]), int(row["keep_best_d_only"]), row["clf_type"])
            keys.add(key)
    return keys


def _append_row(csv_path: str, header: List[str], row: Dict):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow(row)
        f.flush()  # ensure immediate write


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=str, default="", help="Comma-separated list, or leave empty to use built-in list")
    ap.add_argument("--datasets_file", type=str, default="", help="Optional: path to a txt file with one dataset per line")
    ap.add_argument("--threads", type=int, default=0)
    # toggles (tested)
    ap.add_argument("--include_max_options", type=str, default="0,1", help="Comma sep: 0 or 1")
    ap.add_argument("--delta_sigma_options", type=str, default="0.01:0,0.01:0.25,0.02:0.25", help="Comma sep pairs delta:sigma")
    ap.add_argument("--keep_best_d_only_options", type=str, default="1,0", help="Comma sep: 1 or 0")
    ap.add_argument("--clf_types", type=str, default="ridge,logreg", help="Comma sep: ridge,logreg")
    # static (kept defaults as per your spec)
    ap.add_argument("--kernel_budget_total", type=int, default=None)
    ap.add_argument("--shapelet_budget_total", type=int, default=None)
    ap.add_argument("--shapelet_bias_mode", type=str, default="reuse")
    ap.add_argument("--lds_offset", type=float, default=0.0)
    # ridge / logreg grids
    ap.add_argument("--alpha_low", type=float, default=1e-1)
    ap.add_argument("--alpha_high", type=float, default=1e3)
    ap.add_argument("--alpha_points", type=int, default=9)
    ap.add_argument("--C_logreg", type=float, default=1.0)
    ap.add_argument("--max_iter_logreg", type=int, default=200)
    # output
    ap.add_argument("--out_csv", type=str, default=os.path.join("results", "results_tuning.csv"))
    # speed option: approximate K9 kernels by slicing from K369 (skip separate K9 pass)
    ap.add_argument("--approx_k9_from_k369", action="store_true")
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

    include_max_opts = [int(x) for x in args.include_max_options.split(",") if x != ""]
    keep_best_opts = [int(x) for x in args.keep_best_d_only_options.split(",") if x != ""]
    delta_sigma_opts = []
    for pair in args.delta_sigma_options.split(","):
        if not pair:
            continue
        d, s = pair.split(":")
        delta_sigma_opts.append((float(d), float(s)))
    clf_types = [s.strip() for s in args.clf_types.split(",") if s.strip()]

    _ensure_results_dir(args.out_csv)
    done_keys = _load_done_keys(args.out_csv)

    header = [
        "dataset", "include_max", "delta_eta2", "sigma_optimism", "keep_best_d_only", "clf_type",
        "assemble_time_k369", "n_kernels", "n_shapelets",
        "acc_k9_kernels", "acc_k9_shapelets", "acc_k9_both",
        "acc_k369_kernels", "acc_k369_shapelets", "acc_k369_both"
    ]

    print(f"Datasets: {len(datasets)} | include_max={include_max_opts} | delta/sigma={delta_sigma_opts} | keep_best_d_only={keep_best_opts} | clf_types={clf_types}")
    print(f"Writing results to: {args.out_csv}")

    # pre-build ridge alpha grid
    alpha_grid = np.logspace(np.log10(args.alpha_low), np.log10(args.alpha_high), args.alpha_points)

    for ds in datasets:
        # load once per dataset
        try:
            t0 = time.time()
            X_train, y_train, X_test, y_test = load_ucr_uea_sktime(ds)
            load_s = time.time() - t0
            print(f"\n=== {ds} ===  loaded in {load_s:.2f}s | n_tr={len(y_train)} n_te={len(y_test)} L={X_train.shape[1]}")
        except Exception as e:
            print(f"[{ds}] load failed: {e}")
            # optionally write a failed row? skipping for now
            continue

        # loop over parameter combos that affect assembly
        for include_max in include_max_opts:
            for (delta_eta2, sigma_optimism) in delta_sigma_opts:
                for keep_best in keep_best_opts:
                    # key for skipping if done (we'll write one line per clf_type below)
                    # but we only assemble once and reuse for both clfs
                    # We check done per clf in the inner loop.
                    # Assemble K369 once:
                    F369 = fast(
                        include_max_feature=bool(include_max),
                        additional_kernel_families=True,
                        kernel_budget_total=args.kernel_budget_total,
                        shapelet_budget_total=args.shapelet_budget_total,
                        delta_eta2=delta_eta2,
                        sigma_optimism=sigma_optimism,
                        shapelet_bias_mode=args.shapelet_bias_mode,
                        shapelet_subsample_frac=0.20,
                        shapelet_subsample_min=10,
                        shapelet_subsample_max=256,
                        eb_bias_source="single_series",
                        keep_best_dilation_only=bool(keep_best),
                        lds_offset=args.lds_offset,
                    )
                    try:
                        tA = time.time()
                        Xtr_both_369, Xte_both_369, _ = F369.assemble_features(X_train, y_train, X_test, features_used="both")
                        asm_s = time.time() - tA
                    except Exception as e:
                        print(f"[{ds}] assemble K369 failed: {e}")
                        # write rows per clf as failed to allow resume; or skip entirely.
                        for clf_type in clf_types:
                            key = (ds, include_max, delta_eta2, sigma_optimism, keep_best, clf_type)
                            if key in done_keys:
                                continue
                            row = {
                                "dataset": ds, "include_max": include_max,
                                "delta_eta2": delta_eta2, "sigma_optimism": sigma_optimism,
                                "keep_best_d_only": keep_best, "clf_type": clf_type,
                                "assemble_time_k369": "FAIL", "n_kernels": 0, "n_shapelets": 0,
                                "acc_k9_kernels": "NA", "acc_k9_shapelets": "NA", "acc_k9_both": "NA",
                                "acc_k369_kernels": "NA", "acc_k369_shapelets": "NA", "acc_k369_both": "NA",
                            }
                            _append_row(args.out_csv, header, row)
                            done_keys.add(key)
                        continue

                    # prepare feature views for K369 (kernels / shapelets / both)
                    Xtr_ker_369 = F369._Xtr_kernels
                    Xte_ker_369 = F369._Xte_kernels
                    Xtr_shp_369 = F369._Xtr_shapelets
                    Xte_shp_369 = F369._Xte_shapelets

                    # slice K9 shapelets from K369 by family mask
                    shp_masks = _shapelet_family_column_masks(F369)
                    mask_k9_shp = shp_masks.get("K9s3", np.zeros((0,), dtype=bool))
                    Xtr_shp_k9 = Xtr_shp_369[:, mask_k9_shp] if Xtr_shp_369.shape[1] else np.zeros((X_train.shape[0], 0), np.float32)
                    Xte_shp_k9 = Xte_shp_369[:, mask_k9_shp] if Xte_shp_369.shape[1] else np.zeros((X_test.shape[0], 0), np.float32)

                    # kernels K9: either slice from K369 (approx) or do a small K9-only pass (fair baseline)
                    if args.approx_k9_from_k369:
                        mask_k9_ker = _kernel_family_column_mask(F369, "K9s3")
                        Xtr_ker_9 = Xtr_ker_369[:, mask_k9_ker] if Xtr_ker_369.shape[1] else np.zeros((X_train.shape[0], 0), np.float32)
                        Xte_ker_9 = Xte_ker_369[:, mask_k9_ker] if Xte_ker_369.shape[1] else np.zeros((X_test.shape[0], 0), np.float32)
                    else:
                        F9 = fast(
                            include_max_feature=False,
                            additional_kernel_families=False,   # K9 only
                            kernel_budget_total=args.kernel_budget_total,
                            shapelet_budget_total=0,
                        )
                        try:
                            Xtr_ker_9, Xte_ker_9, _ = F9.assemble_features(X_train, y_train, X_test, features_used="kernels")
                        except Exception as e:
                            print(f"[{ds}] assemble K9 kernels failed: {e}")
                            Xtr_ker_9 = np.zeros((X_train.shape[0], 0), dtype=np.float32)
                            Xte_ker_9 = np.zeros((X_test.shape[0], 0), dtype=np.float32)

                    # both views
                    Xtr_both_k9  = np.hstack([Xtr_ker_9,  Xtr_shp_k9 ]).astype(np.float32, copy=False)
                    Xte_both_k9  = np.hstack([Xte_ker_9,  Xte_shp_k9 ]).astype(np.float32, copy=False)
                    Xtr_both_369 = np.hstack([Xtr_ker_369, Xtr_shp_369]).astype(np.float32, copy=False)
                    Xte_both_369 = np.hstack([Xte_ker_369, Xte_shp_369]).astype(np.float32, copy=False)

                    n_kernels = len(getattr(F369, "ranked_parents_", []))
                    n_shapelets = len(getattr(F369, "kept_shapelets_", []))

                    # train heads for both classifiers (without re-assembling)
                    for clf_type in clf_types:
                        key = (ds, include_max, delta_eta2, sigma_optimism, keep_best, clf_type)
                        if key in done_keys:
                            continue

                        # pick grids
                        if clf_type == "ridge":
                            agrid = alpha_grid
                            Clog, maxit = args.C_logreg, args.max_iter_logreg
                        else:
                            agrid = alpha_grid  # unused
                            Clog, maxit = args.C_logreg, args.max_iter_logreg

                        # K9 heads
                        res_k9_ker  = _train_eval(Xtr_ker_9,   y_train, Xte_ker_9,   y_test, clf_type, agrid, maxit, Clog)
                        res_k9_shp  = _train_eval(Xtr_shp_k9,  y_train, Xte_shp_k9,  y_test, clf_type, agrid, maxit, Clog)
                        res_k9_both = _train_eval(Xtr_both_k9, y_train, Xte_both_k9, y_test, clf_type, agrid, maxit, Clog)

                        # K369 heads
                        res_ker369  = _train_eval(Xtr_ker_369,   y_train, Xte_ker_369,   y_test, clf_type, agrid, maxit, Clog)
                        res_shp369  = _train_eval(Xtr_shp_369,   y_train, Xte_shp_369,   y_test, clf_type, agrid, maxit, Clog)
                        res_both369 = _train_eval(Xtr_both_369,  y_train, Xte_both_369,  y_test, clf_type, agrid, maxit, Clog)

                        row = {
                            "dataset": ds,
                            "include_max": include_max,
                            "delta_eta2": delta_eta2,
                            "sigma_optimism": sigma_optimism,
                            "keep_best_d_only": keep_best,
                            "clf_type": clf_type,
                            "assemble_time_k369": f"{asm_s:.4f}",
                            "n_kernels": n_kernels,
                            "n_shapelets": n_shapelets,
                            "acc_k9_kernels": "NA" if np.isnan(res_k9_ker.acc) else f"{res_k9_ker.acc:.4f}",
                            "acc_k9_shapelets": "NA" if np.isnan(res_k9_shp.acc) else f"{res_k9_shp.acc:.4f}",
                            "acc_k9_both": "NA" if np.isnan(res_k9_both.acc) else f"{res_k9_both.acc:.4f}",
                            "acc_k369_kernels": "NA" if np.isnan(res_ker369.acc) else f"{res_ker369.acc:.4f}",
                            "acc_k369_shapelets": "NA" if np.isnan(res_shp369.acc) else f"{res_shp369.acc:.4f}",
                            "acc_k369_both": "NA" if np.isnan(res_both369.acc) else f"{res_both369.acc:.4f}",
                        }
                        _append_row(args.out_csv, header, row)
                        done_keys.add(key)
                        print(f"[{ds} | max={include_max} | d={delta_eta2} | s={sigma_optimism} | keep={keep_best} | {clf_type}] "
                              f"K9(both)={row['acc_k9_both']}, K369(both)={row['acc_k369_both']} | asm={row['assemble_time_k369']}s")

    print("\nAll requested runs attempted. Results appended to:", args.out_csv)


if __name__ == "__main__":
    main()
