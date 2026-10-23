"""Predict breast cancer ER status from somatic mutations in METABRIC.

Hypothesis: the somatic mutation profile of a tumor predicts whether it is
estrogen receptor positive (ER+) or negative (ER-).

For each mutation encoding (see mutation_features.build_feature_sets) an
L1-penalized logistic regression is evaluated with nested, stratified
cross-validation (inner loop tunes the penalty, outer loop scores ROC AUC).
Figures and tables are written to --out_dir.

Usage
-----
    python src/er_mutation_analysis.py \\
        --data_file data/METABRIC_RNA_Mutation.csv --out_dir results
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import mutation_features as mf
import plots

C_GRID = np.logspace(-2, 1, 13)
N_TOP_GENES = 15
N_TOP_COEFS = 10
N_CALIBRATION_BINS = 10


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict ER status from METABRIC somatic mutations.",
        prog="er_mutation_analysis")
    parser.add_argument("--data_file", type=Path, required=True,
                        help="Path to METABRIC_RNA_Mutation.csv")
    parser.add_argument("--out_dir", type=Path, required=True,
                        help="Directory for figures and tables")
    parser.add_argument("--min_gene_freq", type=int, default=10,
                        help="Min patients for a gene feature (default 10)")
    parser.add_argument("--min_hotspot", type=int, default=10,
                        help="Min patients for a hotspot allele (default 10)")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Folds for inner and outer CV (default 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for CV splits (default 42)")
    return parser.parse_args()


def make_model(n_folds, seed):
    """L1 logistic regression with C tuned by stratified CV on ROC AUC.

    Class weights are balanced because ~77% of patients are ER+.
    """
    base = LogisticRegression(l1_ratio=1.0, solver="liblinear",
                              class_weight="balanced", max_iter=1000)
    inner = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    return GridSearchCV(base, {"C": C_GRID}, cv=inner, scoring="roc_auc")


def nested_cv(X, y, n_folds, seed):
    """Nested cross-validation of `make_model` on one feature set.

    Parameters
    ----------
    X : pandas.DataFrame
        Patients x features.
    y : pandas.Series
        0/1 ER status.
    n_folds : int
        Number of outer (and inner) folds.
    seed : int
        Random seed for the splits.

    Returns
    -------
    oof_prob : numpy.ndarray
        Out-of-fold predicted P(ER+) for every patient.
    fold_auc : list of float
        ROC AUC on each outer test fold.
    """
    outer = StratifiedKFold(n_folds, shuffle=True, random_state=seed + 1)
    oof_prob = np.zeros(len(y))
    fold_auc = []
    for train, test in outer.split(X, y):
        model = make_model(n_folds, seed).fit(X.iloc[train], y.iloc[train])
        oof_prob[test] = model.predict_proba(X.iloc[test])[:, 1]
        fold_auc.append(roc_auc_score(y.iloc[test], oof_prob[test]))
    return oof_prob, fold_auc


def summarize(name, X, y, oof_prob, fold_auc):
    """One row of the metrics table for a feature set."""
    return {
        "feature_set": name,
        "n_features": X.shape[1],
        "auc_mean": np.mean(fold_auc),
        "auc_sd": np.std(fold_auc, ddof=1),
        "balanced_accuracy": balanced_accuracy_score(y, oof_prob >= 0.5),
    }


def mutation_frequency(X_gene, y, n_top):
    """Fraction of ER+ and ER- patients mutated in the most mutated genes.

    Returns
    -------
    freq : pandas.DataFrame
        Index gene, columns 'ER+' and 'ER-'.
    """
    top = X_gene.sum().nlargest(n_top).index
    return pd.DataFrame({
        "ER+": X_gene.loc[y == 1, top].mean(),
        "ER-": X_gene.loc[y == 0, top].mean(),
    })


def top_coefficients(model, columns, n_top):
    """Largest positive and negative nonzero coefficients of a fitted model.

    Returns
    -------
    coefs : pandas.Series
        Sorted from most positive to most negative.
    """
    coefs = pd.Series(model.best_estimator_.coef_[0], index=columns)
    coefs = coefs[coefs != 0].sort_values(ascending=False)
    return pd.concat([coefs.head(n_top), coefs.tail(n_top)]).drop_duplicates()


def write_cv_figures(y, oof, fold_rows, out_dir):
    """Write ROC curve and per-fold AUC figures (side effect: files)."""
    roc_points = {name: roc_curve(y, prob)[:2] for name, prob in oof.items()}
    plots.save_figure(plots.plot_roc_curves(roc_points),
                      out_dir / "roc_curves.png")
    plots.save_figure(plots.plot_fold_auc(pd.DataFrame(fold_rows)),
                      out_dir / "fold_auc.png")


def write_best_model_figures(X, y, oof_prob, args):
    """Refit the chosen model on all data; write diagnostics and coefs.

    Side effect: writes tuning, calibration and coefficient figures plus
    coefficients.csv to args.out_dir.
    """
    model = make_model(args.n_folds, args.seed).fit(X, y)
    plots.save_figure(plots.plot_tuning_curve(model.cv_results_),
                      args.out_dir / "tuning_curve.png")
    prob_true, prob_pred = calibration_curve(
        y, oof_prob, n_bins=N_CALIBRATION_BINS, strategy="quantile")
    plots.save_figure(plots.plot_calibration(prob_true, prob_pred),
                      args.out_dir / "calibration.png")
    coefs = top_coefficients(model, X.columns, N_TOP_COEFS)
    plots.save_figure(plots.plot_coefficients(coefs),
                      args.out_dir / "coefficients.png")
    coefs.rename("coefficient").to_csv(args.out_dir / "coefficients.csv")
    return model.best_params_["C"]


def run_analysis(df, args):
    """Evaluate every feature set, then write figures and tables."""
    y = mf.er_labels(df)
    feature_sets = mf.build_feature_sets(df, args.min_gene_freq,
                                         args.min_hotspot)
    freq = mutation_frequency(feature_sets["Gene"], y, N_TOP_GENES)
    plots.save_figure(plots.plot_mutation_frequency(freq),
                      args.out_dir / "mutation_frequency.png")

    summary, fold_rows, oof = [], [], {}
    for name, X in feature_sets.items():
        oof[name], fold_auc = nested_cv(X, y, args.n_folds, args.seed)
        summary.append(summarize(name, X, y, oof[name], fold_auc))
        fold_rows += [{"feature_set": name, "auc": a} for a in fold_auc]
    write_cv_figures(y, oof, fold_rows, args.out_dir)

    summary = pd.DataFrame(summary)
    best = summary.loc[summary["auc_mean"].idxmax(), "feature_set"]
    best_c = write_best_model_figures(feature_sets[best], y, oof[best], args)
    summary.round(3).to_csv(args.out_dir / "cv_metrics.csv", index=False)
    print(summary.round(3).to_string(index=False))
    print(f"\nBest feature set: {best} (refit C = {best_c:.3g})")


def main():
    """Load data, run the analysis and set the exit code."""
    args = parse_args()
    plots.set_classic_style()
    try:
        df = mf.load_metabric(args.data_file)
    except FileNotFoundError:
        print(f"Could not find {args.data_file}. Run `pixi run download`.",
              file=sys.stderr)
        sys.exit(1)
    except KeyError as err:
        print(f"Unexpected data format: {err}", file=sys.stderr)
        sys.exit(1)
    try:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        print(f"Could not create {args.out_dir}: {err}", file=sys.stderr)
        sys.exit(1)
    run_analysis(df, args)


if __name__ == "__main__":
    main()
