"""Minimal, classic-themed figures for the ER status analysis.

No titles or annotations: white background, left/bottom spines only, no grid.

    * set_classic_style - apply the shared matplotlib style
    * save_figure - write a figure to disk and close it
    * plot_mutation_frequency - fraction mutated per gene by ER status
    * plot_roc_curves - out-of-fold ROC curve per feature set
    * plot_fold_auc - per-fold AUC per feature set
    * plot_tuning_curve - inner-CV AUC across the regularization grid
    * plot_calibration - reliability diagram of out-of-fold probabilities
    * plot_coefficients - largest positive / negative model coefficients
"""

import matplotlib.pyplot as plt
import numpy as np

# Categorical slots in fixed order; ink and gray for reference lines.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
ER_COLORS = {"ER+": "#2a78d6", "ER-": "#eb6834"}
INK = "#0b0b0b"
REFERENCE_GRAY = "#9a9994"


def set_classic_style():
    """Apply a classic theme (no grid, no top/right spines) globally."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def save_figure(fig, out_file):
    """Save `fig` to `out_file` and close it (side effect: writes a file)."""
    fig.savefig(out_file)
    plt.close(fig)


def plot_mutation_frequency(freq):
    """Horizontal grouped bars of the fraction of patients mutated per gene.

    Parameters
    ----------
    freq : pandas.DataFrame
        Index is gene, columns 'ER+' and 'ER-' hold fractions mutated.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    y = np.arange(len(freq))
    height = 0.4
    for offset, group in zip([-height / 2, height / 2], ["ER+", "ER-"]):
        ax.barh(y + offset, freq[group], height=height * 0.9,
                color=ER_COLORS[group], label=group)
    ax.set_yticks(y, freq.index.str.upper())
    ax.invert_yaxis()
    ax.set_xlabel("Fraction of patients mutated")
    ax.legend(loc="lower right")
    return fig


def plot_roc_curves(roc_points):
    """Out-of-fold ROC curves, one line per feature set.

    Parameters
    ----------
    roc_points : dict of str to tuple of (fpr, tpr)

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], color=REFERENCE_GRAY, linewidth=1,
            linestyle="--")
    for color, (name, (fpr, tpr)) in zip(SERIES_COLORS, roc_points.items()):
        ax.plot(fpr, tpr, color=color, label=name)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    return fig


def plot_fold_auc(fold_auc):
    """Per-fold ROC AUC (dots) and mean (bar) for each feature set.

    Parameters
    ----------
    fold_auc : pandas.DataFrame
        Columns 'feature_set' and 'auc', one row per outer fold.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(5, 3.5))
    names = list(dict.fromkeys(fold_auc["feature_set"]))
    for i, (name, color) in enumerate(zip(names, SERIES_COLORS)):
        auc = fold_auc.loc[fold_auc["feature_set"] == name, "auc"]
        ax.hlines(auc.mean(), i - 0.25, i + 0.25, color=INK, linewidth=2)
        ax.scatter(np.full(len(auc), i), auc, s=40, color=color,
                   edgecolor="white", linewidth=1, zorder=3)
    ax.axhline(0.5, color=REFERENCE_GRAY, linewidth=1, linestyle="--")
    ax.set_xticks(range(len(names)), names, rotation=20, ha="right")
    ax.set_ylabel("Outer-fold ROC AUC")
    return fig


def plot_tuning_curve(cv_results):
    """Inner-CV ROC AUC (mean +/- 1 sd) across inverse regularization C.

    Parameters
    ----------
    cv_results : dict
        `cv_results_` from a fitted GridSearchCV over 'C'.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    c_values = np.asarray(cv_results["param_C"], dtype=float)
    mean = cv_results["mean_test_score"]
    sd = cv_results["std_test_score"]
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.fill_between(c_values, mean - sd, mean + sd, color=SERIES_COLORS[0],
                    alpha=0.2, linewidth=0)
    ax.plot(c_values, mean, color=SERIES_COLORS[0], marker="o",
            markersize=4)
    ax.set_xscale("log")
    ax.set_xlabel("Inverse L1 penalty, C")
    ax.set_ylabel("Inner-CV ROC AUC")
    return fig


def plot_calibration(prob_true, prob_pred):
    """Reliability diagram: observed ER+ fraction vs mean predicted.

    Parameters
    ----------
    prob_true, prob_pred : numpy.ndarray
        Output of sklearn.calibration.calibration_curve.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], color=REFERENCE_GRAY, linewidth=1,
            linestyle="--")
    ax.plot(prob_pred, prob_true, color=SERIES_COLORS[0], marker="o",
            markersize=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted P(ER+)")
    ax.set_ylabel("Observed fraction ER+")
    return fig


def plot_coefficients(coefs):
    """Horizontal bars of model coefficients (log-odds of ER+).

    Parameters
    ----------
    coefs : pandas.Series
        Coefficients indexed by feature name, already sorted for display.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(5, 0.25 * len(coefs) + 1))
    colors = [ER_COLORS["ER+"] if c > 0 else ER_COLORS["ER-"]
              for c in coefs]
    y = np.arange(len(coefs))
    ax.barh(y, coefs, color=colors, height=0.8)
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_yticks(y, coefs.index.str.upper())
    ax.invert_yaxis()
    ax.set_xlabel("Coefficient (log-odds of ER+)")
    return fig
