import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def smooth_hist_line(values, bins=60, value_range=(-0.5, 2.1), smooth=False):
    hist, edges = np.histogram(values, bins=bins, range=value_range, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    if smooth:
        kernel = np.array([1, 2, 3, 2, 1], dtype=float)
        kernel /= kernel.sum()
        hist = np.convolve(hist, kernel, mode="same")

    return centers, hist


def plot_entropy_comparison(experiment_results, ensemble_sizes=(1, 5, 10),
  save_path="plots/figure3_style_partial_replication.png"):

    blue_colors = {
        1: "#9ecae1",
        5: "#3182bd",
        10: "#08519c"
    }

    red_colors = {
        1: "#fcae91",
        5: "#fb6a4a",
        10: "#cb181d"
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    # ---------- MNIST ----------
    for M in ensemble_sizes:
        x, y = smooth_hist_line(
            experiment_results[("ensemble", M)]["mnist_entropy"].numpy(),
            bins=70,
            value_range=(-0.5, 2.0),
            smooth=True
        )
        axes[0, 0].plot(x, y, color=blue_colors[M], linewidth=2)

    for M in ensemble_sizes:
        x, y = smooth_hist_line(
            experiment_results[("mc_dropout", M)]["mnist_entropy"].numpy(),
            bins=70,
            value_range=(-0.5, 2.0),
            smooth=True
        )
        axes[0, 1].plot(x, y, color=blue_colors[M], linewidth=2)

    # ---------- NotMNIST ----------
    for M in ensemble_sizes:
        x, y = smooth_hist_line(
            experiment_results[("ensemble", M)]["notmnist_entropy"].numpy(),
            bins=70,
            value_range=(-0.5, 2.0),
            smooth=True
        )
        axes[1, 0].plot(x, y, color=red_colors[M], linewidth=2)

    for M in ensemble_sizes:
        x, y = smooth_hist_line(
            experiment_results[("mc_dropout", M)]["notmnist_entropy"].numpy(),
            bins=70,
            value_range=(-0.5, 2.0),
            smooth=True
        )
        axes[1, 1].plot(x, y, color=red_colors[M], linewidth=2)

    # Titles
    axes[0, 0].set_title("Ensemble")
    axes[0, 1].set_title("MC dropout")
    axes[1, 0].set_title("Ensemble")
    axes[1, 1].set_title("MC dropout")

    # Labels
    for ax in axes.flat:
        ax.set_xlabel("entropy values")
        ax.set_xlim(-0.5, 2.0)

    for ax in axes[0]:
      ax.tick_params(axis='x', labelbottom=True)

    axes[0, 0].set_ylabel("Known classes")
    axes[1, 0].set_ylabel("Unknown classes")

    # Legends
    top_handles = [Line2D([0], [0], color=blue_colors[M], lw=2, label=str(M)) for M in ensemble_sizes]
    bottom_handles = [Line2D([0], [0], color=red_colors[M], lw=2, label=str(M)) for M in ensemble_sizes]

    axes[0, 0].legend(handles=top_handles, frameon=False)
    axes[0, 1].legend(handles=top_handles, frameon=False)
    axes[1, 0].legend(handles=bottom_handles, frameon=False)
    axes[1, 1].legend(handles=bottom_handles, frameon=False)

    # Limits
    axes[0, 0].set_ylim(0, 9)
    axes[0, 1].set_ylim(0, 9)
    axes[1, 0].set_ylim(0, 7)
    axes[1, 1].set_ylim(0, 7)

    plt.suptitle("Predictive Entropy Histograms", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()