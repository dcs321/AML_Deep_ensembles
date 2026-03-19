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


# Figure 3
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

    method_order = [
        ("ensemble", "Ensemble"),
        ("ensemble_random", "Ensemble + R"),
        ("ensemble_adversarial", "Ensemble + AT"),
        ("mc_dropout", "MC dropout 0.1"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True)

    # ---------- MNIST ----------
    for col, (method_key, title) in enumerate(method_order):
        for M in ensemble_sizes:
            x, y = smooth_hist_line(
                experiment_results[(method_key, M)]["mnist_entropy"].numpy(),
                bins=100,
                value_range=(-0.5, 2.0),
                smooth=True
            )
            axes[0, col].plot(x, y, color=blue_colors[M], linewidth=2)
        axes[0, col].set_title(title)

    # ---------- NotMNIST ----------
    for col, (method_key, title) in enumerate(method_order):
        for M in ensemble_sizes:
            x, y = smooth_hist_line(
                experiment_results[(method_key, M)]["notmnist_entropy"].numpy(),
                bins=100,
                value_range=(-0.5, 2.0),
                smooth=True
            )
            axes[1, col].plot(x, y, color=red_colors[M], linewidth=2)
        axes[1, col].set_title(title)

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

    for col in range(4):
        axes[0, col].legend(handles=top_handles, frameon=False, loc="upper right")
        axes[1, col].legend(handles=bottom_handles, frameon=False, loc="upper right")

    for col in range(4):
        axes[0, col].set_ylim(0, 14)
        axes[1, col].set_ylim(0, 14)

    plt.suptitle("Predictive Entropy Histograms", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

# Figure 6
def plot_accuracy_vs_confidence(
    experiment_results,
    save_path="plots/figure6_accuracy_vs_confidence.png",
):
    fig, ax = plt.subplots(figsize=(7, 5))

    method_styles = {
        "ensemble": {"label": "Ensemble", "color": "red"},
        "ensemble_random": {"label": "Ensemble + R", "color": "gray"},
        "ensemble_adversarial": {"label": "Ensemble + AT", "color": "blue"},
        "mc_dropout": {"label": "MC dropout", "color": "limegreen"},
    }

    for method_key, style in method_styles.items():
        ax.plot(
            experiment_results[method_key]["thresholds"],
            experiment_results[method_key]["accuracies"],
            marker="o",
            linewidth=2,
            markersize=4,
            label=style["label"],
            color=style["color"],
        )

    ax.set_xlabel(r"Confidence Threshold $\tau$")
    ax.set_ylabel(r"Accuracy on examples $p(y|x) \geq \tau$")
    ax.set_xlim(0.0, 0.9)
    ax.set_ylim(30, 90)
    ax.legend(frameon=False, loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

# qualitative analysis
def plot_qualitative_examples_grid(
    images,
    pred_labels,
    top_indices,
    bottom_indices,
    # title,
    save_path,
):
    top_indices = top_indices.tolist()
    bottom_indices = bottom_indices.tolist()

    ordered_indices = top_indices + bottom_indices

    fig, axes = plt.subplots(4, 10, figsize=(12, 4.8))

    for i, idx in enumerate(ordered_indices):
        row = i // 10
        col = i % 10

        ax = axes[row, col]
        ax.imshow(images[idx].squeeze(), cmap="gray")
        ax.axis("off")
        ax.set_title(str(int(pred_labels[idx])), fontsize=6, pad=1)

    plt.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.96,
        bottom=0.04,
        wspace=0.02,
        hspace=0.10
    )

    plt.tight_layout(rect=[0.04, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
