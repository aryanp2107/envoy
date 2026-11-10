"""
Publication figures for ENVOY.

Two categories:
    Per-dataset figures (produced by `envoy cluster`):
        - sweep_metrics.png: 4-panel sweep diagnostic
        - pca_2x2.png: embedding PCA with medoid overlay + length artifact check
        - pairwise_identity.png: MAFFT-aligned medoid identity heatmap
        - dendrogram.png: UPGMA tree with cluster colors and medoid markers

    Comparison figures (produced by `envoy compare` on multiple datasets):
        - sweep.png: sweep comparison across datasets, sorted by bimodality
        - distance_distributions.png: embedding-distance histograms per dataset
        - identity_embedding.png: aggregate identity-vs-embedding scatter
        - identity_embedding_per_dataset.png: per-dataset identity-vs-embedding

All comparison functions auto-size to however many datasets are passed in.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

# Shared styling constants — keep every figure visually consistent
PRIMARY_COLOR = "#1f4e79"
RANDOM_COLOR = "#999999"
DEPLOY_LINE_COLOR = "#c0392b"
OVERCLUSTER_COLOR = "#f5a623"
EDGE_COLOR = "white"
GRID_ALPHA = 0.25
DPI = 200


# =============================================================================
# Per-dataset figures
# =============================================================================


def plot_sweep_metrics(
    sweep_df: pd.DataFrame,
    deployment_k: int,
    n_sequences: int,
    output_path: str | Path,
    dataset_name: str = "",
) -> None:
    """
    Four-panel sweep diagnostic: silhouette, nearest-medoid, beat-%, inertia vs k.

    Args:
        sweep_df: DataFrame with columns k, silhouette, mean_nearest_medoid_distance,
                  inertia, random_silhouette_mean, random_silhouette_std,
                  random_nearest_mean, random_nearest_std, beat_pct_silhouette,
                  beat_pct_nearest, beat_pct_inertia
        deployment_k: the k chosen as deployment (marked with red dotted line)
        n_sequences: substrate size (for overcluster shading at k > N/4)
        output_path: where to save the figure
        dataset_name: optional title suffix
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    overcluster_threshold = n_sequences / 4

    # Panel 1: silhouette vs k
    ax = axes[0, 0]
    ax.plot(
        sweep_df["k"],
        sweep_df["silhouette"],
        "o-",
        color=PRIMARY_COLOR,
        linewidth=2,
        markersize=6,
        label="PAM",
    )
    if "random_silhouette_mean" in sweep_df.columns:
        ax.plot(
            sweep_df["k"],
            sweep_df["random_silhouette_mean"],
            "--",
            color=RANDOM_COLOR,
            linewidth=1.2,
            label="Random mean",
        )
        ax.fill_between(
            sweep_df["k"],
            sweep_df["random_silhouette_mean"] - sweep_df["random_silhouette_std"],
            sweep_df["random_silhouette_mean"] + sweep_df["random_silhouette_std"],
            color=RANDOM_COLOR,
            alpha=0.2,
            label="Random ±1σ",
        )
    ax.axvline(
        deployment_k,
        color=DEPLOY_LINE_COLOR,
        linestyle=":",
        linewidth=1.2,
        label=f"Deployment k={deployment_k}",
    )
    ax.axvspan(overcluster_threshold, sweep_df["k"].max() + 1, color=OVERCLUSTER_COLOR, alpha=0.08)
    ax.set_title("Silhouette vs k   (higher is better)", fontsize=11)
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(loc="best", fontsize=8)

    # Panel 2: mean nearest-medoid distance
    ax = axes[0, 1]
    ax.plot(
        sweep_df["k"],
        sweep_df["mean_nearest_medoid_distance"],
        "o-",
        color=PRIMARY_COLOR,
        linewidth=2,
        markersize=6,
        label="PAM",
    )
    if "random_nearest_mean" in sweep_df.columns:
        ax.plot(
            sweep_df["k"],
            sweep_df["random_nearest_mean"],
            "--",
            color=RANDOM_COLOR,
            linewidth=1.2,
            label="Random mean",
        )
        ax.fill_between(
            sweep_df["k"],
            sweep_df["random_nearest_mean"] - sweep_df["random_nearest_std"],
            sweep_df["random_nearest_mean"] + sweep_df["random_nearest_std"],
            color=RANDOM_COLOR,
            alpha=0.2,
            label="Random ±1σ",
        )
    ax.axvline(deployment_k, color=DEPLOY_LINE_COLOR, linestyle=":", linewidth=1.2)
    ax.axvspan(overcluster_threshold, sweep_df["k"].max() + 1, color=OVERCLUSTER_COLOR, alpha=0.08)
    ax.set_title("Coverage vs k   (lower is better)", fontsize=11)
    ax.set_xlabel("k")
    ax.set_ylabel("Mean nearest-medoid distance")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(loc="best", fontsize=8)

    # Panel 3: PAM-vs-random beat percentile
    ax = axes[1, 0]
    if "beat_pct_silhouette" in sweep_df.columns:
        ax.plot(
            sweep_df["k"],
            100 * sweep_df["beat_pct_silhouette"],
            "o-",
            color="#2ca02c",
            linewidth=1.5,
            markersize=5,
            label="silhouette",
        )
        ax.plot(
            sweep_df["k"],
            100 * sweep_df["beat_pct_nearest"],
            "s-",
            color="#d62728",
            linewidth=1.5,
            markersize=5,
            label="nearest",
        )
        ax.plot(
            sweep_df["k"],
            100 * sweep_df["beat_pct_inertia"],
            "^-",
            color="#ff7f0e",
            linewidth=1.5,
            markersize=5,
            label="inertia",
        )
    ax.axhline(50, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.axvline(deployment_k, color=DEPLOY_LINE_COLOR, linestyle=":", linewidth=1.2)
    ax.axvspan(overcluster_threshold, sweep_df["k"].max() + 1, color=OVERCLUSTER_COLOR, alpha=0.08)
    ax.set_title("PAM-vs-random   (% baselines beaten)", fontsize=11)
    ax.set_xlabel("k")
    ax.set_ylabel("% random panels beaten")
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(loc="best", fontsize=8)

    # Panel 4: inertia
    ax = axes[1, 1]
    ax.plot(
        sweep_df["k"],
        sweep_df["inertia"],
        "o-",
        color=PRIMARY_COLOR,
        linewidth=2,
        markersize=6,
        label="PAM",
    )
    if "random_silhouette_mean" in sweep_df.columns:
        # Random inertia mean can be derived as current_cost at mean distance
        # but we didn't store it; so skip the random band here
        pass
    ax.axvline(deployment_k, color=DEPLOY_LINE_COLOR, linestyle=":", linewidth=1.2)
    ax.axvspan(
        overcluster_threshold,
        sweep_df["k"].max() + 1,
        color=OVERCLUSTER_COLOR,
        alpha=0.08,
        label="k > N/4 (overcluster)",
    )
    ax.set_title("Inertia vs k   (lower is better)", fontsize=11)
    ax.set_xlabel("k")
    ax.set_ylabel("Total inertia")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(loc="best", fontsize=8)

    suffix = f": {dataset_name}" if dataset_name else ""
    fig.suptitle(
        f"PAM k-sweep on ESM-2 embeddings{suffix} (N = {n_sequences})", fontsize=12, y=1.00
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("[fig] %s", output_path)


def plot_pca_2x2(
    embeddings: np.ndarray,
    accessions: list[str],
    medoid_indices: np.ndarray,
    cluster_assignments: np.ndarray,
    protein_lengths: np.ndarray | None,
    output_path: str | Path,
    dataset_name: str = "",
) -> None:
    """
    Four-panel PCA: PC1-PC2, PC1-PC3, PC2-PC3, and length-vs-PC1 artifact check.

    Args:
        embeddings: (n, embed_dim) matrix
        accessions: (n,) accession list (unused but kept for API)
        medoid_indices: indices into embeddings for deployment medoids
        cluster_assignments: (n,) cluster label per sequence
        protein_lengths: (n,) protein lengths; if None, length panel shows "N/A"
        output_path: where to save
        dataset_name: optional title suffix
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n, _ = embeddings.shape
    k = len(medoid_indices)

    pca = PCA(n_components=3, random_state=0)
    pcs = pca.fit_transform(embeddings)
    var = pca.explained_variance_ratio_ * 100

    # Color by cluster assignment
    cmap = plt.get_cmap("tab20", k)
    point_colors = [cmap(c % 20) for c in cluster_assignments]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # Panel 1: PC1 vs PC2
    ax = axes[0, 0]
    ax.scatter(pcs[:, 0], pcs[:, 1], c=point_colors, s=18, alpha=0.65, edgecolors="none")
    ax.scatter(
        pcs[medoid_indices, 0],
        pcs[medoid_indices, 1],
        marker="*",
        s=180,
        color="black",
        label=f"{k} medoids",
        zorder=5,
    )
    ax.set_xlabel(f"PC1 ({var[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1]:.1f}%)")
    ax.set_title("PC1 vs PC2   (PC1 often length-dominated)", fontsize=10)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(loc="best", fontsize=8)

    # Panel 2: PC1 vs PC3
    ax = axes[0, 1]
    ax.scatter(pcs[:, 0], pcs[:, 2], c=point_colors, s=18, alpha=0.65, edgecolors="none")
    ax.scatter(
        pcs[medoid_indices, 0],
        pcs[medoid_indices, 2],
        marker="*",
        s=180,
        color="black",
        label=f"{k} medoids",
        zorder=5,
    )
    ax.set_xlabel(f"PC1 ({var[0]:.1f}%)")
    ax.set_ylabel(f"PC3 ({var[2]:.1f}%)")
    ax.set_title("PC1 vs PC3", fontsize=10)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(loc="best", fontsize=8)

    # Panel 3: PC2 vs PC3
    ax = axes[1, 0]
    ax.scatter(pcs[:, 1], pcs[:, 2], c=point_colors, s=18, alpha=0.65, edgecolors="none")
    ax.scatter(
        pcs[medoid_indices, 1],
        pcs[medoid_indices, 2],
        marker="*",
        s=180,
        color="black",
        label=f"{k} medoids",
        zorder=5,
    )
    ax.set_xlabel(f"PC2 ({var[1]:.1f}%)")
    ax.set_ylabel(f"PC3 ({var[2]:.1f}%)")
    ax.set_title("PC2 vs PC3   (within-population biology)", fontsize=10)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(loc="best", fontsize=8)

    # Panel 4: length vs PC1 artifact check
    ax = axes[1, 1]
    if protein_lengths is not None:
        r = float(np.corrcoef(protein_lengths, pcs[:, 0])[0, 1])
        ax.scatter(protein_lengths, pcs[:, 0], s=18, alpha=0.55, color="#555555", edgecolors="none")
        # Linear regression line
        coefs = np.polyfit(protein_lengths, pcs[:, 0], 1)
        x = np.linspace(protein_lengths.min(), protein_lengths.max(), 50)
        ax.plot(x, coefs[0] * x + coefs[1], color="red", linewidth=1.5, label=f"r = {r:+.3f}")
        ax.set_xlabel("Protein length (aa)")
        ax.set_ylabel(f"PC1 ({var[0]:.1f}%)")
        ax.set_title("Length vs PC1   (length artifact check)", fontsize=10)
        ax.legend(loc="best", fontsize=8)
    else:
        ax.text(
            0.5,
            0.5,
            "Length data not supplied",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    ax.grid(True, alpha=GRID_ALPHA)

    suffix = f": {dataset_name}" if dataset_name else ""
    fig.suptitle(f"PCA of ESM-2 embeddings{suffix}  (N = {n}, k = {k})", fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("[fig] %s", output_path)


def plot_pairwise_identity(
    identity_matrix: np.ndarray,
    labels: list[str],
    output_path: str | Path,
    dataset_name: str = "",
) -> None:
    """
    Heatmap of pairwise % identity between medoids.

    Args:
        identity_matrix: (k, k) symmetric identity matrix in percent
        labels: (k,) accession labels
        output_path: where to save
        dataset_name: optional title suffix
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    k = identity_matrix.shape[0]
    upper = identity_matrix[np.triu_indices(k, k=1)]
    median_id = np.median(upper) if len(upper) else np.nan
    max_id = upper.max() if len(upper) else np.nan

    fig, ax = plt.subplots(figsize=(1 + k * 0.5, 1 + k * 0.5))
    im = ax.imshow(identity_matrix, cmap="viridis", vmin=50, vmax=100, aspect="equal")
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    cbar = plt.colorbar(im, ax=ax, label="% identity")
    cbar.ax.tick_params(labelsize=8)

    suffix = f": {dataset_name}" if dataset_name else ""
    ax.set_title(
        f"Pairwise identity between {k} medoids{suffix}\n"
        f"median = {median_id:.1f}%   max = {max_id:.1f}%",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("[fig] %s", output_path)


def plot_dendrogram(
    distance_matrix: np.ndarray,
    accessions: list[str],
    medoid_indices: np.ndarray,
    cluster_assignments: np.ndarray,
    output_path: str | Path,
    dataset_name: str = "",
    max_display: int = 500,
) -> None:
    """
    UPGMA dendrogram with cluster-colored leaf labels and medoid asterisks.

    Args:
        distance_matrix: (n, n) precomputed distances
        accessions: (n,) accession list
        medoid_indices: indices of medoids
        cluster_assignments: (n,) cluster labels
        output_path: where to save
        dataset_name: optional title suffix
        max_display: if n > max_display, down-sample leaves to this many (keeping
                     all medoids). Avoids unreadable 3000-leaf dendrograms.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(accessions)
    k = len(medoid_indices)

    # Down-sample if too large
    if n > max_display:
        np.random.seed(0)
        keep_mask = np.zeros(n, dtype=bool)
        keep_mask[medoid_indices] = True
        n_non_medoid_to_keep = max_display - k
        non_medoid_indices = np.setdiff1d(np.arange(n), medoid_indices)
        sampled = np.random.choice(non_medoid_indices, size=n_non_medoid_to_keep, replace=False)
        keep_mask[sampled] = True
        display_idx = np.where(keep_mask)[0]
    else:
        display_idx = np.arange(n)

    # Build linkage from condensed distance vector
    from scipy.spatial.distance import squareform

    D_sub = distance_matrix[np.ix_(display_idx, display_idx)]
    condensed = squareform(D_sub, checks=False)
    linkage_matrix = linkage(condensed, method="average")

    k_eff = k
    cmap = plt.get_cmap("tab20", k_eff)
    label_colors = {}
    labels = []
    for i, orig_idx in enumerate(display_idx):
        c = cluster_assignments[orig_idx]
        is_medoid = orig_idx in medoid_indices
        prefix = "* " if is_medoid else "  "
        acc = accessions[orig_idx]
        labels.append(f"{prefix}{acc}")
        label_colors[acc] = cmap(c % 20)

    height = max(6, min(0.12 * len(display_idx), 24))
    fig, ax = plt.subplots(figsize=(max(14, 0.3 * len(display_idx)), height))
    with plt.rc_context({"lines.linewidth": 0.8}):
        dendrogram(
            linkage_matrix,
            labels=labels,
            leaf_rotation=90,
            leaf_font_size=7,
            color_threshold=0,  # uniform gray branches
            above_threshold_color="#888888",
            ax=ax,
        )

    # Re-color x-tick labels per cluster
    for lbl in ax.get_xticklabels():
        txt = lbl.get_text()
        # Strip prefix ("* " or "  ")
        acc = txt.lstrip("*").strip()
        if acc in label_colors:
            lbl.set_color(label_colors[acc])
        if txt.startswith("*"):
            lbl.set_fontweight("bold")

    suffix = f": {dataset_name}" if dataset_name else ""
    title = f"UPGMA dendrogram{suffix}, k={k}, medoids marked with *" + (
        f"\n(down-sampled: showing {len(display_idx)} of {n} leaves, all medoids kept)"
        if len(display_idx) < n
        else ""
    )
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Cosine distance (UPGMA average linkage)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("[fig] %s", output_path)


# =============================================================================
# Comparison figures (across multiple datasets)
# =============================================================================


def _grid_shape(n: int, max_cols: int = 4) -> tuple[int, int]:
    """Choose a near-square grid for n panels, max `max_cols` columns."""
    ncols = min(max_cols, n)
    nrows = (n + ncols - 1) // ncols
    return nrows, ncols


def plot_sweep_comparison(
    datasets: dict[str, pd.DataFrame],
    deployment_ks: dict[str, int],
    output_path: str | Path,
    y_cap: float = 0.22,
    y_floor: float = -0.15,
) -> None:
    """
    Small-multiples silhouette vs k, one panel per dataset.

    Args:
        datasets: dict mapping dataset_name → sweep metrics DataFrame
        deployment_ks: dict mapping dataset_name → deployment k
        output_path: where to save
        y_cap: shared y-axis ceiling (values above are annotated as "→")
        y_floor: shared y-axis floor
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names = sorted(datasets.keys())
    n = len(names)
    nrows, ncols = _grid_shape(n)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.6 * ncols, 2.8 * nrows), sharex=False, sharey=True
    )
    if n == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for i, name in enumerate(names):
        ax = axes[i]
        df = datasets[name]
        dk = deployment_ks.get(name, df["k"].max())
        n_seq = df["n_sequences"].iloc[0] if "n_sequences" in df.columns else None

        ax.plot(
            df["k"],
            df["silhouette"],
            "o-",
            color=PRIMARY_COLOR,
            linewidth=1.5,
            markersize=4,
            label="PAM",
        )
        if "random_silhouette_mean" in df.columns:
            ax.plot(
                df["k"],
                df["random_silhouette_mean"],
                "--",
                color=RANDOM_COLOR,
                linewidth=1,
                label="Random",
            )
        ax.axvline(dk, color=DEPLOY_LINE_COLOR, linestyle=":", linewidth=1)

        # Clipped-value annotation for out-of-range PAM points
        for xk, ys in zip(df["k"], df["silhouette"]):
            if ys > y_cap:
                ax.annotate(
                    f"{ys:.2f} →",
                    xy=(xk, y_cap * 0.92),
                    fontsize=7,
                    color=PRIMARY_COLOR,
                    fontweight="bold",
                    ha="center",
                )

        header = f"{name}"
        if n_seq is not None:
            header += f" (N={int(n_seq)}, k*={dk})"
        else:
            header += f" (k*={dk})"
        ax.set_title(header, fontsize=10)
        ax.set_xlabel("k", fontsize=8)
        if i % ncols == 0:
            ax.set_ylabel("Silhouette", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=GRID_ALPHA)
        ax.set_ylim(y_floor, y_cap)
        if i == 0:
            ax.legend(loc="best", fontsize=7)

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("PAM k-sweep comparison across datasets, deployment k marked", fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("[fig] %s", output_path)


def plot_distance_distributions(
    datasets: dict[str, np.ndarray],
    output_path: str | Path,
) -> None:
    """
    Small-multiples log-distance histograms, one panel per dataset.

    Args:
        datasets: dict mapping dataset_name → 1D array of pairwise embedding distances
                  (upper triangle of the (k, k) medoid distance matrix)
        output_path: where to save
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort datasets by bimodality ratio (p75 / p50) descending
    ratios = {name: np.percentile(d, 75) / np.percentile(d, 50) for name, d in datasets.items()}
    names = sorted(datasets.keys(), key=lambda n: -ratios[n])
    n = len(names)
    nrows, ncols = _grid_shape(n)

    # Shared x-axis
    all_log = np.concatenate([np.log10(d + 1e-6) for d in datasets.values()])
    x_min = np.floor(all_log.min() * 2) / 2
    x_max = np.ceil(all_log.max() * 2) / 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 2.8 * nrows), sharex=True)
    if n == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    # Share y per row
    for row in range(nrows):
        row_names = names[row * ncols : (row + 1) * ncols]
        row_max_y = 0
        for name in row_names:
            log_d = np.log10(datasets[name] + 1e-6)
            counts, _ = np.histogram(log_d, bins=30, range=(x_min, x_max))
            if counts.max() > row_max_y:
                row_max_y = counts.max()
        for col, name in enumerate(row_names):
            idx = row * ncols + col
            ax = axes[idx]
            log_d = np.log10(datasets[name] + 1e-6)
            ax.hist(
                log_d,
                bins=30,
                range=(x_min, x_max),
                color=PRIMARY_COLOR,
                alpha=0.85,
                edgecolor=EDGE_COLOR,
                linewidth=0.4,
            )
            ax.set_title(name, fontsize=12, pad=14, fontweight="bold")
            ax.text(
                0.5,
                1.02,
                f"n = {len(log_d)},   p75/p50 = {ratios[name]:.1f}",
                transform=ax.transAxes,
                fontsize=9,
                ha="center",
                va="bottom",
                color="#555555",
            )
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(labelsize=9)
            ax.grid(True, alpha=GRID_ALPHA)
            ax.set_axisbelow(True)
            ax.set_ylim(0, row_max_y * 1.1)
            ax.set_xlim(x_min, x_max)

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.supxlabel(r"log$_{10}$(embedding cosine distance between medoid pair)", fontsize=11, y=0.02)
    fig.supylabel("Pair count", fontsize=11, x=0.01)
    fig.tight_layout(rect=[0.02, 0.03, 1.0, 1.0])
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("[fig] %s", output_path)


def plot_identity_vs_embedding(
    datasets: dict[str, dict],
    output_path: str | Path,
) -> None:
    """
    Aggregate scatter of pairwise identity vs embedding distance, colored by dataset.

    Args:
        datasets: dict mapping dataset_name → {'identity': 1D array, 'embedding': 1D array}
                  (arrays of equal length representing the same pairs)
        output_path: where to save
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmap = plt.get_cmap("tab10") if len(datasets) <= 10 else plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(10, 7))

    for i, (name, data) in enumerate(sorted(datasets.items())):
        color = cmap(i % cmap.N)
        ax.scatter(
            data["identity"],
            data["embedding"],
            color=color,
            alpha=0.55,
            s=22,
            label=f"{name} (n={len(data['identity'])})",
            edgecolors="none",
        )

    ax.set_xlabel("Pairwise amino-acid identity (%, MAFFT-aligned)")
    ax.set_ylabel("Embedding cosine distance (ESM-2)")
    ax.set_yscale("log")
    ax.axvline(90, color=DEPLOY_LINE_COLOR, linestyle=":", linewidth=1, alpha=0.5)
    ax.set_title("Identity vs embedding distance across datasets", fontsize=12)
    ax.legend(loc="best", fontsize=8, framealpha=0.85, ncol=2)
    ax.grid(True, alpha=GRID_ALPHA, which="both")

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("[fig] %s", output_path)


def plot_identity_vs_embedding_per_dataset(
    datasets: dict[str, dict],
    output_path: str | Path,
) -> None:
    """
    Small-multiples scatter of identity vs embedding, one panel per dataset.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names = sorted(datasets.keys())
    n = len(names)
    nrows, ncols = _grid_shape(n)

    cmap = plt.get_cmap("tab10") if n <= 10 else plt.get_cmap("tab20")

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.5 * ncols, 3.0 * nrows), sharex=True, sharey=True
    )
    if n == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for i, name in enumerate(names):
        ax = axes[i]
        d = datasets[name]
        color = cmap(i % cmap.N)
        ax.scatter(d["identity"], d["embedding"], color=color, alpha=0.6, s=16, edgecolors="none")
        ax.set_yscale("log")
        ax.set_title(f"{name}  (n={len(d['identity'])})", fontsize=10)
        ax.axvline(90, color=DEPLOY_LINE_COLOR, linestyle=":", linewidth=0.8, alpha=0.5)
        ax.grid(True, alpha=GRID_ALPHA, which="both")
        ax.tick_params(labelsize=8)
        if i % ncols == 0:
            ax.set_ylabel("Embed. dist (log)", fontsize=9)
        if i >= n - ncols:
            ax.set_xlabel("Identity (%)", fontsize=9)

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Identity vs embedding distance per dataset", fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("[fig] %s", output_path)
