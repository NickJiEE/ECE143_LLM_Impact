from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .sunset_analysis import COVID_REMOTE_END, COVID_START, LLM_MARKERS


def plot_group_trends(
    group_term_df: pd.DataFrame,
    groups: list[str],
    with_llm_markers: bool = False,
    save_path: str | Path | None = None,
):
    """Plot merged Sunset department-group trends, with optional LLM markers."""
    ncols = 2
    nrows = int(np.ceil(len(groups) / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 3.6 * nrows), sharey=True)
    axes = np.array(axes).reshape(-1)

    band_colors = {
        "A %": "#2ca02c",
        "B %": "#1f77b4",
        "C %": "#ff7f0e",
        "D/F %": "#d62728",
    }
    llm_colors = {
        "GPT": "#111111",
        "Claude": "#6f4e7c",
        "Gemini": "#1e88e5",
        "Grok": "#ff1493",
    }

    for index, group in enumerate(groups):
        ax = axes[index]
        sub = group_term_df[group_term_df["Group"] == group].sort_values(["Term Date", "Term"])

        for column, color in band_colors.items():
            ax.plot(sub["Term Date"], sub[column], marker="o", linewidth=1.9, markersize=3.7, color=color)

        ax.axvspan(COVID_START, COVID_REMOTE_END, color="#7f7f7f", alpha=0.08)
        ax.axvline(COVID_START, color="#7f7f7f", linestyle=":", linewidth=1.2, alpha=0.9)
        ax.axvline(COVID_REMOTE_END, color="#7f7f7f", linestyle=":", linewidth=1.0, alpha=0.75)

        if with_llm_markers:
            for provider, release_date in LLM_MARKERS.items():
                ax.axvline(release_date, color=llm_colors[provider], linestyle="--", linewidth=1.2, alpha=0.7)

        ax.set_title(group)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.18)
        ax.tick_params(axis="x", rotation=35)

    for index in range(len(groups), len(axes)):
        fig.delaxes(axes[index])

    band_handles = [plt.Line2D([0], [0], color=color, lw=2) for color in band_colors.values()]
    band_labels = list(band_colors.keys())
    covid_handle = plt.Line2D([0], [0], color="#7f7f7f", lw=1.5, linestyle=":")

    if with_llm_markers:
        marker_handles = [plt.Line2D([0], [0], color=color, lw=1.5, linestyle="--") for color in llm_colors.values()]
        marker_labels = [f"{provider} release" for provider in llm_colors]
        fig.legend(
            band_handles + marker_handles + [covid_handle],
            band_labels + marker_labels + ["COVID period marker"],
            loc="upper center",
            ncol=5,
            bbox_to_anchor=(0.5, 1.02),
        )
        fig.suptitle("Merged Departments: Grade-Band Change vs Time (with LLM + COVID markers)", y=1.08)
    else:
        fig.legend(
            band_handles + [covid_handle],
            band_labels + ["COVID period marker"],
            loc="upper center",
            ncol=5,
            bbox_to_anchor=(0.5, 1.02),
        )
        fig.suptitle("Merged Departments: Grade-Band Change vs Time (with COVID marker)", y=1.06)

    fig.supxlabel("Term Date")
    fig.supylabel("Share of Letter Grades (%)")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    return fig, axes


def plot_group_trends_for_provider(
    group_term_df: pd.DataFrame,
    groups: list[str],
    provider: str,
    releases_df: pd.DataFrame,
    save_path: str | Path | None = None,
):
    """Plot merged Sunset department-group trends with version markers for one provider."""
    ncols = 2
    nrows = int(np.ceil(len(groups) / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 3.8 * nrows), sharey=True)
    axes = np.array(axes).reshape(-1)

    band_colors = {
        "A %": "#2ca02c",
        "B %": "#1f77b4",
        "C %": "#ff7f0e",
        "D/F %": "#d62728",
    }
    provider_color = {
        "GPT": "#111111",
        "Claude": "#6f4e7c",
        "Gemini": "#1e88e5",
        "Grok": "#ff1493",
    }.get(provider, "#333333")

    min_date = group_term_df["Term Date"].min()
    max_date = group_term_df["Term Date"].max()
    releases = releases_df[
        (releases_df["Release Date"] >= min_date - pd.Timedelta(days=45))
        & (releases_df["Release Date"] <= max_date + pd.Timedelta(days=45))
    ].copy()

    for index, group in enumerate(groups):
        ax = axes[index]
        sub = group_term_df[group_term_df["Group"] == group].sort_values(["Term Date", "Term"])

        for column, color in band_colors.items():
            ax.plot(sub["Term Date"], sub[column], marker="o", linewidth=1.9, markersize=3.6, color=color)

        ax.axvspan(COVID_START, COVID_REMOTE_END, color="#7f7f7f", alpha=0.08)
        ax.axvline(COVID_START, color="#7f7f7f", linestyle=":", linewidth=1.2, alpha=0.9)
        ax.axvline(COVID_REMOTE_END, color="#7f7f7f", linestyle=":", linewidth=1.0, alpha=0.75)

        for _, row in releases.iterrows():
            ax.axvline(row["Release Date"], color=provider_color, linestyle="--", linewidth=1.1, alpha=0.5)

        ax.set_title(group)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.18)
        ax.tick_params(axis="x", rotation=35)

    for index in range(len(groups), len(axes)):
        fig.delaxes(axes[index])

    if len(releases) and len(groups):
        top_ax = axes[0]
        y_top = top_ax.get_ylim()[1]
        for _, row in releases.iterrows():
            top_ax.text(
                row["Release Date"],
                y_top * 0.98,
                str(row["Model"]),
                rotation=90,
                va="top",
                ha="center",
                color=provider_color,
                fontsize=6.8,
                alpha=0.8,
            )

    band_handles = [plt.Line2D([0], [0], color=color, lw=2) for color in band_colors.values()]
    marker_handle = plt.Line2D([0], [0], color=provider_color, lw=1.5, linestyle="--")
    covid_handle = plt.Line2D([0], [0], color="#7f7f7f", lw=1.5, linestyle=":")
    legend_labels = list(band_colors.keys()) + [f"{provider} version release", "COVID period marker"]
    fig.legend(
        band_handles + [marker_handle, covid_handle],
        legend_labels,
        loc="upper center",
        ncol=6,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.suptitle(f"Merged Departments: Grade-Band Change vs Time with {provider} Releases + COVID marker", y=1.06)
    fig.supxlabel("Term Date")
    fig.supylabel("Share of Letter Grades (%)")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    return fig, axes
