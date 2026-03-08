"""Grouped quarter-level merge for Sunset grades, RMP reviews, and CAPES study hours."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.capes_analysis import clean_capes_analysis_frame, load_capes_data
from src.difficulty_analysis import (
    GROUP_COLORS,
    GROUP_DEPARTMENTS,
    assign_department_group,
    quarter_to_label,
    resolve_project_root,
)
from src.sunset_analysis import clean_sunset_analysis_frame, load_sunset_data, term_to_date


MONTH_TO_QUARTER = {
    1: 0.0,
    2: 0.0,
    3: 0.0,
    4: 0.25,
    5: 0.25,
    6: 0.25,
    7: 0.5,
    8: 0.5,
    9: 0.5,
    10: 0.75,
    11: 0.75,
    12: 0.75,
}

GRADE_BANDS = {
    "A_share": ["A+", "A", "A-"],
    "B_share": ["B+", "B", "B-"],
    "C_share": ["C+", "C", "C-"],
    "DF_share": ["D", "F"],
}


def quarter_num_from_term(term: str | float) -> float | None:
    """Convert Sunset term strings into the common float quarter representation."""
    ts = term_to_date(term)
    if pd.isna(ts):
        return None
    quarter_map = {1: 0.0, 4: 0.25, 9: 0.75}
    return ts.year + quarter_map[ts.month]


def prepare_sunset_group_quarter(root_dir: str | Path) -> pd.DataFrame:
    """Aggregate Sunset grade outcomes by broad group and quarter."""
    root_dir = resolve_project_root(root_dir)
    sunset = clean_sunset_analysis_frame(load_sunset_data(root_dir / "data" / "sunset_data.csv"))
    sunset["Quarter_Num"] = sunset["Term"].apply(quarter_num_from_term)
    sunset["Group"] = sunset["Department"].apply(assign_department_group)
    sunset = sunset.dropna(subset=["Quarter_Num", "Group", "GPA"]).copy()

    work = sunset[["Group", "Quarter_Num", "GPA", *sum(GRADE_BANDS.values(), [])]].copy()
    summary = work.groupby(["Group", "Quarter_Num"], as_index=False).sum(numeric_only=True)
    letter_total = summary[sum(GRADE_BANDS.values(), [])].sum(axis=1)
    summary["Sunset_GPA"] = summary["GPA"]
    for band, grades in GRADE_BANDS.items():
        summary[band] = summary[grades].sum(axis=1).div(letter_total.where(letter_total > 0))
    keep = ["Group", "Quarter_Num", "Sunset_GPA", *GRADE_BANDS.keys()]
    return summary[keep]


def prepare_rmp_group_quarter(root_dir: str | Path) -> pd.DataFrame:
    """Aggregate RMP difficulty and review quality by broad group and quarter."""
    root_dir = resolve_project_root(root_dir)
    rmp = pd.read_csv(root_dir / "data" / "rmp_ucsd_reviews.csv")
    rmp["date"] = pd.to_datetime(
        rmp["date"],
        format="%Y-%m-%d %H:%M:%S +0000 UTC",
        errors="coerce",
    )
    rmp["Quarter_Num"] = rmp["date"].dt.year + rmp["date"].dt.month.map(MONTH_TO_QUARTER)
    rmp["Department"] = rmp["class"].astype(str).str.extract(r"^([A-Z]{2,}(?:\s[A-Z]+)?)")
    rmp["Group"] = rmp["Department"].apply(assign_department_group)
    rmp["difficultyRatingRounded"] = pd.to_numeric(rmp["difficultyRatingRounded"], errors="coerce")
    rmp["qualityRating"] = pd.to_numeric(rmp["qualityRating"], errors="coerce")
    rmp = rmp.dropna(subset=["Quarter_Num", "Group", "difficultyRatingRounded"]).copy()

    summary = (
        rmp.groupby(["Group", "Quarter_Num"], as_index=False)
        .agg(
            RMP_Difficulty=("difficultyRatingRounded", "mean"),
            RMP_Quality=("qualityRating", "mean"),
            RMP_Review_Count=("difficultyRatingRounded", "size"),
        )
    )
    return summary


def prepare_capes_group_quarter(root_dir: str | Path) -> pd.DataFrame:
    """Aggregate CAPES study hours and GPA by broad group and quarter."""
    root_dir = resolve_project_root(root_dir)
    capes = clean_capes_analysis_frame(load_capes_data(root_dir / "data" / "capes_data.csv"))
    capes["Group"] = capes["Department"].apply(assign_department_group)
    capes = capes.dropna(subset=["Quarter_Num", "Group", "Study Hours per Week", "GPA_Received"]).copy()
    summary = (
        capes.groupby(["Group", "Quarter_Num"], as_index=False)
        .agg(
            CAPES_Study_Hours=("Study Hours per Week", "mean"),
            CAPES_GPA=("GPA_Received", "mean"),
            CAPES_Count=("Study Hours per Week", "size"),
        )
    )
    return summary


def build_merged_group_quarter(root_dir: str | Path) -> pd.DataFrame:
    """Build the recommended grouped quarter-level merge.

    Sunset + RMP is the main analysis frame for 2019-2025.
    CAPES is attached as a validation layer for the overlapping period through 2023.
    """
    root_dir = resolve_project_root(root_dir)
    sunset = prepare_sunset_group_quarter(root_dir)
    rmp = prepare_rmp_group_quarter(root_dir)
    capes = prepare_capes_group_quarter(root_dir)

    merged = sunset.merge(rmp, on=["Group", "Quarter_Num"], how="inner")
    merged = merged.merge(capes, on=["Group", "Quarter_Num"], how="left")
    return merged.sort_values(["Group", "Quarter_Num"]).reset_index(drop=True)


def build_rmp_difficulty_effects(merged: pd.DataFrame) -> pd.DataFrame:
    """Summarize RMP difficulty by group across the COVID and LLM windows."""
    work = merged.copy()
    work["Window"] = pd.NA
    work.loc[(work["Quarter_Num"] >= 2019.75) & (work["Quarter_Num"] <= 2021.75), "Window"] = "COVID Window"
    work.loc[(work["Quarter_Num"] >= 2022.0) & (work["Quarter_Num"] <= 2025.75), "Window"] = "LLM Window"
    work = work.dropna(subset=["Window"])

    summary = (
        work.groupby(["Group", "Window"])["RMP_Difficulty"]
        .mean()
        .unstack("Window")
        .reset_index()
    )
    summary["LLM - COVID"] = summary["LLM Window"] - summary["COVID Window"]
    return summary.sort_values("Group")


def plot_rmp_difficulty_windows(
    merged: pd.DataFrame,
    save_path: str | Path,
    group_colors: dict[str, str] | None = None,
) -> None:
    """Plot side-by-side RMP difficulty trends for COVID and LLM-rise windows."""
    group_colors = group_colors or GROUP_COLORS
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

    windows = [
        (
            axes[0],
            merged[(merged["Quarter_Num"] >= 2019.75) & (merged["Quarter_Num"] <= 2021.75)],
            "Difficulty Trend: COVID Years",
        ),
        (
            axes[1],
            merged[(merged["Quarter_Num"] >= 2022.0) & (merged["Quarter_Num"] <= 2025.75)],
            "Difficulty Trend: LLM Evolution Years",
        ),
    ]

    legend_handles = []
    legend_labels = []

    for ax, window_df, title in windows:
        for group in window_df["Group"].dropna().unique():
            sub = window_df[window_df["Group"] == group].sort_values("Quarter_Num")
            (line,) = ax.plot(
                sub["Quarter_Num"],
                sub["RMP_Difficulty"],
                marker="o",
                markersize=3,
                linewidth=2,
                label=group,
                color=group_colors.get(group),
            )
            if group not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(group)

        tick_values = sorted(window_df["Quarter_Num"].unique())
        ax.set_xticks(tick_values)
        ax.set_xticklabels(
            [quarter_to_label(value) for value in tick_values],
            rotation=45,
            ha="right",
        )
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Quarter")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("RMP Difficulty")
    fig.legend(
        legend_handles,
        legend_labels,
        title="Group",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=min(4, len(legend_labels)),
        frameon=False,
    )
    fig.subplots_adjust(top=0.80)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
