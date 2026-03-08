"""Reusable helpers for the sub-hypothesis notebook."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

SEASON_ORDER = ["Fall", "Winter", "Spring", "Summer"]
FOCUS_DEPARTMENTS = ["CSE", "MATH", "ECE", "ECON"]
SIZE_BINS = [0, 20, 40, 75, 120, 200, 350, 600]
SIZE_LABELS = ["≤20", "21-40", "41-75", "76-120", "121-200", "201-350", "351+"]
SEASON_COLORS = {
    "Fall": "#e74c3c",
    "Winter": "#3498db",
    "Spring": "#2ecc71",
    "Summer": "#f39c12",
}


def prepare_class_size_frame(df: pd.DataFrame, max_enrollment: int = 500) -> pd.DataFrame:
    """Keep rows needed for enrollment-based analyses."""
    frame = df.dropna(subset=["Pct_Rec_Class", "Pct_Rec_Prof", "Total Enrolled in Course"]).copy()
    return frame[frame["Total Enrolled in Course"] <= max_enrollment].copy()


def plot_class_size_hexbin(df_size: pd.DataFrame, save_path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    specs = [
        (axes[0], "Pct_Rec_Class", "Enrollment vs % Recommend Class", "steelblue"),
        (axes[1], "Pct_Rec_Prof", "Enrollment vs % Recommend Professor", "coral"),
    ]
    for ax, y_col, title, cmap in specs:
        hb = ax.hexbin(df_size["Total Enrolled in Course"], df_size[y_col], gridsize=30, cmap=cmap, mincnt=1)
        plt.colorbar(hb, ax=ax, label="Count")
        ax.set_xlabel("Total Enrolled in Course")
        ax.set_ylabel(y_col.replace("_", " "))
        ax.set_title(title)
        ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def add_size_bins(
    df_size: pd.DataFrame,
    bins: list[int] | None = None,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Add human-readable enrollment bins."""
    bins = bins or SIZE_BINS
    labels = labels or SIZE_LABELS
    out = df_size.copy()
    out["Size_Bin"] = pd.cut(out["Total Enrolled in Course"], bins=bins, labels=labels)
    return out


def summarize_size_bins(df_size: pd.DataFrame) -> pd.DataFrame:
    """Average recommendation and GPA metrics by enrollment bin."""
    return (
        df_size.groupby("Size_Bin", observed=True)[["Pct_Rec_Class", "Pct_Rec_Prof", "GPA_Received"]]
        .mean()
    )


def plot_size_bin_summary(bin_stats: pd.DataFrame, save_path: str | Path) -> None:
    x = range(len(bin_stats))
    width = 0.3

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar([i - width for i in x], bin_stats["Pct_Rec_Class"], width=width, label="% Rec Class", color="steelblue", edgecolor="white")
    ax.bar(x, bin_stats["Pct_Rec_Prof"], width=width, label="% Rec Prof", color="darkorange", edgecolor="white")
    ax.bar([i + width for i in x], bin_stats["GPA_Received"], width=width, label="Avg GPA", color="seagreen", edgecolor="white")
    ax.set_xticks(list(x))
    ax.set_xticklabels(bin_stats.index)
    ax.set_ylabel("Average Value")
    ax.set_title("Class Size Bins vs Satisfaction and GPA", fontsize=14)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def prepare_season_frame(df: pd.DataFrame, season_order: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Keep rows belonging to the four standard academic seasons."""
    season_order = season_order or SEASON_ORDER
    frame = df.dropna(subset=["Season_Name"]).copy()
    frame = frame[frame["Season_Name"].isin(season_order)].copy()
    return frame, season_order


def summarize_season_gpa(df_season: pd.DataFrame, season_order: list[str]) -> pd.DataFrame:
    """Average GPA by season."""
    return (
        df_season.groupby("Season_Name")["GPA_Received"]
        .agg(["mean", "std", "count"])
        .reindex(season_order)
    )


def plot_season_gpa(season_gpa: pd.DataFrame, save_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [SEASON_COLORS[season] for season in season_gpa.index]
    ax.bar(season_gpa.index, season_gpa["mean"], yerr=season_gpa["std"], capsize=5, color=colors, edgecolor="white")
    ax.set_title("Average GPA by Quarter Season", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("Average GPA Received")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def focus_departments(df: pd.DataFrame, focus: list[str] | None = None) -> list[str]:
    """Return the subset of focus departments present in the frame."""
    focus = focus or FOCUS_DEPARTMENTS
    available = set(df["Department"].dropna().unique())
    return [dept for dept in focus if dept in available]


def build_season_department_heatmap(
    df_season: pd.DataFrame,
    departments: list[str],
    season_order: list[str],
) -> pd.DataFrame:
    """Return department x season GPA matrix."""
    return (
        df_season[df_season["Department"].isin(departments)]
        .groupby(["Department", "Season_Name"])["GPA_Received"]
        .mean()
        .unstack("Season_Name")
        .reindex(columns=[season for season in season_order if season in df_season["Season_Name"].unique()])
    )


def plot_season_department_heatmap(pivot: pd.DataFrame, save_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=2.8, vmax=4.0, linewidths=0.5, ax=ax)
    ax.set_title("Average GPA by Department and Season", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def summarize_season_recommendations(df_season: pd.DataFrame, season_order: list[str]) -> pd.DataFrame:
    """Average class and professor recommendation rates by season."""
    return (
        df_season.dropna(subset=["Pct_Rec_Class", "Pct_Rec_Prof"])
        .groupby("Season_Name")[["Pct_Rec_Class", "Pct_Rec_Prof"]]
        .mean()
        .reindex(season_order)
    )


def plot_season_recommendations(season_rec: pd.DataFrame, save_path: str | Path) -> None:
    x = range(len(season_rec))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], season_rec["Pct_Rec_Class"], width=width, label="% Rec Class", color="steelblue", edgecolor="white")
    ax.bar([i + width / 2 for i in x], season_rec["Pct_Rec_Prof"], width=width, label="% Rec Prof", color="darkorange", edgecolor="white")
    ax.set_xticks(list(x))
    ax.set_xticklabels(season_rec.index)
    ax.set_ylabel("Percentage")
    ax.set_title("Recommendation Rates by Quarter Season", fontsize=14)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def filter_season_hours(df_season: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with study-hours data."""
    return df_season.dropna(subset=["Study Hours per Week"]).copy()


def plot_season_study_hours(
    df_hours: pd.DataFrame,
    season_order: list[str],
    save_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(
        data=df_hours,
        x="Season_Name",
        y="Study Hours per Week",
        order=[season for season in season_order if season in df_hours["Season_Name"].unique()],
        palette=SEASON_COLORS,
        ax=ax,
        showfliers=False,
    )
    ax.set_title("Study Hours per Week by Quarter Season", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("Study Hours / Week")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def print_sub_hypothesis_summary(df_size: pd.DataFrame, season_gpa: pd.DataFrame, df_hours: pd.DataFrame) -> None:
    """Print summary tables used at the end of the notebook."""
    print("=== Class Size vs Satisfaction ===")
    print(f"  Enrollment vs Rec Class:    r = {df_size['Total Enrolled in Course'].corr(df_size['Pct_Rec_Class']):.3f}")
    print(f"  Enrollment vs Rec Prof:     r = {df_size['Total Enrolled in Course'].corr(df_size['Pct_Rec_Prof']):.3f}")
    print(f"  Enrollment vs GPA:          r = {df_size['Total Enrolled in Course'].corr(df_size['GPA_Received']):.3f}")

    print("\n=== GPA by Season ===")
    print(season_gpa[["mean", "std", "count"]].round(3))

    print("\n=== Study Hours by Season ===")
    print(df_hours.groupby("Season_Name")["Study Hours per Week"].describe().round(3))
