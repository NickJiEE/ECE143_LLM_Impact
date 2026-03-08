"""Reusable helpers for the department analysis notebook."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DIVISION_ORDER = [
    "Lower Division (1-99)",
    "Upper Division (100-199)",
    "Graduate (200+)",
]
FOCUS_DEPARTMENTS = ["CSE", "MATH", "ECE", "ECON"]
LLM_CUTOFF = 2022 + 11 / 12


def top_department_counts(df: pd.DataFrame, top_n: int = 15) -> tuple[pd.Series, list[str]]:
    """Return the most frequent departments and their labels."""
    counts = df["Department"].value_counts().head(top_n)
    return counts, counts.index.tolist()


def plot_department_record_counts(
    counts: pd.Series,
    save_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Top 15 Departments by Number of CAPES Records", fontsize=14)
    ax.set_xlabel("Department")
    ax.set_ylabel("Number of Records")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def summarize_department_gpa(df: pd.DataFrame, departments: list[str]) -> pd.DataFrame:
    """Summarize mean/std/count GPA for selected departments."""
    return (
        df[df["Department"].isin(departments)]
        .groupby("Department")["GPA_Received"]
        .agg(["mean", "std", "count"])
        .sort_values("mean", ascending=False)
    )


def plot_department_gpa_summary(
    dept_gpa: pd.DataFrame,
    overall_mean: float,
    save_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(
        dept_gpa.index,
        dept_gpa["mean"],
        yerr=dept_gpa["std"],
        capsize=4,
        color="coral",
        edgecolor="white",
    )
    ax.axhline(
        overall_mean,
        color="navy",
        linestyle="--",
        linewidth=1.5,
        label=f"Overall avg ({overall_mean:.2f})",
    )
    ax.set_title("Average GPA by Department (Top 15)", fontsize=14)
    ax.set_xlabel("Department")
    ax.set_ylabel("Average GPA Received")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def focus_departments(df: pd.DataFrame, focus: list[str] | None = None) -> list[str]:
    """Keep only focus departments that are present in the data."""
    focus = focus or FOCUS_DEPARTMENTS
    available = set(df["Department"].dropna().unique())
    return [dept for dept in focus if dept in available]


def build_department_trend(df: pd.DataFrame, departments: list[str]) -> pd.DataFrame:
    """Aggregate mean GPA by quarter and department."""
    return (
        df[df["Department"].isin(departments)]
        .groupby(["Quarter_Num", "Department"])["GPA_Received"]
        .mean()
        .reset_index()
    )


def plot_department_trend(
    trend: pd.DataFrame,
    departments: list[str],
    llm_events: list[tuple[float, str, str]],
    llm_colors: dict[str, str],
    save_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    palette = sns.color_palette("tab10", len(departments))

    for dept, color in zip(departments, palette):
        sub = trend[trend["Department"] == dept].sort_values("Quarter_Num")
        ax.plot(
            sub["Quarter_Num"],
            sub["GPA_Received"],
            marker="o",
            linewidth=2,
            label=dept,
            color=color,
        )

    y_top = ax.get_ylim()[1]
    for quarter_num, model, brand in llm_events:
        ax.axvline(quarter_num, color=llm_colors[brand], linestyle="--", alpha=0.35)
        ax.text(
            quarter_num,
            y_top,
            model,
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            color=llm_colors[brand],
        )

    ax.set_title("Average GPA Over Time for Selected Departments", fontsize=14)
    ax.set_xlabel("Quarter Number")
    ax.set_ylabel("Average GPA Received")
    ax.grid(alpha=0.3)
    ax.legend(title="Department")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def summarize_department_recommendations(df: pd.DataFrame, departments: list[str]) -> pd.DataFrame:
    """Average class and professor recommendation rates by department."""
    return (
        df[df["Department"].isin(departments)]
        .groupby("Department")[["Pct_Rec_Class", "Pct_Rec_Prof"]]
        .mean()
        .sort_values("Pct_Rec_Prof", ascending=False)
    )


def plot_department_recommendations(dept_rec: pd.DataFrame, save_path: str | Path) -> None:
    x = range(len(dept_rec))
    width = 0.4

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(
        [i - width / 2 for i in x],
        dept_rec["Pct_Rec_Class"],
        width=width,
        label="% Recommended Class",
        color="steelblue",
        edgecolor="white",
    )
    ax.bar(
        [i + width / 2 for i in x],
        dept_rec["Pct_Rec_Prof"],
        width=width,
        label="% Recommended Professor",
        color="darkorange",
        edgecolor="white",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(dept_rec.index, rotation=45)
    ax.set_ylabel("Percentage")
    ax.set_title("Recommendation Rates by Department (Top 15)", fontsize=14)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def filter_division_frame(
    df: pd.DataFrame,
    division_order: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Return rows for the supported division labels."""
    division_order = division_order or DIVISION_ORDER
    return df[df["Division"].isin(division_order)].copy(), division_order


def plot_division_gpa_distribution(
    df_div: pd.DataFrame,
    division_order: list[str],
    save_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(
        data=df_div,
        x="Division",
        y="GPA_Received",
        order=division_order,
        palette="Set2",
        ax=ax,
        inner="quartile",
    )
    ax.set_title("GPA Received Distribution by Course Division", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("GPA Received")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def filter_division_hours(df_div: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with study-hours data for division plots."""
    return df_div.dropna(subset=["Study Hours per Week"]).copy()


def plot_division_study_hours(
    df_div_hours: pd.DataFrame,
    division_order: list[str],
    save_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=df_div_hours,
        x="Division",
        y="Study Hours per Week",
        order=division_order,
        palette="Set3",
        ax=ax,
        showfliers=False,
    )
    ax.set_title("Study Hours per Week by Course Division", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("Study Hours / Week")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def build_division_trend(df_div: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean GPA by quarter and course division."""
    return (
        df_div.groupby(["Quarter_Num", "Division"])["GPA_Received"]
        .mean()
        .reset_index()
    )


def plot_division_trend(
    div_trend: pd.DataFrame,
    division_order: list[str],
    llm_events: list[tuple[float, str, str]],
    llm_colors: dict[str, str],
    save_path: str | Path,
) -> None:
    div_palette = {
        "Lower Division (1-99)": "royalblue",
        "Upper Division (100-199)": "darkorange",
        "Graduate (200+)": "seagreen",
    }

    fig, ax = plt.subplots(figsize=(14, 6))
    for division in division_order:
        sub = div_trend[div_trend["Division"] == division].sort_values("Quarter_Num")
        ax.plot(
            sub["Quarter_Num"],
            sub["GPA_Received"],
            marker="o",
            linewidth=2,
            label=division,
            color=div_palette[division],
        )

    y_top = ax.get_ylim()[1]
    for quarter_num, model, brand in llm_events:
        ax.axvline(quarter_num, color=llm_colors[brand], linestyle="--", alpha=0.35)
        ax.text(
            quarter_num,
            y_top,
            model,
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            color=llm_colors[brand],
        )

    ax.set_title("Average GPA Over Time by Course Division", fontsize=14)
    ax.set_xlabel("Quarter Number")
    ax.set_ylabel("Average GPA Received")
    ax.grid(alpha=0.3)
    ax.legend(title="Division")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def build_department_division_heatmap(
    df: pd.DataFrame,
    departments: list[str],
    division_order: list[str],
) -> pd.DataFrame:
    """Return department x division mean GPA matrix."""
    pivot = (
        df[
            df["Department"].isin(departments)
            & df["Division"].isin(division_order)
        ]
        .groupby(["Department", "Division"])["GPA_Received"]
        .mean()
        .unstack("Division")
    )
    available = [division for division in division_order if division in pivot.columns]
    return pivot[available]


def plot_department_division_heatmap(pivot: pd.DataFrame, save_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=2.8,
        vmax=4.0,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Average GPA by Department and Division", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def add_era_column(df: pd.DataFrame, cutoff: float = LLM_CUTOFF) -> pd.DataFrame:
    """Annotate rows as pre/post LLM release era."""
    out = df.copy()
    out["Era"] = out["Quarter_Num"].apply(
        lambda value: "Post-LLM (after Nov 2022)" if value >= cutoff else "Pre-LLM"
    )
    return out


def summarize_department_era_delta(df: pd.DataFrame, departments: list[str]) -> pd.DataFrame:
    """Compute post-minus-pre GPA delta by department."""
    era_dept = (
        df[df["Department"].isin(departments)]
        .groupby(["Department", "Era"])["GPA_Received"]
        .mean()
        .unstack("Era")
    )
    era_dept["Delta"] = era_dept.get("Post-LLM (after Nov 2022)", 0) - era_dept.get("Pre-LLM", 0)
    return era_dept.sort_values("Delta", ascending=False)


def plot_department_era_delta(era_dept: pd.DataFrame, save_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(era_dept.index, era_dept["Delta"], color="mediumpurple", edgecolor="white")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Department GPA Change: Post-LLM minus Pre-LLM", fontsize=14)
    ax.set_xlabel("Department")
    ax.set_ylabel("GPA Delta")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def print_department_summary(df_clean: pd.DataFrame, division_order: list[str]) -> None:
    """Print the summary tables used at the end of the notebook."""
    print("=== Overall GPA Statistics ===")
    print(df_clean["GPA_Received"].describe().round(3))

    print("\n=== GPA by Division ===")
    print(
        df_clean[df_clean["Division"].isin(division_order)]
        .groupby("Division")["GPA_Received"]
        .agg(["mean", "median", "std", "count"])
        .round(3)
    )

    print("\n=== Pre vs Post LLM (all departments) ===")
    print(df_clean.groupby("Era")["GPA_Received"].agg(["mean", "count"]).round(3))
