"""Reusable descriptive analysis helpers for the Sunset notebooks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def available_grade_columns(df: pd.DataFrame) -> list[str]:
    """Return grade columns in canonical order when present."""
    grade_order = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "F"]
    return [grade for grade in grade_order if grade in df.columns]


def overall_grade_distribution(df: pd.DataFrame, grade_columns: list[str]) -> pd.Series:
    """Compute overall letter-grade percentages."""
    totals = df[grade_columns].sum()
    return totals / totals.sum() * 100


def plot_overall_grade_distribution(
    grade_pct: pd.Series,
    save_path: str | Path,
) -> None:
    colors = ["#2ecc71"] * 3 + ["#3498db"] * 3 + ["#e67e22"] * 3 + ["#e74c3c"] * 2
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(grade_pct.index, grade_pct.values, color=colors[: len(grade_pct)], edgecolor="white")
    ax.set_title("Overall Grade Distribution - Sunset Dataset", fontsize=14)
    ax.set_xlabel("Grade")
    ax.set_ylabel("Percentage of Letter Grades")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def top_departments_by_frequency(df: pd.DataFrame, top_n: int = 15) -> pd.Index:
    """Return the most frequent departments in the Sunset data."""
    return df["Department"].value_counts().head(top_n).index


def summarize_department_gpa(df: pd.DataFrame, departments: list[str] | pd.Index) -> pd.DataFrame:
    """Average GPA summary for selected departments."""
    return (
        df[df["Department"].isin(departments)]
        .groupby("Department")["GPA"]
        .agg(["mean", "std", "count"])
        .sort_values("mean", ascending=False)
    )


def plot_department_gpa_summary(
    dept_gpa: pd.DataFrame,
    overall_mean: float,
    save_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(dept_gpa.index, dept_gpa["mean"], yerr=dept_gpa["std"], capsize=4, color="steelblue", edgecolor="white")
    ax.axhline(overall_mean, color="red", linestyle="--", linewidth=1.5, label=f"Overall avg ({overall_mean:.2f})")
    ax.set_title("Average GPA by Department - Sunset Dataset", fontsize=14)
    ax.set_xlabel("Department")
    ax.set_ylabel("Average GPA")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def summarize_term_gpa(df: pd.DataFrame) -> pd.DataFrame:
    """Average GPA by cleaned term label."""
    return df.groupby("Term_clean")["GPA"].agg(["mean", "count"]).sort_index()


def plot_term_gpa(term_gpa: pd.DataFrame, save_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(term_gpa.index, term_gpa["mean"], color="mediumseagreen", edgecolor="white")
    ax.set_title("Avg GPA by Term - Sunset Dataset", fontsize=14)
    ax.set_xlabel("Term")
    ax.set_ylabel("Avg GPA")
    ax.set_ylim(2.8, 4.0)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def focus_departments(df: pd.DataFrame, focus: list[str] | None = None) -> list[str]:
    """Return selected focus departments that exist in the frame."""
    focus = focus or ["CSE", "MATH", "ECE", "PHYS", "CHEM"]
    available = set(df["Department"].dropna().unique())
    return [department for department in focus if department in available]


def build_department_grade_heatmap(
    df: pd.DataFrame,
    departments: list[str],
    grade_columns: list[str],
) -> pd.DataFrame:
    """Compute department-level letter-grade percentages."""
    heatmap_data = (
        df[df["Department"].isin(departments)]
        .groupby("Department")[grade_columns]
        .sum()
    )
    return heatmap_data.div(heatmap_data.sum(axis=1), axis=0) * 100


def plot_department_grade_heatmap(heatmap_pct: pd.DataFrame, save_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 4))
    sns.heatmap(heatmap_pct, annot=True, fmt=".1f", cmap="YlOrRd_r", linewidths=0.5, ax=ax)
    ax.set_title("Department-Level Grade Distribution (%)", fontsize=13)
    ax.set_xlabel("Grade")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def compare_sunset_and_capes_by_department(
    sunset_df: pd.DataFrame,
    capes_df: pd.DataFrame,
    departments: list[str] | pd.Index,
) -> pd.DataFrame:
    """Build a side-by-side department GPA comparison table."""
    capes = capes_df.copy()
    capes["GPA_Received"] = capes["Average Grade Received"].str.extract(r"\(([\d.]+)\)").astype(float)
    capes["Department"] = capes["Course"].str.extract(r"^([A-Z]+(?:\s[A-Z]+)?)\s+\d")
    capes_clean = capes.dropna(subset=["GPA_Received", "Department"])

    comparison = pd.concat(
        [
            capes_clean[capes_clean["Department"].isin(departments)]
            .groupby("Department")["GPA_Received"]
            .mean()
            .rename("CAPES (2007-2023)"),
            sunset_df[sunset_df["Department"].isin(departments)]
            .groupby("Department")["GPA"]
            .mean()
            .rename("Sunset (2023+)"),
        ],
        axis=1,
    ).dropna()
    comparison["Delta"] = comparison["Sunset (2023+)"] - comparison["CAPES (2007-2023)"]
    return comparison.sort_values("Delta", ascending=False)


def plot_sunset_vs_capes(comparison: pd.DataFrame, save_path: str | Path) -> None:
    x = range(len(comparison))
    width = 0.4

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar([i - width / 2 for i in x], comparison["CAPES (2007-2023)"], width=width, label="CAPES (2007-2023)", color="steelblue", edgecolor="white")
    ax.bar([i + width / 2 for i in x], comparison["Sunset (2023+)"], width=width, label="Sunset (2023+)", color="darkorange", edgecolor="white")
    ax.set_xticks(list(x))
    ax.set_xticklabels(comparison.index, rotation=45)
    ax.set_ylabel("Average GPA")
    ax.set_title("Department GPA: Sunset vs Historical CAPES", fontsize=14)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
