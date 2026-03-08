"""Reusable helpers for the professor analysis notebook."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_rmp_data(root_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load RateMyProfessor summary and review exports."""
    root_dir = Path(root_dir)
    return (
        pd.read_csv(root_dir / "data" / "rmp_ucsd_professors.csv"),
        pd.read_csv(root_dir / "data" / "rmp_ucsd_reviews.csv"),
    )


def summarize_professors(df: pd.DataFrame, min_courses: int = 5) -> pd.DataFrame:
    """Aggregate core professor-level CAPES metrics."""
    return (
        df.groupby("Instructor")
        .agg(
            avg_gpa=("GPA_Received", "mean"),
            std_gpa=("GPA_Received", "std"),
            n_courses=("GPA_Received", "count"),
            avg_rec=("Pct_Rec_Prof", "mean"),
            department=("Department", lambda values: values.mode().iloc[0]),
        )
        .query("n_courses >= @min_courses")
        .sort_values("avg_gpa", ascending=False)
    )


def top_and_bottom_professors(
    prof_stats: pd.DataFrame,
    n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return top and bottom professors by mean GPA."""
    return prof_stats.head(n), prof_stats.tail(n).sort_values("avg_gpa")


def plot_top_bottom_professors(
    top10: pd.DataFrame,
    bottom10: pd.DataFrame,
    save_path: str | Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].barh(top10.index[::-1], top10["avg_gpa"][::-1], color="seagreen", edgecolor="white")
    axes[0].set_title("Top 10 Professors by Avg GPA")
    axes[0].set_xlabel("Avg GPA Received")
    axes[0].grid(axis="x", alpha=0.3)

    axes[1].barh(bottom10.index[::-1], bottom10["avg_gpa"][::-1], color="indianred", edgecolor="white")
    axes[1].set_title("Bottom 10 Professors by Avg GPA")
    axes[1].set_xlabel("Avg GPA Received")
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def top_professors_by_count(prof_stats: pd.DataFrame, n: int = 5) -> list[str]:
    """Return the most frequently reviewed professors."""
    return prof_stats.nlargest(n, "n_courses").index.tolist()


def plot_professor_trends(
    df: pd.DataFrame,
    professors: list[str],
    llm_events: list[tuple[float, str, str]],
    save_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    palette = sns.color_palette("tab10", len(professors))

    for professor, color in zip(professors, palette):
        sub = (
            df[df["Instructor"] == professor]
            .groupby("Quarter_Num")["GPA_Received"]
            .mean()
            .reset_index()
            .sort_values("Quarter_Num")
        )
        ax.plot(sub["Quarter_Num"], sub["GPA_Received"], marker="o", linewidth=2, label=professor, color=color)

    y_top = ax.get_ylim()[1]
    llm_palette = {"GPT": "green", "Claude": "purple", "Gemini": "blue", "Grok": "orange"}
    for quarter_num, model, brand in llm_events:
        ax.axvline(quarter_num, color=llm_palette.get(brand, "gray"), linestyle="--", alpha=0.35)
        ax.text(quarter_num, y_top, model, rotation=90, va="top", ha="right", fontsize=8, color=llm_palette.get(brand, "gray"))

    ax.set_title("Average GPA Over Time for Top Professors", fontsize=14)
    ax.set_xlabel("Quarter Number")
    ax.set_ylabel("Average GPA Received")
    ax.grid(alpha=0.3)
    ax.legend(title="Instructor")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_professor_gpa_vs_recommendation(
    prof_stats: pd.DataFrame,
    save_path: str | Path,
) -> float:
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(
        prof_stats["avg_gpa"],
        prof_stats["avg_rec"],
        c=prof_stats["n_courses"],
        cmap="viridis",
        alpha=0.5,
        s=20,
        edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label="Number of Courses")
    ax.set_xlabel("Avg GPA Received")
    ax.set_ylabel("Avg % Recommended Professor")
    ax.set_title("Professor: GPA vs Recommendation Rate", fontsize=14)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    return prof_stats["avg_gpa"].corr(prof_stats["avg_rec"])


def clean_rmp_frame(rmp_profs: pd.DataFrame) -> pd.DataFrame:
    """Coerce RMP numeric fields for analysis."""
    rmp = rmp_profs.copy()
    numeric_columns = [
        "avgRating",
        "avgDifficulty",
        "wouldTakeAgainPercent",
        "numRatings",
    ]
    for column in numeric_columns:
        rmp[column] = pd.to_numeric(rmp[column], errors="coerce")
    return rmp


def plot_rmp_distributions(rmp: pd.DataFrame, save_path: str | Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(rmp["avgRating"].dropna(), bins=20, color="steelblue", edgecolor="white")
    axes[0].set_title("RMP Rating Distribution")
    axes[0].set_xlabel("Average Rating")

    axes[1].hist(rmp["avgDifficulty"].dropna(), bins=20, color="darkorange", edgecolor="white")
    axes[1].set_title("RMP Difficulty Distribution")
    axes[1].set_xlabel("Average Difficulty")

    axes[2].hist(rmp["wouldTakeAgainPercent"].dropna(), bins=20, color="seagreen", edgecolor="white")
    axes[2].set_title("Would Take Again Distribution")
    axes[2].set_xlabel("Would Take Again (%)")

    for ax in axes:
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def summarize_rmp_departments(rmp: pd.DataFrame, top_n: int = 15) -> tuple[pd.DataFrame, pd.Index]:
    """Return average RMP metrics for the most frequent departments."""
    top_depts = rmp["department"].value_counts().head(top_n).index
    summary = (
        rmp[rmp["department"].isin(top_depts)]
        .groupby("department")[["avgRating", "avgDifficulty"]]
        .mean()
        .sort_values("avgRating", ascending=False)
    )
    return summary, top_depts


def plot_rmp_department_summary(dept_rating: pd.DataFrame, save_path: str | Path) -> None:
    x = range(len(dept_rating))
    width = 0.4

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar([i - width / 2 for i in x], dept_rating["avgRating"], width=width, label="Avg Rating", color="steelblue", edgecolor="white")
    ax.bar([i + width / 2 for i in x], dept_rating["avgDifficulty"], width=width, label="Avg Difficulty", color="indianred", edgecolor="white")
    ax.set_xticks(list(x))
    ax.set_xticklabels(dept_rating.index, rotation=45)
    ax.set_ylabel("Average Score")
    ax.set_title("RMP Scores by Department (Top 15)", fontsize=14)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def filter_rmp_for_scatter(rmp: pd.DataFrame, min_ratings: int = 10) -> pd.DataFrame:
    """Keep professors with enough RMP data for scatter analysis."""
    return rmp[rmp["numRatings"] >= min_ratings].dropna(
        subset=["avgRating", "avgDifficulty", "wouldTakeAgainPercent"]
    )


def plot_rmp_difficulty_vs_rating(rmp_filtered: pd.DataFrame, save_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(
        rmp_filtered["avgDifficulty"],
        rmp_filtered["avgRating"],
        c=rmp_filtered["wouldTakeAgainPercent"],
        cmap="RdYlGn",
        alpha=0.6,
        s=30,
        edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label="Would Take Again (%)")
    ax.set_xlabel("Avg Difficulty")
    ax.set_ylabel("Avg Rating")
    ax.set_title("RMP: Difficulty vs Rating", fontsize=14)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def build_capes_rmp_match(df_clean: pd.DataFrame, rmp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Match CAPES and RMP professor summaries by last name."""
    capes_prof = (
        df_clean.groupby("Instructor")
        .agg(capes_gpa=("GPA_Received", "mean"), capes_rec=("Pct_Rec_Prof", "mean"))
        .reset_index()
    )
    capes_prof["last_name"] = capes_prof["Instructor"].str.split().str[0].str.rstrip(",")

    rmp_summary = rmp[
        ["lastName", "avgRating", "avgDifficulty", "wouldTakeAgainPercent", "numRatings"]
    ].rename(columns={"lastName": "last_name"}).copy()
    rmp_summary["last_name"] = rmp_summary["last_name"].str.strip()

    merged = capes_prof.merge(rmp_summary, on="last_name", how="inner")
    return capes_prof, rmp_summary, merged


def plot_capes_rmp_comparison(merged: pd.DataFrame, save_path: str | Path) -> tuple[float, float]:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(merged["avgRating"], merged["capes_gpa"], alpha=0.5, color="steelblue", s=25, edgecolors="none")
    axes[0].set_xlabel("RMP Avg Rating")
    axes[0].set_ylabel("CAPES Avg GPA Received")
    axes[0].set_title("RMP Rating vs CAPES GPA", fontsize=12)
    axes[0].grid(alpha=0.3)
    corr_rating = merged["avgRating"].corr(merged["capes_gpa"])
    axes[0].text(0.05, 0.92, f"r = {corr_rating:.3f}", transform=axes[0].transAxes, fontsize=11)

    axes[1].scatter(
        merged["wouldTakeAgainPercent"],
        merged["capes_rec"],
        alpha=0.5,
        color="darkorange",
        s=25,
        edgecolors="none",
    )
    axes[1].set_xlabel("RMP Would Take Again (%)")
    axes[1].set_ylabel("CAPES % Recommended Professor")
    axes[1].set_title("RMP WTA vs CAPES Recommendation", fontsize=12)
    axes[1].grid(alpha=0.3)
    corr_rec = merged["wouldTakeAgainPercent"].corr(merged["capes_rec"])
    axes[1].text(0.05, 0.92, f"r = {corr_rec:.3f}", transform=axes[1].transAxes, fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    return corr_rating, corr_rec
