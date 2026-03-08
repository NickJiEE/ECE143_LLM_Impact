"""Difficulty analysis helpers for comparing COVID-era and LLM-era effects."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.capes_analysis import clean_capes_analysis_frame, load_capes_data

COVID_START_Q = 2020.25
COVID_REMOTE_END_Q = 2021.75
LLM_START_Q = 2022.9167
ERA_ORDER = ["Pre-COVID", "COVID Remote", "Post-COVID / Pre-LLM", "Post-LLM"]
ERA_COLORS = {
    "Pre-COVID": "#4c78a8",
    "COVID Remote": "#e45756",
    "Post-COVID / Pre-LLM": "#72b7b2",
    "Post-LLM": "#54a24b",
}
GROUP_DEPARTMENTS = {
    "Engineering": ["CSE", "ECE", "MAE", "SE", "CENG", "BENG"],
    "Biology": ["BILD", "BIMM", "BIBC", "BICD", "BIPN"],
    "Languages & Linguistics": ["LIGN", "JAPN", "CHIN", "LTEN", "LTWR"],
    "Social Sciences": ["ECON", "PSYC", "SOCI", "POLI", "COMM", "USP"],
}
GROUP_COLORS = {
    "Engineering": "#4c78a8",
    "Biology": "#54a24b",
    "Languages & Linguistics": "#b279a2",
    "Social Sciences": "#f58518",
}


def resolve_project_root(start_path: str | Path = ".") -> Path:
    """Return the project root regardless of whether the notebook starts in repo or notebooks/."""
    start = Path(start_path).resolve()
    candidates = [start, *start.parents]
    for candidate in candidates:
        if (candidate / "data" / "capes_data.csv").exists():
            return candidate
    raise FileNotFoundError("Could not locate project root containing data/capes_data.csv")


def quarter_to_label(value: float) -> str:
    """Convert the float quarter representation into a readable label."""
    season_map = {0.0: "WI", 0.25: "SP", 0.5: "SU", 0.75: "FA"}
    year = int(value)
    season = season_map.get(round(value - year, 2), "Q")
    return f"{season}{str(year)[-2:]}"


def assign_era(quarter_num: float) -> str:
    """Map a quarter number into the project eras used in the comparison."""
    if quarter_num < COVID_START_Q:
        return "Pre-COVID"
    if quarter_num < COVID_REMOTE_END_Q:
        return "COVID Remote"
    if quarter_num < LLM_START_Q:
        return "Post-COVID / Pre-LLM"
    return "Post-LLM"


def prepare_capes_difficulty(root_dir: str | Path) -> pd.DataFrame:
    """Return CAPES rows with study-hours workload and era labels."""
    root_dir = Path(root_dir)
    capes = clean_capes_analysis_frame(load_capes_data(root_dir / "data" / "capes_data.csv"))
    capes = capes.dropna(subset=["Quarter_Num", "Study Hours per Week"]).copy()
    capes["Era"] = capes["Quarter_Num"].apply(assign_era)
    capes["Source"] = "CAPES Study Hours"
    capes["Difficulty"] = capes["Study Hours per Week"]
    return capes


def assign_department_group(
    department: str | float,
    group_departments: dict[str, list[str]] | None = None,
) -> str | None:
    """Map a department code into one of the broad analysis groups."""
    if pd.isna(department):
        return None
    group_departments = group_departments or GROUP_DEPARTMENTS
    for group, departments in group_departments.items():
        if department in departments:
            return group
    return None


def prepare_group_difficulty(
    root_dir: str | Path,
    group_departments: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Return CAPES difficulty data labeled with broad department groups."""
    capes = prepare_capes_difficulty(root_dir)
    work = capes.copy()
    work["Group"] = work["Department"].apply(
        lambda value: assign_department_group(value, group_departments)
    )
    return work.dropna(subset=["Group"]).copy()


def prepare_rmp_difficulty(root_dir: str | Path) -> pd.DataFrame:
    """Return timestamped RMP review difficulty with quarter and era labels."""
    root_dir = Path(root_dir)
    reviews = pd.read_csv(root_dir / "data" / "rmp_ucsd_reviews.csv")
    reviews["date"] = pd.to_datetime(
        reviews["date"],
        format="%Y-%m-%d %H:%M:%S +0000 UTC",
        errors="coerce",
    )
    reviews = reviews.dropna(subset=["date", "difficultyRatingRounded"]).copy()

    month_to_quarter = {
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
    reviews["Quarter_Num"] = reviews["date"].dt.year + reviews["date"].dt.month.map(month_to_quarter)
    reviews["Era"] = reviews["Quarter_Num"].apply(assign_era)
    reviews["Source"] = "RMP Review Difficulty"
    reviews["Difficulty"] = pd.to_numeric(reviews["difficultyRatingRounded"], errors="coerce")
    return reviews.dropna(subset=["Difficulty"])


def build_timeline_summary(
    df: pd.DataFrame,
    value_col: str = "Difficulty",
    min_quarter: float | None = None,
) -> pd.DataFrame:
    """Aggregate a dataset into quarter-level means and counts."""
    work = df.copy()
    if min_quarter is not None:
        work = work[work["Quarter_Num"] >= min_quarter]
    summary = (
        work.groupby("Quarter_Num")[value_col]
        .agg(["mean", "count", "std"])
        .reset_index()
        .sort_values("Quarter_Num")
    )
    summary["sem"] = summary["std"] / summary["count"].pow(0.5)
    summary["Quarter_Label"] = summary["Quarter_Num"].apply(quarter_to_label)
    return summary


def build_group_timeline_summary(
    df: pd.DataFrame,
    min_quarter: float = 2018.0,
) -> pd.DataFrame:
    """Aggregate quarter-level mean study hours for each broad group."""
    work = df[df["Quarter_Num"] >= min_quarter].copy()
    summary = (
        work.groupby(["Group", "Quarter_Num"])["Difficulty"]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values(["Group", "Quarter_Num"])
    )
    summary["Quarter_Label"] = summary["Quarter_Num"].apply(quarter_to_label)
    return summary


def build_era_summary(df: pd.DataFrame, value_col: str = "Difficulty") -> pd.DataFrame:
    """Return era-level summary statistics for the requested measure."""
    summary = (
        df.groupby("Era")[value_col]
        .agg(["mean", "median", "std", "count"])
        .reindex(ERA_ORDER)
        .reset_index()
    )
    summary["sem"] = summary["std"] / summary["count"].pow(0.5)
    return summary


def plot_difficulty_timeline(
    capes_timeline: pd.DataFrame,
    rmp_timeline: pd.DataFrame,
    save_path: str | Path,
) -> None:
    """Plot quarter-level difficulty trends with COVID and LLM markers."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    panels = [
        (axes[0], capes_timeline, "CAPES Study Hours per Week", "Study Hours / Week", "#4c78a8"),
        (axes[1], rmp_timeline, "RMP Review Difficulty", "Difficulty Rating", "#f58518"),
    ]

    for ax, timeline, title, ylabel, color in panels:
        ax.plot(timeline["Quarter_Num"], timeline["mean"], color=color, linewidth=2)
        ax.fill_between(
            timeline["Quarter_Num"],
            timeline["mean"] - timeline["sem"],
            timeline["mean"] + timeline["sem"],
            color=color,
            alpha=0.18,
        )
        ax.axvspan(COVID_START_Q, COVID_REMOTE_END_Q, color="#bdbdbd", alpha=0.18)
        ax.axvline(LLM_START_Q, color="#2f2f2f", linestyle="--", linewidth=1.5)
        ax.set_title(title, fontsize=13)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

    axes[0].text(COVID_START_Q + 0.05, axes[0].get_ylim()[1] * 0.98, "COVID period", va="top", fontsize=9)
    axes[0].text(LLM_START_Q + 0.03, axes[0].get_ylim()[1] * 0.98, "GPT-3.5 release", va="top", fontsize=9)

    tick_values = sorted(set(capes_timeline["Quarter_Num"]).union(rmp_timeline["Quarter_Num"]))
    tick_values = [value for value in tick_values if value >= 2018.0]
    axes[1].set_xticks(tick_values[::2])
    axes[1].set_xticklabels([quarter_to_label(value) for value in tick_values[::2]], rotation=45, ha="right")
    axes[1].set_xlabel("Quarter")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_era_comparison(
    capes_era: pd.DataFrame,
    rmp_era: pd.DataFrame,
    save_path: str | Path,
) -> None:
    """Compare average difficulty across the four project eras."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    panels = [
        (axes[0], capes_era, "CAPES Workload by Era", "Study Hours / Week"),
        (axes[1], rmp_era, "RMP Difficulty by Era", "Difficulty Rating"),
    ]

    for ax, summary, title, ylabel in panels:
        colors = [ERA_COLORS[era] for era in summary["Era"]]
        ax.bar(summary["Era"], summary["mean"], yerr=summary["sem"], color=colors, edgecolor="white", capsize=4)
        ax.set_title(title, fontsize=13)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def build_effect_table(capes_era: pd.DataFrame, rmp_era: pd.DataFrame) -> pd.DataFrame:
    """Create a compact table of era means and simple deltas for interpretation."""
    capes_map = capes_era.set_index("Era")["mean"].to_dict()
    rmp_map = rmp_era.set_index("Era")["mean"].to_dict()

    rows = [
        {
            "Metric": "CAPES Study Hours",
            "Pre-COVID": capes_map["Pre-COVID"],
            "COVID Remote": capes_map["COVID Remote"],
            "Post-COVID / Pre-LLM": capes_map["Post-COVID / Pre-LLM"],
            "Post-LLM": capes_map["Post-LLM"],
        },
        {
            "Metric": "RMP Difficulty",
            "Pre-COVID": rmp_map["Pre-COVID"],
            "COVID Remote": rmp_map["COVID Remote"],
            "Post-COVID / Pre-LLM": rmp_map["Post-COVID / Pre-LLM"],
            "Post-LLM": rmp_map["Post-LLM"],
        },
    ]
    table = pd.DataFrame(rows)
    table["COVID - Pre-COVID"] = table["COVID Remote"] - table["Pre-COVID"]
    table["Post-LLM - Post-COVID / Pre-LLM"] = table["Post-LLM"] - table["Post-COVID / Pre-LLM"]
    table["Post-LLM - Pre-COVID"] = table["Post-LLM"] - table["Pre-COVID"]
    return table


def build_group_effect_table(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize pre-COVID, COVID, and post-LLM mean study hours by broad group."""
    summary = (
        df.groupby(["Group", "Era"])["Difficulty"]
        .mean()
        .unstack("Era")
        .reindex(columns=ERA_ORDER)
        .reset_index()
    )
    summary["COVID - Pre-COVID"] = summary["COVID Remote"] - summary["Pre-COVID"]
    summary["Post-LLM - Pre-COVID"] = summary["Post-LLM"] - summary["Pre-COVID"]
    summary["Post-LLM - Post-COVID / Pre-LLM"] = (
        summary["Post-LLM"] - summary["Post-COVID / Pre-LLM"]
    )
    return summary.sort_values("COVID - Pre-COVID", ascending=False)


def plot_group_difficulty_trends(
    group_timeline: pd.DataFrame,
    save_path: str | Path,
    group_colors: dict[str, str] | None = None,
) -> None:
    """Plot quarter-level study-hour trends for the four broad groups."""
    group_colors = group_colors or GROUP_COLORS
    fig, ax = plt.subplots(figsize=(14, 6))

    for group in group_timeline["Group"].dropna().unique():
        sub = group_timeline[group_timeline["Group"] == group].sort_values("Quarter_Num")
        ax.plot(
            sub["Quarter_Num"],
            sub["mean"],
            marker="o",
            markersize=3,
            linewidth=2,
            label=group,
            color=group_colors.get(group),
        )

    ax.axvspan(COVID_START_Q, COVID_REMOTE_END_Q, color="#bdbdbd", alpha=0.18)
    ax.axvline(LLM_START_Q, color="#2f2f2f", linestyle="--", linewidth=1.5)
    ax.text(COVID_START_Q + 0.05, ax.get_ylim()[1] * 0.99, "COVID period", va="top", fontsize=9)
    ax.text(LLM_START_Q + 0.03, ax.get_ylim()[1] * 0.99, "GPT-3.5", va="top", fontsize=9)

    tick_values = sorted(group_timeline["Quarter_Num"].unique())
    ax.set_xticks(tick_values[::2])
    ax.set_xticklabels([quarter_to_label(value) for value in tick_values[::2]], rotation=45, ha="right")
    ax.set_title("Difficulty Trend by Broad Department Group", fontsize=14)
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Study Hours per Week")
    ax.grid(alpha=0.25)
    ax.legend(title="Group")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_group_difficulty_windows(
    group_timeline: pd.DataFrame,
    save_path: str | Path,
    group_colors: dict[str, str] | None = None,
) -> None:
    """Plot side-by-side difficulty trends for the COVID window and the LLM-rise window."""
    group_colors = group_colors or GROUP_COLORS
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

    windows = [
        (
            axes[0],
            group_timeline[
                (group_timeline["Quarter_Num"] >= 2018.0)
                & (group_timeline["Quarter_Num"] <= COVID_REMOTE_END_Q)
            ],
            "Difficulty Trend: Pre to During COVID",
        ),
        (
            axes[1],
            group_timeline[group_timeline["Quarter_Num"] >= 2021.75],
            "Difficulty Trend: Post-COVID During LLM Rise",
        ),
    ]

    for ax, window_df, title in windows:
        for group in window_df["Group"].dropna().unique():
            sub = window_df[window_df["Group"] == group].sort_values("Quarter_Num")
            ax.plot(
                sub["Quarter_Num"],
                sub["mean"],
                marker="o",
                markersize=3,
                linewidth=2,
                label=group,
                color=group_colors.get(group),
            )

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

    axes[0].set_ylabel("Study Hours per Week")
    axes[1].legend(title="Group")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
