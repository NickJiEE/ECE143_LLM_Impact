import re

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_reviews():
    print("hello world!")


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    df_clean["date"] = df_clean["date"].str.replace(r"\s+UTC$", "", regex=True)
    df_clean["date"] = pd.to_datetime(df_clean["date"], utc=True).dt.tz_localize(None)
    return df_clean


def filter_courses(
    df: pd.DataFrame,
    course: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    df_filtered = df[
        (df["class"].str.upper() == course.upper())
        & (df["date"] >= start_date)
        & (df["date"] <= end_date)
    ].sort_values("date").copy()
    return df_filtered


def smooth_ratings(input_df: pd.DataFrame, span: int = 15) -> pd.DataFrame:
    df = input_df.copy()
    df["quality_avg"] = df["quality"].ewm(span=span, adjust=False).mean()
    df["difficulty_avg"] = df["difficulty"].ewm(span=span, adjust=False).mean()
    return df


def extract_department_code(course: str | float) -> str | None:
    if pd.isna(course):
        return None

    match = re.match(r"^([A-Z]{3})", str(course).strip().upper())
    if match is None:
        return None
    return match.group(1)


def add_department_codes(
    df: pd.DataFrame,
    class_col: str = "class",
    department_col: str = "department",
) -> pd.DataFrame:
    df_with_departments = df.copy()
    df_with_departments[department_col] = df_with_departments[class_col].apply(
        extract_department_code
    )
    return df_with_departments


def summarize_department_ratings(
    df: pd.DataFrame,
    department_col: str = "department",
) -> pd.DataFrame:
    department_summary = (
        df.dropna(subset=[department_col, "quality", "difficulty"])
        .groupby(department_col)[["quality", "difficulty"]]
        .agg(["mean", "count"])
    )
    department_summary.columns = [
        "quality_mean",
        "quality_count",
        "difficulty_mean",
        "difficulty_count",
    ]
    department_summary = department_summary.reset_index()
    department_summary["review_count"] = department_summary[
        ["quality_count", "difficulty_count"]
    ].min(axis=1)
    department_summary = department_summary.sort_values(
        ["review_count", "quality_mean"], ascending=[False, False]
    ).reset_index(drop=True)
    return department_summary


def major_departments(
    df: pd.DataFrame,
    include: list[str] | None = None,
    top_n: int = 8,
    department_col: str = "department",
) -> list[str]:
    include = [department.upper() for department in (include or [])]
    counts = (
        df[department_col]
        .dropna()
        .astype(str)
        .str.upper()
        .value_counts()
    )

    departments = []
    for department in include:
        if department in counts.index and department not in departments:
            departments.append(department)

    for department in counts.head(top_n).index.tolist():
        if department not in departments:
            departments.append(department)

    return departments


def filter_departments(
    df: pd.DataFrame,
    departments: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    department_col: str = "department",
) -> pd.DataFrame:
    normalized_departments = [department.upper() for department in departments]
    df_filtered = df[
        df[department_col].isin(normalized_departments)
        & (df["date"] >= start_date)
        & (df["date"] <= end_date)
    ].sort_values("date").copy()
    return df_filtered


def build_department_ratings(
    df: pd.DataFrame,
    department_col: str = "department",
) -> pd.DataFrame:
    department_ratings = (
        df.dropna(subset=[department_col, "date", "quality", "difficulty"])
        .groupby([department_col, "date"])[["quality", "difficulty"]]
        .mean()
        .reset_index()
        .sort_values([department_col, "date"])
    )
    return department_ratings


def smooth_department_ratings(
    df: pd.DataFrame,
    department_col: str = "department",
    span: int = 15,
) -> pd.DataFrame:
    smoothed_groups = []

    for department, group in df.groupby(department_col, sort=False):
        group = group.sort_values("date").copy()
        group["quality_avg"] = group["quality"].ewm(span=span, adjust=False).mean()
        group["difficulty_avg"] = group["difficulty"].ewm(span=span, adjust=False).mean()
        smoothed_groups.append(group)

    if not smoothed_groups:
        return df.copy()

    return pd.concat(smoothed_groups, ignore_index=True)


def plot_course_ratings(df: pd.DataFrame, course: str):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(df["date"], df["quality"], marker="o", label="quality")
    ax.plot(df["date"], df["difficulty"], marker="o", label="difficulty")

    ax.set_ylim(1, 5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Rating Score")
    ax.set_title(f"{course} Ratings Over Time")
    ax.legend()

    ax.set_xticks([])

    fig.tight_layout()
    return fig, ax


def plot_smoothed_ratings(df: pd.DataFrame, course: str):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(df["date"], df["quality_avg"], linewidth=3, label="quality")
    ax.plot(df["date"], df["difficulty_avg"], linewidth=3, label="difficulty")

    ax.set_ylim(1, 5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Rating Score")
    ax.set_title(f"{course} Ratings Over Time (Smoothed)")
    ax.legend()

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=0, labelsize=9)

    fig.tight_layout()
    return fig, ax


def plot_department_ratings(
    df: pd.DataFrame,
    department: str,
    department_col: str = "department",
):
    department_df = df[df[department_col] == department].copy()

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(department_df["date"], department_df["quality_avg"], linewidth=3, label="quality")
    ax.plot(
        department_df["date"],
        department_df["difficulty_avg"],
        linewidth=3,
        label="difficulty",
    )

    ax.set_ylim(1, 5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Rating Score")
    ax.set_title(f"{department} Ratings Over Time (Smoothed)")
    ax.legend()

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=0, labelsize=9)

    fig.tight_layout()
    return fig, ax


def load_releases(csv_path) -> pd.DataFrame:
    releases = pd.read_csv(csv_path)
    releases["Time"] = pd.to_datetime(releases["Time"], format="mixed")
    return releases


def overlay_releases(ax, releases_df: pd.DataFrame) -> None:
    for _, row in releases_df.iterrows():
        ax.axvline(
            row["Time"],
            linestyle="--",
            linewidth=1.2,
            alpha=0.6,
            color="black",
        )

    y_top = ax.get_ylim()[1]
    for _, row in releases_df.iterrows():
        ax.text(
            row["Time"],
            y_top * 0.98,
            str(row["Model"]),
            rotation=90,
            va="top",
            ha="center",
            fontsize=10,
            alpha=0.8,
            color="red",
            fontweight="bold",
        )


def build_overall_ratings(df: pd.DataFrame) -> pd.DataFrame:
    overall_ratings = (
        df.dropna(subset=["date", "quality", "difficulty"])
        .groupby("date")[["quality", "difficulty"]]
        .mean()
        .reset_index()
        .sort_values("date")
    )
    return overall_ratings


def smooth_overall_ratings(df: pd.DataFrame, span: int = 15) -> pd.DataFrame:
    overall_df = df.sort_values("date").copy()
    overall_df["quality_avg"] = overall_df["quality"].ewm(span=span, adjust=False).mean()
    overall_df["difficulty_avg"] = overall_df["difficulty"].ewm(span=span, adjust=False).mean()
    return overall_df


def _format_release_period_label(
    current_model: str | None,
    next_model: str | None,
) -> str:
    if current_model is None and next_model is None:
        return "All Dates"
    if current_model is None:
        return f"Pre-{next_model}"
    if next_model is None:
        return f"{current_model}+"
    return f"{current_model} to {next_model}"


def assign_release_periods(
    df: pd.DataFrame,
    releases_df: pd.DataFrame,
    date_col: str = "date",
    period_col: str = "release_period",
) -> pd.DataFrame:
    if df.empty:
        df_with_periods = df.copy()
        df_with_periods[period_col] = pd.Series(dtype="object")
        df_with_periods["release_order"] = pd.Series(dtype="int64")
        return df_with_periods

    releases = releases_df.sort_values("Time").reset_index(drop=True).copy()
    boundaries = releases["Time"].tolist()
    models = releases["Model"].astype(str).tolist()

    bins = [pd.Timestamp.min, *boundaries, pd.Timestamp.max]
    labels = []
    for idx in range(len(bins) - 1):
        current_model = models[idx - 1] if idx > 0 else None
        next_model = models[idx] if idx < len(models) else None
        labels.append(_format_release_period_label(current_model, next_model))

    df_with_periods = df.copy()
    df_with_periods[period_col] = pd.cut(
        df_with_periods[date_col],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )
    order_map = {label: idx for idx, label in enumerate(labels)}
    df_with_periods["release_order"] = df_with_periods[period_col].map(order_map)
    return df_with_periods


def summarize_release_period_ratings(
    df: pd.DataFrame,
    releases_df: pd.DataFrame,
    date_col: str = "date",
    period_col: str = "release_period",
) -> pd.DataFrame:
    df_with_periods = assign_release_periods(
        df=df,
        releases_df=releases_df,
        date_col=date_col,
        period_col=period_col,
    )

    period_summary = (
        df_with_periods.dropna(subset=[period_col, "quality", "difficulty"])
        .groupby(period_col, observed=True)
        .agg(
            release_order=("release_order", "first"),
            quality_mean=("quality", "mean"),
            quality_count=("quality", "count"),
            difficulty_mean=("difficulty", "mean"),
            difficulty_count=("difficulty", "count"),
        )
        .reset_index()
        .sort_values("release_order")
    )

    if period_summary.empty:
        return pd.DataFrame(
            columns=[
                period_col,
                "release_order",
                "quality_mean",
                "quality_count",
                "difficulty_mean",
                "difficulty_count",
                "review_count",
            ]
        )

    period_summary["review_count"] = period_summary[
        ["quality_count", "difficulty_count"]
    ].min(axis=1)
    return period_summary.reset_index(drop=True)


def classify_department_release_behavior(
    department_period_summary: pd.DataFrame,
    stable_threshold: float = 0.15,
) -> str | None:
    if len(department_period_summary) < 2:
        return None

    ordered = department_period_summary.sort_values("release_order").reset_index(drop=True)
    quality_delta = (
        ordered.iloc[-1]["quality_mean"] - ordered.iloc[0]["quality_mean"]
    )
    difficulty_delta = (
        ordered.iloc[-1]["difficulty_mean"] - ordered.iloc[0]["difficulty_mean"]
    )

    if (
        abs(quality_delta) <= stable_threshold
        and abs(difficulty_delta) <= stable_threshold
    ):
        return "stable"
    if quality_delta > stable_threshold and difficulty_delta < -stable_threshold:
        return "improving"
    if quality_delta < -stable_threshold and difficulty_delta > stable_threshold:
        return "inverting"
    return "stable"


def summarize_department_release_behaviors(
    df: pd.DataFrame,
    releases_df: pd.DataFrame,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    department_col: str = "department",
    stable_threshold: float = 0.15,
) -> pd.DataFrame:
    review_df = df.dropna(subset=[department_col, "date", "quality", "difficulty"]).copy()

    if start_date is not None:
        review_df = review_df[review_df["date"] >= start_date]
    if end_date is not None:
        review_df = review_df[review_df["date"] <= end_date]

    behavior_rows = []
    for department, department_reviews in review_df.groupby(department_col):
        period_summary = summarize_release_period_ratings(
            department_reviews,
            releases_df=releases_df,
        )
        behavior = classify_department_release_behavior(
            period_summary,
            stable_threshold=stable_threshold,
        )
        if behavior is None:
            continue

        ordered = period_summary.sort_values("release_order").reset_index(drop=True)
        behavior_rows.append(
            {
                department_col: department,
                "release_period_count": len(ordered),
                "review_count": int(ordered["review_count"].sum()),
                "first_release_period": ordered.iloc[0]["release_period"],
                "last_release_period": ordered.iloc[-1]["release_period"],
                "quality_delta": ordered.iloc[-1]["quality_mean"]
                - ordered.iloc[0]["quality_mean"],
                "difficulty_delta": ordered.iloc[-1]["difficulty_mean"]
                - ordered.iloc[0]["difficulty_mean"],
                "behavior": behavior,
            }
        )

    if not behavior_rows:
        return pd.DataFrame(
            columns=[
                department_col,
                "release_period_count",
                "review_count",
                "first_release_period",
                "last_release_period",
                "quality_delta",
                "difficulty_delta",
                "behavior",
            ]
        )

    return pd.DataFrame(behavior_rows).sort_values(
        ["behavior", "review_count", department_col],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def count_department_release_behaviors(
    department_behavior_summary: pd.DataFrame,
) -> pd.DataFrame:
    behavior_order = ["inverting", "stable", "improving"]
    counts = (
        department_behavior_summary["behavior"]
        .value_counts()
        .reindex(behavior_order, fill_value=0)
        .rename_axis("behavior")
        .reset_index(name="department_count")
    )
    return counts


def build_release_period_windows(
    releases_df: pd.DataFrame,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    period_col: str = "release_period",
) -> pd.DataFrame:
    releases = releases_df.sort_values("Time").reset_index(drop=True).copy()
    models = releases["Model"].astype(str).tolist()
    boundaries = releases["Time"].tolist()

    if start_date is None:
        start_date = boundaries[0]
    if end_date is None:
        end_date = boundaries[-1]

    windows = []
    current_start = start_date

    for idx, boundary in enumerate(boundaries):
        if current_start < boundary:
            windows.append(
                {
                    period_col: _format_release_period_label(
                        models[idx - 1] if idx > 0 else None,
                        models[idx],
                    ),
                    "release_order": idx,
                    "start": current_start,
                    "end": min(boundary, end_date),
                }
            )
        current_start = max(current_start, boundary)
        if current_start >= end_date:
            break

    if current_start < end_date:
        windows.append(
            {
                period_col: _format_release_period_label(models[-1], None),
                "release_order": len(models),
                "start": current_start,
                "end": end_date,
            }
        )

    return pd.DataFrame(windows)


def plot_release_period_ratings(
    period_summary: pd.DataFrame,
    releases_df: pd.DataFrame,
    title: str,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    period_col: str = "release_period",
):
    fig, ax = plt.subplots(figsize=(12, 5))

    if period_summary.empty:
        ax.set_title(title)
        ax.set_ylim(1, 5)
        ax.set_xlabel("Date")
        ax.set_ylabel("Average Rating")
        fig.tight_layout()
        return fig, ax

    windows = build_release_period_windows(
        releases_df=releases_df,
        start_date=start_date,
        end_date=end_date,
        period_col=period_col,
    )
    plot_df = period_summary.merge(
        windows,
        on=[period_col, "release_order"],
        how="left",
    ).dropna(subset=["start", "end"])

    quality_labeled = False
    difficulty_labeled = False
    for _, row in plot_df.iterrows():
        ax.hlines(
            y=row["quality_mean"],
            xmin=row["start"],
            xmax=row["end"],
            colors="tab:blue",
            linewidth=3,
            label="quality" if not quality_labeled else None,
        )
        ax.hlines(
            y=row["difficulty_mean"],
            xmin=row["start"],
            xmax=row["end"],
            colors="tab:orange",
            linewidth=3,
            label="difficulty" if not difficulty_labeled else None,
        )
        quality_labeled = True
        difficulty_labeled = True

    ax.set_ylim(1, 5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Rating")
    ax.set_title(title)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=0, labelsize=9)
    ax.legend()

    fig.tight_layout()
    return fig, ax


def plot_overall_ratings(df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(df["date"], df["quality_avg"], linewidth=3, label="quality")
    ax.plot(df["date"], df["difficulty_avg"], linewidth=3, label="difficulty")

    ax.set_ylim(1, 5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Rating Score")
    ax.set_title(title)
    ax.legend()

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=0, labelsize=9)

    fig.tight_layout()
    return fig, ax
