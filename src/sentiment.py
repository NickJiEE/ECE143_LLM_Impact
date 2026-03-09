import re

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
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
