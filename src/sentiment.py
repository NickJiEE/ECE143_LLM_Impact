import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math


def parse_reviews():
    print("hello world!")


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    df_clean["date"] = df_clean["date"].str.replace(r'\s+UTC$', '', regex=True)
    df_clean["date"] = pd.to_datetime(df_clean["date"], utc=True).dt.tz_localize(None)
    return df_clean

def filter_courses(df: pd.DataFrame, 
                   course: str,
                   start_date: pd.Timestamp,
                   end_date: pd.Timestamp) -> pd.DataFrame:
    df_filtered = df[
        (df["class"].str.upper() == course.upper()) &
        (df["date"] >= start_date) &
        (df["date"] <= end_date)
    ].sort_values("date").copy()
    return df_filtered

def smooth_ratings(input_df: pd.DataFrame) -> pd.DataFrame:
    df = input_df.copy()
    df["quality_avg"] = df["quality"].ewm(span=15, adjust=False).mean()
    df["difficulty_avg"] = df["difficulty"].ewm(span=15, adjust=False).mean()
    return df
    
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
