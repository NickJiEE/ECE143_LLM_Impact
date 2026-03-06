import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
import os

def clean_dataframe(df):
    df = df.copy()

    # Remove % and convert to float
    df["Percentage Recommended Class"] = (
        df["Percentage Recommended Class"]
        .str.replace("%", "", regex=False)
        .astype(float)
    )

    df["Percentage Recommended Professor"] = (
        df["Percentage Recommended Professor"]
        .str.replace("%", "", regex=False)
        .astype(float)
    )

    # Extract GPA numbers inside parentheses
    df["Average Grade Expected (GPA)"] = (
        df["Average Grade Expected"]
        .str.extract(r"\((.*?)\)")
        .astype(float)
    )

    df["Average Grade Received (GPA)"] = (
        df["Average Grade Received"]
        .str.extract(r"\((.*?)\)")
        .astype(float)
    )

    return df

def plot_grade_vs_recommend(df):
    plt.figure(figsize=(8, 6))

    plt.hexbin(
        df["Average Grade Received (GPA)"],
        df["Percentage Recommended Professor"],
        gridsize=40,
        cmap="viridis"
    )

    plt.colorbar(label="Count")
    plt.xlabel("Average Grade Received (GPA)")
    plt.ylabel("Recommended Professor (%)")
    plt.title("Grade vs Professor Recommendation Density")

    plt.tight_layout()
    plt.show()

def plot_by_course(df):
    top_courses = (
        df["Course"]
        .value_counts()
        .head(10)
        .index
    )

    subset = df[df["Course"].isin(top_courses)]

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=subset,
        x="Course",
        y="Percentage Recommended Professor"
    )

    plt.xticks(rotation=45)
    plt.title("Top 10 Courses - Professor Recommendation Distribution")
    plt.tight_layout()
    plt.show()

def plot_study_hours_distribution(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 6))

    sns.histplot(
        df["Study Hours per Week"],
        bins=30,
        kde=True
    )

    plt.xlabel("Study Hours per Week")
    plt.ylabel("Count")
    plt.title("Distribution of Study Hours")

    plt.tight_layout()
    plt.show()

def plot_study_hours_vs_gpa(df):
    plt.figure(figsize=(8, 6))

    plt.hexbin(
        df["Study Hours per Week"],
        df["Average Grade Received (GPA)"],
        gridsize=40,
        cmap="viridis"
    )

    plt.colorbar(label="Count")
    plt.xlabel("Study Hours Per Week")
    plt.ylabel("GPA")
    plt.title("Study Hours vs GPA Density")

    plt.tight_layout()
    plt.show()

# LLM TIMELINE HELPERS

# Maps quarter prefix → (month, day) for approximate mid-quarter date
_SEASON_MAP = {
    "FA": (9, 1),   # Fall   — starts ~September
    "WI": (1, 1),   # Winter — starts ~January
    "SP": (4, 1),   # Spring — starts ~April
    "S1": (6, 1),   # Summer Session 1
    "S2": (7, 1),   # Summer Session 2
    "S3": (8, 1),   # Summer Session 3
    "SU": (6, 15),  # General summer
}

# Only the milestone models that fall within the dataset window (2007–SP23).
# Feel free to extend this list; dates beyond the data range are silently ignored.
_LLM_MILESTONES = {
    "ChatGPT": {
        "color": "#10a37f",
        "models": [
            ("2018-06-01", "GPT-1"),
            ("2019-02-01", "GPT-2"),
            ("2020-06-01", "GPT-3"),
            ("2022-11-30", "GPT-3.5\n(ChatGPT)"),
            ("2023-03-14", "GPT-4"),
        ],
    },
    "Claude": {
        "color": "#d97706",
        "models": [
            ("2023-03-01", "Claude 1"),
        ],
    },
}

# Key historical events to annotate as shaded bands
_HISTORICAL_EVENTS = [
    {
        "label": "COVID-19\nPandemic\n(remote)",
        "start": "2020-03-01",
        "end":   "2021-09-01",
        "color": "#ef4444",
        "alpha": 0.08,
    },
]


def _quarter_to_date(quarter: str) -> pd.Timestamp:
    """Convert a UCSD quarter code (e.g. 'FA22', 'SP23') to a Timestamp."""
    prefix = quarter[:2]
    year = int(quarter[2:]) + 2000
    month, day = _SEASON_MAP.get(prefix, (6, 1))
    return pd.Timestamp(year=year, month=month, day=day)


def _prepare_timeline_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the raw CAPES dataframe to one row per quarter with:
      - mean GPA received / expected
      - mean % recommending professor
      - rolling 3-quarter smoothed GPA
    """
    work = df.copy()
    work["Date"] = work["Quarter"].apply(_quarter_to_date)
    work["GPA_Received"] = (
        work["Average Grade Received"].str.extract(r"\((.*?)\)").astype(float)
    )
    work["GPA_Expected"] = (
        work["Average Grade Expected"].str.extract(r"\((.*?)\)").astype(float)
    )
    work["Pct_Rec_Prof"] = (
        work["Percentage Recommended Professor"]
        .str.replace("%", "", regex=False)
        .astype(float)
    )

    agg = (
        work.groupby("Date")
        .agg(
            GPA_Received=("GPA_Received", "mean"),
            GPA_Expected=("GPA_Expected", "mean"),
            Pct_Rec_Prof=("Pct_Rec_Prof", "mean"),
            n=("GPA_Received", "count"),
        )
        .reset_index()
        .sort_values("Date")
    )
    agg["GPA_Smoothed"] = agg["GPA_Received"].rolling(3, center=True, min_periods=1).mean()
    return agg


def _draw_llm_lines(ax, data_end: pd.Timestamp, label_offset_pts: int = 8):
    """
    Draw vertical dashed lines + labels for LLM milestones that fall within
    the data range.  Returns legend handles for the company colours.
    """
    legend_handles = {}
    ymin, ymax = ax.get_ylim()
    label_y = ymax - (ymax - ymin) * 0.03  # just below the top edge

    for company, info in _LLM_MILESTONES.items():
        color = info["color"]
        for date_str, model_name in info["models"]:
            dt = pd.Timestamp(date_str)
            if dt > data_end:
                continue
            ax.axvline(dt, color=color, linewidth=1.1, linestyle="--", alpha=0.75, zorder=3)
            ax.text(
                dt, label_y, model_name,
                rotation=90, fontsize=7, color=color,
                va="top", ha="right",
                fontweight="semibold",
            )
            if company not in legend_handles:
                legend_handles[company] = Line2D(
                    [], [], color=color, linewidth=1.5, linestyle="--", label=company
                )
    return list(legend_handles.values())


def _draw_historical_bands(ax):
    """Shade background bands for key historical events and return legend handles."""
    handles = []
    for event in _HISTORICAL_EVENTS:
        start = pd.Timestamp(event["start"])
        end   = pd.Timestamp(event["end"])
        ax.axvspan(start, end, color=event["color"], alpha=event["alpha"], zorder=1)
        handles.append(
            mpatches.Patch(
                facecolor=event["color"], alpha=event["alpha"] * 5,
                label=event["label"].replace("\n", " "),
            )
        )
    return handles


def plot_gpa_over_time(df: pd.DataFrame, figsize=(14, 5)):
    """
    Line chart of average GPA received (raw + smoothed) over every quarter,
    with vertical markers for key LLM releases and a COVID band.

    Parameters
    ----------
    df : raw CAPES dataframe (as loaded from capes_data.csv)
    figsize : tuple, default (14, 5)
    """
    agg = _prepare_timeline_df(df)

    fig, ax = plt.subplots(figsize=figsize)

    band_handles = _draw_historical_bands(ax)

    # Raw quarterly line
    ax.plot(
        agg["Date"], agg["GPA_Received"],
        color="#94a3b8", linewidth=0.9, alpha=0.6, label="Quarterly avg GPA",
    )
    # 3-quarter rolling average
    ax.plot(
        agg["Date"], agg["GPA_Smoothed"],
        color="#1e40af", linewidth=2.2, label="3-quarter rolling avg",
    )

    # LLM milestone lines
    llm_handles = _draw_llm_lines(ax, data_end=agg["Date"].max())

    # Formatting 
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.set_xlabel("Quarter")
    ax.set_ylabel("Avg GPA Received")
    ax.set_title("UCSD Average GPA Received Over Time\nwith LLM Release Milestones", fontsize=13)
    ax.set_ylim(bottom=ax.get_ylim()[0] - 0.02)

    # Combined legend
    data_handles = [
        Line2D([], [], color="#94a3b8", linewidth=0.9, alpha=0.8, label="Quarterly avg GPA"),
        Line2D([], [], color="#1e40af", linewidth=2.2, label="3-qtr rolling avg"),
    ]
    ax.legend(
        handles=data_handles + llm_handles + band_handles,
        loc="lower left", fontsize=8, framealpha=0.85,
    )

    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    plt.show()

# For each department
_DEPARTMENTS = {
    "Computer Science":  ["CSE"],
    "Mathematics":       ["MATH"],
    "Physics":           ["PHYS"],
    "Humanities":        ["HUM"],
    "Linguistics":       ["LIGN"],
    "Cognitive Science": ["COGS"],
    "Life Sciences":     ["BILD"],
    "Visual Arts":       ["VIS"],
    "Economics":         ["ECON"],
    "Engineering":       ["ECE", "MAE"],
}

# Distinct colours for each department
_DEPT_COLORS = {
    "Computer Science":  "#2563eb",
    "Mathematics":       "#7c3aed",
    "Physics":           "#0891b2",
    "Humanities":        "#b45309",
    "Linguistics":       "#059669",
    "Cognitive Science": "#d97706",
    "Life Sciences":     "#16a34a",
    "Visual Arts":       "#db2777",
    "Economics":         "#dc2626",
    "Engineering":       "#6d28d9",
}


def _prepare_dept_df(df: pd.DataFrame, prefixes: list) -> pd.DataFrame:
    """Filter to given course prefixes, parse dates & GPA, aggregate by quarter."""
    work = df.copy()
    work["Date"] = work["Quarter"].apply(_quarter_to_date)
    work["GPA_Received"] = (
        work["Average Grade Received"].str.extract(r"\((.*?)\)").astype(float)
    )
    work["dept_prefix"] = work["Course"].str.split(" ").str[0]
    sub = work[work["dept_prefix"].isin(prefixes)]
    agg = (
        sub.groupby("Date")["GPA_Received"]
        .mean()
        .reset_index()
        .sort_values("Date")
        .rename(columns={"GPA_Received": "GPA"})
    )
    agg["GPA_Smoothed"] = agg["GPA"].rolling(3, center=True, min_periods=1).mean()
    return agg


def plot_dept_gpa_over_time(df: pd.DataFrame, figsize=(16, 7)):
    """
    One subplot per department showing average GPA received over time,
    with LLM release markers and COVID band.  All 10 panels are arranged
    in a 2-row × 5-column grid so they can be compared side-by-side.

    Parameters
    ----------
    df      : raw CAPES dataframe (before clean_dataframe)
    figsize : overall figure size, default (16, 7)
    """
    ncols, nrows = 5, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=False, sharex=True)
    axes_flat = axes.flatten()

    dept_list = list(_DEPARTMENTS.items())

    for idx, (dept_name, prefixes) in enumerate(dept_list):
        ax = axes_flat[idx]
        color = _DEPT_COLORS[dept_name]
        agg = _prepare_dept_df(df, prefixes)
        data_end = agg["Date"].max()

        # Historical band
        for event in _HISTORICAL_EVENTS:
            ax.axvspan(
                pd.Timestamp(event["start"]), pd.Timestamp(event["end"]),
                color=event["color"], alpha=event["alpha"], zorder=1,
            )

        # LLM milestone lines
        for company, info in _LLM_MILESTONES.items():
            for date_str, model_name in info["models"]:
                dt = pd.Timestamp(date_str)
                if dt > data_end:
                    continue
                ax.axvline(dt, color=info["color"], linewidth=0.9,
                           linestyle="--", alpha=0.7, zorder=3)
                ymin, ymax = ax.get_ylim() or (0, 1)
                ax.text(
                    dt, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else agg["GPA_Smoothed"].max(),
                    model_name, rotation=90, fontsize=5.5,
                    color=info["color"], va="top", ha="right", fontweight="semibold",
                )

        ax.plot(agg["Date"], agg["GPA"],
                color=color, linewidth=0.7, alpha=0.35)
        ax.plot(agg["Date"], agg["GPA_Smoothed"],
                color=color, linewidth=2.0)

        prefix_str = "/".join(prefixes)
        ax.set_title(f"{dept_name}\n({prefix_str})", fontsize=8.5, fontweight="bold")
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("'%y"))
        ax.tick_params(axis="x", labelsize=7, rotation=45)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_ylabel("Avg GPA" if idx % ncols == 0 else "", fontsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_ylim(
            max(0, agg["GPA"].min() - 0.15),
            min(4.0, agg["GPA"].max() + 0.25),
        )

    legend_elements = [
        Line2D([], [], color="#94a3b8", linewidth=0.8, alpha=0.6, label="Quarterly avg GPA"),
        Line2D([], [], color="#374151", linewidth=2.0, label="3-qtr rolling avg"),
    ]
    for company, info in _LLM_MILESTONES.items():
        legend_elements.append(
            Line2D([], [], color=info["color"], linewidth=1.2,
                   linestyle="--", label=company)
        )
    legend_elements.append(
        mpatches.Patch(facecolor="#ef4444", alpha=0.4, label="COVID-19 (remote)")
    )

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=len(legend_elements),
        fontsize=7.5,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle(
        "Average GPA Received by Department Over Time\nwith LLM Release Milestones",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    plt.show()