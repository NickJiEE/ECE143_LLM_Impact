from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


QUARTER_ORDER = {"WI": 1, "SP": 2, "SU": 3, "FA": 4}
SEASON_NAME = {"WI": "Winter", "SP": "Spring", "SU": "Summer", "FA": "Fall"}
PROVIDER_RELEASE_FILES = {
    "GPT": "chatgpt_model_updates.csv",
    "Claude": "claude_model_updates.csv",
    "Gemini": "gemini_model_updates.csv",
    "Grok": "grok_model_updates.csv",
}


def extract_department(course: str | float) -> str | None:
    """Extract the department code from a CAPES course string."""
    if pd.isna(course):
        return None

    match = re.match(r"^([A-Z]+(?:\s[A-Z]+)?)\s+\d", str(course))
    return match.group(1) if match else None


def extract_course_number(course: str | float) -> float | None:
    """Extract the numeric course number from a CAPES course string."""
    if pd.isna(course):
        return None

    match = re.search(r"[A-Z]+(?:\s[A-Z]+)?\s+(\d+)", str(course))
    return float(match.group(1)) if match else None


def classify_division(course_number: float | int | None) -> str:
    """Bucket a course number into lower, upper, or graduate division."""
    if pd.isna(course_number):
        return "Unknown"
    if course_number < 100:
        return "Lower Division (1-99)"
    if course_number < 200:
        return "Upper Division (100-199)"
    return "Graduate (200+)"


def parse_quarter_num(quarter: str | float) -> float | None:
    """
    Convert a quarter code like ``FA23`` to a sortable numeric value.

    Example:
    ``FA23`` -> ``2024.0`` and ``WI23`` -> ``2023.25``.
    """
    match = re.match(r"([A-Z]+)(\d{2})$", str(quarter))
    if not match:
        return None

    season, year_suffix = match.groups()
    year = 2000 + int(year_suffix)
    return year + QUARTER_ORDER.get(season, 0) / 4


def add_capes_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add parsed GPA, department, division, and quarter features to CAPES data."""
    work = df.copy()

    work["Pct_Rec_Class"] = (
        work["Percentage Recommended Class"].str.replace("%", "", regex=False).astype(float)
    )
    work["Pct_Rec_Prof"] = (
        work["Percentage Recommended Professor"].str.replace("%", "", regex=False).astype(float)
    )
    work["GPA_Expected"] = (
        work["Average Grade Expected"].str.extract(r"\(([\d.]+)\)").astype(float)
    )
    work["GPA_Received"] = (
        work["Average Grade Received"].str.extract(r"\(([\d.]+)\)").astype(float)
    )
    work["Department"] = work["Course"].apply(extract_department)
    work["Course_Number"] = work["Course"].apply(extract_course_number)
    work["Division"] = work["Course_Number"].apply(classify_division)
    work["Season"] = work["Quarter"].str.extract(r"^([A-Z]+)")
    work["Season_Name"] = work["Season"].map(SEASON_NAME)
    work["Year"] = work["Quarter"].str.extract(r"(\d{2})$").astype(float) + 2000
    work["Quarter_Num"] = work["Quarter"].apply(parse_quarter_num)

    return work


def clean_capes_analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a CAPES analysis frame with the shared derived columns used by notebooks."""
    work = add_capes_features(df)
    return work.dropna(subset=["GPA_Received", "Quarter_Num"])


def load_capes_data(csv_path: str | Path) -> pd.DataFrame:
    """Load the raw CAPES CSV."""
    return pd.read_csv(csv_path)


def load_provider_release_frames(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load all provider release CSV files from the project data directory."""
    data_dir = Path(data_dir)
    frames: dict[str, pd.DataFrame] = {}
    for provider, filename in PROVIDER_RELEASE_FILES.items():
        frames[provider] = pd.read_csv(data_dir / filename)
    return frames


def provider_frames_to_quarter_events(
    frames: dict[str, pd.DataFrame],
    min_year: float = 2020.0,
    max_year: float = 2024.0,
) -> list[tuple[float, str, str]]:
    """
    Convert release-data frames into quarter-scale event tuples.

    Returns tuples of ``(quarter_num_like_value, model_name, provider_name)``.
    """
    events: list[tuple[float, str, str]] = []

    for provider, frame in frames.items():
        dates = pd.to_datetime(frame.iloc[:, 0], errors="coerce", format="mixed")
        models = frame.iloc[:, 1].astype(str)
        provider_events = zip(dates.dt.year + dates.dt.month / 12, models)
        events.extend(
            (quarter_num, model, provider)
            for quarter_num, model in provider_events
            if min_year <= quarter_num <= max_year
        )

    return sorted(events, key=lambda item: item[0])
