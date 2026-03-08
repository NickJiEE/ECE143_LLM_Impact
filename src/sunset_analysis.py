from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


MERGED_GROUPS = {
    "Computing & EECS": {"CSE", "DSC", "ECE"},
    "Engineering": {"MAE", "BENG", "NANO"},
    "Physical Sciences": {"CHEM", "PHYS", "SIO"},
    "Life Sciences": {"BILD", "BICD", "BIEB", "BIMM", "BIBC", "BIPN"},
    "Math & Logic": {"MATH", "LIGN"},
    "Econ & Management": {"ECON", "MGT"},
    "Social Sciences": {"POLI", "PSYC", "SOCI", "USP", "COGS", "HDS"},
    "Writing & Humanities": {"CAT", "DOC", "WCWP", "MCWP", "HUM", "MMW", "HILD", "ETHN", "PHIL"},
    "Arts & Media": {"MUS", "VIS", "TDGE", "TDMV"},
    "Languages": {"JAPN", "CHIN", "LTEN", "LIGM", "LIFR", "LISP", "LTWL", "LTKO", "LTSP", "LTEA", "LTWR"},
}
DEPT_TO_GROUP = {dept: group for group, depts in MERGED_GROUPS.items() for dept in depts}

LLM_MARKERS = {
    "GPT": pd.Timestamp("2022-11-30"),
    "Claude": pd.Timestamp("2023-03-14"),
    "Gemini": pd.Timestamp("2023-12-06"),
    "Grok": pd.Timestamp("2023-11-04"),
}

COVID_START = pd.Timestamp("2020-03-01")
COVID_REMOTE_END = pd.Timestamp("2021-09-01")

GRADE_POINTS = {
    "A+": 4.0,
    "A": 4.0,
    "A-": 3.7,
    "B+": 3.3,
    "B": 3.0,
    "B-": 2.7,
    "C+": 2.3,
    "C": 2.0,
    "C-": 1.7,
    "D+": 1.3,
    "D": 1.0,
    "D-": 0.7,
    "F": 0.0,
}

BAND_TO_GRADES = {
    "A": ["A+", "A", "A-"],
    "B": ["B+", "B", "B-"],
    "C": ["C+", "C", "C-"],
    "D/F": ["D", "F"],
}

MIN_QUARTERS = 12
MAX_GROUPS = 10
QUALITY_RATIO_RANGE = (0.95, 1.05)


def load_sunset_data(csv_path: str | Path) -> pd.DataFrame:
    """Load the raw Sunset CSV."""
    return pd.read_csv(csv_path)


def preview_sunset_frame(df: pd.DataFrame, rows: int = 5) -> pd.DataFrame:
    """
    Return a notebook-safe preview without professor or user-identifying columns.
    """
    preview_cols = [col for col in ["Term", "Course", "Grade distribution"] if col in df.columns]
    return df.loc[:, preview_cols].head(rows)


def extract_department(course: str | float) -> str | None:
    """Extract the leading department code from a Sunset course string."""
    if pd.isna(course):
        return None
    match = re.match(r"^\s*([A-Za-z&]+)", str(course))
    return match.group(1).upper() if match else None


def parse_grade_distribution(raw: str | float) -> dict[str, int]:
    """
    Parse a Sunset grade distribution string into a grade-count mapping.

    Repeated keys are summed because the source data occasionally duplicates entries.
    """
    if pd.isna(raw):
        return {}

    text = str(raw).strip()
    lowered = text.lower()
    if not text or "not available" in lowered or "temporarily unavailable" in lowered:
        return {}

    parsed: dict[str, int] = {}
    for piece in text.split(","):
        if ":" not in piece:
            continue
        key, value = piece.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "Class GPA":
            continue
        try:
            parsed[key] = parsed.get(key, 0) + int(value)
        except ValueError:
            continue

    return parsed


def calculate_gpa(grade_dist: dict[str, int]) -> float | None:
    """Compute GPA from a parsed Sunset grade distribution."""
    total_points = 0.0
    total_count = 0
    for grade, count in grade_dist.items():
        if grade in GRADE_POINTS:
            total_points += GRADE_POINTS[grade] * count
            total_count += count
    return total_points / total_count if total_count else None


def clean_sunset_analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the Sunset dataframe used by the descriptive analysis notebook."""
    work = df.copy()
    work.columns = work.columns.str.strip()
    work["grade_dist"] = work["Grade distribution"].apply(parse_grade_distribution)
    work["GPA"] = work["grade_dist"].apply(calculate_gpa)

    grade_cols = pd.DataFrame(work["grade_dist"].tolist(), index=work.index).fillna(0)
    for grade in GRADE_POINTS:
        if grade not in grade_cols.columns:
            grade_cols[grade] = 0
    work = pd.concat([work, grade_cols], axis=1)
    work["Department"] = work["Course"].str.extract(r"^([A-Z]+)")
    work["Term_clean"] = work["Term"].str.replace(" Qtr", "", regex=False).str.strip()

    return work.dropna(subset=["GPA", "Department"])


def term_to_date(term: str | float) -> pd.Timestamp:
    """Convert quarter-like Sunset term strings into sortable timestamps."""
    match = re.match(r"^(Winter|Spring|Fall)\s+Qtr\s+(\d{4})$", str(term))
    if not match:
        return pd.NaT

    quarter_to_month = {"Winter": 1, "Spring": 4, "Fall": 9}
    quarter, year = match.group(1), int(match.group(2))
    return pd.Timestamp(year=year, month=quarter_to_month[quarter], day=1)


def prepare_group_term_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Build the merged-group term-level frame used by the Sunset trend notebook.

    Returns:
    ``(group_term_frame, coverage_table, selected_groups)``.
    """
    work = df[["Term", "Course", "Professor", "Grade distribution"]].copy()
    work["Department"] = work["Course"].apply(extract_department)
    work["Group"] = work["Department"].map(DEPT_TO_GROUP)
    work = work.dropna(subset=["Grade distribution", "Group"])
    work = work.drop_duplicates(subset=["Term", "Course", "Professor", "Grade distribution"])
    work["Term Date"] = work["Term"].apply(term_to_date)
    work = work.dropna(subset=["Term Date"])

    parsed = work["Grade distribution"].apply(parse_grade_distribution)
    parsed = parsed[parsed.map(bool)]
    work = work.loc[parsed.index].copy()

    parsed_df = parsed.apply(pd.Series).fillna(0)
    parsed_df = parsed_df.apply(pd.to_numeric, errors="coerce").fillna(0)

    total_students = parsed_df.get("Total Students", pd.Series(0, index=parsed_df.index))
    observed_outcomes = parsed_df.drop(columns=["Total Students"], errors="ignore").sum(axis=1)
    quality_ratio = observed_outcomes.div(total_students.where(total_students > 0))

    min_ratio, max_ratio = QUALITY_RATIO_RANGE
    valid = total_students.gt(0) & quality_ratio.between(min_ratio, max_ratio)
    parsed_df = parsed_df.loc[valid].copy()
    work = work.loc[valid].copy()

    band_counts = pd.DataFrame(index=parsed_df.index)
    for band, grades in BAND_TO_GRADES.items():
        band_counts[band] = parsed_df.reindex(columns=grades, fill_value=0).sum(axis=1)

    records = work[["Group", "Term", "Term Date"]].join(band_counts)
    records["Class Count"] = 1
    records["Letter Total"] = records[list(BAND_TO_GRADES.keys())].sum(axis=1)
    records = records[records["Letter Total"] > 0].copy()

    group_term = (
        records.groupby(["Group", "Term", "Term Date"], as_index=False)[
            ["A", "B", "C", "D/F", "Class Count", "Letter Total"]
        ].sum()
    )
    group_term = group_term.sort_values(["Group", "Term Date", "Term"]).reset_index(drop=True)

    for band in BAND_TO_GRADES:
        group_term[f"{band} %"] = (group_term[band] / group_term["Letter Total"]) * 100

    coverage = (
        group_term.groupby("Group", as_index=False)
        .agg(
            Quarter_Count=("Term", "nunique"),
            First_Term=("Term", "first"),
            Last_Term=("Term", "last"),
            Class_Count=("Class Count", "sum"),
            Letter_Outcomes=("Letter Total", "sum"),
        )
        .sort_values(["Quarter_Count", "Class_Count"], ascending=[False, False])
        .reset_index(drop=True)
    )

    eligible = coverage[coverage["Quarter_Count"] >= MIN_QUARTERS].head(MAX_GROUPS)
    selected_groups = eligible["Group"].tolist()
    filtered = group_term[group_term["Group"].isin(selected_groups)].copy()
    filtered = filtered.sort_values(["Group", "Term Date", "Term"]).reset_index(drop=True)

    return filtered, coverage, selected_groups


def load_provider_releases(provider: str, csv_path: str | Path) -> pd.DataFrame:
    """Load a provider release CSV into a normalized release dataframe."""
    updates = pd.read_csv(csv_path)
    if "Time" not in updates.columns or "Model" not in updates.columns:
        return pd.DataFrame(columns=["Provider", "Model", "Release Date"])

    releases = updates[["Model", "Time"]].copy()
    releases["Release Date"] = pd.to_datetime(releases["Time"], errors="coerce", format="mixed")
    releases = releases.dropna(subset=["Release Date"]).sort_values("Release Date").reset_index(drop=True)
    releases["Provider"] = provider
    return releases[["Provider", "Model", "Release Date"]]
