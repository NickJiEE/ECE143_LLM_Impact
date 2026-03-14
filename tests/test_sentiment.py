import shutil
import uuid
from pathlib import Path

import pandas as pd
import pytest

from src.sentiment import (
    add_department_codes,
    assign_release_periods,
    build_overall_ratings,
    classify_department_release_behavior,
    count_department_release_behaviors,
    filter_courses,
    find_project_root,
    load_prepared_reviews,
    normalize_dates,
    prepare_course_analysis,
    prepare_reviews,
    smooth_ratings,
    summarize_release_period_ratings,
)


def test_prepare_reviews_keeps_expected_columns_and_valid_courses():
    raw = pd.DataFrame(
        {
            "date": ["2024-01-01 UTC", "2024-01-02 UTC"],
            "class": ["ECE35", "not-a-course"],
            "qualityRating": [4.5, 1.0],
            "difficultyRatingRounded": [3.0, 2.0],
            "grade": ["A", "B"],
            "comment": ["good", "bad"],
            "extra": [1, 2],
        }
    )

    prepared = prepare_reviews(raw)

    assert list(prepared.columns) == [
        "date",
        "class",
        "quality",
        "difficulty",
        "grade",
        "comment",
    ]
    assert prepared["class"].tolist() == ["ECE35"]


def test_normalize_dates_removes_utc_suffix_and_timezone():
    df = pd.DataFrame({"date": ["2024-01-01 12:30:00 UTC"]})

    normalized = normalize_dates(df)

    assert str(normalized.loc[0, "date"]) == "2024-01-01 12:30:00"
    assert normalized.loc[0, "date"].tzinfo is None


def test_filter_courses_applies_course_and_date_filters():
    df = pd.DataFrame(
        {
            "class": ["ECE35", "ECE35", "CSE100"],
            "date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-01-15"]),
            "quality": [4.0, 3.5, 5.0],
            "difficulty": [3.0, 4.0, 2.0],
        }
    )

    filtered = filter_courses(
        df=df,
        course="ece35",
        start_date=pd.Timestamp("2024-01-15"),
        end_date=pd.Timestamp("2024-12-31"),
    )

    assert filtered["class"].tolist() == ["ECE35"]
    assert filtered["date"].tolist() == [pd.Timestamp("2024-02-01")]


def test_smooth_ratings_adds_expected_average_columns():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "quality": [2.0, 4.0],
            "difficulty": [5.0, 3.0],
        }
    )

    smoothed = smooth_ratings(df, span=2)

    assert "quality_avg" in smoothed.columns
    assert "difficulty_avg" in smoothed.columns
    assert smoothed.loc[0, "quality_avg"] == pytest.approx(2.0)
    assert smoothed.loc[1, "quality_avg"] > smoothed.loc[0, "quality_avg"]


def test_prepare_course_analysis_returns_raw_and_smoothed_frames():
    df = pd.DataFrame(
        {
            "class": ["ECE35", "ECE35", "CSE100"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-03"]),
            "quality": [3.0, 4.0, 5.0],
            "difficulty": [4.0, 3.0, 2.0],
        }
    )

    raw, smoothed = prepare_course_analysis(
        df=df,
        course="ECE35",
        start_date=pd.Timestamp("2024-01-01"),
        end_date=pd.Timestamp("2024-12-31"),
        span=2,
    )

    assert len(raw) == 2
    assert len(smoothed) == 2
    assert {"quality_avg", "difficulty_avg"}.issubset(smoothed.columns)


def test_assign_release_periods_labels_reviews_in_order():
    reviews = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-12-01", "2024-02-01", "2024-07-01"]),
            "quality": [3.0, 4.0, 5.0],
            "difficulty": [3.0, 2.5, 2.0],
        }
    )
    releases = pd.DataFrame(
        {
            "Time": pd.to_datetime(["2024-01-01", "2024-06-01"]),
            "Model": ["GPT-A", "GPT-B"],
        }
    )

    assigned = assign_release_periods(reviews, releases)

    assert assigned["release_period"].astype(str).tolist() == [
        "Pre-GPT-A",
        "GPT-A to GPT-B",
        "GPT-B+",
    ]
    assert assigned["release_order"].tolist() == [0, 1, 2]


def test_summarize_release_period_ratings_counts_and_means():
    reviews = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-12-01", "2024-02-01", "2024-03-01"]),
            "quality": [3.0, 4.0, 2.0],
            "difficulty": [3.0, 2.0, 4.0],
        }
    )
    releases = pd.DataFrame(
        {
            "Time": pd.to_datetime(["2024-01-01"]),
            "Model": ["GPT-A"],
        }
    )

    summary = summarize_release_period_ratings(reviews, releases)

    assert summary["release_period"].astype(str).tolist() == ["Pre-GPT-A", "GPT-A+"]
    assert summary.loc[0, "review_count"] == 1
    assert summary.loc[1, "review_count"] == 2
    assert summary.loc[1, "quality_mean"] == pytest.approx(3.0)


def test_classify_department_release_behavior_detects_improving():
    period_summary = pd.DataFrame(
        {
            "release_order": [0, 1],
            "quality_mean": [3.0, 4.0],
            "difficulty_mean": [4.0, 3.0],
        }
    )

    behavior = classify_department_release_behavior(period_summary, stable_threshold=0.15)

    assert behavior == "improving"


def test_count_department_release_behaviors_respects_output_order():
    summary = pd.DataFrame({"behavior": ["stable", "improving", "stable", "inverting"]})

    counts = count_department_release_behaviors(summary)

    assert counts["behavior"].tolist() == ["inverting", "stable", "improving"]
    assert counts["department_count"].tolist() == [1, 2, 1]


def test_build_overall_ratings_averages_by_day():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
            "quality": [3.0, 5.0, 4.0],
            "difficulty": [2.0, 4.0, 3.0],
        }
    )

    overall = build_overall_ratings(df)

    assert overall["date"].tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
    ]
    assert overall.loc[0, "quality"] == pytest.approx(4.0)
    assert overall.loc[0, "difficulty"] == pytest.approx(3.0)


def _make_workspace_temp_dir() -> Path:
    """Create a temp directory inside the repo workspace for tests."""
    temp_dir = Path("tests") / "_tmp" / str(uuid.uuid4())
    temp_dir.mkdir(parents=True, exist_ok=False)
    return temp_dir


def test_find_project_root_returns_matching_parent():
    temp_dir = _make_workspace_temp_dir()
    project_root = temp_dir / "repo"
    nested_dir = project_root / "notebooks" / "analysis"
    (project_root / "src").mkdir(parents=True)
    (project_root / "data").mkdir()
    nested_dir.mkdir(parents=True)

    try:
        found = find_project_root(nested_dir)
        assert found == project_root.resolve()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_load_prepared_reviews_reads_and_cleans_csv():
    temp_dir = _make_workspace_temp_dir()
    csv_path = temp_dir / "reviews.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-01 UTC", "2024-01-02 UTC"],
            "class": ["ECE35", "bad course"],
            "qualityRating": [4.0, 2.0],
            "difficultyRatingRounded": [3.0, 4.0],
            "grade": ["A", "B"],
            "comment": ["x", "y"],
        }
    ).to_csv(csv_path, index=False)

    try:
        prepared = load_prepared_reviews(csv_path)
        prepared = add_department_codes(normalize_dates(prepared))

        assert len(prepared) == 1
        assert prepared.loc[0, "department"] == "ECE"
        assert prepared.loc[0, "quality"] == pytest.approx(4.0)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
