import pytest
import pandas as pd

from src.sentiment import add_department_codes, parse_reviews, summarize_department_ratings

def test_compile():
    print('Basic compile test')
    parse_reviews()
    pass


def test_department_summary():
    df = pd.DataFrame(
        {
            "class": ["ECE35", "ECE45", "CSE100"],
            "quality": [4.0, 2.0, 5.0],
            "difficulty": [3.0, 5.0, 2.0],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        }
    )

    df = add_department_codes(df)
    summary = summarize_department_ratings(df)

    ece_row = summary.loc[summary["department"] == "ECE"].iloc[0]
    assert ece_row["quality_mean"] == pytest.approx(3.0)
    assert ece_row["difficulty_mean"] == pytest.approx(4.0)
    assert ece_row["review_count"] == 2

