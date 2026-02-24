import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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