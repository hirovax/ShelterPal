
import pandas as pd
import numpy as np
from map import map_sex, age_group, status


def date_window(df, months=3, days=7):
    df = df.copy()
    intakes = pd.to_datetime(df["intake_datetime"]).sort_values().values
    outcomes = pd.to_datetime(df["outcome_datetime"]).sort_values().values

    t = pd.to_datetime(df["intake_datetime"]).values
    start = t - np.timedelta64(months * 30, 'D')
    intakes_count = np.searchsorted(intakes, t, side="left") - np.searchsorted(intakes, start, side="left")
    outcomes_count = np.searchsorted(outcomes, t, side="left") - np.searchsorted(outcomes, start, side="left")

    start_d = t - np.timedelta64(days, 'D')
    intakes_count_d = np.searchsorted(intakes, t, side="left") - np.searchsorted(intakes, start_d, side="left")
    outcomes_count_d = np.searchsorted(outcomes, t, side="left") - np.searchsorted(outcomes, start_d, side="left")

    df[f"balance_{months}m"] = intakes_count - outcomes_count
    df[f"balance_{days}d"] = intakes_count_d - outcomes_count_d

    return df


def group_categories_ratio(df, min_ratio=0.01, other_label="Other"):
    freq = df.value_counts(normalize=True)
    keep = freq[freq >= min_ratio].index
    return df.where(df.isin(keep), other_label)


def load_and_clean_data():
    # Load dataset
    df = pd.read_csv("data/austin/raw/aac_intakes_outcomes.csv")

    # Remove duplicate records
    df = df.drop_duplicates()

    # Drop rows with missing values
    df = df.dropna(subset=["outcome_type", "sex_upon_intake", "sex_upon_outcome"])
    df = df.reset_index(drop=True)

    return df


def feature_engineering(df):
    # Coat color
    df["is_mixed_color"] = df["color"].str.contains("/|Tricolor", na=False).astype(int)

    # Outcome type
    df["outcome_type"] = df["outcome_type"].replace({
        "Euthanasia": "Death",
        "Died": "Death",
        "Disposal": "Death",
        "Rto-Adopt": "Adoption",
        "Missing": "Transfer",
        "Relocate": "Transfer",
    })

    # Animal type
    df["animal_type"] = group_categories_ratio(df["animal_type"], min_ratio=0.01)

    df = df[df["animal_type"] != "Bat"]

    # Sex and neuter/spay status
    df["sex"] = df["sex_upon_intake"].apply(map_sex)
    df["intact"] = df["sex_upon_intake"].apply(status)

    df = df[df["sex"] != "Unknown"]
    df = df[df["intact"] != "Unknown"]

    # Animal breed
    df["is_mixed_breed"] = df["breed"].str.contains("/|Mix", na=False).astype(int)
    df["primary_breed"] = df["breed"].str.split("/|Mix").str[0].str.strip()

    df["primary_breed_top"] = group_categories_ratio(df["primary_breed"], min_ratio=0.02)

    # Intake condition
    df["intake_condition"] = group_categories_ratio(df["intake_condition"], min_ratio=0.01)

    # Age group
    df["age_group"] = df.apply(age_group, axis=1)

    # Previous shelter admissions
    df["is_repeat"] = (df["intake_number"] > 1).astype(int)

    # Feature interactions
    df["young_and_dog"] = (
        (df["animal_type"].str.contains("Dog", na=False)) &
        (df["age_group"].str.contains("Young", na=False))
    ).astype(int)

    df["young_and_cat"] = (
        (df["animal_type"].str.contains("Cat", na=False)) &
        (df["age_group"].str.contains("Young", na=False))
    ).astype(int)

    df["stray_and_dog"] = (
        (df["intake_type"] == "Stray") & (df["animal_type"] == "Dog")
    ).astype(int)

    df["stray_and_intact"] = (
        (df["intake_type"] == "Stray") & (df["intact"] == "Intact")
    ).astype(int)

    df["young_and_intact"] = (
        (df["age_group"].str.contains("Young", na=False)) & (df["intact"] == "Intact")
    ).astype(int)

    # Date-related features
    df["intake_datetime"] = pd.to_datetime(df["intake_datetime"])
    df["day_of_week"] = df["intake_datetime"].dt.day_name()
    df["month"] = df["intake_datetime"].dt.month
    df["year"] = df["intake_datetime"].dt.year
    df["is_weekend"] = df["intake_datetime"].dt.dayofweek.isin([5, 6]).astype(int)

    df = date_window(df, months=3, days=30)

    df = df[(df["intake_type"] != "Euthanasia Request") & (df["intake_type"] != "Wildlife")]

    # Drop unnecessary columns
    df = df.drop(columns=[
        "date_of_birth", "dob_year", "age_upon_outcome_(years)",
        "age_upon_outcome_(days)", "age_upon_outcome", "animal_id_outcome",
        "outcome_weekday", "outcome_hour", "outcome_number", "dob_month",
        "dob_monthyear", "age_upon_intake", "count", "found_location",
        "sex_upon_intake", "breed", "primary_breed", "intake_monthyear",
        "outcome_datetime", "outcome_monthyear", "age_upon_outcome_age_group",
        "outcome_month", "outcome_year", "intake_month", "intake_hour",
        "intake_year", "animal_id_intake", "color", "intake_weekday"
    ])

    df = df.reset_index(drop=True)

    return df
