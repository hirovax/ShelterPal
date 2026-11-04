
import pandas as pd


def season(x):
    """
    Assigns a season based on the month number.
    """
    if pd.isnull(x):
        return "Unknown"

    if x in [12, 1, 2]:
        return "Winter"
    elif x in [3, 4, 5]:
        return "Spring"
    elif x in [6, 7, 8]:
        return "Summer"
    elif x in [9, 10, 11]:
        return "Autumn"

    return "Unknown"


def location(x):
    """
    Groups locations into the most frequently occurring areas.
    """
    if pd.isnull(x):
        return "Unknown"

    loc = str(x).lower()

    if "outside jurisdiction" in loc:
        return "Outside Jurisdiction"
    elif "travis" in loc:
        return "Travis"
    elif "del valle" in loc:
        return "Del Valle"
    elif "austin" in loc:
        return "Austin"

    return "Other"


def map_sex(x):
    """
    Extracts the animal's sex from the sex_upon field.
    """
    if pd.isnull(x):
        return "Unknown"

    s = str(x).lower()

    if "female" in s:
        return "Female"
    elif "male" in s:
        return "Male"

    return "Unknown"


def status(x):
    """
    Extracts the spay/neuter status from the sex_upon field.
    """
    if pd.isnull(x):
        return "Unknown"

    status = str(x).lower()

    if "intact" in status:
        return "Intact"
    elif "neutered" in status:
        return "Neutered"
    elif "spayed" in status:
        return "Spayed"

    return "Unknown"


def age_group(row):
    """
    Assigns an age group based on the animal's age and species.
    """
    age = row["age_upon_intake_(years)"]
    animal = row["animal_type"]

    if pd.isnull(age):
        return "Unknown"

    if animal in ["Dog", "Other"]:
        if age < 2:
            return "Young"
        elif age < 8:
            return "Adult"
        else:
            return "Senior"

    elif animal == "Cat":
        if age < 3:
            return "Young"
        elif age < 10:
            return "Adult"
        else:
            return "Senior"

    else:
        return "Unknown"
