from typing import List

import numpy as np
import pandas as pd


def get_skewed_columns(df_in: pd.DataFrame, top_percentage: float = 99.0) -> List:
    """
    Selects columns from a DataFrame where the top value in the value counts
    represents at least top_percentage% of the data.
    """

    sk_columns = []

    for column in df_in.columns:
        value_counts = df_in[column].value_counts(normalize=True) * 100
        if value_counts.iloc[0] >= top_percentage:
            sk_columns.append(column)

    return sk_columns


def preprocess_features(df_: pd.DataFrame):
    return (
        df_.pipe(_features_missing_values)
        .pipe(_features_drop_values)
        .pipe(_features_drop_columns)
        .pipe(_features_map_age_values)
        .pipe(_features_map_diag_values)
        .pipe(_features_map_medical_specialty_values)
        .pipe(_features_map_race_values)
        .pipe(_features_map_admission_type_values)
        .pipe(_features_map_discharche_id_values)
    )


def preprocess_features_encode(df_: pd.DataFrame):
    return (
        df_.pipe(_features_encode_diabetesMed_values)
        .pipe(_features_encode_med_values)
        .pipe(_features_encode_a1cresult_values)
        .pipe(_features_encode_change_values)
    )


def _features_missing_values(df_features: pd.DataFrame) -> pd.DataFrame:
    return df_features.replace("?", np.nan)


def _features_drop_values(df_features: pd.DataFrame) -> pd.DataFrame:
    # drop rows with bad missing values
    mask = df_features["gender"] != "Unknown/Invalid"
    return df_features.loc[mask]


def _features_drop_columns(df_: pd.DataFrame) -> pd.DataFrame:
    # drop irrelevant columns
    # drop columns with high number of missing values
    # X_processed[X_processed.notna().sum().sort_values().index].info()
    # drop columns with low predictive value
    # columns = get_skewed_columns(X_processed)

    return (
        df_.drop(
            ["encounter_id", "patient_nbr"],
            axis=1,
            inplace=False,
        )
        .drop(
            ["weight", "payer_code", "admission_source_id"],
            axis=1,
            inplace=False,
        )
        .drop(
            [
                "acarbose",
                "acetohexamide",
                "chlorpropamide",
                "citoglipton",
                "examide",
                "miglitol",
                "nateglinide",
                "tolazamide",
                "tolbutamide",
                "troglitazone",
                "glyburide-metformin",
                "glipizide-metformin",
                "glimepiride-pioglitazone",
                "metformin-rosiglitazone",
                "metformin-pioglitazone",
            ],
            axis=1,
            inplace=False,
        )
    )


def _features_map_admission_type_values(df_features: pd.DataFrame):
    df_features["admission_type_id"] = (
        df_features["admission_type_id"]
        .map(
            {
                1: "Urgent",
                2: "Urgent",
                3: "Elective",
                4: "Newborn",
            }
        )
        .fillna("Other")
    )
    return df_features


def _features_map_discharche_id_values(df_features: pd.DataFrame):
    df_features["discharge_disposition_id"] = df_features["discharge_disposition_id"].astype("object")
    return df_features


def _features_map_race_values(df_features: pd.DataFrame):
    df_features["race"] = df_features["race"].fillna("Other")
    return df_features


def _features_encode_a1cresult_values(df_features: pd.DataFrame):
    df_features["A1Cresult"] = (
        df_features["A1Cresult"]
        .replace(["Norm", ">7", ">8"], ["1", "2", "2"], inplace=False)
        .fillna("0")
        .astype("int")
    )
    return df_features


def _features_encode_med_values(df_features: pd.DataFrame):
    cols = [
        "metformin",
        "repaglinide",
        "glimepiride",
        "glipizide",
        "glyburide",
        "pioglitazone",
        "rosiglitazone",
        "insulin",
    ]
    df_features[cols] = (
        df_features[cols]
        .replace(["No", "Steady", "Down", "Up"], ["0", "1", "2", "2"], inplace=False)
        .astype("int")
    )
    return df_features


def _features_encode_change_values(df_features: pd.DataFrame):
    df_features["change"] = (
        df_features["change"].replace(["No", "Ch"], ["0", "1"], inplace=False).astype("int")
    )
    return df_features


def _features_encode_diabetesMed_values(df_features: pd.DataFrame):
    df_features["diabetesMed"] = (
        df_features["diabetesMed"].replace(["No", "Yes"], ["0", "1"], inplace=False).astype("int")
    )
    return df_features


def _features_map_age_values(df_features: pd.DataFrame):
    df_features["age"] = df_features["age"].map(
        {
            "[0-10)": 5,
            "[10-20)": 15,
            "[20-30)": 25,
            "[30-40)": 35,
            "[40-50)": 45,
            "[50-60)": 55,
            "[60-70)": 65,
            "[70-80)": 75,
            "[80-90)": 85,
            "[90-100)": 95,
        },
    )
    return df_features


def _features_map_medical_specialty_values(df_features: pd.DataFrame):
    speciality_map = {
        "Internal medicine and related": [
            "Cardiology",
            "Gastroenterology",
            "Hematology",
            "Hematology/Oncology",
            "InternalMedicine",
            "Nephrology",
            "Pulmonology",
            "Oncology",
            "Rheumatology",
        ],
        "Emergency and critical care": [
            "Emergency/Trauma",
            "IntensiveCare",
            "Pediatrics-CriticalCare",
        ],
        "General practice": [
            "Family/GeneralPractice",
        ],
        "Surgery": [
            "Orthopedics",
            "Orthopedics-Reconstructive",
            "Surgery-General",
            "Surgery-Cardiovascular",
            "Surgery-Cardiovascular/Thoracic",
            "Surgery-Colon&Rectal",
            "Surgery-Neuro",
            "Surgery-Plastic",
            "Surgery-PlasticwithinHeadandNeck",
            "Surgery-Pediatric",
            "Surgery-Vascular",
            "Surgery-Maxillofacial",
            "Surgery-Thoracic",
            "SurgicalSpecialty",
        ],
    }

    cat_map = {val: key for key, val_list in speciality_map.items() for val in val_list}

    df_features["medical_specialty"] = df_features["medical_specialty"].map(cat_map).fillna("Other")

    return df_features


def category_from_icd9_code(icd9_code):
    try:
        icd9_code = icd9_code.split(".")[0]
        code = int(icd9_code)
    except Exception:
        return "Other"

    if code == 250:
        return "Diabetes"
    elif code >= 400 and code < 460:
        return "Cardiovascular disease"
    elif code >= 460 and code < 520:
        return "Respiratory disease"
    elif code >= 710 and code < 740:
        return "Musculoskeletal disease"
    else:
        return "Other"


def _features_map_diag_values(df_features: pd.DataFrame):
    for col_name in ["diag_1", "diag_2", "diag_3"]:
        df_features[col_name] = df_features[col_name].apply(category_from_icd9_code)
    return df_features


def preprocess_targets(df_targets: pd.DataFrame) -> pd.DataFrame:
    return (
        df_targets["readmitted"].replace(["NO", ">30", "<30"], ["0", "0", "1"], inplace=False).astype("int")
    )
