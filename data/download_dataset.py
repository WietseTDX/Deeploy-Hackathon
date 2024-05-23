from ucimlrepo import fetch_ucirepo

"""
Fetch dataset
[1] Beata Strack, Jonathan P. DeShazo, Chris Gennings, Juan L. Olmo, Sebastian Ventura, Krzysztof J. Cios, and John N. Clore,
“Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records,” BioMed Research International, vol. 2014, Article ID 781670, 11 pages, 2014.
"""

diabetes_130_us_hospitals_for_years_1999_2008 = fetch_ucirepo(id=296)

diabetes_130_us_hospitals_for_years_1999_2008.data.original.to_parquet(
    path="./01_raw/diabetes_1999_2008_original.parquet"
)
diabetes_130_us_hospitals_for_years_1999_2008.data.features.to_parquet(
    path="./01_raw/diabetes_1999_2008_features.parquet"
)
diabetes_130_us_hospitals_for_years_1999_2008.data.targets.to_parquet(
    path="./01_raw/diabetes_1999_2008_targets.parquet"
)
