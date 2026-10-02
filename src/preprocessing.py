import pandas as pd


def clean_telemetry(df):

    df = df.copy()

    # Remove duplicate records
    df = df.drop_duplicates()

    # Handle missing numerical values
    df = df.fillna(df.median(numeric_only=True))

    return df
