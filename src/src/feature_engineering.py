import pandas as pd


def extract_features(df, window=10):

    df = df.copy()

    columns = ["CPU", "Memory", "Latency", "Traffic"]

    for col in columns:

        df[f"{col}_mean"] = (
            df[col].rolling(window=window, min_periods=1).mean()
        )

        df[f"{col}_std"] = (
            df[col].rolling(window=window, min_periods=1).std()
        )

        df[f"{col}_max"] = (
            df[col].rolling(window=window, min_periods=1).max()
        )

    df = df.fillna(0)

    return df
