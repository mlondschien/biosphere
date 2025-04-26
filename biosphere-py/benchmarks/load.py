# From https://github.com/xhochy/nyc-taxi-fare-prediction-deployment-example
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

DATA_PATH = Path(__file__).resolve().parent / "data"
NYC_TAXI_DATASET_PATH = DATA_PATH / "nyc_taxi_data.parquet"
NYC_TAXI_DATASET_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2019-01.parquet"
)
NYC_TAXI_BURROWS_PATH = DATA_PATH / "nyc_taxi_burrows.parquet"
NYC_TAXI_BURROWS_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"


def load_nyc_taxi_raw():
    if NYC_TAXI_DATASET_PATH.exists():
        return pd.read_parquet(NYC_TAXI_DATASET_PATH)
    else:
        df = pd.read_parquet(NYC_TAXI_DATASET_URL)
        df.to_parquet(NYC_TAXI_DATASET_PATH)
        return df


def load_nyc_taxi_burrows():
    if NYC_TAXI_BURROWS_PATH.exists():
        return pd.read_parquet(NYC_TAXI_BURROWS_PATH)
    else:
        df = pd.read_csv(NYC_TAXI_BURROWS_URL, index_col=False,)
        df.to_parquet(NYC_TAXI_BURROWS_PATH)
        return df


def split_pickup_datetime(df):
    return pd.DataFrame(
        {
            "pickup_dayofweek": df["tpep_pickup_datetime"].dt.dayofweek,
            "pickup_hour": df["tpep_pickup_datetime"].dt.hour,
            "pickup_minute": df["tpep_pickup_datetime"].dt.minute,
        }
    )


def burrow(df):
    df = df.merge(
        load_nyc_taxi_burrows().rename(columns={"Borough": "pickup_borough"}),
        left_on="PULocationID",
        right_on="LocationID",
        how="left",
    )
    df["pickup_borough"] = df["pickup_borough"].fillna("Unknown")
    return df[["pickup_borough"]]


burrow_categories = [
    "Bronx",
    "Brooklyn",
    "EWR",
    "Manhattan",
    "Queens",
    "Staten Island",
    "Unknown",
]


def feature_enginering():
    return make_column_transformer(
        (FunctionTransformer(), ["passenger_count", "trip_distance"]),
        (FunctionTransformer(func=split_pickup_datetime), ["tpep_pickup_datetime"],),
        (
            Pipeline(
                steps=[
                    ("pickup_borough", FunctionTransformer(burrow)),
                    ("one_hot_encoder", OneHotEncoder(categories=[burrow_categories])),
                ]
            ),
            ["PULocationID"],
        ),
    )


def load_nyc_taxi(train_size, test_size):
    df = load_nyc_taxi_raw().head(train_size + test_size)
    df = df[lambda x: x["fare_amount"] > 0]
    df["fare_amount"] = np.log(df["fare_amount"])

    fare_amount_mean = df["fare_amount"].mean()
    fare_amount_std = df["fare_amount"].std()
    df = df[
        lambda x: x["fare_amount"].between(
            fare_amount_mean - 4 * fare_amount_std,
            fare_amount_mean + 4 * fare_amount_std,
        )
    ]

    y = df.pop("fare_amount").to_numpy()
    X = feature_enginering().fit_transform(df)

    return train_test_split(X, y, test_size=test_size / (train_size + test_size))
