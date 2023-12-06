from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import holidays

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.neighbors import BallTree

df_train = pd.read_parquet(Path("data") / "train.parquet")
df_test = pd.read_parquet(Path("data") / "final_test.parquet")

X_train = df_train.drop(columns=['log_bike_count', 'bike_count'])
y_train = df_train['log_bike_count']
X_test = df_test

def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])

def _drop_init_cols(X):
    return X.drop(columns=['coordinates',
                           'site_name',
                           'site_id',
                           'counter_technical_id',
                           'counter_installation_date'
                           ])

def _merge_external_data(X):
    file_path = Path("Notebooks") / "weather_v1.csv"
    df_ext = pd.read_csv(file_path, parse_dates=["date"])

    X = X.copy()
    # When using merge_asof left frame need to be sorted
    X["orig_index"] = np.arange(X.shape[0])
    X = pd.merge_asof(
        X.sort_values("date"), df_ext[['date', 'numer_sta', 'pmer', 'tend', 'cod_tend', 
                                       'dd', 'ff', 't', 'td','u', 'vv', 'ww',  
                                       'n', 'nbas', 'pres', 'tend24', 'raf10', 
                                       'per',  'rr24']].sort_values("date"), on="date"
    ) #'w1', 'w2', 'rafper', 'etat_sol', 'ht_neige', 'rr1', 'rr3', 'rr6', 'rr12',
    # Sort back to the original order
    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X

def _add_holiday_column(X):
    years=[2020, 2021]
    fr_holidays = holidays.France(years=years)
    X = X.copy()

    def is_holiday(date):
        if date in fr_holidays:
            return 1
        else:
            return 0

    X['is_holiday'] = X['date'].apply(is_holiday)

    return X

def _merge_connected_roads(X):
    file_path = Path("Notebooks") / "reseau_cyclable_v1.csv"
    df_cyclable_roads = pd.read_csv(file_path)

    lat_lon_cyclable_roads = np.deg2rad(df_cyclable_roads[['latitude', 'longitude']].values) # converting lat and lon to rad (required for haversine calculation)
    lat_lon_original = np.deg2rad(X[['latitude', 'longitude']].values)

    tree = BallTree(lat_lon_cyclable_roads, metric='haversine') # create a BallTree with cyclable roads' coordinates

    radius = 100 / 6371000 # here, we define a search radius of 100 and convert it to radians (according to earth radius)

    # Query the tree for roads within the radius for each bike traffic point
    indices = tree.query_radius(lat_lon_original, r=radius)

    X = X.copy() # copy the DataFrame to avoid modifying the original one

    # Count the number of roads within the radius for each site and add to DataFrame
    X['number_of_connected_roads'] = [len(index) for index in indices]

    return X

def _merge_velib_info(X):
    file_path = Path("Notebooks") / "info_velib_v1.csv"
    df_velib = pd.read_csv(file_path)

    # Convert lat/lon to radians for haversine distance calculation
    lat_lon_stations = np.deg2rad(df_velib[['latitude', 'longitude']].values)
    lat_lon_original = np.deg2rad(X[['latitude', 'longitude']].values)

    # Create a BallTree with station coordinates
    tree = BallTree(lat_lon_stations, metric='haversine')

    # Define your search radius in meters and convert to radians (Earth radius is approximately 6371 km)
    radius = 200 / 6371000  # Example radius of 500 meters

    # Query the tree for stations within the radius for each point in X
    indices = tree.query_radius(lat_lon_original, r=radius)

    X = X.copy() # copy the DataFrame to avoid modifying the original one

    # Calculate the sum of capacities for stations within the radius for each site in X
    X['total_nearby_station_capacity'] = [df_velib.iloc[index]['Capacité de la station'].sum() for index in indices]
    X['number_of_nearby_stations'] = [len(index) for index in indices]

    return X

date_encoder = FunctionTransformer(_encode_dates)
is_holiday_col = FunctionTransformer(_add_holiday_column)
init_data_eng = FunctionTransformer(_drop_init_cols)
merge_external_data = FunctionTransformer(_merge_external_data, validate=False)
merge_connected_roads = FunctionTransformer(_merge_connected_roads)
merge_velib_info = FunctionTransformer(_merge_velib_info)

categorical_columns = ['counter_id', 'counter_name']
one_hot = OneHotEncoder(handle_unknown='ignore')

def one_hot_encode_and_concat(X):
    one_hot_encoded_data = one_hot.fit_transform(X[categorical_columns])

    one_hot_encoded_df = pd.DataFrame(one_hot_encoded_data.toarray(), 
                                      columns=one_hot.get_feature_names_out(categorical_columns))

    X_dropped = X.drop(columns=categorical_columns)
    X_encoded = pd.concat([X_dropped.reset_index(drop=True), one_hot_encoded_df.reset_index(drop=True)], axis=1)

    return X_encoded

one_hot_transformer = FunctionTransformer(one_hot_encode_and_concat)

regressor = RandomForestRegressor(max_depth=20, 
                                  n_estimators=100, 
                                  random_state=42)

pipeline = Pipeline([
    ("merging_ext_data", merge_external_data),
    ("adding_is_holiday_column", is_holiday_col),
    ("merging_connected_roads_data", merge_connected_roads),
    ("merging_velib_info", merge_velib_info),
    ("encoding_dates", date_encoder),
    ("dropping_redundant_columns_in_initial_data", init_data_eng),
    ("one_hot_encoding", one_hot_transformer),
    ("regressor", regressor)
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

results = pd.DataFrame(
    dict(
        Id=np.arange(y_pred.shape[0]),
        log_bike_count=y_pred,
    )
)

results.to_csv("submission.csv", index=False)