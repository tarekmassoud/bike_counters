from pathlib import Path

import numpy as np
import pandas as pd
import holidays

from sklearn.preprocessing import FunctionTransformer
from sklearn.neighbors import BallTree
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder

df_train = pd.read_parquet(Path("../data") / "train.parquet")
df_test = pd.read_parquet(Path("../data") / "final_test.parquet")

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
    return X.drop(columns=['counter_name',
                           'coordinates',
                           'site_name',
                           'site_id',
                           'counter_technical_id',
                           'counter_installation_date'
                           ])

def _merge_external_data(X):
    file_path = Path("../Notebooks") / "weather_v1.csv"
    df_ext = pd.read_csv(file_path, parse_dates=["date"])

    X = X.copy()
    # When using merge_asof left frame need to be sorted
    X["orig_index"] = np.arange(X.shape[0])
    X = pd.merge_asof(
        X.sort_values("date"), df_ext[['date', 'pres_diff', 'pmer', 'tend', 'cod_tend',
                                       'dd', 'ff', 't', 'td','u', 'vv', 'tend24', 'ww', 
                                       'n', 'pres', 'raf10',  'nbas',
                                       'ht_neige', 'rr1', 'rr6']].sort_values("date"), on="date"
    )  

    # Sort back to the original order
    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X

def _add_holiday_column(X):
    years=[2020, 2021]
    fr_holidays = holidays.France(years=years)
    X = X.copy()

    def is_holiday(date):
        weekday = date.weekday()
        if weekday > 4 or date in fr_holidays:
            return 1
        else:
            return 0

    X['is_holiday'] = X['date'].apply(is_holiday)

    return X

def _merge_connected_roads(X):
    file_path = Path("../Notebooks") / "reseau_cyclable_v1.csv"
    df_cyclable_roads = pd.read_csv(file_path)

    lat_lon_cyclable_roads = np.deg2rad(df_cyclable_roads[['latitude', 'longitude']].values)
    lat_lon_original = np.deg2rad(X[['latitude', 'longitude']].values) # here, we converted the latitude and longitude to radians for the next computation
    tree = BallTree(lat_lon_cyclable_roads, metric='haversine')

    radius = 100 / 6371000 # for a radius of 100m (converted in radians)
    indices = tree.query_radius(lat_lon_original, r=radius)# search the tree for stations that are within the defined radius

    X = X.copy()
    X['number_of_connected_roads'] = [len(index) for index in indices] # here, we count the number of roads within the radius for each site and add to the new feature

    return X

def _merge_velib_info(X):
    file_path = Path("../Notebooks") / "info_velib_v1.csv"
    df_velib = pd.read_csv(file_path)

    lat_lon_stations = np.deg2rad(df_velib[['latitude', 'longitude']].values)
    lat_lon_original = np.deg2rad(X[['latitude', 'longitude']].values) # here, again, we converted the latitude and longitude to radians for the next computation
    tree = BallTree(lat_lon_stations, metric='haversine')

    radius = 150 / 6371000  
    indices = tree.query_radius(lat_lon_original, r=radius) 

    X = X.copy() 

    X['total_nearby_station_capacity'] = [df_velib.iloc[index]['Capacité de la station'].sum() for index in indices]
    X['number_of_nearby_stations'] = [len(index) for index in indices] # here, we create two new features to add to our final dataset

    return X

def _process_column_id(X):
    # Here, we drop the first part of the counter_id feature and add to it the longitude and latitude features; the resulting column will be one hot encoded later on
    X = X.copy() 
    
    def process_id(counter_id): # we added this function because we noticed some rows contained non-numeric values
        parts = counter_id.split('-')
        return parts[1] if parts[1].isdigit() else parts[0]
    
    X['counter_id'] = X['counter_id'].apply(process_id).astype(str)

    X['longitude'] = X['longitude'].astype(str)
    X['latitude'] = X['latitude'].astype(str)

    X['counter_id'] = X['counter_id'] + '_' + X['longitude'] + '_' + X['latitude']

    return X.drop(columns=['latitude', 'longitude'])

def _merge_fuel_index(X):
    # To merge the fuel index dataset on the date 
    file_path = Path("../Notebooks") / "fuel_index_v1.csv"
    df_fuel = pd.read_csv(file_path)
    
    X['date'] = pd.to_datetime(X['date'])
    X['year'] = X['date'].dt.year
    X['month'] = X['date'].dt.month

    X = X.copy()

    X = pd.merge(
        X, 
        df_fuel, 
        on=['year', 'month'],
        how='left'
    )

    return X.drop(['year', 'month'], axis=1)

def one_hot_encode_and_concat(X):
    # One hot encoder function
    categorical_columns = ['counter_id'] 
    one_hot = OneHotEncoder(handle_unknown='ignore')

    one_hot_encoded_data = one_hot.fit_transform(X[categorical_columns])

    one_hot_encoded_df = pd.DataFrame(one_hot_encoded_data.toarray(), 
                                      columns=one_hot.get_feature_names_out(categorical_columns))

    X_dropped = X.drop(columns=categorical_columns)
    X_encoded = pd.concat([X_dropped.reset_index(drop=True), one_hot_encoded_df.reset_index(drop=True)], axis=1)

    return X_encoded

one_hot_transformer = FunctionTransformer(one_hot_encode_and_concat)
date_encoder = FunctionTransformer(_encode_dates)
is_holiday_col = FunctionTransformer(_add_holiday_column)
init_data_eng = FunctionTransformer(_drop_init_cols)
merge_external_data = FunctionTransformer(_merge_external_data, validate=False)
merge_connected_roads = FunctionTransformer(_merge_connected_roads)
merge_velib_info = FunctionTransformer(_merge_velib_info)
process_column_id = FunctionTransformer(_process_column_id)
merge_fuel_index = FunctionTransformer(_merge_fuel_index)

regressor = XGBRegressor(
    max_depth=8, 
    n_estimators=200,
    learning_rate=0.08, 
    random_state=None
)

pipeline = Pipeline([
    ("merging_ext_data", merge_external_data),
    ("adding_is_holiday_column", is_holiday_col),
    ("merging_connected_roads_data", merge_connected_roads),
    ("merging_velib_info", merge_velib_info),
    ("merging_fuel_index_info", merge_fuel_index),
    ("encoding_dates", date_encoder),
    ("dropping_redundant_columns_in_initial_data", init_data_eng),
    ("processing_column_id", process_column_id),
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