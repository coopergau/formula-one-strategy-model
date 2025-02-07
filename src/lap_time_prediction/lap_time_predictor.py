import fastf1
import fastf1.plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

"""
Module uses sklearn to create a model that estimates lap time based on track, driver, tire compound, tire age, and fuel load (current lap).
Assumptions: Car is in free air with ideal weather conditions. Fuel usage is uniform throughout the race.

Found that using one driver to predict themselves was better even though its less data.
"""

def process_data(track_name, driver, years):
    # Get all data
    lap_dfs = []
    for year in years:
        race = fastf1.get_session(year, track_name, 'R')
        race.load()
        laps = race.laps.pick_drivers([driver]).pick_quicklaps() 
        laps = laps[["LapTime", "LapNumber", "TyreLife", "Compound"]]
        lap_dfs.append(laps)
    laps_df = pd.concat(lap_dfs, ignore_index=True)
    laps_df["LapTime"] = laps_df["LapTime"].dt.total_seconds()

    # One hot encode tire compunds
    encoder = OneHotEncoder(sparse_output=False)
    tire_encoded = encoder.fit_transform(laps_df[['Compound']])
    tire_columns = encoder.get_feature_names_out(['Compound'])
    laps_df[tire_columns] = tire_encoded

    # Return train and test data
    X = laps_df[["LapNumber", "TyreLife"] + list(tire_columns)]
    y = laps_df["LapTime"]
    return train_test_split(X, y, test_size=0.2, random_state=2)
    
def train_model(X_train, y_train):
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=2)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"MAE: {mae:.3f} sec, RMSE: {rmse:.3f} sec")
    
    fig, ax = plt.subplots()
    ax.scatter(range(len(y_pred)), y_pred, color="blue", label="Predicted")
    ax.scatter(range(len(y_pred)), y_test, color="red", label="Actual")
    for i in range(len(y_pred)):
        plt.plot([i, i], [y_pred[i], y_test.iloc[i]], color='black')
    
    ax.set_ylabel("Lap Time (seconds)")
    ax.legend()

    plt.show()
    return mae, rmse
