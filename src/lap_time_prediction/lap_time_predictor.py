import fastf1
import fastf1.plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

"""
Module uses sklearn to create a model that estimates lap time based on track, driver, tire compound, tire age, and fuel load (current lap).
Assumptions: Car is in free air with ideal weather conditions. Fuel usage is uniform throughout the race.

Found that using one driver to predict themselves was better even though its less data.
"""

def process_data(track_name, driver, years, test_year):
    # Get race data
    lap_dfs = []
    for year in years + test_year:
        race = fastf1.get_session(year, track_name, 'R')
        race.load()

        # Get fastest lap of last year in training data
        if year == years[-1]:
            fastest_lap = race.laps.pick_fastest()["LapTime"].total_seconds()

        # Get tire and lap data for the driver 
        laps = race.laps.pick_drivers([driver]).pick_quicklaps()
        laps = laps[["LapStartDate", "LapTime", "LapNumber", "TyreLife", "Compound"]]
        lap_dfs.append(laps)
    laps_df = pd.concat(lap_dfs, ignore_index=True)
    laps_df["LapTime"] = laps_df["LapTime"].dt.total_seconds()
    laps_df['LapStartDate'] = pd.to_datetime(laps_df['LapStartDate'])

    # One hot encode tire compunds
    encoder = OneHotEncoder(sparse_output=False)
    tire_encoded = encoder.fit_transform(laps_df[['Compound']])
    tire_columns = encoder.get_feature_names_out(['Compound'])
    laps_df[tire_columns] = tire_encoded

    # Return train and test data
    train = laps_df[laps_df['LapStartDate'].dt.year != test_year[0]]
    test = laps_df[laps_df['LapStartDate'].dt.year == test_year[0]]
    X_train = train[["LapNumber", "TyreLife"] + list(tire_columns)]
    X_test = test[["LapNumber", "TyreLife"] + list(tire_columns)]
    y_train = train["LapTime"]
    y_test = test["LapTime"]

    return X_train, X_test, y_train, y_test, fastest_lap
    
def train_model(X_train, y_train):
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=1)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, adjustment, track_name, visual=False):
    y_pred = model.predict(X_test)
    y_pred -= adjustment
    me = np.mean(y_pred - y_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Predictions were: {me:.3f} sec slower than actual.")
    print(f"MAE: {mae:.3f} sec, RMSE: {rmse:.3f} sec")
    
    if visual:
        fig, ax = plt.subplots()
        ax.scatter(range(len(y_pred)), y_pred, color="blue", label="Predicted From Last Season")
        ax.scatter(range(len(y_pred)), y_test, color="red", label="Actual")
        for i in range(len(y_pred)):
            plt.plot([i, i], [y_pred[i], y_test.iloc[i]], color='black')
        
        ax.set_ylabel("Lap Time (seconds)")
        ax.set_title(track_name)
        ax.legend()

        plt.show(block=False)
    return me, mae, rmse

def get_avg_errors(track_names, driver, years, test_year):
    """
    Function gets avg ME, MAE, and RMSE for a list of races. Avg ME can be used as an offset for future
    lap time predictions. The errors are converted into percentages of the fastest lap of previous year.
    """
    me_percents = []
    for track_name in track_names:
        X_train, X_test, y_train, y_test, fastest_lap = process_data(track_name, driver, years, test_year)
        lap_time_predictor = train_model(X_train, y_train)
        me, mae, rmse = evaluate_model(lap_time_predictor, X_test, y_test, 0, track_name, True)
        me_percentage = me / fastest_lap
        me_percents.append(me_percentage)
    
    fig, ax = plt.subplots()
    ax.scatter(track_names, me_percents, color="blue")
    
    ax.set_ylabel("Error (% of fastest lap)")
    ax.set_xlabel("Race")
    ax.set_title("Average Error in Predicted Lap Times")
    ax.legend()

    plt.show(block=False)

    return me_percents