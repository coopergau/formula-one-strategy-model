import numpy as np
import fastf1
fastf1.Cache.enable_cache('./cache')

# Custom Modules
from tire_strats import plot_tire_strat
from lap_pace import plot_lap_pace_comparison
from lap_time_prediction.lap_time_predictor import process_data, train_model, evaluate_model, get_avg_errors
from lap_time_prediction.quali_analysis import get_quali_lap_difference

def main():

    track_name = "Emilia-Romagna"
    driver = "LEC"
    '''race = fastf1.get_session(2023, track_name, 'R')
    race.load()
    plot_tire_strat(race)
    race = fastf1.get_session(2024, track_name, 'R')
    race.load()
    plot_tire_strat(race)'''
    last_year = 2023
    current_year = 2024
    track_names = ["Bahrain", "Saudi Arabia", "Miami", "Emilia-Romagna", "Monaco", "Spain", "Canada"
                   "Austria", "Great Britain", "Hungary"]
    laptime_offset_multiplier = get_quali_lap_difference(track_name, driver, last_year, current_year)
    print(laptime_offset_multiplier)
    '''avg_mes = get_avg_errors(track_names, driver, years, test_year)'''
    '''X_train, X_test, y_train, y_test, fastest_lap = process_data(track_name, driver, years, test_year)
    print(fastest_lap)
    lap_time_predictor = train_model(X_train, y_train)
    adjustment = 2.514 
    me, mae, rmse = evaluate_model(lap_time_predictor, X_test, y_test, 0, fastest_lap, True)
    me, mae, rmse = evaluate_model(lap_time_predictor, X_test, y_test, adjustment, fastest_lap, True)'''

    input("Press Enter to exit") # To keep the visuals open

if __name__ == "__main__":
    main()