import fastf1
fastf1.Cache.enable_cache('./cache')

# Custom Modules
from tire_strats import plot_tire_strat
from lap_pace import plot_lap_pace_comparison
from lap_time_prediction.lap_time_predictor import process_data, train_model, evaluate_model

def main():
    race = fastf1.get_session(2024, 'Saudi Arabia', 'R')
    race.load()

    driver = "NOR"
    track_name = "Saudi Arabia"
    years = [2024, 2023]
    X_train, X_test, y_train, y_test = process_data(track_name, driver, years)
    lap_time_predictor = train_model(X_train, y_train)
    mae, rmse = evaluate_model(lap_time_predictor, X_test, y_test)

    #input("Press Enter to exit") # To keep the visuals open

if __name__ == "__main__":
    main()