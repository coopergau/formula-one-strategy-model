import fastf1
fastf1.Cache.enable_cache('./cache')

# Custom Modules
from tire_strats import plot_tire_strat
from lap_pace import plot_lap_pace_comparison

def main():
    race = fastf1.get_session(2024, 'Miami', 'R')
    race.load()

    plot_tire_strat(race)

    driver1 = "NOR"
    driver2 = "VER"
    plot_lap_pace_comparison(race, driver1, driver2)

    input("Press Enter to exit") # To keep the visuals open

if __name__ == "__main__":
    main()