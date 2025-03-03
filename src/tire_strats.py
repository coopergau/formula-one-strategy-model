import fastf1
import fastf1.plotting
import matplotlib.pyplot as plt

def plot_tire_strat(race):
    """
    Plots a visual of the different tire strategies of each driver for a given race.
    """
    # Get driver names
    drivers = race.drivers
    drivers = [race.get_driver(driver)["Abbreviation"] for driver in drivers]
    
    # Get driver stint data
    laps = race.laps
    stints = laps[["Driver", "Stint", "Compound", "LapNumber"]]
    stints = stints.groupby(["Driver", "Stint", "Compound"])
    stints = stints.count().reset_index()
    stints = stints.rename(columns={"LapNumber": "StintLength"})

    # Get driver total race times
    driver_lap_times = laps[["Driver", "LapTime"]]
    driver_lap_times = driver_lap_times.groupby(["Driver"])
    driver_race_times = driver_lap_times.sum().reset_index()
    driver_race_times = driver_race_times.rename(columns={"LapTime": "RaceTime"})
    driver_race_times["RaceTime"] = driver_race_times["RaceTime"].dt.total_seconds() # Converts to seconds
  
    # Plot stint data
    fig, ax = plt.subplots()
    for driver in drivers:
        driver_stints = stints.loc[stints["Driver"] == driver]

        previous_stint_end = 0
        for _, row in driver_stints.iterrows():
            tire_colour = fastf1.plotting.get_compound_color(row["Compound"], session=race)
            
            plt.barh(
                y=driver,
                width=row["StintLength"],
                left=previous_stint_end,
                color=tire_colour,
                edgecolor="black",
                fill=True
            )

            previous_stint_end += row["StintLength"]
    
    # Add total race times as text annotations
    max_x = stints.groupby("Driver")["StintLength"].sum().max()
    for _, row in driver_race_times.iterrows():
        ax.text(
            x=max_x + 2,  # Position to the right of the longest stint
            y=row["Driver"],
            s=f"{row['RaceTime']:.1f}s",
            va="center",
            fontsize=10,
            color="black"
        )
    
    plt.title(f"{race.event["EventName"]} {race.event["OfficialEventName"][-4:]}")
    plt.xlabel("Lap Number")
    ax.invert_yaxis() # Top down in order of finishing
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.show(block=False)