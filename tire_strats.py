import fastf1
import fastf1.plotting
import matplotlib.pyplot as plt

def tire_strat_chart(session):
    # Get driver names
    drivers = session.drivers
    drivers = [session.get_driver(driver)["Abbreviation"] for driver in drivers]
    
    # Get driver stint data
    laps = session.laps
    stints = laps[["Driver", "Stint", "Compound", "LapNumber"]]
    stints = stints.groupby(["Driver", "Stint", "Compound"])
    stints = stints.count().reset_index()
    stints = stints.rename(columns={"LapNumber": "StintLength"})

    # Get driver total race times
    driver_lap_times = laps[["Driver", "LapTime"]]
    print(driver_lap_times)
    driver_lap_times = driver_lap_times.groupby(["Driver"])
    driver_race_times = driver_lap_times.sum().reset_index()
    driver_race_times = driver_race_times.rename(columns={"LapTime": "RaceTime"})
    print(driver_race_times)
  
    # Plot stint data
    fig, ax = plt.subplots()
    for driver in drivers:
        driver_stints = stints.loc[stints["Driver"] == driver]
    
        previous_stint_end = 0
        for idx, row in driver_stints.iterrows():
            tire_colour = fastf1.plotting.get_compound_color(row["Compound"], session=session)
            
            plt.barh(
                y=driver,
                width=row["StintLength"],
                left=previous_stint_end,
                color=tire_colour,
                edgecolor="black",
                fill=True
            )

            previous_stint_end += row["StintLength"]
    
    plt.title(f"{session.event["EventName"]} {session.event["OfficialEventName"][-4:]}")
    plt.xlabel("Lap Number")
    ax.invert_yaxis() # Top down in order of finishing
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.show()