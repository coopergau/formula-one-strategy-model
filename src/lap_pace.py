import fastf1
import fastf1.plotting
import matplotlib.pyplot as plt

def plot_lap_pace_comparison(race, driver1, driver2):
    """
    Plots a scatter comparison of each lap time for two drivers in a given session. 
    The plots only include quick laps (no pit stops or yellow/red flags) to avoid distortion.
    """
    # Enable Matplotlib patches for plotting timedelta values and load
    # FastF1's dark color scheme
    fastf1.plotting.setup_mpl(mpl_timedelta_support=True, misc_mpl_mods=False,
                            color_scheme='fastf1')
    
    # Get driver lap times
    driver1_laps = race.laps.pick_driver(driver1).pick_quicklaps()
    driver2_laps = race.laps.pick_driver(driver2).pick_quicklaps()

    # Plot the data
    fig, ax = plt.subplots()
    ax.plot(driver1_laps["LapNumber"], driver1_laps["LapTime"].dt.total_seconds(), color="blue", label=driver1)
    ax.plot(driver2_laps["LapNumber"], driver2_laps["LapTime"].dt.total_seconds(), color="red", label=driver2)
    
    ax.set_xlabel("Lap Number")
    ax.set_ylabel("Lap Time (seconds)")
    ax.set_title("Lap Times Comparison")
    ax.legend()

    plt.show(block=False)