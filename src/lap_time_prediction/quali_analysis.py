import numpy as np
import fastf1
fastf1.Cache.enable_cache('./cache')

"""
Module compares previous years qualifying lap times to current years qualifying times to gauge
the difference between season laptimes from change in circuit, car, or something else.
"""

def get_quali_laps(track, driver, year):
    quali = fastf1.get_session(year, track, 'Q')
    quali.load()
    laps = quali.laps.pick_drivers([driver]).pick_quicklaps()
    lap_times = laps["LapTime"].dt.total_seconds()
    return lap_times

def get_quali_lap_difference(track, driver, prev_year, curr_year):
    prev_lap_times = get_quali_laps(track, driver, prev_year)
    curr_lap_times = get_quali_laps(track, driver, curr_year)

    prev_avg = np.mean(prev_lap_times)
    curr_avg = np.mean(curr_lap_times)

    multiplier = curr_avg / prev_avg
    return multiplier
    