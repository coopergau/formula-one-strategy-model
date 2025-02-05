import fastf1
fastf1.Cache.enable_cache('./cache')

# Custom Modules
from tire_strats import tire_strat_chart

def main():
    session = fastf1.get_session(2024, 'Bahrain', 'R')  # 'R' = Race, 'Q' = Quali
    session.load()
    tire_strat_chart(session)

if __name__ == "__main__":
    main()