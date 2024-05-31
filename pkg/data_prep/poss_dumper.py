import pandas as pd

from data_io.meta import Schema
from data_io.utils import DataFrameUtils

pd.options.mode.chained_assignment = None

if __name__ == '__main__':
    summary = {}
    utils = DataFrameUtils()

    clean_num_poss = 0

    df = pd.DataFrame()
    for game_id in range(1, 11, 1):
        game = f"game{game_id}"
        try:
            gdf = pd.read_hdf(path_or_buf=f"full-game.h5", key=game)
            complete = 0
            total = 0
            for name, coords in gdf.groupby("possession"):
                possession_label = name
                poss_duration = coords.size
                full_trajectories = coords.loc[
                    coords[Schema.ACTIVE_PLAYERS_COLUMN] == coords[Schema.ACTIVE_SENSORS_COLUMN]]
                if poss_duration == full_trajectories.size:
                    complete = complete + 1
                    df = pd.concat([df, coords])
                total += 1
            clean_num_poss += complete
            summary[game] = {"complete_possessions": complete, "total_possession": total,
                             "percent": (complete / (total * 1.0))}
        except Exception as e:
            print(f"Something happened processing game {game}")
    string_fields = ["player_0", "player_1", "player_2", "player_3", "player_4", "player_5", "analysis", "game_phases",
                     "possession", "tactical_situation", "possession_result",
                     "prev2_possession_result", "passive_alert", "passive_alert_result", "offense_type", "throw_zone",
                     "organized_game", "sequence_label"]
    for i in range(5):
        string_fields += [f"prev{i+1}_possession_result"]

    for column in df.columns:
        if column in string_fields:
            continue
            #df[column] = df[column].astype("string")
        else:
            df[column] = pd.to_numeric(df[column], errors='coerce')
    df.to_excel("handball.xlsx")
    df.to_hdf(path_or_buf="handball.h5", key="pos")
