import pandas as pd
import json

from data_io.meta import Schema
from data_io.utils import DataFrameUtils

if __name__ == '__main__':
    summary = {}
    utils = DataFrameUtils()

    clean_num_poss = 0

    for game_id in range(1, 11, 1):
        game = f"game{game_id}"
        try:
            df = pd.read_hdf(path_or_buf=f"full-game.h5", key=game)

            complete = 0
            total = 0
            for name, coords in df.groupby("possession"):
                possession_label = name
                poss_duration = coords.size
                full_trajectories = coords.loc[coords[Schema.ACTIVE_PLAYERS_COLUMN] == coords[Schema.ACTIVE_SENSORS_COLUMN]]
                # print(f"{possession_label}: {poss_duration} - {full_trajectories.size}")
                if poss_duration == full_trajectories.size:
                    complete = complete + 1
                total += 1
            clean_num_poss += complete
            summary[game] = {"complete_possessions": complete, "total_possession": total, "percent" : (complete/(total*1.0))}
        except:
            print(f"Something happened processing game {game}")

    summary["clean_num_poss"] = clean_num_poss
    json_summary = json.dumps(summary, indent=2)
    print(json_summary)
