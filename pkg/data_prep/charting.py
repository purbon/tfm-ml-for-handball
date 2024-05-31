import os
from itertools import cycle

import numpy as np
import pandas as pd

from data_io.meta import Schema
from data_io.utils import DataFrameUtils
from data_io.view import HandballPlot

pd.options.mode.chained_assignment = None


if __name__ == '__main__':
    game_direction_map = pd.read_csv(filepath_or_buffer="./data/game_map.csv")
    game_direction_map.set_index("game", inplace=True)

    utils = DataFrameUtils()
    game = "game1"
    df = pd.read_hdf(path_or_buf=f"full-game.h5", key=game)
    #df = df.loc[df[Schema.ACTIVE_PLAYERS_COLUMN] == df[Schema.ACTIVE_SENSORS_COLUMN]]

    #df_pos = utils.transform_locations(df)
    #coordinates = df_pos["locations"].reset_index()["locations"].squeeze().values

    os.makedirs(f"charts/{game}/trajectory/1", exist_ok=True)
    os.makedirs(f"charts/{game}/trajectory/2", exist_ok=True)

    j = 0
    cycol = cycle('bgrcmk')
    for name, coords in df.groupby(["HALF", "possession"]):
        half_label = int(name[0])
        possession_label = name[1]
        game_direction = game_direction_map.loc[game][f"{half_label}h"]
        title = f'Half: {half_label} - Poss: {possession_label} - Direction: {game_direction}'
        handball_plot = HandballPlot().handball_plot(title=title)
        active_players = coords[Schema.ACTIVE_SENSORS_COLUMN][0]
        for i in range(active_players):
            #player = coords[f"player_{i}"]
            #xs = player.map(lambda x: x[0]).to_numpy()
            #ys = player.map(lambda x: x[1]).to_numpy()
            xs = coords[f"player_{i}_x"].to_numpy()
            ys = coords[f"player_{i}_y"].to_numpy()

            index = np.argwhere(xs == "") # required in case there is a change and the following player does not have a sensor or her
            xs = np.delete(xs, index)

            index = np.argwhere(ys == "")
            ys = np.delete(ys, index)

            player_label = coords[f"player_{i}"][0]
            #if possession_label == "1D":
            #    print(player_label)
            #    print(xs)
            #    print(ys)
            #    print("***")
            # plot_positions(xy=XY(xy_pos), frame=1, ball=False, ax=ax)
            handball_plot.add_trajectories(x=xs, y=ys, label=player_label)
        handball_plot.add_legend()
        trajectory_charts = f"charts/{game}/trajectory/{half_label}"
        chart_path = os.path.join(trajectory_charts, f'{possession_label}.png')
        handball_plot.save(chart_path=chart_path)
