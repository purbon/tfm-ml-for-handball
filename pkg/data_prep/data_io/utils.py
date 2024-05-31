import h5py
import pandas as pd
import numpy as np


def print_players(record):
    for player_id in range(6):
        player = record[f"player_{player_id}"]
        print(f"{player} ", end="")
    print()


class Hdf5Utils:
    @staticmethod
    def get_keys(path):
        f = h5py.File(path, 'r')
        return [key for key in f.keys()]

    @staticmethod
    def parse_ranges(path, keys):
        keys = Hdf5Utils.get_keys(path=path)
        for key in keys:
            print(f'For {key}')
            df = pd.read_hdf(path, key=key)
            range = f"{df['time'].min()} {df['time'].max()}"
            print(range)


class DataFrameUtils:

    def map_coordinates_to_array(self, current):
        for player_id in range(6):
            x_coord = 20.0 if isinstance(current[f'player_{player_id}_x'], str) else current[f'player_{player_id}_x']
            y_coord = 10.0 if isinstance(current[f'player_{player_id}_y'], str) else current[f'player_{player_id}_y']
            current[f'player_{player_id}'] = np.array([x_coord, y_coord], dtype=np.float64)
        return current

    def transform_locations(self, df):
        player_columns = ["player_0_x", "player_0_y",
                          "player_1_x", "player_1_y",
                          "player_2_x", "player_2_y",
                          "player_3_x", "player_3_y",
                          "player_4_x", "player_4_y",
                          "player_5_x", "player_5_y"]

        columns = ["HALF",
                   "analysis",
                   "possession",
                   "game_phases",
                   "player_0_x", "player_0_y",
                   "player_1_x", "player_1_y",
                   "player_2_x", "player_2_y",
                   "player_3_x", "player_3_y",
                   "player_4_x", "player_4_y",
                   "player_5_x", "player_5_y"]

        df_pos = df[columns]

        for player_id in range(6):
            df_pos[f'player_{player_id}'] = np.nan

        df_pos = df_pos.apply(lambda current: self.map_coordinates_to_array(current), axis=1)
        df_pos["locations"] = df_pos.apply(
            lambda dp: np.array(
                [dp["player_0"], dp["player_1"], dp["player_2"], dp["player_3"], dp["player_4"], dp["player_5"]],
                dtype=np.float64),
            axis=1)
        df_pos = df_pos.drop(columns=player_columns)
        # df_pos = df_pos.drop(columns=["player_0", "player_1", "player_2", "player_3", "player_4", "player_5"])

        return df_pos


class FileUtils:

    @staticmethod
    def convert(src, trg):
        df = pd.read_excel(io=src)
        df.to_csv(path_or_buf=trg, index=False)
