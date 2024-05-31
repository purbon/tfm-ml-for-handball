import unittest

import pandas as pd
from pandas._testing import assert_frame_equal, assert_series_equal

from data_io.meta import Schema
from data_pre.embeddings import DataGameGenerator


class TestPossessionRotation(unittest.TestCase):
    def test_successful_rotation(self):
        df = pd.read_hdf(path_or_buf=f"../handball.h5", key="pos")

        df.drop(columns=["player_0", "player_1", "player_2", "player_3", "player_4", "player_5"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        timesteps = 20

        df = df[df[Schema.ACTIVE_PLAYERS_COLUMN] == 6]
        filter = []
        for i in range(6):
            labels = [f"player_{i}_x",
                      f"player_{i}_y"]  # , f"player_{i}_dist_last_5sec", f"player_{i}_speed_last_5sec" ]
            for label in labels:
                filter.append(label)

        group_df = df.groupby(["GAME", "possession"])
        possession = None
        for game, data_points in group_df:
            possession = data_points[filter]
            break

        gen = DataGameGenerator(df=df, timesteps=timesteps)
        rotated_possession = gen.rotate_a_possession(possession=possession)
        for i in range(6):
            assert_series_equal(rotated_possession[f"player_{i}_x"], possession[f"player_{i}_y"], check_names=False)
            assert_series_equal(rotated_possession[f"player_{i}_y"], possession[f"player_{i}_x"], check_names=False)