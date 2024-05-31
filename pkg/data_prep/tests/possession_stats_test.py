import unittest

import numpy as np
import pandas as pd
from pandas._testing import assert_frame_equal, assert_series_equal

from data_io.file import PlayerRotations
from data_io.meta import Schema
from data_pre.embeddings import DataGameGenerator


class TestPossessionRotation(unittest.TestCase):

    def test_successful_mapping(self):
        df = pd.read_excel(io="../tests/resources/poss-file.xlsx")
        df_count = self.possession_phase_stats(df=df)
        print(df_count)

    def possession_phase_stats(self, df):
        df = df[["game_phases", "possession"]]

        poss_group = df.groupby("possession")

        keys = list(poss_group.indices)
        stats = {}
        for h in keys:
            v = poss_group.indices[h]
            low_index, high_index = self.find_possession_index(df=df, groups=poss_group, key=h)
            sdf = df.iloc[low_index:high_index]
            # {'GF': 0, 'PEN': 0}
            attrs = ["GF", "PEN", "PREV_TM", "POST_TM"]
            counts = [0, 0, 0, 0]
            active = True

            pre_tm = self.check_for_tm(df=df, base_index=v[0], diff=-1)
            counts[2] = 1 if pre_tm == 1 else 0
            post_tm = self.check_for_tm(df=df, base_index=v[-1], diff=1)
            counts[3] = 1 if post_tm == 1 else 0

            for i in range(len(sdf)):
                game_phase = sdf.iloc[i, 0]
                if active and game_phase in attrs:
                    index = attrs.index(game_phase)
                    counts[index] += 1
                    active = False
                elif not active and (game_phase == "DEF" or game_phase == "AT"):
                    active = True
            stats[h] = counts
        return pd.DataFrame.from_dict(data=stats, orient='index', columns=["FK_count", "PEN_count", "Prev_TM", "Post_TM"])

    def check_for_tm(self, df, base_index, diff):
        try:
            pre_row = df.iloc[base_index + diff]
            return 1 if pre_row["game_phases"] == "TM" else 0
        except:
            return -1

    def find_possession_index(self, df, groups, key):
        v = groups.indices[key]
        last_index = v[-1]
        max_rows = df.shape[0]
        next_possession = ""
        current_possession = df.iloc[last_index]["possession"]
        i = last_index + 1
        while i < max_rows and next_possession == "":
            row_possession = df.iloc[i]["possession"]
            if isinstance(row_possession, str) and row_possession != "" and next_possession != current_possession:
                next_possession = row_possession
            i = i + 1
        last_index = i if next_possession != "" else v[-1]
        return v[0], last_index

