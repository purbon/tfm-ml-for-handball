import unittest

import numpy as np
import pandas as pd
from pandas._testing import assert_frame_equal, assert_series_equal

from data_io.file import PlayerRotations
from data_io.meta import Schema
from data_pre.embeddings import DataGameGenerator


class TestSegmentationProcess(unittest.TestCase):

    def test_successful_mapping(self):
        df = pd.read_excel(io="../tests/resources/game-segments.xlsx")
        df = self.game_segmentation(df=df)
        df.to_excel("segmented.xlsx")

    def game_segmentation(self, df):

        offense_possession_id = 0
        defense_possession_id = 0
        last_game_phase = ""
        las_analysis = ""
        sequence_id = 0
        break_happened = False
        game_init_tags = ["IP", "I2P"]
        for index, row in df.iterrows():
            current_phase = str(row['game_phases']).replace('-', '')
            current_analysis = row['analysis']
            current_possession = row['possession']

            if game_init_tags.__contains__(current_analysis):  # Game start
                offense_possession_id = 0
                defense_possession_id = 0
                last_game_phase = ""
                last_analysis = ""
            elif current_analysis == "BLJ":  # Something is happening
                if last_game_phase == "":  # coming from an empty phase (initial set)
                    if current_phase == "DEF":
                        defense_possession_id = defense_possession_id + 1
                    elif current_phase == "AT":
                        offense_possession_id = offense_possession_id + 1
                elif last_game_phase != current_phase:  # change of phases (DEF<->AT) while the game is still on
                    if current_phase == "DEF":
                        defense_possession_id = defense_possession_id + 1
                    elif current_phase == "AT":
                        offense_possession_id = offense_possession_id + 1
                    sequence_id = 0
                elif last_game_phase == current_phase:
                    # staying in the same game_phase, but there could have been breaks in between
                    pass
                last_game_phase = current_phase
                df.loc[index, 'sequence_label'] = f"{current_possession}{sequence_id}"
                df.loc[index, 'sequences'] = sequence_id
            elif current_analysis == "JP":  # Game is stop
                if last_analysis != "JP":
                    sequence_id += 1
            print(f"{current_analysis} {current_phase} {defense_possession_id} {offense_possession_id} {sequence_id}")
            last_analysis = current_analysis
        return df