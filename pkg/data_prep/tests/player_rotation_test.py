import unittest

import pandas as pd

from data_io.file import PlayerRotations

pd.options.mode.chained_assignment = None


class TestPlayerRotations(unittest.TestCase):

    def test_single_change_in_sequence_rotation(self):
        player_rotations = PlayerRotations(game="game_test", players_metadata_loc="../data/players.csv")
        player_rotations.load_players_in_game_for(path="../data/test/players-in-game.xlsx")
        player_rotations.map_players()

        previous = []
        for row_id, rotation in player_rotations.rotations.iterrows():
            current = rotation
            if len(previous) != 0 and not current.equals(previous):
                changes_count = self.__count_changes(previous=previous, current=current)
                self.assertEqual(1, changes_count)
            previous = current

    def __count_changes(self, previous, current):
        changes = 0
        for player_id in range(6):
            p_i = previous[f"player_{player_id}"]
            c_i = current[f"player_{player_id}"]
            if p_i != c_i:
                changes += 1
        return changes
