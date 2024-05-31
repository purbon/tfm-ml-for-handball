import pandas as pd
import re


class Resources:
    def __init__(self):
        self.games = {
            "game1": Resource("game1"),
            "game2": Resource("game2"),
            "game3": Resource("game3"),
            "game4": Resource("game4"),
            "game5": Resource("game5"),
            "game6": Resource("game6"),
            "game7": Resource("game7"),
            "game8": Resource("game8"),
            "game9": Resource("game9"),
            "game10": Resource("game10"),
        }
        self.player_attributes = [Schema.X_COLUMN, Schema.Y_COLUMN]

    def get(self, game):
        return self.games[game]


class Resource:
    def __init__(self, game):
        pcode = self.map_code(game)
        self.players_in_game = f"./data/{pcode}/players-in-game.xlsx"
        self.enrich_results = f"results/{game}.csv"
        self.game_labels = f"data/{pcode}/game.xlsx"
        self.passive_labels = f"data/{pcode}/passive-game-alert.xlsx"
        self.analysis = f"data/game-analysis.xlsx"

    def map_code(self, game):
        index = int(re.sub('\D', '', game))
        return f'P{index}'


class Schema:
    ACTIVE_PLAYERS_COLUMN = "active_players"
    ACTIVE_SENSORS_COLUMN = "active_sensors"
    X_COLUMN = "x"
    Y_COLUMN = "y"
    POS_COLUMN = "pos"
    ANGLE_COLUMN = "angle"
    CUM_DIST_COLUMN = "cumdist"
    DIST_COLUMN = "dist_last_5sec"
    SPEED_COLUMN = "speed_last_5sec"


class Metadata:
    def __init__(self, players_metadata_loc):
        self.players_metadata_loc = players_metadata_loc
        self.codes = pd.read_csv(filepath_or_buffer=self.players_metadata_loc)
        self.codes.index = self.codes['Code']

    def is_a_goalkeeper(self, label):
        if label.startswith('P'):
            label = label[1:]
        return self.codes.loc[label]['Position'] == 'P'
