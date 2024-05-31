import numpy as np
import pandas as pd

from data_io.file import PlayerRotations
from data_io.meta import Metadata, Schema, Resources
from data_io.utils import Hdf5Utils
from data_io.math import Vector


class PointExtension:

    def __init__(self):
        self.metadata = Metadata(players_metadata_loc="./data/players.csv")

    def __append(self, row, df, tail=5):
        table = df[max(row.name - tail + 1, 0):row.name + 1]
        target = table.tail(1)["pos"].iloc[0]
        origin = table.head(1)["pos"].iloc[0]

        v = Vector()
        row['angle'] = v.angle(a=[target[0], target[1]])
        row[Schema.DIST_COLUMN] = v.diff_norm(y=[target[0], target[1]], x=[origin[0], origin[1]])
        time_in_sec = (table.tail(1)["time"].iloc[0] - table.head(1)["time"].iloc[0]).total_seconds()
        row[Schema.SPEED_COLUMN] = row[Schema.DIST_COLUMN] / time_in_sec if time_in_sec > 0 else np.nan
        row[Schema.CUM_DIST_COLUMN] = 0
        return row

    def update_game(self, game):
        GAME_COORDINATES_FILE = f"results/{game}-coordinates.h5"
        GAME_ATTR_FILE = f"results/{game}_attr.h5"
        keys = Hdf5Utils.get_keys(path=GAME_COORDINATES_FILE)

        for key in keys:
            if self.metadata.is_a_goalkeeper(label=key):
                continue
            print(f'Updating {key}')
            df = pd.read_hdf(GAME_COORDINATES_FILE, key=key)
            df = df.aggregate(lambda row: self.__append(row=row, df=df, tail=5), axis=1)
            df[Schema.CUM_DIST_COLUMN] = df[Schema.DIST_COLUMN].cumsum()
            df.to_hdf(GAME_ATTR_FILE, key=key, index=False)
        return GAME_ATTR_FILE


class GameBuilder:

    def __init__(self, debug=False, display_changes=False):
        self.game = None
        self.player_rotations = None
        self.metadata = Metadata(players_metadata_loc="./data/players.csv")
        self.resources = Resources()
        self.player_attributes = [Schema.X_COLUMN, Schema.Y_COLUMN]

        self.DEBUG = debug
        self.DISPLAY_CHANGES = display_changes

    def build(self, game):
        self.game = game
        self.player_rotations = PlayerRotations(game=game)
        players_path = self.resources.get(game).players_in_game

        self.player_rotations.load_players_in_game_for(path=players_path)
        self.player_rotations.map_players()
        return self

    def update_attributes(self, path):
        self.player_rotations.update_attributes(path=path)
        return self

    def save_rotations(self):
        self.player_rotations.save_rotations()
        return self

    def save(self, path="full-game.h5"):
        self.player_rotations.save_full_game(path=path)

    def update_labels(self):
        game_labels_path = self.resources.get(self.game).game_labels
        game_df = pd.read_excel(game_labels_path)
        game_df = self.__game_columns_rename(game_df)

        game_passive_labels_path = self.resources.get(self.game).passive_labels
        passives_df = pd.read_excel(game_passive_labels_path)
        passives_df = self.__passives_columns_rename(df=passives_df)

        game_df = game_df.join(other=passives_df, on='possession')
        self.player_rotations.add_labels(game_df)

    def update_analysis(self):
        resource = self.resources.get(self.game)
        game_id = self.game[4:]
        df = pd.read_excel(io=resource.analysis, sheet_name=1)
        df = df[df['GAME'] == int(game_id)]
        columns = ["misses", "throws", "goal"]
        for column in columns:
            df[column] = df[column].fillna(0)
        df["organized_game"] = df["organized_game"].fillna("no")
        self.player_rotations.add_analysis(df)

    def __game_columns_rename(self, df):
        map = {
            'WHOLE GAME': 'WHOLE_GAME',
            'ANALISIS': 'analysis',
            'ATAC/DEF': 'game_phases',
            'P.E. INV.': 'score_team_a',
            'P.RIV': 'score_team_b',
            'DIF': 'score_diff',
            'POSESIÓN': 'possession',
            'JUG. E. INVEST': 'in_game_players_team_a',
            'JUG. E. RIVAL': 'in_game_players_team_b',
        }
        df = df.rename(columns=map)
        df["game_phases"] = df["game_phases"].str.strip()
        columns_to_drop = ['PARTIDO', 'PARTE', 'TJ CADA PARTE', 'TJ CRONO MARCHA', 'TJ EFECTIVO']
        df.drop(columns=columns_to_drop, inplace=True)
        df['WHOLE_GAME'] = pd.to_timedelta(df['WHOLE_GAME'], unit='s')
        df.set_index('WHOLE_GAME', inplace=True)
        return df

    def __passives_columns_rename(self, df):
        map = {
            'PARTIDO': 'game',
            'NÚMERO POSESION': 'possession',
            'AVISO DE PASIVO': 'passive_alert',
            'PASES DE FINALIZACIÓN': "number_of_passes",
            'LANZAMIENTO O PERDIDA': 'passive_alert_result'
        }
        df = df.rename(columns=map)
        columns_to_drop = ['game']
        df.drop(columns=columns_to_drop, inplace=True)
        df.set_index('possession', inplace=True)
        return df
