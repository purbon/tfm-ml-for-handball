import os
import re

import numpy as np
import pandas as pd

from data_io.meta import Resources, Metadata, Schema
from data_io.utils import Hdf5Utils


class Points:

    def __init__(self, debug=False):
        self.coordinates = None
        self.debug = debug

    def process_player(self, path, player_code, start_point=[1.5, 1.5]):
        print(f'Processing player {player_code}')

        df = pd.read_excel(io=path)
        df["x"] = df["x"] - start_point[0]
        df["y"] = df["y"] - start_point[1]
        df["pos"] = list(zip(df["x"], df["y"]))

        game_start_time = df.iloc[0]
        df["time"] = (df["time"] - game_start_time["time"])
        for field in ["x", "y"]:
            df[field] = pd.to_numeric(df[field], errors='coerce')

        print(df.describe(include='all'))
        return df

    def load(self, dir, todo_players=[]):
        p = re.compile('(\\d+)(\\w+).xlsx')
        players = {}
        for file in os.listdir(dir):
            m = p.match(file)
            if m:
                id = m.group(1)
                player_code = f'P{id}{m.group(2).strip()}'
                if id not in players:
                    players[id] = {'player_code': player_code}
                players[id]["path"] = os.path.join(dir, file)
        self.coordinates = {}

        for player_id, dataset_ref in players.items():
            player_code = dataset_ref['player_code']
            if len(todo_players) == 0 or player_code in todo_players:
                self.coordinates[player_code] = self.process_player(path=dataset_ref['path'],
                                                                    player_code=player_code)

    def save_coordinates(self, path):
        for player_code, coordinates in self.coordinates.items():
            coordinates.to_hdf(path, key=player_code, index=False)
            if self.debug:
                coordinates.to_excel(f'{player_code}.xlsx', index=False)


class PlayerRotations:

    def __init__(self, game, debug=False, players_metadata_loc="./data/players.csv"):
        self.keys = None
        self.metadata = Metadata(players_metadata_loc=players_metadata_loc)
        self.resources = Resources()
        self.debug = debug
        self.game = game
        self.rotations = None

    def load_players_in_game_for(self, path):
        self.rotations = pd.read_excel(io=path)
        self.__players_in_game_column_rename()

        for player_id in range(6):
            self.rotations[f"player_{player_id}"] = ""
        self.rotations[Schema.ACTIVE_PLAYERS_COLUMN] = 0
        self.rotations[Schema.ACTIVE_SENSORS_COLUMN] = 0
        self.keys = [code for code in self.rotations.columns if code.startswith("P")]
        return self.rotations

    def update_attributes(self, path):
        for player_id in range(6):
            for attribute in self.resources.player_attributes:
                column_name = f"player_{player_id}_{attribute}"
                self.rotations[column_name] = ""

        df_index = {}
        for key in self.keys:
            try:
                player_df = pd.read_hdf(path_or_buf=path, key=key)
                player_df.index = player_df['time']
                df_index[key] = player_df
            except:
                df_index[key] = None

        self.rotations = self.rotations.apply(lambda row: self.map_attributes(row, df_index), axis=1)
        self.rotations.set_index("WHOLE_GAME", inplace=True)

    def add_labels(self, game_df):
        df = self.rotations.join(other=game_df)
        df['tactical_situation'] = df.apply(lambda r: self.__update_tactical_situation(r), axis=1)

        phase_stats_df = self.__possession_phase_stats(df=df)

        possession_time = df.groupby("possession").size().to_frame()
        possession_time.rename(columns={0: 'time_in_seconds'}, inplace=True)
        df = df.join(other=possession_time, on="possession")
        df = df.join(other=phase_stats_df, on="possession")

        df['live_possession_duration_in_sec'] = np.nan
        self.rotations = self.__add_possession_duration_live(df)

        self.rotations = self.__add_game_segmentation(df=df)

        df.reset_index(inplace=True)
        df['possession_result'] = df.apply(lambda r: self.__update_possession_result(current=r, df=df), axis=1)
        df = self.__add_prev_poss_status(df=df)
        df = self.__add_prev_poss_status(df=df, retrieved_key="score_diff", default_val=np.nan)

        df.set_index('WHOLE_GAME', inplace=True)

    def __add_prev_poss_status(self, df, retrieved_key="possession_result", default_val=""):
        first_poss_df = df.groupby("possession", as_index=True).head(n=1)
        last_poss_df = df.groupby("possession", as_index=True).tail(n=1)

        fdf = first_poss_df.apply(lambda x: self.__group_entries_for(x=x), axis=1, result_type="reduce")
        edf = last_poss_df.apply(lambda x: self.__group_entries_for(x=x, is_first=False), axis=1, result_type="reduce")
        fdf.set_index("possession", inplace=True)
        edf.set_index("possession", inplace=True)

        mdf = fdf.merge(edf, on="possession")
        mdf.drop(columns=["end_x", "start_y"], inplace=True)
        mdf.rename(columns={"start_x": "start", "end_y": "end"}, inplace=True)

        last_poss_df.set_index("possession", inplace=True)

        for index, row in df.iterrows():
            current_possession = row["possession"]
            if isinstance(current_possession, str) and current_possession != "":
                possession_index = mdf.index.get_loc(current_possession)
                for i in range(5):
                    i_limit = i + 1
                    prev_possession = None if possession_index < i_limit else mdf.index.tolist()[
                        possession_index - i_limit]
                    if prev_possession is not None:
                        prev_val = last_poss_df.loc[prev_possession][retrieved_key]
                    else:
                        prev_val = default_val
                    df.at[index, f"prev{i_limit}_{retrieved_key}"] = prev_val
        return df

    def __group_entries_for(self, x, is_first=True):
        start_value = x.name if is_first else 0
        end_value = 0 if is_first else x.name
        d = {"possession": x["possession"], "start": start_value, "end": end_value}
        return pd.Series(data=d)

    def add_analysis(self, analysis_df):
        self.rotations = pd.merge(self.rotations, analysis_df, on=["GAME", "possession", "sequence_label"], how="left")
        self.rotations.reset_index(names="WHOLE_GAME", inplace=True)

    def map_attributes(self, row, df_index):
        timedelta = row['WHOLE_GAME']
        detected_sensors = 0
        for player_id in range(6):
            player_column_name = f"player_{player_id}"
            player_label = row[player_column_name]
            if not "".__eq__(player_label):
                df = df_index[player_label]
                if not df is None:
                    try:
                        record = df.loc[timedelta]
                        detected_sensors += 1
                        for attribute in self.resources.player_attributes:
                            column_name = f"player_{player_id}_{attribute}"
                            row[column_name] = record[attribute]
                    except:
                        print(f"Missing timestamp: {timedelta}")
        row[Schema.ACTIVE_SENSORS_COLUMN] = detected_sensors
        return row

    def __update_tactical_situation(self, r):
        team_a_players = int(r['in_game_players_team_a'].strip() or 0) if type(r['in_game_players_team_a']) == str else \
            r['in_game_players_team_a']
        team_b_players = int(r['in_game_players_team_b'].strip() or 0) if type(r['in_game_players_team_b']) == str else \
            r[
                'in_game_players_team_b']
        if team_a_players > team_b_players:
            return "superiority"
        elif team_a_players < team_b_players:
            return "inferiority"
        else:
            return "equal"

    def __possession_phase_stats(self, df):
        df = df[["game_phases", "possession"]]

        poss_group = df.groupby("possession")

        keys = list(poss_group.indices)
        stats = {}
        for h in keys:
            v = poss_group.indices[h]
            low_index, high_index = self.__find_possession_index(df=df, groups=poss_group, key=h)
            sdf = df.iloc[low_index:high_index]
            # {'GF': 0, 'PEN': 0}
            attrs = ["GF", "PEN", "TM", "PREV_TM", "POST_TM"]
            counts = [0, 0, 0, 0, 0]
            active = True
            pre_tm = self.__check_for_tm(df=df, base_index=v[0], diff=-1)
            counts[3] = 1 if pre_tm == 1 else 0
            post_tm = self.__check_for_tm(df=df, base_index=v[-1], diff=1)
            counts[4] = 1 if post_tm == 1 else 0

            for i in range(len(sdf)):
                game_phase = sdf.iloc[i, 0]
                if active and game_phase in attrs:
                    index = attrs.index(game_phase)
                    counts[index] += 1
                    active = False
                elif not active and (game_phase == "DEF" or game_phase == "AT"):
                    active = True
            stats[h] = counts
        df_columns = ["FK_count", "PEN_count", "TM_count", "Prev_TM", "Post_TM"]
        return pd.DataFrame.from_dict(data=stats, orient='index', columns=df_columns)

    def __add_game_segmentation(self, df):
        offense_possession_id = 0
        defense_possession_id = 0
        last_game_phase = ""
        las_analysis = ""
        sequence_id = 1
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
                    sequence_id = 1
                elif last_game_phase == current_phase:
                    # staying in the same game_phase, but there could have been breaks in between
                    pass
                last_game_phase = current_phase
                df.loc[index, 'sequence_label'] = f"{current_possession}{sequence_id}"
                df.loc[index, 'sequences'] = sequence_id
            elif current_analysis == "JP":  # Game is stop
                if last_analysis != "JP":
                    sequence_id += 1
            if self.debug:
                print(
                    f"{current_analysis} {current_phase} {defense_possession_id} {offense_possession_id} {sequence_id}")
            last_analysis = current_analysis
        return df

    def __check_for_tm(self, df, base_index, diff):
        try:
            pre_row = df.iloc[base_index + diff]
            return 1 if pre_row["game_phases"] == "TM" else 0
        except:
            return -1

    def __find_possession_index(self, df, groups, key):
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

    # game_phases score_diff
    def __update_possession_result(self, current, df):
        last = df.iloc[max(current.name - 1, 0)]
        last_score_diff = last['score_diff']
        current_score_diff = current['score_diff']
        try:
            game_phase = current['game_phases'].strip()
            if game_phase == 'DEF':
                return 'TRUE' if last_score_diff == current_score_diff else 'FALSE'
            elif game_phase == 'AT':
                return 'TRUE' if current_score_diff > last_score_diff else 'FALSE'
            else:
                return 'unknown'
        except Exception as e:
            return 'unknown'

    def __add_possession_duration_live(self, df):
        current_poss = ""
        counter = 0
        for index, row in df.iterrows():
            label = row['possession']

            if pd.isna(label):
                continue
            else:
                if label == current_poss:
                    counter += 1
                else:
                    counter = 1
                df.at[index, "live_possession_duration_in_sec"] = counter
                current_poss = label
        return df

    def __players_in_game_column_rename(self):
        map = self.metadata.codes[["Label", "Code"]]
        map = map.set_index("Label")
        map = {k: f"P{v['Code']}" for k, v in map.to_dict(orient='index').items()}

        map['PARTIDO'] = "GAME"
        map['PARTE'] = 'HALF'
        map['WHOLE GAME'] = 'WHOLE_GAME'
        map['TJ CADA PARTE'] = "TIME_PER_HALF_IN_SECONDS"
        map['TJ CRONO MARCHA'] = "STOPWATCH_TIME_IN_SECONDS"
        map['TJ EFECTIVO'] = "EFFECTIVE_TIME_IN_SECONDS"
        self.rotations.rename(columns=map, inplace=True)

        self.rotations['WHOLE_GAME'] = pd.to_timedelta(self.rotations['WHOLE_GAME'], unit='s')

    def __select_players(self, current, last, keys):
        current_active_players = self.active_players_filter(current[keys])
        last_active_players = list(last.loc["player_0":"player_5"])
        new_players = []
        for player_id in range(len(current_active_players)):
            current_player = current_active_players[player_id]
            last_player = ""
            try:
                last_player = last_active_players[player_id]
            except IndexError as e:
                continue
            if current_player == last_player:
                current[f"player_{player_id}"] = current_player
            else:
                try:
                    last_index = last_active_players.index(current_player)
                    current[f"player_{last_index}"] = current_player
                except ValueError as ve:
                    # element was not found in the previous list, so it is a new player.
                    # we should assign them later on to the empty slots.
                    new_players.append(current_player)
                    # current[f"player_{player_id}"] = ""
        if len(new_players) > 0:
            i = 0
            for player_id in range(6):
                if i < len(new_players) and current[f"player_{player_id}"] == "":
                    current[f"player_{player_id}"] = new_players[i]
                    i = i + 1
        current[Schema.ACTIVE_PLAYERS_COLUMN] = len(current_active_players)
        print(list(current)) if self.debug else None
        return current

    def active_players_filter(self, current):
        current = current.loc[lambda x: x == 1.0]
        if current.empty:
            return set()
        else:
            squad = current.index.tolist()
            squad = [player for player in squad if not self.metadata.is_a_goalkeeper(label=player)]
            # squad.sort(key=self.__get_player_id)
            return squad

    def __get_player_id(self, player):
        p = re.compile('P(\\d+)(\\w+)')
        m = p.match(player)
        if m:
            player_id = m.group(1)
            return int(player_id)
        return 0

    # TODO: should handle changes in structure and keep previous order
    ## (1)  P5LARO	P2PALU	P6JUOL	P13ELCU	P15CRHE	P10ANHE
    ##      P5LARO	P6JUOL	P15CRHE	P3PAGA	P4ALSE	P13ELCU
    ## should be
    ## (1)  P5LARO	P2PALU	P6JUOL	P13ELCU	P15CRHE	P10ANHE
    ##      P5LARO	P3PAGA	P6JUOL	P13ELCU	P15CRHE	P4ALSE
    def map_players(self):
        coordinates_path = f"results/{self.game}-coordinates.h5"
        if self.debug:
            keys2 = Hdf5Utils.get_keys(path=coordinates_path)
            diff = list(set(self.keys) - set(keys2))
            print(diff)
            diff2 = list(set(keys2) - set(self.keys))
            print(diff2)

        for i in range(len(self.rotations)):
            current = self.rotations.iloc[i]
            last = self.rotations.iloc[max(current.name - 1, 0)]
            self.rotations.iloc[i] = self.__select_players(current=current, last=last, keys=self.keys)
        self.rotations.drop(columns=self.keys, inplace=True)

    def save_rotations(self):
        game_partial_enrich_path = self.resources.get(self.game).enrich_results
        self.rotations.reset_index().to_excel(f"{self.game}.xlsx")
        self.rotations.to_csv(path_or_buf=game_partial_enrich_path)

    def save_full_game(self, path="full-game.h5"):
        self.rotations.reset_index().to_excel(f"full-{self.game}.xlsx")
        self.rotations.to_hdf(path, key=self.game)
