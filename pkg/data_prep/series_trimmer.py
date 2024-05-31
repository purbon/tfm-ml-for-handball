import os
import re

import pandas as pd


def process_player(path, game_filter_table, player_code):
    df = pd.read_excel(io=path)
    start_point = game_filter_table.loc[player_code][0]
    return df[start_point:]


def load(base_dir, game_code, todo_players=[]):
    p = re.compile('(\\d+)(\\w+).xlsx')
    players = {}
    source_path = os.path.join(base_dir, "loc")
    target_path = os.path.join(base_dir, "tloc")
    for file in os.listdir(source_path):
        m = p.match(file)
        if m:
            id = m.group(1)
            player_code = f'P{id}{m.group(2).strip()}'
            if id not in players:
                players[id] = {'player_code': player_code}
            players[id]["path"] = os.path.join(source_path, file)
            players[id]["tpath"] = os.path.join(target_path, file)

    game_filter_df = pd.read_csv(f"{base_dir}/{game_code}-filter.csv")
    game_filter_df.set_index("player", inplace=True)
    for player_id, dataset_ref in players.items():
        player_code = dataset_ref['player_code']
        if len(todo_players) == 0 or player_code in todo_players:
            filtered_df = process_player(path=dataset_ref['path'],
                                         player_code=player_code,
                                         game_filter_table=game_filter_df)
            #tmp_path = f"{dataset_ref['path']}_tmp.xlsx"
            filtered_df.to_excel(dataset_ref['tpath'], index=False)


if __name__ == '__main__':
    game_label = "game5"
    source_path = "./data/P5"

    load(base_dir=source_path, game_code=game_label, todo_players=[])
