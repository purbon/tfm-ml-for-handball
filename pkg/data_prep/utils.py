import os
import re

import pandas as pd


def parse_player(path):
    df = pd.read_excel(io=path)
    return df.loc[0]["time"]


def load(base_dir, game_code):
    p = re.compile('(\\d+)(\\w+).xlsx')
    players = {}
    source_path = os.path.join(base_dir, "tloc")
    for file in os.listdir(source_path):
        m = p.match(file)
        if m:
            id = m.group(1)
            player_code = f'P{id}{m.group(2).strip()}'
            if id not in players:
                players[id] = {'player_code': player_code}
            players[id]["path"] = os.path.join(source_path, file)

    df = pd.DataFrame()
    for player_id, dataset_ref in players.items():
        player_code = dataset_ref['player_code']
        start_time = parse_player(path=dataset_ref['path'])
        new_row = {'gameID': game_code, 'player': player_code, 'start_time': start_time}
        df = df.append(new_row, ignore_index=True)
    return df


if __name__ == '__main__':

    df = pd.DataFrame()
    for game_id in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        game_label = f"game{game_id}"
        print(game_label)
        source_path = f"./data/P{game_id}"
        gdf = load(base_dir=source_path, game_code=game_label)
        df = pd.concat([df, gdf])
    df.to_excel("game_start_times.xlsx")