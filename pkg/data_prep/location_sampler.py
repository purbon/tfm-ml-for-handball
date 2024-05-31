import os
import re

import pandas as pd


def resample_coordinate_file(path, label):
    df = pd.read_excel(io=path)
    df['time'] = pd.to_datetime(df['TIME '], format="%H:%M:%S %f")
    df.rename(columns={df.columns[1]: label}, inplace=True)
    df.drop(columns=[df.columns[0], df.columns[2], df.columns[3]], axis=1, inplace=True)
    df = df.resample("1S", on="time").first().bfill()
    df = df.reset_index()
    return df


def resample_player(x_path, y_path):
    resampled_dx = resample_coordinate_file(path=x_path, label="x")
    resampled_dy = resample_coordinate_file(path=y_path, label="y")
    resampled_dy.rename(columns={resampled_dy.columns[0]: "time2"}, inplace=True)
    return resampled_dx, resampled_dy


def load(dir, dest_path, todo_players=[]):
    p = re.compile('(\\d+)(\\w+)\\s([\\w|\\d]+)\\s([X|Y]).xlsx')
    players = {}
    for file in os.listdir(dir):
        m = p.match(file)
        if m:
            id = m.group(1)
            player_code = f'{id}{m.group(2).strip()}'
            axe = m.group(4)
            if id not in players:
                players[id] = {'player_code': player_code}
            players[id][axe] = os.path.join(dir, file)

    for player_id, dataset_ref in players.items():
        player_code = dataset_ref['player_code']
        if len(todo_players) == 0 or player_code in todo_players:
            print(f"Processing {player_code}")
            resampled_dx, resampled_dy = resample_player(x_path=dataset_ref['X'], y_path=dataset_ref['Y'])
            df = pd.concat([resampled_dx, resampled_dy], axis=1, join="inner")
            df.drop(df.columns[[2]], axis=1, inplace=True)
            coord_dest = os.path.join(dest_path, f"{player_code}.xlsx")
            df.to_excel(coord_dest, index=False)


if __name__ == '__main__':
    # P15CRHE	P5LARO	P10ANHE	P1ALCA	P12VIJO	P16NAAR # P6
    players = ["18FRAG"]

    game_label = "game3"
    source_path = "./raw-data/P3/loc"
    dest_path = "./data/P3/loc"

    load(dir=source_path, dest_path=dest_path, todo_players=players)
