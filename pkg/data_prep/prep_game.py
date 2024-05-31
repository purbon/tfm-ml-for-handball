import os

import pandas as pd

from data_io.file import Points
from data_io.transform import GameBuilder, PointExtension

pd.options.mode.chained_assignment = None


def process_game(game_label, source_path):
    os.makedirs("./results", exist_ok=True)
    coordinates_path = f"results/{game_label}-coordinates.h5"

    points_io = Points()
    points_io.load(dir=source_path)
    points_io.save_coordinates(path=coordinates_path)

    game_builder = (GameBuilder()
                    .build(game=game_label)
                    .update_attributes(path=coordinates_path))

    game_builder.save_rotations()
    game_builder.update_labels()
    game_builder.update_analysis()
    game_builder.save()


if __name__ == '__main__':
    available_game_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for game_id in available_game_ids:
        game_label = f"game{game_id}"
        source_path = f"./data/P{game_id}/tloc"
        process_game(game_label=game_label, source_path=source_path)
