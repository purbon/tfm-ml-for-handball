import numpy as np
import pandas as pd

from data_io.meta import Schema
from handy.models.centroids import CentroidsModel
from handy.models.kinetics import Kinetics


def flush_dataset(aggregation_key):
    df = pd.read_hdf(path_or_buf=f"handball.h5", key="pos")
    df = df[df[Schema.ACTIVE_PLAYERS_COLUMN] == 6]

    new_df_columns = ["GAME", aggregation_key]
    extra_attributes = ["game_phases", "score_team_a", "score_team_b", "score_diff", "tactical_situation",
                        "time_in_seconds", "possession_result",
                        "passive_alert", "number_of_passes", "passive_alert_result",
                        "FK_count", "PEN_count", "TM_count", "Prev_TM",
                        "Post_TM", "offense_type", "misses", "throws", "goal", "throw_zone",
                        "organized_game", "sequences", "possession"]

    agg_fields = ["misses", "throws", "goal"]

    try:
        extra_attributes.remove(aggregation_key)
    except ValueError:
        pass

    for i in range(5):
        extra_attributes += [f"prev{i + 1}_possession_result"]
        extra_attributes += [f"prev{i + 1}_score_diff"]

    for i in range(6):
        new_df_columns.append(f"p{i}_x_centroid")
        new_df_columns.append(f"p{i}_y_centroid")
    new_df_columns += extra_attributes

    flatten_view_df = pd.DataFrame([], columns=new_df_columns)

    cm = CentroidsModel()
    km = Kinetics()
    for game, data_points in df.groupby(["GAME", aggregation_key]):
        print(f"Processing {game[0]} {game[1]}")
        first_record_in_series = data_points.head(1)
        last_record_in_series = data_points.tail(1)
        phase = first_record_in_series["game_phases"].values[0]

        cdf = cm.fit(possession_array=data_points)
        kdf = km.fit(poss_array=data_points)
        cdf = pd.concat((cdf, kdf), axis=1)

        cdf["GAME"] = game[0]
        cdf[aggregation_key] = game[1]

        for extra_attribute in extra_attributes:
            if extra_attribute == "game_phases":
                val = first_record_in_series[extra_attribute].iloc[0]
            else:
                if aggregation_key == "possession" and extra_attribute in agg_fields:
                    values_arr = data_points.groupby("sequence_label")[extra_attribute].unique().values
                    val = np.array([value_arr[0] for value_arr in values_arr]).sum()
                elif aggregation_key == "possession" and phase == "AT" and extra_attribute == "organized_game":
                    values_arr = data_points.groupby("sequence_label")[extra_attribute].unique().values
                    val_arr = np.array([value_arr[0] for value_arr in values_arr])
                    val = "yes" if "yes" in val_arr else "no"
                else:
                    val = last_record_in_series[extra_attribute].iloc[0]
            cdf[extra_attribute] = val

        flatten_view_df = pd.concat((flatten_view_df, cdf), ignore_index=True)
    string_fields = ["possession", "game_phases", "tactical_situation", "passive_alert_result",
                     "possession_result", "passive_alert",
                     "offense_type", "throw_zone", "organized_game", "sequence_label"]
    for i in range(5):
        string_fields += [f"prev{i + 1}_possession_result"]

    for column in flatten_view_df.columns:
        if column in string_fields:
            continue
            # df[column] = df[column].astype("string")
        else:
            flatten_view_df[column] = pd.to_numeric(flatten_view_df[column], errors='coerce')

    print("Saving....")
    fl_key = "dv0_p6" if aggregation_key == "possession" else "dv0_p6s"
    excel_file = "handball-vp6.xlsx" if aggregation_key == "possession" else "handball-vp6s.xlsx"
    flatten_view_df.to_hdf(path_or_buf="handball.h5", key=fl_key)
    flatten_view_df.to_excel(excel_file, index=False)
    print("done!")


if __name__ == '__main__':
    for aggregation_key in ["possession", "sequence_label"]:
        flush_dataset(aggregation_key=aggregation_key)
