import os
import time

import numpy as np

from data_io.view import HandballPlot
from handy.datasets import centroid_handball_possession, ConfigFilter, flattened_handball_possessions

if __name__ == "__main__":

    target_attribute = "possession_result"
    phase = "AT"
    include_sequence = False
    class_action = "upsample"
    current_time = time.time()

    config_filter = ConfigFilter(_id=1,
                                 include_centroids=True,
                                 include_distance=True,
                                 include_sequences=True,
                                 include_prev_possession_results=True,
                                 include_breaks=True,
                                 include_acl=True,
                                 include_vel=True,
                                 include_faults=True,
                                 include_metadata=True,
                                 include_scores=True,
                                 include_prev_score_diff=True)

    action_type = "centroids"
    lse_size = 128
    timesteps = 30

    trajectory_charts = f"charts/examples/centroids"
    os.makedirs(trajectory_charts, exist_ok=True)

    data, classes, _ = centroid_handball_possession(target_attribute=target_attribute,
                                                    phase=phase,
                                                    include_sequences=False,
                                                    filter=config_filter)

    # p2_x_centroid p2_y_centroid [0..5]
    # team_x_centroid team_y_centroid

    for i in range(1, 11, 1):
        df = data.iloc[i]
        title = f"possession {i}"
        handball_plot = HandballPlot().handball_plot(title=title)
        for j in range(6):
            x = df[f"p{j}_x_centroid"]
            y = df[f"p{j}_y_centroid"]
            handball_plot.plot_positions(x=x, y=y, label=f"player_{j}")
        x = df[f"team_x_centroid"]
        y = df[f"team_y_centroid"]

        handball_plot.plot_positions(x=x, y=y, label=f"team_centroids")

        (handball_plot
         .add_legend()
         .save(chart_path=f"{trajectory_charts}/possession_{i}.png"))


    sample_size = 30
    trajectory_charts = f"charts/examples/flatten/{sample_size}"
    os.makedirs(trajectory_charts, exist_ok=True)

    data, classes, _ = flattened_handball_possessions(target_attribute=target_attribute,
                                                   length=sample_size,
                                                   phase=phase,
                                                   include_sequences=False,
                                                   filter=config_filter)


    # 'player_0_x0', 'player_0_y0', 'player_0_x1', 'player_0_y1'

    for i in range(1, 11, 1):
        df = data.iloc[i]
        title = f"possession {i}"
        handball_plot = HandballPlot().handball_plot(title=title)
        for j in range(6):
            xs = np.zeros(0)
            ys = np.zeros(0)
            for k in range(sample_size):
                x_val = df[f"player_{j}_x{k}"]
                y_val = df[f"player_{j}_y{k}"]
                xs = np.append(xs, x_val)
                ys = np.append(ys, y_val)
            handball_plot.add_trajectories(x=xs, y=ys, label=f"player_{i}")

        (handball_plot
         .add_legend()
         .save(chart_path=f"{trajectory_charts}/possession_{i}.png"))