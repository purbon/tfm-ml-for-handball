import os

import pandas as pd

from data_io.file import Points
from data_io.view import HandballPlot


def plot_trajectory(start_point, end_point, player, points_io):
    title = f'start {start_point}'
    handball_plot = HandballPlot().handball_plot(title=title)

    pc = points_io.coordinates[player]
    xs = pc["x"][start_point:end_point]
    ys = pc["y"][start_point:end_point]

    handball_plot.add_trajectories(x=xs, y=ys, label=player)
    handball_plot.plot_positions(x=xs[end_point - 1], y=ys[end_point - 1])
    handball_plot.plot_positions(x=xs[end_point - 2], y=ys[end_point - 2])
    handball_plot.plot_positions(x=xs[end_point - 3], y=ys[end_point - 3])

    (handball_plot
     .add_legend()
     .save(chart_path=f"{trajectory_charts}/sync_{player}-{start_point}-{end_point}.png"))
    if should_append:
        data = {"player": player, "start_point": start_point}
        df = pd.DataFrame()
        df = df.append(data, ignore_index=True)
        df.to_csv(f'{source_path}/{game_label}-filter.csv', mode='a', index=False, header=False)

    print(start_point)
    pc = points_io.coordinates[player]
    time = pc["time"][start_point]
    print(f'{player}: {time}')


def plot_trajectories(points_io, start_point, interval, increment, player, longitude=None):
    if longitude is None:
        longitude = increment
    for i in range(start_point, interval, increment):
        start_point = i
        end_point = start_point + longitude  # 17
        plot_trajectory(start_point=start_point, end_point=end_point, points_io=points_io, player=player)


should_append = False

if __name__ == '__main__':
    player = "P9JOSA"
    game_label = "game10"
    source_path = "./data/P10"

    points_io = Points()
    dir = os.path.join(source_path, "loc")
    points_io.load(dir=dir, todo_players=[player])

    trajectory_charts = f"charts/{game_label}/sync"
    os.makedirs(trajectory_charts, exist_ok=True)

    plot_trajectories(points_io=points_io,
                      start_point=2310, interval=2330, increment=1, longitude=15,
                      player=player)
