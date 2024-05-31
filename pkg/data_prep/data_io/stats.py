import json
from json import JSONEncoder

import numpy as np


class StatsEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.dtypes.Float64DType):
            return o.to_numeric()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o.__dict__


class Stats:

    def __init__(self):
        self.possessions = None
        self.possessions_time_in_seconds = None
        self.possession_score_team_a = None
        self.possession_score_team_b = None
        self.possession_score_team_diff = None
        self.game_phases = None
        self.analysis = None

    def as_df(self):
        pass

class StatsCollector:

    def __init__(self):
        pass

    @classmethod
    def describe(cls, df):
        stats = Stats()

        stats.possessions = df["possession"].nunique()
        stats.game_phases = df.groupby("game_phases")["game_phases"].count().to_dict()
        stats.analysis = df.groupby("analysis")["analysis"].count().to_dict()

        possession_time = df.groupby("possession").size().to_frame()
        possession_time.rename(columns={0: 'time_in_seconds'}, inplace=True)
        stats.possessions_time_in_seconds = cls.get_series_stats(possession_time["time_in_seconds"]) #possession_time.describe().to_dict()

        possession_data = (df.groupby("possession")
                           .last()[["WHOLE_GAME", "time_in_seconds", "score_team_a", "score_team_b", "score_diff"]]
                           .sort_values(by=['WHOLE_GAME']))

        stats.possession_score_team_a = cls.get_series_stats(possession_data["score_team_a"])
        stats.possession_score_team_b = cls.get_series_stats(possession_data["score_team_b"])
        stats.possession_score_team_diff = cls.get_series_stats(possession_data["score_diff"])

        json_str = json.dumps(stats, indent=4, cls=StatsEncoder)
        print(json_str)
        return stats

    @classmethod
    def get_series_stats(cls, series):
        base_stats = {
            'count': series.count().item(),
            'quantiles': series.quantile([.25, .5, .75]).to_dict(),
            'min': series.min().item(),
            'avg': series.mean().item(),
            'max': series.max().item(),
            'skew': series.skew().item(),
            'var': series.var().item(),
            'stddev': series.std().item()
        }
        return base_stats
