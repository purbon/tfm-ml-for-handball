import pandas as pd


class ConfigFilter:
    def __init__(self,
                 _id=0,
                 include_centroids=False,
                 include_distance=False,
                 include_vel=False,
                 include_acl=False,
                 include_metadata=False,
                 include_faults=False,
                 include_breaks=False,
                 include_sequences=False,
                 include_scores=False,
                 include_prev_possession_results=False,
                 include_prev_score_diff=False
                 ):
        self.id = _id
        self.include_centroids = include_centroids
        self.include_distance = include_distance
        self.include_vel = include_vel
        self.include_acl = include_acl
        self.include_metadata = include_metadata
        self.include_faults = include_faults
        self.include_breaks = include_breaks
        self.include_sequences = include_sequences
        self.include_scores = include_scores
        self.include_prev_possession_results = include_prev_possession_results
        self.include_prev_score_diff = include_prev_score_diff

    def to_dict(self):
        data = {
            "id": self.id,
            "include_centroids": self.include_centroids,
            "include_distance": self.include_distance,
            "include_vel": self.include_vel,
            "include_acl": self.include_acl,
            "include_metadata": self.include_metadata,
            "include_faults": self.include_faults,
            "include_breaks": self.include_breaks,
            "include_sequences": self.include_sequences,
            "include_scores": self.include_scores,
            "include_prev_possession_results": self.include_prev_possession_results,
            "include_prev_score_diff": self.include_prev_score_diff
        }
        return data


class KerasConfig:

    def __init__(self,
                 id,
                 timesteps,
                 batch_size,
                 epochs,
                 kfold_strategy,
                 n_splits,
                 do_augment,
                 class_action="none"):
        self.id = id
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.epochs = epochs
        self.kfold_strategy = kfold_strategy
        self.n_splits = n_splits
        self.do_augment = do_augment
        self.class_action = class_action

    def to_dict(self):
        data = {
            "id": self.id,
            "timesteps": self.timesteps,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "kfoldstrategy": self.kfold_strategy,
            "n_splits": self.n_splits,
            "do_augment": self.do_augment,
            "class_action": self.class_action
        }
        return data

    def __str__(self):
        return str(self.to_dict())

def flattened_handball_possessions(data_home=f"handball.h5",
                                   n_players=6,
                                   target_attribute=None,
                                   phase=None,
                                   include_sequences=False,
                                   length=20,
                                   filter=ConfigFilter(),
                                   game_column="GAME"):
    key = f"dv0_p6s-norm{length}" if include_sequences else f"dv0_p6-norm{length}"
    df = pd.read_hdf(path_or_buf=data_home, key=key)
    if phase is not None:
        df = df[df["game_phases"] == phase]

    if target_attribute is None:
        return df, None, None
    else:
        columns = []
        for column in df.columns:
            if column.startswith('player_'):
                columns.append(column)
        filter.include_centroids = False
        columns += __cget_classification_data(filter=filter, n_players=0)
        try:
            columns.remove(target_attribute)
        except ValueError:
            pass
        G = df[game_column]
        return df[columns], df[target_attribute], G


def centroid_handball_possession(data_home=f"handball.h5",
                                 n_players=6,
                                 include_sequences=False,
                                 target_attribute=None,
                                 phase=None,
                                 filter=ConfigFilter(),
                                 game_column="GAME"):
    key = "dv0_p6s" if include_sequences else "dv0_p6"
    df = pd.read_hdf(path_or_buf=data_home, key=key)
    if phase == "AT" and "offense_type" in df.columns:
        df.dropna(subset="offense_type", inplace=True)

    if phase is not None:
        df = df[df["game_phases"] == phase]
    if target_attribute is None:
        return df, None, None
    else:
        y = df[target_attribute]
        x_columns = __cget_classification_data(n_players=n_players, filter=filter)
        try:
            x_columns.remove(target_attribute)
        except ValueError:
            pass
        X = df[x_columns]
        G = df[game_column]
        return X, y, G


def latent_space_encoded_possession(method_type, lse_size,
                                    data_home=f"handball.h5",
                                    target_attribute=None,
                                    phase=None,
                                    timesteps=25,
                                    filter=ConfigFilter(),
                                    game_column="GAME"):
    key_label = f"{method_type}_autoenc_{lse_size}-{timesteps}"
    df = pd.read_hdf(path_or_buf=data_home, key=key_label)
    if phase == "AT" and "offense_type" in df.columns:
        df.dropna(subset="offense_type", inplace=True)

    if phase is not None:
        df = df[df["game_phases"] == phase]
    if target_attribute is None:
        return df, None, None
    else:
        y = df[target_attribute]
        filter.include_centroids = False
        filter.include_distance = False
        filter.include_vel = False
        filter.include_acl = False
        x_columns = __cget_classification_data(n_players=0, filter=filter)

        try:
            x_columns.remove(target_attribute)
        except ValueError:
            pass
        lse_columns = [i for i in range(lse_size)]
        x_columns += lse_columns

        X = df[x_columns]
        X.columns = X.columns.astype(str)
        G = df[game_column]
        return X, y, G


def __get_classification_data(include_centroids=True, include_distance=False,
                              include_vel=False, include_acl=False, n_players=6,
                              include_metadata=False, include_faults=False):
    configFilter = ConfigFilter()
    configFilter.include_centroids = include_centroids
    configFilter.include_distance = include_distance
    configFilter.include_vel = include_vel
    configFilter.include_acl = include_acl
    configFilter.include_metadata = include_metadata
    configFilter.include_faults = include_faults
    return __cget_classification_data(filter=configFilter, n_players=n_players)


def __cget_classification_data(filter=ConfigFilter(), n_players=6):
    columns = []
    for i in range(n_players):
        if filter.include_centroids:
            columns += [f"p{i}_x_centroid", f"p{i}_y_centroid"]
        if filter.include_distance:
            columns += [f"p{i}_dist_to_center", f"p{i}_dist"]
        if filter.include_vel:
            columns += [f"p{i}_avg_vel", f"p{i}_p90_vel"]
        if filter.include_acl:
            columns += [f"p{i}_avg_acc", f"p{i}_p90_acc"]
    if filter.include_scores:
        columns += ["score_diff"]
    if filter.include_faults:
        columns += ["FK_count", "PEN_count"]
    if filter.include_metadata:
        columns += ["game_phases", "passive_alert", "throw_zone", "tactical_situation", "misses", "possession_result"]
    if filter.include_centroids:
        columns += ["team_x_centroid", "team_y_centroid"]
    if filter.include_breaks:
        columns += ["FK_count", "PEN_count", "TM_count"]
    if filter.include_sequences:
        columns += ["sequences"]
    if filter.include_prev_possession_results:
        for i in range(5):
            columns += [f"prev{i + 1}_possession_result"]
    if filter.include_prev_score_diff:
        for i in range(2):
            columns += [f"prev{i + 1}_score_diff"]
    return list(set(columns))
