import math
import os

import keras.utils
import numpy as np
import pandas as pd
import skimage
from PIL import Image
from imblearn.over_sampling import SMOTE
from pandas.core.dtypes.common import is_string_dtype
from skimage.color import rgb2gray
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from skimage import io

from data_io.meta import Schema
from data_io.view import HandballPlot, SimpleHandballPlot
from skimage.transform import resize


class EmbeddingsBuilder:

    def __init__(self):
        pass


# player_0_x	player_0_y	player_0_dist_last_5sec	player_0_cumdist	player_0_angle	player_0_speed_last_5sec

class DataGameGenerator(keras.utils.Sequence):

    def __init__(self, df,
                 timesteps=60,
                 batch_size=10,
                 start=0,
                 end=-1,
                 display_labels=False,
                 augment=False,
                 normalizer=False,
                 game_phase=None):
        self.batch_size = batch_size
        self.normalizer = normalizer
        self.timesteps = timesteps
        self.data = []
        self.origins = []
        self.labels = []
        self.truth = []
        self.filter = []
        for i in range(6):
            labels = [f"player_{i}_x",
                      f"player_{i}_y"]  # , f"player_{i}_dist_last_5sec", f"player_{i}_speed_last_5sec" ]
            for label in labels:
                self.filter.append(label)

        group_df = df.groupby(["GAME", "possession"])
        if game_phase is not None:
            end = -1
        for game, data_points in group_df:
            my_game_phase = data_points["game_phases"].head(1).iloc[0]
            if game_phase is not None and my_game_phase != game_phase:
                continue
            self.origins.append(f"{game[0]}_{game[1]}")
            self.labels.append(data_points["organized_game"].tail(1))
            possession = data_points[self.filter]
            np_data, np_truth = self.map_a_possession(possession=possession, timesteps=timesteps)
            self.data.append(np_data)
            self.truth.append(np_truth)
            if augment:
                augmented_possession = self.rotate_a_possession(possession=possession)
                np_data, np_truth = self.map_a_possession(possession=augmented_possession, timesteps=timesteps)
                self.data.append(np_data)
                self.truth.append(np_truth)

        self.truth = np.array(self.data[start:end])
        self.data = np.array(self.data[start:end])
        self.origins = np.array(self.origins[start:end])
        self.labels = np.array(self.labels)

        self.indices = np.arange(self.data.shape[0])

        if display_labels:
            for origin in self.origins:
                print(origin)
        self.numPossessions = len(self.data)
        self.on_epoch_end()

    def map_a_possession(self, possession, timesteps):
        np_array = possession.to_numpy()
        np_data = np_array  # [:-1]
        np_truth = np_array[-1:]
        if np_data.shape[0] < timesteps:
            diff = timesteps - np_data.shape[0]
            np_data = np.pad(np_data, ((0, diff), (0, 0)), 'constant', constant_values=(0,))
        elif np_data.shape[0] > timesteps:
            np_data = np_data[0:timesteps]
        return np_data, np_truth

    def rotate_a_possession(self, possession):
        rotated = possession.copy()
        original_columns = rotated.columns
        new_columns = []
        i = 0
        while i < len(original_columns):
            first_column = original_columns[i + 1]
            second_column = original_columns[i]
            new_columns.append(first_column)
            new_columns.append(second_column)
            i += 2
        df = rotated[new_columns]

        def column_rename(column_name):
            if column_name.endswith('_y'):
                return column_name.replace('_y', '_x')
            else:
                return column_name.replace('_x', '_y')

        df.rename(lambda x: column_rename(x), axis='columns', inplace=True)
        return df

    def __getitem__(self, index):
        # lower_bound = index * self.batch_size
        # upper_bound = (index + 1) * self.batch_size
        inds = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        x = self.data[inds].astype('float32')  # data
        y = self.truth[inds].astype('float32')  # ground truth
        if self.normalizer:
            x = (np.rint(x * 10_000)).astype('int')
            x = x.reshape((self.batch_size, self.timesteps * len(self.filter)))
        return x, y

    def __getorigins__(self, index):
        # lower_bound = index * self.batch_size
        # upper_bound = (index + 1) * self.batch_size
        inds = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        return self.origins[inds]

    def __getlabels__(self, index):
        inds = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        return self.labels[inds].flatten()

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(self.numPossessions / float(self.batch_size)))


def raw_handball_possessions(target_class,
                             timesteps,
                             game_phase=None,
                             augment=False,
                             normalizer=False,
                             population_field=None):
    df = pd.read_hdf(path_or_buf=f"handball.h5", key="pos")
    df = df[df[Schema.ACTIVE_PLAYERS_COLUMN] == 6]

    scaler = MinMaxScaler()
    label_encoder = LabelEncoder()

    data = []
    truth = []
    filter = []
    games = []
    sensitive_attribute = []
    for i in range(6):
        labels = [f"player_{i}_x",
                  f"player_{i}_y"]
        for label in labels:
            filter.append(label)

    df[filter] = scaler.fit_transform(df[filter])
    df[target_class] = label_encoder.fit_transform(df[target_class])

    def field_encode(df_value):
        if is_string_dtype(df_value):
            return label_encoder.fit_transform(df_value)
        return scaler.fit_transform(df_value)

    if population_field:
        df[population_field].fillna('', inplace=True)
    #    df[population_field] = field_encode(df_value=df[population_field])

    group_df = df.groupby(["GAME", "possession"])
    for game, data_points in group_df:
        my_game_phase = data_points["game_phases"].head(1).iloc[0]
        if game_phase is not None and my_game_phase != game_phase:
            continue
        possession = data_points[filter]
        games.append(data_points["GAME"].to_numpy()[-1])
        np_data, np_truth = __map_a_possession(possession=possession, timesteps=timesteps)
        data.append(np_data)
        truth.append(data_points[target_class].tail(1))
        if population_field:
            sensitive_attribute.append(data_points[population_field].tail(1))
        if augment:
            augmented_possession = __rotate_a_possession(possession=possession)
            np_data, np_truth = __map_a_possession(possession=augmented_possession, timesteps=timesteps)
            data.append(np_data)
            truth.append(data_points[target_class].tail(1))
            if population_field:
                sensitive_attribute.append(data_points[population_field].tail(1))
            games.append(data_points["GAME"].to_numpy()[-1])

    X, y, G = np.array(data), np.array(truth), np.array(games)
    if normalizer:
        X = (np.rint(X * 10_000)).astype('int')
        shape_x = X.shape[0]
        X = X.reshape((shape_x, timesteps * len(filter)))
        y = y.reshape((-1, 1))
        G = G.reshape((-1, 1))
    return X, y, G, np.array(sensitive_attribute)

def __map_a_possession(possession, timesteps):
    np_array = possession.to_numpy()
    np_data = np_array  # [:-1]
    np_truth = np_array[-1:]
    if np_data.shape[0] < timesteps:
        diff = timesteps - np_data.shape[0]
        np_data = np.pad(np_data, ((0, diff), (0, 0)), 'constant', constant_values=(0,))
    elif np_data.shape[0] > timesteps:
        np_data = np_data[0:timesteps]
    return np_data, np_truth


def __rotate_a_possession(possession):
    rotated = possession.copy()
    original_columns = rotated.columns
    new_columns = []
    i = 0
    while i < len(original_columns):
        first_column = original_columns[i + 1]
        second_column = original_columns[i]
        new_columns.append(first_column)
        new_columns.append(second_column)
        i += 2
    return rotated[new_columns]


class GameDataGeneratorV2(keras.utils.Sequence):

    def __init__(self, df,
                 timesteps=60,
                 batch_size=10,
                 start=0,
                 end=-1,
                 display_labels=False,
                 augment=False,
                 normalizer=False,
                 game_phase=None,
                 target_class="",
                 resample=False):
        self.batch_size = batch_size
        self.normalizer = normalizer
        self.timesteps = timesteps
        self.target_class = target_class
        self.data = []
        self.origins = []
        self.labels = []
        self.truth = []
        self.filter = []
        self.resample = resample
        for i in range(6):
            labels = [f"player_{i}_x",
                      f"player_{i}_y"]
            for label in labels:
                self.filter.append(label)

        group_df = df.groupby(["GAME", "possession"])
        if game_phase is not None:
            end = -1
        for game, data_points in group_df:
            my_game_phase = data_points["game_phases"].head(1).iloc[0]
            if game_phase is not None and my_game_phase != game_phase:
                continue
            self.origins.append(f"{game[0]}_{game[1]}")
            self.labels.append(data_points["organized_game"].tail(1))
            possession = data_points[self.filter]
            np_data, np_truth = self.map_a_possession(possession=possession, timesteps=timesteps)
            self.data.append(np_data)
            self.truth.append(data_points[self.target_class].tail(1))
            if augment:
                augmented_possession = self.rotate_a_possession(possession=possession)
                np_data, np_truth = self.map_a_possession(possession=augmented_possession, timesteps=timesteps)
                self.data.append(np_data)
                self.truth.append(data_points[self.target_class].tail(1))

        self.truth = np.array(self.truth[start:end])
        self.data = np.array(self.data[start:end])
        self.origins = np.array(self.origins[start:end])
        self.labels = np.array(self.labels)

        self.indices = np.arange(self.data.shape[0])

        if display_labels:
            for origin in self.origins:
                print(origin)
        self.numPossessions = len(self.data)
        self.on_epoch_end()

    def map_a_possession(self, possession, timesteps):
        np_array = possession.to_numpy()
        np_data = np_array  # [:-1]
        np_truth = np_array[-1:]
        if np_data.shape[0] < timesteps:
            diff = timesteps - np_data.shape[0]
            np_data = np.pad(np_data, ((0, diff), (0, 0)), 'constant', constant_values=(0,))
        elif np_data.shape[0] > timesteps:
            np_data = np_data[0:timesteps]
        return np_data, np_truth

    def rotate_a_possession(self, possession):
        rotated = possession.copy()
        original_columns = rotated.columns
        new_columns = []
        i = 0
        while i < len(original_columns):
            first_column = original_columns[i + 1]
            second_column = original_columns[i]
            new_columns.append(first_column)
            new_columns.append(second_column)
            i += 2
        df = rotated[new_columns]

        def column_rename(column_name):
            if column_name.endswith('_y'):
                return column_name.replace('_y', '_x')
            else:
                return column_name.replace('_x', '_y')

        df.rename(lambda x: column_rename(x), axis='columns', inplace=True)
        return df

    def __getitem__(self, index):
        lower_bound = index * self.batch_size
        upper_bound = min((index + 1) * self.batch_size, self.indices.size)
        inds = self.indices[lower_bound: upper_bound]
        x = self.data[inds].astype('float32')  # data
        y = self.truth[inds].astype('float32')  # ground truth
        if self.normalizer:
            x = (np.rint(x * 10_000)).astype('int')
            shape_x = x.shape[0]
            x = x.reshape((shape_x, self.timesteps * len(self.filter)))
            y = y.reshape((-1, 1))
        return x, y

    def __getorigins__(self, index):
        # lower_bound = index * self.batch_size
        # upper_bound = (index + 1) * self.batch_size
        inds = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        return self.origins[inds]

    def __getlabels__(self, index):
        inds = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        return self.labels[inds].flatten()

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(self.numPossessions / float(self.batch_size)))


class PossessionImageGenerator(keras.utils.Sequence):

    def __init__(self, df, batch_size=10, start=0, end=-1):
        self.batch_size = batch_size
        self.data = []
        self.origins = []
        self.labels = []
        self.truth = []
        self.filter = []

        self.scaler = MinMaxScaler()
        # df[fields] = scaler.fit_transform(df[fields])

        for i in range(6):
            for label in [f"player_{i}_x", f"player_{i}_y"]:
                self.filter.append(label)
        i = 0
        for game, data_points in df:
            if start <= i < end:
                origins_label = f"{game[0]}_{game[1]}"
                self.origins.append(origins_label)
                self.labels.append(data_points["game_phases"].head(1))
                possession = data_points[self.filter]
                np_array = possession.to_numpy()
                poss_path = plot_a_possession(X=np_array, label=origins_label, override=True)
                img_bytes = read_img(img_path=poss_path, should_resize=False, make_gray=False)
                img_bytes = img_bytes / 255.
                self.data.append(img_bytes)
                self.truth.append(img_bytes)
            i += 1

        self.truth = np.array(self.truth)
        self.data = np.array(self.data)
        self.origins = np.array(self.origins)
        self.labels = np.array(self.labels)

        self.indices = np.arange(self.data.shape[0])
        self.numPossessions = len(self.data)
        self.on_epoch_end()

    def __getitem__(self, index):
        inds = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        x = self.data[inds]  # data
        y = self.truth[inds]  # ground truth
        return x, y

    def __getorigins__(self, index):
        # lower_bound = index * self.batch_size
        # upper_bound = (index + 1) * self.batch_size
        inds = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        return self.origins[inds]

    def __getlabels__(self, index):
        inds = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        return self.labels[inds].flatten()

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(self.numPossessions / float(self.batch_size)))


def plot_a_possession(X, label, outdir="/tmp", override=False):
    handball_plot = SimpleHandballPlot().handball_plot(title=f"")

    for player_id in range(6):
        xs = X[:, 2 * player_id]
        ys = X[:, 2 * player_id + 1]
        # xs = np.array([x if x <= 20 else x - 20 for x in X[:, 2 * player_id]])
        # ys = np.array([y if y <= 20 else y - 20 for y in X[:, 2 * player_id + 1]])
        player_label = f"p{player_id}"
        handball_plot.add_trajectories(x=xs, y=ys, label=player_label)

    # handball_plot.add_legend()
    chart_path = os.path.join(f"{outdir}", f'{label}.png')
    if override or not os.path.exists(chart_path):
        handball_plot.save(chart_path=chart_path)
    return chart_path


def read_img(img_path, should_resize=True, make_gray=False):
    img = io.imread(img_path)
    # img = skimage.color.rgba2rgb(img_rgba)
    if make_gray:
        img = rgb2gray(img)
    if should_resize:
        img = resize(img, (200, 400), anti_aliasing=True)
    return img
