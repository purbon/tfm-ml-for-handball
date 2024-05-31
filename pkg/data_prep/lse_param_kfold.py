import math
import os

import numpy as np
import pandas as pd
from keras_tuner import HyperParameters, RandomSearch, GridSearch, BayesianOptimization
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold, KFold, ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from create_conv_encoding import save_trained_model, load_trained_model
from data_io.meta import Schema
from data_io.view import HandballPlot
from data_pre.embeddings import DataGameGenerator, PossessionImageGenerator
from handy.encodings.auto_encoder import LSTMLatentSpace, ConvolutionalLatentSpace


def plot_a_possession(X, label):
    handball_plot = HandballPlot().handball_plot(title="P0")

    for player_id in range(6):
        xs = X[:, 2 * player_id]
        ys = X[:, 2 * player_id + 1]
        player_label = f"p{player_id}"
        handball_plot.add_trajectories(x=xs, y=ys, label=player_label)

    handball_plot.add_legend()
    trajectory_charts = f"charts/test_embeddings"
    chart_path = os.path.join(trajectory_charts, f'{label}.png')
    handball_plot.save(chart_path=chart_path)


if __name__ == '__main__':
    df = pd.read_hdf(path_or_buf=f"handball.h5", key="pos")

    df.drop(columns=["player_0", "player_1", "player_2", "player_3", "player_4", "player_5"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = df[df[Schema.ACTIVE_PLAYERS_COLUMN] == 6]

    timesteps = 20
    n_features = 12

    create_plots = False
    debug = False
    dump = True
    plot = False
    do_data_augmentation = False
    model = "conv"

    total_size = df.groupby(["GAME", "possession"]).count().shape[0]

    trainSize = int(math.ceil(0.7 * total_size * 2)) if do_data_augmentation else int(math.ceil(0.7 * total_size))
    testSize = int(math.ceil(0.2 * total_size * 2)) if do_data_augmentation else int(math.ceil(0.2 * total_size))
    valSize = int(math.ceil(0.1 * total_size * 2)) if do_data_augmentation else int(math.ceil(0.1 * total_size))

    train_start, train_end = 0, trainSize
    test_start, test_end = trainSize, (trainSize + testSize)
    val_start, val_end = (trainSize + testSize), (trainSize + testSize + valSize)

    print(f"{train_start} {train_end}, {test_start} {test_end}, {val_start} {val_end}")

    fields = []
    for i in range(6):
        fields.append(f"player_{i}_x")
        fields.append(f"player_{i}_y")

    print(f"trainSize={trainSize}, testSize={testSize}, valSize={valSize}")
    if model == "lstm":
        scaler = MinMaxScaler()
        df[fields] = scaler.fit_transform(df[fields])
        gen = DataGameGenerator(df=df, timesteps=timesteps,
                                start=train_start, end=val_end,
                                augment=do_data_augmentation)
    else:
        possessions_df = df.groupby(["GAME", "possession"])
        gen = PossessionImageGenerator(df=possessions_df, start=train_start, end=val_end)

    X = gen.data
    y = gen.truth
    n_splits = 10
    kFold = KFold(n_splits=n_splits, shuffle=True)

    def evaluate_model(x_train, x_test, y_train, y_test):
        ref_model = LSTMLatentSpace()
        encoder, the_decoder, model = ref_model.build(lstm_units=64, lr=0.001,
                                                      timesteps=timesteps,
                                                      n_features=n_features)
        model.fit(x=x_train,
                  y=y_train,
                  epochs=500,
                  batch_size=32)

        return model.evaluate(x=x_test, y=y_test, verbose=0)


    def evaluate_conv_model(x_train, x_test, y_train, y_test):
        ref_model = ConvolutionalLatentSpace()
        encoder, the_decoder, model = ref_model.build(lhs_size=128, lr=0.001)
        model.fit(x=x_train,
                  y=y_train,
                  epochs=100,
                  batch_size=32)

        return model.evaluate(x=x_test, y=y_test, verbose=0)


    fold_id = 0
    acc_score = 0
    for train, test in kFold.split(X=X, y=y):
        x_train, x_test, y_train, y_test = X[train], X[test], y[train], y[test]
        if model == "lstm":
            score = evaluate_model(x_train, x_test, y_train, y_test)
        else:
            score = evaluate_conv_model(x_train, x_test, y_train, y_test)

        print(f"AccScore {score[2]} FoldId={fold_id}")
        fold_id += 1
        acc_score += score[2]
    avg_score = acc_score / fold_id
    print(f"Avg score for {fold_id} folds is {avg_score}")

    # ref_model = LSTMLatentSpace()
    # encoder, the_decoder, autoencoder = ref_model.build(lstm_units=64,
    #                                                    timesteps=timesteps,
    #                                                    n_features=n_features)
