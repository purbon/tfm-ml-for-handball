import math
import os

import numpy as np
import pandas as pd
from keras_tuner import HyperParameters, RandomSearch, GridSearch, BayesianOptimization
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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
        trainGenerator = DataGameGenerator(df=df, timesteps=timesteps,
                                           start=train_start, end=train_end,
                                           augment=do_data_augmentation)

        testGenerator = DataGameGenerator(df=df, timesteps=timesteps,
                                          start=test_start, end=test_end,
                                          augment=do_data_augmentation)
        valGenerator = DataGameGenerator(df=df, timesteps=timesteps,
                                         start=val_start,
                                         end=val_end,
                                         display_labels=False,
                                         augment=do_data_augmentation)


        def build_model(hp):
            units = hp.Int("units", min_value=32, max_value=128, step=32)
            learning_rate = hp.Float("lr", min_value=0.001, max_value=0.1, sampling="log")

            ref_model = LSTMLatentSpace()
            encoder, the_decoder, autoencoder = ref_model.build(lstm_units=units, lr=learning_rate, timesteps=timesteps,
                                                                n_features=n_features)
            return autoencoder

    if model == "conv":
        possessions_df = df.groupby(["GAME", "possession"])
        trainGenerator = PossessionImageGenerator(df=possessions_df, start=train_start, end=train_end)
        testGenerator = PossessionImageGenerator(df=possessions_df, start=test_start, end=test_end)
        valGenerator = PossessionImageGenerator(df=possessions_df,
                                                start=val_start,
                                                end=val_end)

        def build_model(hp):
            lhs_size = hp.Int("lhs_size", min_value=32, max_value=128, step=32)
            learning_rate = hp.Float("lr", min_value=0.001, max_value=0.1, sampling="log")

            ref_model = ConvolutionalLatentSpace()
            the_encoder, the_decoder, autoencoder = ref_model.build(lhs_size=lhs_size, lr=learning_rate)
            return autoencoder

    autoencoder = build_model(hp=HyperParameters())

    tuner = BayesianOptimization(
        hypermodel=build_model,
        objective="val_acc",
        max_trials=10,
        executions_per_trial=2,
        overwrite=True,
        directory="my_dir",
        project_name="helloworld")

    tuner.search(trainGenerator,
                 epochs=100,
                 validation_data=valGenerator)

    models = tuner.get_best_models(num_models=2)
    best_model = models[0]

    #best_model.build(input_shape=(None, timesteps, n_features))
    #print(best_model.summary())

    print(tuner.results_summary())
