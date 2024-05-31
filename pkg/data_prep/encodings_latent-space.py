import math
import os
import pickle

import numpy as np
import pandas as pd
from keras import Model, Sequential, saving, layers
from keras.src.layers import Dense, LSTM, RepeatVector, TimeDistributed
from keras.src.optimizers import Adam
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

from create_conv_encoding import save_trained_model, load_trained_model
from data_io.meta import Schema
from data_io.view import HandballPlot
from data_pre.embeddings import DataGameGenerator
from handy.encodings.auto_encoder import lstm_masked, lstm_repeatvector


# lstm_1(LSTM)(None, 64)
# repeat_vector(RepeatVecto(None, 20, 64)
# lstm_2(LSTM)(None, 20, 64)
def lstm_autoencoder_m2(timesteps, n_features):
    encoder_input = layers.Input(shape=(timesteps, n_features), name="encoder_input")
    x = layers.Masking(mask_value=0)(encoder_input)
    x = LSTM(128, activation='relu', return_sequences=True)(x)
    encoder_output = lstm_masked(lstm_units=64)(x)

    decoded = lstm_repeatvector(time_steps=timesteps)(encoder_output)
    encoder = Model(encoder_input, encoder_output)

    y = LSTM(64, activation='relu', return_sequences=True)(decoded)
    y = LSTM(128, activation='relu', return_sequences=True)(y)
    y = TimeDistributed(Dense(n_features))(y)

    autoencoder = Model(encoder_input, y)
    autoencoder.compile(optimizer="adam", loss='mean_squared_error', metrics=["mae", "acc"])

    return encoder, None, autoencoder


# 256, 128, 64, 32, 16
def lstm_autoencoder_m3(timesteps, n_features, lstm_units=128, levels=2):
    encoder_input = layers.Input(shape=(timesteps, n_features), name="encoder_input")
    x = layers.Masking(mask_value=0)(encoder_input)

    units_count = lstm_units
    for i in range(levels - 1):
        x = LSTM(units_count, activation='relu', return_sequences=True)(x)
        units_count = int(units_count / 2)

    encoder_output = lstm_masked(lstm_units=units_count)(x)
    encoder = Model(encoder_input, encoder_output)

    # decoded = lstm_repeatvector(time_steps=timesteps)(encoder_output)

    # y = LSTM(lstm_units, activation='relu', return_sequences=True)(decoded)
    # y = LSTM(128, activation='relu', return_sequences=True)(y)
    # y = TimeDistributed(Dense(n_features))(y)

    the_decoder = Sequential()
    the_decoder.add(layers.Input(shape=(units_count,), name="encoder_input"))
    the_decoder.add(lstm_repeatvector(time_steps=timesteps))
    for i in range(levels - 1):
        the_decoder.add(LSTM(units_count, activation='relu', return_sequences=True))
        units_count = int(units_count * 2)
    the_decoder.add(LSTM(lstm_units, activation='relu', return_sequences=True))
    the_decoder.add(TimeDistributed(Dense(n_features)))

    opt = Adam(learning_rate=0.01)
    autoencoder = Model(inputs=[encoder_input], outputs=the_decoder(encoder_output))
    autoencoder.compile(optimizer="adam", loss='mean_squared_error', metrics=["mae", "acc"])

    return encoder, the_decoder, autoencoder


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


def flush_dataset(lstm_units, levels, timesteps=25, epochs=500):
    df = pd.read_hdf(path_or_buf=f"handball.h5", key="pos")

    df.drop(columns=["player_0", "player_1", "player_2", "player_3", "player_4", "player_5"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = df[df[Schema.ACTIVE_PLAYERS_COLUMN] == 6]

    n_features = 12

    create_plots = False
    debug = False
    dump = False
    plot = False

    total_size = df.groupby(["GAME", "possession"]).count().shape[0]

    trainSize = int(math.ceil(0.7 * total_size * 2))
    testSize = int(math.ceil(0.2 * total_size * 2))
    valSize = int(math.ceil(0.1 * total_size * 2))

    fields = []
    for i in range(6):
        fields.append(f"player_{i}_x")
        fields.append(f"player_{i}_y")

    scaler = MinMaxScaler()
    df[fields] = scaler.fit_transform(df[fields])
    # df = scale(df=df, fields=fields)
    lse_size = int(lstm_units / pow(2, levels - 1))

    print(f"trainSize={trainSize}, testSize={testSize}, valSize={valSize}, lse_size={lse_size}")
    trainGenerator = DataGameGenerator(df=df, timesteps=timesteps,
                                       start=0, end=trainSize,
                                       augment=True)
    testGenerator = DataGameGenerator(df=df, timesteps=timesteps,
                                      start=trainSize, end=(trainSize + testSize),
                                      augment=True)
    valGenerator = DataGameGenerator(df=df, timesteps=timesteps,
                                     start=(trainSize + testSize),
                                     end=(trainSize + testSize + valSize),
                                     display_labels=False,
                                     augment=True)
    model_name = f"keras.{lse_size}_{lstm_units}-{levels}-{timesteps}.model"
    encoder, decoder, model = lstm_autoencoder_m3(timesteps=timesteps, n_features=n_features,
                                                  lstm_units=lstm_units, levels=levels)

    if not os.path.exists(model_name):
        model.summary()
        trainHistory = model.fit(trainGenerator,
                                 epochs=epochs,
                                 batch_size=32,
                                 validation_data=valGenerator)

        score = model.evaluate(testGenerator, verbose=0)
        print(model.metrics_names)
        print(score)
        save_trained_model(model_name, model, trainHistory, overwrite=True)
    else:
        model, history = load_trained_model(model_name)
        print(model.summary())
        print(model.layers[-1].summary())
    ## plot the results on the test dataset

    # for i, l in enumerate(model.layers):
    #   print(f'layer {i}: {l}')
    #   print(f'has input shape: {l.input_shape}')
    #   print(f'has output shape: {l.output_shape}')
    #    print(f'has input mask: {l.input_mask}')
    #    print(f'has output mask: {l.output_mask}')

    if create_plots:
        os.makedirs(f"charts/test_embeddings", exist_ok=True)
        for j in range(len(testGenerator)):
            X, y = testGenerator.__getitem__(j)
            O = testGenerator.__getorigins__(j)
            prediction = model.predict(X)
            # print(prediction.shape)
            for i in range(len(X)):
                # plit X0
                p0 = scaler.inverse_transform(prediction[i])
                fx = X[i][~np.all(X[i] == 0, axis=1)]
                x0 = scaler.inverse_transform(fx)
                plot_a_possession(X=x0, label=f"x{O[i]}")
                plot_a_possession(X=p0, label=f"p{O[i]}")

    if debug:
        X, y = testGenerator.__getitem__(0)
        O = testGenerator.__getorigins__(0)
        embeddings = encoder.predict(X)
        print(len(embeddings[0]))
        print(embeddings[0])
        print("****")
        reconstructed = decoder.predict(embeddings)
        print(len(reconstructed[0]))
        print(reconstructed[0])

        p0 = scaler.inverse_transform(reconstructed[0])
        plot_a_possession(X=p0, label=f"r{O[0]}")

    if dump:
        mdf = pd.DataFrame()
        print(total_size)
        dataGenerator = DataGameGenerator(df=df, timesteps=timesteps, start=0, end=total_size)
        print(df.shape)

        def map_row(row, i):
            return row.origin.split("_")[i]

        for j in range(len(dataGenerator)):
            X, y = dataGenerator.__getitem__(j)
            embeddings = encoder.predict(X)
            O = dataGenerator.__getorigins__(j)
            ope = pd.DataFrame(embeddings)
            oo = pd.DataFrame(O, columns=['origin'])
            oo["GAME"] = oo.apply(lambda row: map_row(row, 0), axis=1)
            oo["possession"] = oo.apply(lambda row: map_row(row, 1), axis=1)
            oo.drop(columns='origin', inplace=True)
            idf = ope.join(oo)
            mdf = pd.concat([mdf, idf])
            mdf["GAME"] = mdf["GAME"].astype(int)

        handball_df = pd.read_hdf(path_or_buf=f"handball.h5", key="dv0_p6")

        def filter_columns(column):
            pcount = [f"p{i}" for i in range(6)]
            team = ["team_x_centroid", "team_y_centroid"]
            return not column.startswith("player_") and column[:2] not in pcount and column not in team

        columns = list(filter(filter_columns, handball_df.columns))
        handball_df = handball_df[columns]
        mdf = pd.merge(handball_df.reset_index(), mdf, on=['GAME', 'possession'], how='inner')
        mdf.to_excel(f"lstm-autoencoder-embeddings-{lse_size}-{timesteps}.xlsx", index=False)
        mdf.to_hdf(path_or_buf="handball.h5", key=f"lstm_autoenc_{lse_size}-{timesteps}")

    if plot:
        labels = None
        embeddings = None
        dataGenerator = DataGameGenerator(df=df,
                                          start=0, timesteps=timesteps, end=total_size,
                                          game_phase="AT",
                                          augment=False)
        for j in range(len(dataGenerator)):
            X, y = dataGenerator.__getitem__(j)
            O = dataGenerator.__getorigins__(j)
            L = dataGenerator.__getlabels__(j)
            embedding = encoder.predict(X)
            if embeddings is None:
                embeddings = embedding
                labels = L
            else:
                embeddings = np.concatenate((embeddings, embedding))
                labels = np.concatenate((labels, L))

        # print(len(embeddings[0]))
        print(embeddings.shape)
        # print(embeddings[0])
        pca = PCA(n_components=2)
        proj = pca.fit_transform(embeddings)
        print(np.sum(pca.explained_variance_ratio_))
        df = pd.DataFrame({'X': proj[:, 0], 'Y': proj[:, 1]})
        df.insert(2, 'Labels', labels, True)
        import seaborn as sns

        sns.scatterplot(data=df, x='X', y='Y', hue="Labels")
        plt.savefig("lstm-pca.png")
        plt.clf()

        pca = PCA(n_components=50)
        proj = pca.fit_transform(embeddings)
        print(np.sum(pca.explained_variance_ratio_))
        tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000)
        proj = tsne.fit_transform(proj)
        df = pd.DataFrame({'X': proj[:, 0], 'Y': proj[:, 1]})
        df.insert(2, 'Labels', labels, True)
        import seaborn as sns

        sns.scatterplot(data=df, x='X', y='Y', hue="Labels")
        plt.savefig("lstm-tsne.png")
    # data = np.array([[ 0.1858418, -0.22647993, 0.51035756]])
    # reconstructed = decoder.predict(data)
    # p0 = scaler.inverse_transform(reconstructed[0])
    # plot_a_possession(X=p0, label=f"madeup")


if __name__ == '__main__':
    #configs = [(128, 2), (256, 2), (512, 2), (512, 3)]
    configs = [(512, 2)]
    for config in configs:
        flush_dataset(lstm_units=config[0], levels=config[1], timesteps=25)
        #flush_dataset(lstm_units=config[0], levels=config[1], timesteps=30)
