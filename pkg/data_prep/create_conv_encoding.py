import math
import os
import pickle

import numpy as np
import pandas as pd
from PIL import Image
from keras import Model, Sequential, saving, layers, Input
from keras.src.layers import Dense, LSTM, RepeatVector, TimeDistributed, Conv2D, MaxPooling2D, UpSampling2D, Dropout, \
    BatchNormalization, Flatten, Reshape
from keras.src.optimizers import RMSprop
from keras.src.optimizers.schedules import ExponentialDecay
from matplotlib import pyplot as plt
from skimage.io import imsave, imshow
from skimage.transform import resize
from skimage.util import img_as_uint
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

from data_io.meta import Schema
from data_io.view import HandballPlot
from data_pre.embeddings import DataGameGenerator, PossessionImageGenerator, plot_a_possession


# def lstm_autoencoder_m3(timesteps, n_features, lstm_units=64):
#    encoder_input = layers.Input(shape=(timesteps, n_features), name="encoder_input")
#    x = layers.Masking(mask_value=0)(encoder_input)
#    x = LSTM(128, activation='relu', return_sequences=True)(x)
#    encoder_output = lstm_masked(lstm_units=lstm_units)(x)
#    encoder = Model(encoder_input, encoder_output)

# decoded = lstm_repeatvector(time_steps=timesteps)(encoder_output)

# y = LSTM(lstm_units, activation='relu', return_sequences=True)(decoded)
# y = LSTM(128, activation='relu', return_sequences=True)(y)
# y = TimeDistributed(Dense(n_features))(y)

#    the_decoder = Sequential([
#        layers.Input(shape=(lstm_units,), name="encoder_input"),
#        lstm_repeatvector(time_steps=timesteps),
#        LSTM(lstm_units, activation='relu', return_sequences=True),
#        LSTM(128, activation='relu', return_sequences=True),
#        TimeDistributed(Dense(n_features))
#    ])
#    autoencoder = Model(inputs=[encoder_input], outputs=the_decoder(encoder_output))
#    autoencoder.compile(optimizer="adam", loss='mean_squared_error', metrics=["mae", "acc"])

#   return encoder, the_decoder, autoencoder

def conv_autoencoder(input_shape=(32, 48, 3), lhs_size=32):
    encoder_input = layers.Input(shape=input_shape, name="encoder_input")
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoder_input)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(2)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    encoder_output = MaxPooling2D(2)(x)
    encoder_output = Dropout(0.2)(encoder_output)
    encoder_output = Flatten()(encoder_output)
    encoder_output = Dense(lhs_size)(encoder_output)

    the_encoder = Model(encoder_input, encoder_output)

    the_decoder = Sequential([
        layers.Input(shape=(lhs_size,), name="decoder_input"),
        Dense(4 * 6 * 256),
        Reshape((4, 6, 256)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        UpSampling2D(2),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        UpSampling2D(2),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        UpSampling2D(2),
        Conv2D(3, (3, 3), activation='sigmoid', padding='same')
    ])

    print("***")
    print(the_encoder.summary())
    print("***")

    print("***")
    print(the_decoder.summary())
    print("***")

    autoencoder = Model(inputs=encoder_input, outputs=the_decoder(encoder_output))
    opt = RMSprop()
    autoencoder.compile(optimizer=opt, loss='binary_crossentropy', metrics=["mse", "mae", "acc"])

    return the_encoder, the_decoder, autoencoder


def fully_con_autoencoder(input_shape=576):
    encoder_input = Input(shape=(input_shape,))
    encoded = Dense(128, activation='relu')(encoder_input)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)

    the_encoder = Model(encoder_input, encoded)
    the_decoder = Sequential([
        Input(shape=(None, 32), name="decoder_input"),
        Dense(64, activation='relu'),
        Dense(128, activation='relu'),
        Dense(input_shape, activation='sigmoid')
    ])

    autoencoder = Model(inputs=encoder_input, outputs=the_decoder(encoded))
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=["mse", "mae", "acc"])
    return the_encoder, the_decoder, autoencoder


def save_trained_model(fileName, theModel, trainHistory, overwrite=False):
    if not overwrite and os.path.exists(fileName):
        return None
    print(f'Overwriting the model {fileName}')
    saving.save_model(theModel, fileName, overwrite=True)
    with open('keras.history', 'wb') as file:
        pickle.dump(trainHistory.history, file)


def load_trained_model(fileName="model.keras"):
    model = saving.load_model(fileName)
    with open('keras.history', "rb") as file:
        history = pickle.load(file)
    return model, history


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

    total_size = df.groupby(["GAME", "possession"]).count().shape[0]

    trainSize = int(math.ceil(0.7 * total_size))
    testSize = int(math.ceil(0.2 * total_size))
    valSize = int(math.ceil(0.1 * total_size))

    fields = []
    for i in range(6):
        fields.append(f"player_{i}_x")
        fields.append(f"player_{i}_y")

    # scaler = MinMaxScaler()
    # df[fields] = scaler.fit_transform(df[fields])
    # df = scale(df=df, fields=fields)

    print(f"trainSize={trainSize}, testSize={testSize}, valSize={valSize}")
    possessions_df = df.groupby(["GAME", "possession"])

    print("Loading .....")
    trainGenerator = PossessionImageGenerator(df=possessions_df, start=0, end=trainSize)
    testGenerator = PossessionImageGenerator(df=possessions_df, start=trainSize, end=(trainSize + testSize))
    valGenerator = PossessionImageGenerator(df=possessions_df,
                                            start=(trainSize + testSize),
                                            end=(trainSize + testSize + valSize))
    print("done!")
    lhs_size = 64
    encoder, decoder, model = conv_autoencoder(lhs_size=lhs_size)
    model_name = f"keras.conv{lhs_size}.model"

    if not os.path.exists(model_name):
        model.summary()
        trainHistory = model.fit(trainGenerator,
                                 epochs=300,
                                 batch_size=32,
                                 validation_data=valGenerator)

        score = model.evaluate(testGenerator, verbose=0)
        print(score)
        save_trained_model(model_name, model, trainHistory, overwrite=True)
    else:
        model, history = load_trained_model(model_name)

    if create_plots:
        os.makedirs(f"charts/test_embeddings", exist_ok=True)
        for j in range(len(testGenerator)):
            X, y = testGenerator.__getitem__(j)
            O = testGenerator.__getorigins__(j)
            prediction = model.predict(X)
            for i in range(len(O)):
                img = X[i]
                x_path = os.path.join("charts/test_embeddings", f"x{O[i]}.png")
                p_path = os.path.join("charts/test_embeddings", f"p{O[i]}.png")
                img = Image.fromarray((img * 255).astype(np.uint8))
                img = np.array(img)
                imsave(x_path, img)
                img = prediction[0]
                img = Image.fromarray((img * 255).astype(np.uint8))
                img = np.array(img)
                imsave(p_path, img)
    if debug:
        labels = None
        embeddings = None
        dataGenerator = PossessionImageGenerator(df=possessions_df, start=0, end=total_size)
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
        plt.savefig("conv-pca.png")
        plt.clf()

        pca = PCA(n_components=50)
        proj = pca.fit_transform(embeddings)
        print(np.sum(pca.explained_variance_ratio_))
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        proj = tsne.fit_transform(proj)
        df = pd.DataFrame({'X': proj[:, 0], 'Y': proj[:, 1]})
        df.insert(2, 'Labels', labels, True)
        import seaborn as sns

        sns.scatterplot(data=df, x='X', y='Y', hue="Labels")
        plt.savefig("conv-tsne.png")

    if dump:
        mdf = pd.DataFrame()
        print(total_size)
        dataGenerator = PossessionImageGenerator(df=possessions_df, start=0, end=total_size)
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
        mdf.to_excel(f"conv-autoencoder_{lhs_size}_embeddings.xlsx", index=False)
        mdf.to_hdf(path_or_buf="handball.h5", key=f"conv_autoenc_{lhs_size}")
