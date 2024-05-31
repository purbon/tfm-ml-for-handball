import keras
from keras import Model, Sequential, layers
from keras.src.layers import Dense, LSTM, RepeatVector, TimeDistributed, Conv2D, BatchNormalization, MaxPooling2D, \
    Reshape, Dropout, Flatten, UpSampling2D
from keras.src.optimizers import Adam


class AutoEncoderModel:

    def __init__(self, debug):
        self.debug = debug

    def build(self):
        pass


class LSTMLatentSpace(AutoEncoderModel):

    def __init__(self, debug=False):
        super().__init__(debug)

    def build(self, timesteps, n_features, lstm_units=64, lr=0.001):
        encoder_input = layers.Input(shape=(timesteps, n_features), name="encoder_input")
        x = layers.Masking(mask_value=0)(encoder_input)
        x = LSTM(128, activation='relu', return_sequences=True)(x)
        encoder_output = lstm_masked(lstm_units=lstm_units)(x)
        encoder = Model(encoder_input, encoder_output)

        # decoded = lstm_repeatvector(time_steps=timesteps)(encoder_output)
        # y = LSTM(lstm_units, activation='relu', return_sequences=True)(decoded)
        # y = LSTM(128, activation='relu', return_sequences=True)(y)
        # y = TimeDistributed(Dense(n_features))(y)

        the_decoder = keras.Sequential()
        the_decoder.add(layers.Input(shape=(lstm_units,), name="encoder_input"))
        the_decoder.add(lstm_repeatvector(time_steps=timesteps))
        the_decoder.add(LSTM(lstm_units, activation='relu', return_sequences=True))
        the_decoder.add(LSTM(128, activation='relu', return_sequences=True))
        the_decoder.add(TimeDistributed(Dense(n_features)))

        opt = Adam(learning_rate=lr)
        autoencoder = Model(inputs=[encoder_input], outputs=the_decoder(encoder_output))
        autoencoder.compile(optimizer=opt, loss='mean_squared_error', metrics=["mae", "acc"])

        return encoder, the_decoder, autoencoder


class ConvolutionalLatentSpace(AutoEncoderModel):

    def __init__(self, debug=False):
        super().__init__(debug)

    def build(self, input_shape=(32, 48, 3), lhs_size=32, lr=0.001, dropout=0.2):
        encoder_input = layers.Input(shape=input_shape, name="encoder_input")
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoder_input)
        x = BatchNormalization()(x)
        x = MaxPooling2D(2)(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(2)(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        encoder_output = MaxPooling2D(2)(x)
        encoder_output = Dropout(dropout)(encoder_output)
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

        if self.debug:
            self.__log_encoders(the_encoder=the_encoder, the_decoder=the_decoder)

        autoencoder = Model(inputs=encoder_input, outputs=the_decoder(encoder_output))
        opt = Adam(learning_rate=lr)
        autoencoder.compile(optimizer=opt, loss='binary_crossentropy', metrics=["mse", "mae", "acc"])

        return the_encoder, the_decoder, autoencoder

    def __log_encoders(self, the_encoder, the_decoder):
        self.__log_encoder(encoder=the_encoder)
        self.__log_encoder(encoder=the_decoder)

    def __log_encoder(self, encoder):
        print("***")
        print(encoder.summary())
        print("***")


class lstm_bottleneck(layers.Layer):
    def __init__(self, lstm_units, time_steps, **kwargs):
        self.lstm_units = lstm_units
        self.time_steps = time_steps
        self.lstm_layer = LSTM(lstm_units, return_sequences=False)
        self.repeat_layer = RepeatVector(time_steps)
        super(lstm_bottleneck, self).__init__(**kwargs)

    def call(self, inputs):
        # just call the two initialized layers
        return self.repeat_layer(self.lstm_layer(inputs))

    def compute_mask(self, inputs, mask=None):
        # return the input_mask directly
        return mask

class lstm_masked(layers.Layer):
    def __init__(self, lstm_units, **kwargs):
        self.lstm_units = lstm_units
        self.lstm_layer = LSTM(lstm_units, return_sequences=False)
        super(lstm_masked, self).__init__(**kwargs)

    def call(self, inputs):
        # just call the two initialized layers
        return self.lstm_layer(inputs)

    def compute_mask(self, inputs, mask=None):
        # return the input_mask directly
        return mask

class lstm_repeatvector(layers.Layer):
    def __init__(self, time_steps, **kwargs):
        self.time_steps = time_steps
        self.repeat_layer = RepeatVector(time_steps)
        super(lstm_repeatvector, self).__init__(**kwargs)

    def call(self, inputs):
        return self.repeat_layer(inputs)

    def compute_mask(self, inputs, mask=None):
        # return the input_mask directly
        return mask
