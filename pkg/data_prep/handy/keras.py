import os
import pickle

from keras import saving


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