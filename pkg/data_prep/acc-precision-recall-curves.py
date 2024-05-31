import math
import random

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, f1_score, \
    precision_recall_curve, PrecisionRecallDisplay


def prediction_metrics_for(y_true, y_pred):
    precision = precision_score(y_true=y_true, y_pred=y_pred, zero_division=1)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    auc_score = roc_auc_score(y_true, y_pred)

    return {
        'precision': math.floor(precision * 1000) / 1000.0,
        'recall': math.floor(recall * 1000) / 1000.0,
        'f1': math.floor(f1 * 1000) / 1000.0,
        'accuracy': math.floor(accuracy * 1000) / 1000.0,
        'auc_score': math.floor(auc_score * 1000) / 1000.0
    }


def print_predictions(y_true, y_pred):
    prediction_metrics = prediction_metrics_for(y_true=y_true, y_pred=y_pred)
    print(prediction_metrics)


def example1():
    y_true = [1, 0, 1, 0, 1, 1, 0, 0, 0, 0]  # P = 4, N = 6

    y_pred = [0, 1, 0, 1, 0, 0, 1, 1, 1, 1]  # TP=0, TN=0, FP=6, FN=4
    print_predictions(y_true=y_true, y_pred=y_pred)

    print("True negative baseline: ")
    y_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # TP=0, TN=6, FP=0, FN=4
    print_predictions(y_true=y_true, y_pred=y_pred)

    y_pred = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # TP=1, TN=6, FP=0, FN=3
    print_predictions(y_true=y_true, y_pred=y_pred)

    y_pred = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]  # TP=2, TN=6, FP=0, FN=2
    print_predictions(y_true=y_true, y_pred=y_pred)

    y_pred = [1, 0, 1, 0, 1, 0, 0, 0, 0, 0]  # TP=3, TN=6, FP=0, FN=1
    print_predictions(y_true=y_true, y_pred=y_pred)

    y_pred = [1, 0, 1, 0, 1, 1, 0, 0, 0, 0]  # TP=4, TN=6, FP=0, FN=0
    print_predictions(y_true=y_true, y_pred=y_pred)

    print("True positive baseline: ")
    y_pred = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # TP=4, TN=0, FP=6, FN=0
    print_predictions(y_true=y_true, y_pred=y_pred)

    y_pred = [1, 0, 1, 1, 1, 1, 1, 1, 1, 1]  # TP=4, TN=1, FP=5, FN=0
    print_predictions(y_true=y_true, y_pred=y_pred)

    y_pred = [1, 0, 1, 0, 1, 1, 1, 1, 1, 1]  # TP=4, TN=2, FP=4, FN=0
    print_predictions(y_true=y_true, y_pred=y_pred)

    y_pred = [1, 0, 1, 0, 1, 1, 0, 1, 1, 1]  # TP=4, TN=3, FP=3, FN=0
    print_predictions(y_true=y_true, y_pred=y_pred)

    y_pred = [1, 0, 1, 0, 1, 1, 0, 0, 1, 1]  # TP=4, TN=4, FP=2, FN=0
    print_predictions(y_true=y_true, y_pred=y_pred)

    y_pred = [1, 0, 1, 0, 1, 1, 0, 0, 0, 1]  # TP=4, TN=5, FP=1, FN=0
    print_predictions(y_true=y_true, y_pred=y_pred)

    y_pred = [1, 0, 1, 0, 1, 1, 0, 0, 0, 0]  # TP=4, TN=6, FP=0, FN=0
    print_predictions(y_true=y_true, y_pred=y_pred)


def make_y_pred(default_size=20, ones_index=[], init_value=0, add_noice_for=0):
    y_pred = np.zeros(default_size) if init_value == 0 else np.ones(default_size)
    other_value = 1 if init_value == 0 else 0
    for one_index in ones_index:
        np.put(y_pred, one_index, other_value)

    rand = random.Random()
    for i in range(add_noice_for):
        maybe_noice = rand.randint(0, default_size)
        while ones_index.__contains__(maybe_noice):
            maybe_noice = rand.randint(0, default_size)
        np.put(y_pred, maybe_noice, other_value)
    return y_pred


def sample_maybe_ones(number_of_ones, default_size):
    predefined_ones = []
    rand = random.Random()
    while number_of_ones > 0:
        maybe_one = rand.randint(0, default_size)
        if not predefined_ones.__contains__(maybe_one):
            predefined_ones.append(maybe_one)
            number_of_ones -= 1
    return predefined_ones


##
# Precision: Relevant retrieved instances / All retrieved instances - TP / (TP + FP)
# Recall: Relevant retrieved instances / All  - TP / (TP + FN)
# Accuracy: (TP + TN) / (P + N)

if __name__ == "__main__":
    number_of_ones = 10
    default_size = 200

    predefined_ones = sample_maybe_ones(number_of_ones=number_of_ones, default_size=default_size)
    y_true = make_y_pred(default_size=default_size, ones_index=predefined_ones)
    y_pred = make_y_pred(default_size=default_size, ones_index=predefined_ones[0:7], add_noice_for=0)

    print_predictions(y_true=y_true, y_pred=y_pred)

    # print roc curve display
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='')
    display.plot()
    plt.show()

    # print prediction recall display
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    plt.show()