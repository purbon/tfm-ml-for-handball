import json
import math
import os
import random
from functools import partial

import keras
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate, false_negative_rate, false_positive_rate, count, \
    true_positive_rate, true_negative_rate
from keras_tuner.src.backend.io import tf
from matplotlib import pyplot as plt
from pandas.core.dtypes.common import is_numeric_dtype, is_string_dtype
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, ConfusionMatrixDisplay, confusion_matrix, \
    accuracy_score
from sklearn.model_selection import StratifiedKFold, GroupKFold

import shap
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from data_pre.embeddings import raw_handball_possessions
from handy.configs import get_experimental_configs, methods_configs
from handy.datasets import centroid_handball_possession, flattened_handball_possessions, latent_space_encoded_possession
from handy.experimenter import up_down_sample, method_estimators, grid_search_estimator
from handy.metrics import get_score_rates
from experiment_keras_kf import get_knn_experiment_configs, get_ann_model


def get_data(mode,
             target_attribute,
             include_sequence,
             config,
             phase="AT",
             lse_size=64,
             timesteps=30,
             normalizer=False,
             population_field=None):
    data, classes, games, sensitive_attribute = None, None, None, None
    if mode == "centroids":
        data, classes, games = centroid_handball_possession(target_attribute=target_attribute,
                                                            phase=phase,
                                                            include_sequences=include_sequence,
                                                            filter=config)
    if mode == "flat":
        data, classes, games = flattened_handball_possessions(target_attribute=target_attribute,
                                                              length=timesteps,
                                                              phase=phase,
                                                              include_sequences=include_sequence,
                                                              filter=config)
    if mode == "ls":
        data, classes, games = latent_space_encoded_possession(method_type="lstm",
                                                               lse_size=lse_size,
                                                               phase=phase,
                                                               timesteps=timesteps,
                                                               target_attribute=target_attribute,
                                                               filter=config)
    if mode == "raw":
        data, classes, games, sensitive_attribute = raw_handball_possessions(target_class=target_attribute,
                                                                             timesteps=config.timesteps,
                                                                             game_phase="AT",
                                                                             augment=config.do_augment,
                                                                             normalizer=normalizer,
                                                                             population_field=population_field)

    return data, classes.to_frame() if mode != "raw" else classes, games, sensitive_attribute


def explain_the_outcome(model, X, y_test, y_pred, method_name, labels, fold_id=1, output_dir="."):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    display.plot()
    fig_path = os.path.join(output_dir, f"cfm_{method_name}-{fold_id}.jpg")
    plt.savefig(fig_path)
    plt.clf()

    def f(X):
        y_pred = model.predict(X)
        y_pred[y_pred <= 0.5] = 0.
        y_pred[y_pred > 0.5] = 1.
        return y_pred

    #Xt = shap.sample(X, nsamples=200, random_state=random.randint(0, 1000))
    Xt = X
    explainer = shap.KernelExplainer(f, Xt)  # (136,360)

    shap_values = explainer.shap_values(Xt)

    # shap.summary_plot(shap_values, Xt[:20, :], show=False)
    #fig_path = os.path.join(output_dir, f"shap-summary_{method_name}-{fold_id}.html")

    # shap.force_plot(explainer.expected_value, shap_values[0], Xt[:20, :], show=False).savefig(fig_path)
    # plt.clf()

    shap.summary_plot(shap_values, Xt, show=False)
    fig_path = os.path.join(output_dir, f"shap-summary_{method_name}-{fold_id}.jpg")
    plt.savefig(fig_path)
    plt.clf()

    #shap.save_html(fig_path, shap.force_plot(explainer.expected_value, shap_values[0], Xt[:5, :], show=False))

    # features = X.columns
    # shap.plots.force(explainer.expected_value[1], shap_values[1][0, :], features=features, matplotlib=False, show=False)
    # fig_path = os.path.join(output_dir, f"shap-force_{method_name}-{fold_id}.jpg")
    # plt.savefig(fig_path)
    # plt.clf()


def run(output_dir, method_name, config, target_attribute, population_field, normalizer, kfold_strategy):
    lse_size, timesteps = 64, 25

    data, classes, games, sensitive_attribute = get_data(mode=mode,
                                                         target_attribute=target_attribute,
                                                         include_sequence=include_sequence,
                                                         config=config,
                                                         lse_size=lse_size,
                                                         timesteps=timesteps,
                                                         population_field=population_field,
                                                         normalizer=normalizer)

    label_encoder = LabelEncoder()
    class_encoder = LabelEncoder()
    std_encoder = MinMaxScaler()  # StandardScaler()

    X = data
    y = classes

    k_folder = StratifiedKFold(n_splits=n_splits, shuffle=True)
    if kfold_strategy != "stratified":
        k_folder = GroupKFold(n_splits=n_splits)

    fold_id = 0

    # population_classes = [class_encoder.classes_[v] for v in X[pupul_class].unique()]
    # default_values = [{'count': 0, 'fnr': 0, 'fpr': 0, 'tnr': 0, 'tpr': 0, 'bacc': 0} for i in
    #                  range(len(population_classes))]
    # fairness_values = dict(zip(population_classes, default_values))
    metric_values = {'precision': 0, 'recall': 0, 'f1': 0}

    if not normalizer:
        if class_action == "upsample":
            config.do_augment = True
        elif class_action == "none":
            config.do_augment = False

    callbacks = [keras.callbacks.EarlyStopping(patience=10, monitor="loss", restore_best_weights=True)]

    population_classes = np.unique(sensitive_attribute)
    default_values = [{'count': 0, 'fnr': 0, 'fpr': 0, 'tnr': 0, 'tpr': 0, 'acc': 0} for i in
                      range(len(population_classes))]
    fairness_values = dict(zip(population_classes, default_values))
    differences_df = pd.DataFrame()

    # print(X.columns)
    for train_index, test_index in k_folder.split(X=X, y=y, groups=games):
        X_train = X[train_index, :]
        y_train = y[train_index, :]
        s_train = sensitive_attribute[train_index, :]
        X_test = X[test_index, :]
        y_test = y[test_index, :]
        s_test = sensitive_attribute[test_index, :]

        print(fold_id)
        # X_train, y_train = up_down_sample(X=X_train, y=y_train, class_action=class_action)
        # X_test.reset_index(inplace=True, drop=True)

        if normalizer:
            X_train, y_train = up_down_sample(X=X_train, y=y_train, class_action=config.class_action)

        model = get_ann_model(model_type=method_name, timesteps=config.timesteps, n_fields=12)

        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            optimizer='adam',
            # optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-4),
            metrics=["binary_accuracy",
                     tf.keras.metrics.AUC(from_logits=False),
                     tf.keras.metrics.Precision(),
                     tf.keras.metrics.Recall()],
        )
        # model.summary()

        model.fit(
            X_train,
            y_train,
            epochs=config.epochs,
            batch_size=config.batch_size,
            callbacks=callbacks,
            verbose=0
        )

        score = model.evaluate(X_test,
                               y_test,
                               batch_size=config.batch_size,
                               verbose=0)
        print(model.metrics_names)
        print(score)

        precision_num = score[3]  # precision and recall should be weighted to be rightful compared
        recall_num = score[4]
        if (precision_num + recall_num) == 0:
            f1_num = 0
        else:
            f1_num = 2 * (precision_num * recall_num) / (precision_num + recall_num)
        auc_num = score[2]

        y_pred = model.predict(x=X_test)
        y_pred[y_pred <= 0.5] = 0.
        y_pred[y_pred > 0.5] = 1.
        metric_values['precision'] += precision_score(y_true=y_test, y_pred=y_pred, average='weighted')
        metric_values['recall'] += recall_score(y_true=y_test, y_pred=y_pred, average='weighted')
        metric_values['f1'] += f1_score(y_true=y_test, y_pred=y_pred, average='weighted')

        labels = np.unique(y_pred)
        if explain_the_output and fold_id < 2:
            try:
                feature_names = []
                for i_time in range(config.timesteps):
                    for i in range(6):
                        feature_names += [f"player_{i}_x{i_time}", f"player_{i}_y{i_time}"]
                X_train_df = pd.DataFrame(X_train, columns=feature_names)
                explain_the_outcome(model=model,
                                    X=X_train_df,
                                    y_test=y_test,
                                    y_pred=y_pred,
                                    method_name=method_name,
                                    fold_id=fold_id,
                                    labels=labels,
                                    output_dir=output_dir)
            except Exception as error:
                print(error)
        metrics = {
            "accuracy": accuracy_score,
            "precision": precision_score,
            "recall": recall_score,
            "false positive rate": false_positive_rate,
            "false negative rate": false_negative_rate,
            "true positive rate": true_positive_rate,
            "true negative rate": true_negative_rate,
            "selection rate": selection_rate,
            "count": count,
        }
        reverse_map = {
            'count': 'count',
            'false negative rate': 'fnr',
            'false positive rate': 'fpr',
            'true negative rate': 'tnr',
            'true positive rate': 'tpr',
            'binary_accuracy': 'acc',
            'accuracy': 'acc'
        }

        mf = MetricFrame(metrics=metrics, y_true=y_test, y_pred=y_pred, sensitive_features=s_test)

        mf_groups = mf.by_group
        mf_diff = mf.difference()

        for row in mf_groups.iterrows():
            protected_class = row[0]
            metrics_values = row[1]
            for index, value in metrics_values.items():
                property_key = reverse_map.get(index, None)
                if property_key:
                    fairness_values[protected_class][property_key] += value

        mf_groups.to_excel(os.path.join(output_dir, f"mf-{method_name}-groups-{fold_id}.xlsx"), index=True)
        mf_groups.to_latex(os.path.join(output_dir, f"mf-{method_name}-groups-{fold_id}.tex"), index=True)

        differences_df[fold_id] = mf_diff

        if explain_the_output:
            mf.by_group.plot.bar(
                subplots=True,
                layout=[3, 3],
                legend=False,
                figsize=[12, 8],
                title="Show all metrics",
            )

            fig_path = os.path.join(output_dir, f"mf_{method_name}-{fold_id}.jpg")
            plt.savefig(fig_path)
            plt.clf()

        fold_id += 1

    for key, value in fairness_values.items():
        for other_key in ['fnr', 'fpr', 'tnr', 'tpr', 'acc']:
            fairness_values[key][other_key] = fairness_values[key][other_key] / n_splits

    df_fairness = pd.DataFrame.from_dict(fairness_values)
    df_fairness.to_excel(f"{output_dir}/fairness-{population_field}.xlsx")
    df_fairness.to_latex(f"{output_dir}/fairness-{population_field}.tex")

    for key, value in metric_values.items():
        metric_values[key] = metric_values[key] / n_splits

    with open(f"{output_dir}/metrics.json", 'w+') as f:
         json.dump(metric_values, f)

    differences_df.to_excel(os.path.join(output_dir, f"mf-{method_name}-differences.xlsx"), index=True)
    differences_df.to_latex(os.path.join(output_dir, f"mf-{method_name}-differences.tex"), index=True)


if __name__ == "__main__":
    mode = "raw"
    #target_attribute = "organized_game"  # possession_result organized_game
    include_sequence = True
    kfold_strategy = "group"
    n_splits = 9
    class_action = "upsample"
    explain_the_output = True
    # population_field = "throw_zone"  # tactical_situation throw_zone possession_result

    # method_name = "dense"

    configs = get_knn_experiment_configs()
    config = configs[3]

    target_attributes = ["organized_game"]
    population_fields = ["throw_zone"] #["throw_zone", "tactical_situation"]
    models = ["dense"] #["dense", "lstm", "lstm2", "transformer"]

    for method_name in models:
        for target_attribute in target_attributes:
            for population_field in population_fields:
                output_dir = os.path.join("explain",
                                          "dml-2",
                                          target_attribute,
                                          mode,
                                          method_name,
                                          population_field,
                                          class_action)
                os.makedirs(output_dir, exist_ok=True)
                normalizer = (method_name != "lstm2" and method_name != "transformer")

                run(output_dir=output_dir,
                    method_name=method_name,
                    config=config,
                    kfold_strategy=kfold_strategy,
                    target_attribute=target_attribute,
                    population_field=population_field,
                    normalizer=normalizer)
