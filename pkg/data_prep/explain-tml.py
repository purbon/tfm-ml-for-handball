import json
import math
import os

import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate, false_negative_rate, false_positive_rate, count, \
    true_positive_rate, true_negative_rate, demographic_parity_ratio, equalized_odds_ratio
from matplotlib import pyplot as plt
from pandas.core.dtypes.common import is_numeric_dtype, is_string_dtype
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, ConfusionMatrixDisplay, confusion_matrix, \
    accuracy_score
from sklearn.model_selection import StratifiedKFold, GroupKFold

import shap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from handy.configs import get_experimental_configs, methods_configs
from handy.datasets import centroid_handball_possession, flattened_handball_possessions, latent_space_encoded_possession
from handy.experimenter import up_down_sample, method_estimators, grid_search_estimator
from handy.metrics import get_score_rates


def get_data(mode, target_attribute, phase, include_sequence, config, lse_size=64, timesteps=30):
    data, classes, games = None, None, None
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
    return data, classes.to_frame(), games


def explain_the_outcome(model, X, y_test, y_pred, method_name, fold_id=1, output_dir="."):
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    display.plot()
    fig_path = os.path.join(output_dir, f"cfm_{method_name}-{fold_id}.jpg")
    plt.savefig(fig_path)
    plt.clf()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap.summary_plot(shap_values, X, show=False)
    fig_path = os.path.join(output_dir, f"shap-summary_{method_name}-{fold_id}.jpg")
    plt.savefig(fig_path)
    plt.clf()

    # features = X.columns
    # shap.plots.force(explainer.expected_value[1], shap_values[1][0, :], features=features, matplotlib=False, show=False)
    # fig_path = os.path.join(output_dir, f"shap-force_{method_name}-{fold_id}.jpg")
    # plt.savefig(fig_path)
    # plt.clf()


def run(mode, method_name, population_field, config, output_dir, n_splits, lse_size=64, timesteps=25):
    data, classes, games = get_data(mode=mode,
                                    target_attribute=target_attribute,
                                    include_sequence=include_sequence,
                                    phase="AT",
                                    config=config,
                                    lse_size=lse_size,
                                    timesteps=timesteps)

    label_encoder = LabelEncoder()
    class_encoder = LabelEncoder()
    std_encoder = MinMaxScaler()  # StandardScaler()
    X = data

    na_columns = ["sequences", "passive_alert", "tactical_situation", "misses"]
    for na_column in na_columns:
        if na_column in X.columns:
            X[na_column].fillna(0, inplace=True)

    for column in X.columns:
        if is_numeric_dtype(X[column]):
            X[[column]] = std_encoder.fit_transform(X[[column]])
        if is_string_dtype(X[column]):
            if column == population_field:
                X[column] = class_encoder.fit_transform(X[column])
            else:
                X[column] = label_encoder.fit_transform(X[column])

    y = label_encoder.fit_transform(classes)

    k_folder = StratifiedKFold(n_splits=n_splits, shuffle=True)
    if kfold_strategy != "stratified":
        k_folder = GroupKFold(n_splits=n_splits)

    method_configs = methods_configs()
    method_config = method_configs[method_name]
    fold_id = 0

    try:
        population_classes = [class_encoder.classes_[v] for v in X[population_field].unique()]
    except:
        population_classes = X[population_field].unique()
    default_values = [{'count': 0, 'fnr': 0, 'fpr': 0, 'tnr': 0, 'tpr': 0, 'bacc': 0} for i in
                      range(len(population_classes))]
    fairness_values = dict(zip(population_classes, default_values))
    metric_values = {'precision': 0, 'recall': 0, 'f1': 0}

    differences_df = pd.DataFrame()
    fairness_metrics_df = pd.DataFrame()

    # print(X.columns)
    for train_index, test_index in k_folder.split(X=X, y=y, groups=games):
        X_train = X.iloc[train_index, :]
        y_train = y[train_index]
        X_test = X.iloc[test_index, :]
        y_test = y[test_index]
        X_train, y_train = up_down_sample(X=X_train, y=y_train, class_action=class_action)

        X_test.reset_index(inplace=True, drop=True)

        estimator = method_estimators(method_name)
        model = grid_search_estimator(estimator=estimator, params=method_config["params"],
                                      name=f"{method_name}",
                                      X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                      doTest=False,
                                      log=False)

        params = model.get_params()
        # n_estimators=params['estimator__n_estimators'],
        # learning_rate = params['estimator__learning_rate']
        if method_name == "randomforest":
            model = RandomForestClassifier(n_estimators=params['estimator__n_estimators'],
                                           criterion=params['estimator__criterion'],
                                           max_depth=params['estimator__max_depth'])
        elif method_name == "adaboost":
            model = AdaBoostClassifier(n_estimators=params['estimator__n_estimators'],
                                       learning_rate=params['estimator__learning_rate'])
        elif method_name == "hist-lgbm":
            model = HistGradientBoostingClassifier(learning_rate=params['estimator__learning_rate'])
        elif method_name == "knn":
            model = KNeighborsClassifier(
                n_neighbors=params['estimator__n_neighbors'],
                weights=params['estimator__weights'],
                p=params['estimator__p']
            )

        model.fit(X=X_train, y=y_train)

        y_pred = model.predict(X=X_test)
        precision_num = precision_score(y_true=y_test, y_pred=y_pred, average='weighted')
        recall_num = recall_score(y_true=y_test, y_pred=y_pred, average='weighted')
        f1_num = f1_score(y_true=y_test, y_pred=y_pred, average='weighted')

        metric_values['precision'] += precision_num
        metric_values['recall'] += recall_num
        metric_values['f1'] += f1_num

        if explain_the_output:
            try:
                explain_the_outcome(model=model,
                                    X=X_test,
                                    y_test=y_test,
                                    y_pred=y_pred,
                                    method_name=method_name,
                                    fold_id=fold_id,
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

        sensiv_feature = X_test[population_field]
        mf = MetricFrame(metrics=metrics, y_true=y_test, y_pred=y_pred, sensitive_features=sensiv_feature)

        eor = equalized_odds_ratio(y_true=y_test, y_pred=y_pred, sensitive_features=sensiv_feature)
        dpr = demographic_parity_ratio(y_true=y_test, y_pred=y_pred, sensitive_features=sensiv_feature)

        mf_groups = mf.by_group
        mf_diff = mf.difference()

        differences_df[fold_id] = mf_diff

        fairness_metrics_df[fold_id] = pd.Series({"equalized_odds_ratio": eor, "demographic_parity_ratio": dpr})

        mf_groups.to_excel(os.path.join(output_dir, f"mf-{method_name}-groups-{fold_id}.xlsx"), index=True)
        mf_groups.to_latex(os.path.join(output_dir, f"mf-{method_name}-groups-{fold_id}.tex"), index=True)

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

        population_classes = X_test[population_field].unique()
        for population_clazz in population_classes:
            print(f"X.{population_field} == {population_clazz}")
            pX_test = X_test[X_test[population_field] == population_clazz]
            pX_index = X_test[X_test[population_field] == population_clazz].index.to_numpy()
            y_test = y[pX_index]

            y_pred = model.predict(X=pX_test)

            fnr, fpr, tnr, tpr, bacc = get_score_rates(y_true=y_test, y_pred=y_pred, labels=[0, 1])

            try:
                label_pop_class = class_encoder.classes_[population_clazz]
            except:
                label_pop_class = population_clazz
            fairness_values[label_pop_class]['count'] += len(y_test)
            if not math.isnan(fnr):
                fairness_values[label_pop_class]['fnr'] += fnr
            if not math.isnan(fpr):
                fairness_values[label_pop_class]['fpr'] += fpr
            if not math.isnan(tnr):
                fairness_values[label_pop_class]['tnr'] += tnr
            if not math.isnan(tpr):
                fairness_values[label_pop_class]['tpr'] += tpr
            if not math.isnan(bacc):
                fairness_values[label_pop_class]['bacc'] += bacc

            # print(f"fnr={fnr}, tnr={tnr}, fpr={fpr}, tpr={tpr}")
            # print(f"bacc={bacc}")
            # print("")

        fold_id += 1

    for key, value in fairness_values.items():
        for other_key in ['fnr', 'fpr', 'tnr', 'tpr', 'bacc']:
            fairness_values[key][other_key] = fairness_values[key][other_key] / n_splits

    for key, value in metric_values.items():
        metric_values[key] = metric_values[key] / n_splits

    df_fairness = pd.DataFrame.from_dict(fairness_values)
    df_fairness.to_excel(f"{output_dir}/fairness-{population_field}.xlsx")
    df_fairness.to_latex(f"{output_dir}/fairness-{population_field}.tex")

    with open(f"{output_dir}/metrics.json", 'w+') as f:
        json.dump(metric_values, f)

    fairness_metrics_df.to_excel(os.path.join(output_dir, f"mf-{method_name}-metrics.xlsx"), index=True)
    fairness_metrics_df.to_latex(os.path.join(output_dir, f"mf-{method_name}-metrics.tex"), index=True)

    differences_df.to_excel(os.path.join(output_dir, f"mf-{method_name}-differences.xlsx"), index=True)
    differences_df.to_latex(os.path.join(output_dir, f"mf-{method_name}-differences.tex"), index=True)


if __name__ == "__main__":

    include_sequence = True
    kfold_strategy = "group"
    n_splits = 9
    class_action = "upsample"
    explain_the_output = True

    configs = get_experimental_configs()
    configs = configs[1: 5]  # 1 2 3 4

    target_attributes = ["organized_game", "possession_result"]
    methods = ["hist-lgbm", "randomforest", "adaboost", "knn"]
    population_fields = ["throw_zone", "tactical_situation", "passive_alert"]

    modes = ["centroids", "flat", "ls"]
    lse_options = [(64, 25), (64, 30), (128, 25), (128, 30)]

    for target_attribute in target_attributes:
        for mode in modes:
            for method_name in methods:
                for population_field in population_fields:
                    for config in configs:
                        output_dir = os.path.join("explain",
                                                  "ml-91",
                                                  target_attribute,
                                                  mode,
                                                  method_name,
                                                  str(config.id),
                                                  population_field,
                                                  class_action)
                        if mode == "lse":
                            for lse_option in lse_options:
                                output_dir = os.path.join(output_dir, f"{lse_option[0]}_{lse_option[1]}")
                                os.makedirs(output_dir, exist_ok=True)
                                run(mode=mode,
                                    method_name=method_name,
                                    population_field=population_field,
                                    config=config,
                                    n_splits=n_splits,
                                    output_dir=output_dir,
                                    lse_size=lse_option[0],
                                    timesteps=lse_option[1])
                        else:
                            os.makedirs(output_dir, exist_ok=True)
                            run(mode=mode,
                                method_name=method_name,
                                population_field=population_field,
                                config=config,
                                n_splits=n_splits,
                                output_dir=output_dir)
