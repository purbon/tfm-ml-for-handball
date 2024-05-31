import json
import os

import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NeighbourhoodCleaningRule
from pandas.core.dtypes.common import is_numeric_dtype, is_string_dtype
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, \
    HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_score, \
    recall_score, f1_score, PrecisionRecallDisplay, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, permutation_test_score, GroupKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

from handy.configs import methods_configs
from handy.datasets import centroid_handball_possession, flattened_handball_possessions, \
    latent_space_encoded_possession, ConfigFilter


def grid_search_estimator(estimator, params, name,
                          X_train, y_train, X_test, y_test,
                          doTest=False, output_path="./", log=True):
    print(f"Running estimator: {estimator}")
    clf = GridSearchCV(estimator, params,
                       cv=5,
                       scoring="f1_weighted",  # f1_weighted
                       refit="f1_weighted",
                       n_jobs=10)
    clf.fit(X=X_train, y=y_train)

    if doTest:
        score, perm_scores, pvalue = permutation_test_score(estimator=clf.best_estimator_,
                                                            X=X_train, y=y_train,
                                                            cv=5,
                                                            scoring="f1_weighted",
                                                            n_permutations=1000)
    if log:
        with open(f"{output_path}/{name}-scores.json", 'a+') as f:
            scores_dict = {
                'best_estimator': str(clf.best_estimator_),
                'best_score': clf.best_score_,
                'test_score': clf.score(X=X_test, y=y_test)
            }
        if doTest:
            scores_dict['perm_test'] = {'score': score, 'pvalue': pvalue}
        json.dump(scores_dict, f)
    return clf


def method_estimators(name):
    config = {
        'knn': KNeighborsClassifier(),
        'randomforest': RandomForestClassifier(),
        "gradientboosting": GradientBoostingClassifier(),
        "adaboost": AdaBoostClassifier(),
        "hist-lgbm": HistGradientBoostingClassifier(),
        "bernoulli": BernoulliNB(),
    }
    return config[name]


def run_experiment(X_train, y_train, X_test, y_test,
                   output_path="./",
                   show_confmatrix=False,
                   doTest=False):
    method_configs = methods_configs()

    for method_name, method_config in method_configs.items():
        estimator = method_estimators(method_name)
        model = grid_search_estimator(estimator=estimator, params=method_config["params"],
                                      name=method_name, output_path=output_path,
                                      X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                      doTest=doTest)

        if show_confmatrix:
            show_conf_matrix(model=model, name=method_name,
                             X_test=X_test, y_test=y_test, output_path=output_path)


def do_kfold_experiment(X, y,
                        k_folder=None,
                        class_action=None,
                        do_test=False,
                        n_splits=5,
                        games=None,
                        output_path="./"):
    if k_folder is None:
        k_folder = StratifiedKFold(n_splits=n_splits, shuffle=True)
    fold_id = 0
    acc_scores = {}

    method_configs = methods_configs()
    for train_index, test_index in k_folder.split(X=X, y=y, groups=games):
        X_train = X.iloc[train_index, :]
        y_train = y[train_index]
        X_test = X.iloc[test_index, :]
        y_test = y[test_index]
        X_train, y_train = up_down_sample(X=X_train, y=y_train, class_action=class_action)
        for method_name, method_config in method_configs.items():
            estimator = method_estimators(method_name)
            model = grid_search_estimator(estimator=estimator, params=method_config["params"],
                                          name=f"{method_name}-{fold_id}", output_path=output_path,
                                          X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                          doTest=do_test)
            fscore = train_and_test(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                    fold_id=fold_id,
                                    model=model,
                                    name=method_name,
                                    output_path=output_path)
            if acc_scores.get(method_name, None) is None:
                acc_scores[method_name] = 0
            acc_scores[method_name] += fscore
        fold_id += 1
    for method_name, acc_score in acc_scores.items():
        avg_score = acc_score / fold_id
        with open(f"{output_path}/{method_name}-avg-score.txt", 'a+') as f:
            print(f"Avg score for {fold_id} folds is {avg_score}", file=f)


def show_conf_matrix(model, name, X_test, y_test, output_path="./"):
    predictions = model.predict(X_test)
    with open(f"{output_path}/{name}.json", 'a+') as f:
        class_report_dict = classification_report(y_true=y_test, y_pred=predictions, output_dict=True)
        json.dump(class_report_dict, f)
    cm = confusion_matrix(y_test, predictions, labels=model.classes_)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    display.plot()
    plt.savefig(f"{output_path}/{name}.png")


def train_and_test(X_train, y_train, X_test, y_test, fold_id, output_path="./", model=None, name="kfold"):
    if model is None:
        model = RandomForestClassifier(n_estimators=150)
        name = "randomforest"
    model.fit(X=X_train, y=y_train)
    y_pred = model.predict(X=X_test)
    precision_num = precision_score(y_true=y_test, y_pred=y_pred, average='weighted')
    recall_num = recall_score(y_true=y_test, y_pred=y_pred, average='weighted')
    f1_num = f1_score(y_true=y_test, y_pred=y_pred, average='weighted')
    auc_num = roc_auc_score(y_true=y_test, y_score=model.predict_proba(X_test)[:, 1], average='weighted')
    with open(f"{output_path}/{name}-train_and_test.txt", 'a+') as f:
        print(f"Model {name} Precision {precision_num} Recall {recall_num} F1-score {f1_num} FoldId={fold_id} Auc={auc_num}",
              file=f)

    show_conf_matrix(model=model, X_test=X_test, y_test=y_test, name=f"{name}-{fold_id}", output_path=output_path)
    dump_precision_recall_curve(model, X_test=X_test, y_test=y_test, name=f"{name}-prc-{fold_id}",
                                output_path=output_path)
    return f1_num


def dump_precision_recall_curve(estimator, X_test, y_test, name, output_path="./"):
    PrecisionRecallDisplay.from_estimator(estimator=estimator, X=X_test, y=y_test)
    plt.savefig(f"{output_path}/{name}.png")


def config_dump(output_path, config, extra_params):
    with open(f"{output_path}/config.json", 'a+') as f:
        data = {"config": config, "extra_params": extra_params}
        json.dump(data, f)


def up_down_sample(X, y, class_action=""):
    if class_action == "undersample":
        rus = NeighbourhoodCleaningRule()  # RandomUnderSampler()
        X, y = rus.fit_resample(X, y)
    elif class_action == "upsample":
        rus = SMOTE()
        X, y = rus.fit_resample(X, y)
    return X, y


def run_experiments(config,
                    mode,
                    _id,
                    target_attribute,
                    current_time,
                    should_normalize=True,
                    should_onehotenc=False,
                    train_test_split_method='split',
                    kfold_strategy="stratified",
                    n_splits=5,
                    phase=None,
                    include_sequence=False,
                    class_action="X",
                    lse_size=64,
                    timesteps=30,
                    ):
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

    output_path = f"experiments/{train_test_split_method}/{phase}/{target_attribute}/{class_action}/{current_time}/{_id}"
    os.makedirs(name=output_path, exist_ok=True)

    extra_params = {
        "mode": mode,
        "_id": _id,
        "target_attribute": target_attribute,
        "should_normalize": should_normalize,
        "should_onehotenc": should_onehotenc,
        "train_test_split_method": train_test_split_method,
        "kfold_strategy": kfold_strategy,
        "n_splits": n_splits,
        "phase": phase,
        "include_sequence": include_sequence,
        "class_action": class_action,
        "lse_size": lse_size,
        "timesteps": timesteps

    }
    config_dump(output_path=output_path, config=config.to_dict(), extra_params=extra_params)
    print(data.shape)
    label_encoder = LabelEncoder()
    std_encoder = MinMaxScaler()  # StandardScaler()
    X = data

    na_columns = ["sequences", "passive_alert", "tactical_situation", "misses"]
    for na_column in na_columns:
        if na_column in X.columns:
            X[na_column].fillna(0, inplace=True)

    if should_normalize:
        for column in X.columns:
            if is_numeric_dtype(X[column]):
                X[[column]] = std_encoder.fit_transform(X[[column]])

    if not should_onehotenc:
        for column in X.columns:
            if is_string_dtype(X[column]):
                X[column] = label_encoder.fit_transform(X[column])
    else:
        cat_cols = [column for column in X.columns if is_string_dtype(X[column])]
        cat_cols_encoded = []
        for col in cat_cols:
            cat_cols_encoded += [f"{col[0]}_{cat}" for cat in list(X[col].unique())]
        oh_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded_cols = oh_encoder.fit_transform(X[cat_cols])
        df_enc = pd.DataFrame(encoded_cols, columns=cat_cols_encoded)
        X = X.reset_index().join(df_enc, how='inner')
        X.drop(columns=cat_cols, inplace=True)

    print(X.shape)
    y = label_encoder.fit_transform(classes)

    if train_test_split_method == 'split':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y)
        X_train, y_train = up_down_sample(X=X_train, y=y_train, class_action=class_action)
        print(X.shape)

        run_experiment(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                       show_confmatrix=True,
                       doTest=False,
                       output_path=output_path)
    elif train_test_split_method == "ekflod":
        k_folder = StratifiedKFold(n_splits=n_splits, shuffle=True)
        if kfold_strategy != "stratified":
            k_folder = GroupKFold(n_splits=n_splits)
        do_kfold_experiment(X=X, y=y,
                            k_folder=k_folder,
                            games=games,
                            class_action=class_action,
                            do_test=False,
                            output_path=output_path)
    elif train_test_split_method == 'kfold':
        k_folder = StratifiedKFold(n_splits=n_splits, shuffle=True)
        if kfold_strategy != "stratified":
            k_folder = GroupKFold(n_splits=n_splits)
        fold_id = 0
        acc_score = 0
        for train_index, test_index in k_folder.split(X, y, groups=games):
            X_train = X.iloc[train_index, :]
            y_train = y[train_index]
            X_test = X.iloc[test_index, :]
            y_test = y[test_index]
            X_train, y_train = up_down_sample(X=X_train, y=y_train, class_action=class_action)
            fscore = train_and_test(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                    fold_id=fold_id,
                                    output_path=output_path)
            fold_id += 1
            acc_score += fscore
        avg_score = acc_score / fold_id
        with open(f"{output_path}/avg-score.txt", 'a+') as f:
            print(f"Avg score for {fold_id} folds is {avg_score}", file=f)