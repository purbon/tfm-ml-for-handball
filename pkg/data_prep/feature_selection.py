import os
import time

from pandas.core.dtypes.common import is_numeric_dtype, is_string_dtype
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile, f_classif, mutual_info_classif, SelectFdr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from handy.datasets import centroid_handball_possession, ConfigFilter, flattened_handball_possessions, \
    latent_space_encoded_possession
from handy.experimenter import up_down_sample, run_experiment

if __name__ == "__main__":
    target_attribute = "possession_result"
    phase = "AT"
    include_sequence = False
    class_action = "upsample"
    current_time = time.time()

    config_filter = ConfigFilter(_id=1,
                                 include_centroids=True,
                                 include_distance=True,
                                 include_sequences=True,
                                 include_prev_possession_results=True,
                                 include_breaks=True,
                                 include_acl=True,
                                 include_vel=True,
                                 include_faults=True,
                                 include_metadata=True,
                                 include_scores=True,
                                 include_prev_score_diff=True)

    action_type = "centroids"
    lse_size = 128
    timesteps = 30

    if action_type == "centroids":
        data, classes = centroid_handball_possession(target_attribute=target_attribute,
                                                     phase=phase,
                                                     include_sequences=include_sequence,
                                                     filter=config_filter)
    if action_type == "flat":
        data, classes = flattened_handball_possessions(target_attribute=target_attribute,
                                                       length=25,
                                                       phase=phase,
                                                       include_sequences=include_sequence,
                                                       filter=config_filter)
    if action_type == "ls":
        data, classes = latent_space_encoded_possession(method_type="lstm",
                                                        lse_size=lse_size,
                                                        phase=phase,
                                                        timesteps=timesteps,
                                                        target_attribute=target_attribute,
                                                        filter=config_filter)

    std_encoder = MinMaxScaler()
    label_encoder = LabelEncoder()

    X = data
    y = label_encoder.fit_transform(classes)

    for column in X.columns:
        if is_numeric_dtype(X[column]):
            X[column].fillna(0, inplace=True)
            X[[column]] = std_encoder.fit_transform(X[[column]])
        if is_string_dtype(X[column]):
            X[column].fillna("na", inplace=True)
            X[column] = label_encoder.fit_transform(X[column])

    print(data.shape)
    # kbest = SelectKBest(f_classif, k=10)
    kbest = SelectPercentile(mutual_info_classif, percentile=10)
    X_new = kbest.fit_transform(X, y)
    print(X_new.shape)

    feature_names = kbest.get_feature_names_out()
    print(feature_names)

    output_path = f"experiments/feature_selection/{phase}/{target_attribute}/{action_type}/{class_action}/{current_time}/1"
    os.makedirs(name=output_path, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.20, stratify=y)
    X_train, y_train = up_down_sample(X=X_train, y=y_train, class_action=class_action)

    run_experiment(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                   show_confmatrix=True,
                   doTest=False,
                   output_path=output_path)
