from pandas.core.dtypes.common import is_numeric_dtype, is_string_dtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from handy.configs import get_experimental_configs
from handy.datasets import centroid_handball_possession, flattened_handball_possessions, latent_space_encoded_possession
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
    return data, classes, games


if __name__ == "__main__":
    mode = "centroids"
    target_attribute = "organized_game"
    include_sequence = False
    kfold_strategy = "group"
    n_splits = 9
    class_action = "upsample"
    should_normalize = True
    method_name = "randomforest"

    configs = get_experimental_configs()
    config = configs[3]
    lse_size, timesteps = 64, 25

    data, classes, games = get_data(mode=mode,
                                    target_attribute=target_attribute,
                                    include_sequence=include_sequence,
                                    phase="AT",
                                    config=config,
                                    lse_size=lse_size,
                                    timesteps=timesteps)

    label_encoder = LabelEncoder()
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
            X[column] = label_encoder.fit_transform(X[column])

    y = label_encoder.fit_transform(classes)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X=X_train, y=y_train)

    y_pred = model.predict(X=X_test)

    fnr, fpr, tnr, tpr, bacc = get_score_rates(y_true=y_test, y_pred=y_pred)

    print(f"fnr={fnr}, tnr={tnr}, fpr={fpr}, tpr={tpr}")
    print(f"bacc={bacc}")