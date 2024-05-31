import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

pd.options.mode.chained_assignment = None


def print_metrics(labels, target_values, title):
    print(title)
    acc = accuracy_score(y_pred=labels, y_true=target_values)
    try:
        pre = precision_score(y_pred=labels, y_true=target_values, average="binary")
    except:
        pre = -1
    try:
        rec = recall_score(y_pred=labels, y_true=target_values)
    except:
        rec = -1
    try:
        f1s = f1_score(y_pred=labels, y_true=target_values)
    except:
        f1s = -1

    print(f"Acc={acc}, Precision={pre}, Recall={rec}, F1-Score={f1s}")


def flattenned_clustering(avg_length, include_distances=False, only_dist=False):
    df = pd.read_hdf(path_or_buf=f"handball.h5", key=f"dv0_p6-norm{avg_length}")
    target_values = label_encoder.fit_transform(df["game_phases"])

    columns = []
    for n_player in range(n_players):
        for n_row in range(avg_length):
            columns += [f"player_{n_player}_x{n_row}", f"player_{n_player}_y{n_row}"]

    if only_dist:
        columns = []
    if include_distances:
        ## add velocities and accelerations
        for n_player in range(n_players):
            columns += [f"p{n_player}_dist", f"p{n_player}_avg_vel", f"p{n_player}_p90_vel", f"p{n_player}_avg_acc", f"p{n_player}_p90_acc"]

    df = df[columns]
    X = scaler.fit_transform(X=df)

    print(" ")
    print(f"Flattening(s) {avg_length}")
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
    print_metrics(labels=kmeans.labels_, target_values=target_values, title="KMeans n_clusters=2, auto (flattening)")

    clustering = AgglomerativeClustering(n_clusters=2).fit(X)
    print_metrics(labels=clustering.labels_, target_values=target_values,
                  title="Agglomerative n_clusters=2 (flattening)")


if __name__ == '__main__':
    n_players = 6
    df = pd.read_hdf(path_or_buf=f"handball.h5", key="dv0_p6")

    columns_to_drop = ["GAME", "possession", "game_phases",
                       "score_team_a", "score_team_b", "score_diff",
                       "tactical_situation",
                       "time_in_seconds"]

    label_encoder = LabelEncoder()
    target_values = label_encoder.fit_transform(df["game_phases"])

    df.drop(columns=columns_to_drop, inplace=True)
    df[np.isnan(df)] = 0

    columns_to_drop = ["team_x_centroid", "team_y_centroid"]
    for i in range(6):
        columns_to_drop += [f"p{i}_dist_to_center"]
    xy_df = df.drop(columns=columns_to_drop)

    scaler = StandardScaler()
    X = scaler.fit_transform(X=df)
    X_xy = scaler.fit_transform(X=xy_df)

    print("Centroid(s) and distances")
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
    print_metrics(labels=kmeans.labels_, target_values=target_values, title="KMeans n_clusters=2, auto")

    clustering = AgglomerativeClustering(n_clusters=2).fit(X)
    print_metrics(labels=clustering.labels_, target_values=target_values, title="Agglomerative n_clusters=2")

    clustering = DBSCAN().fit(X_xy)
    print_metrics(labels=clustering.labels_, target_values=target_values, title="DBScan default (x,y)")

    print(" ")
    print("Centroid(s)")
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X_xy)
    print_metrics(labels=kmeans.labels_, target_values=target_values, title="KMeans n_clusters=2, auto (x,y)")

    clustering = AgglomerativeClustering(n_clusters=2).fit(X_xy)
    print_metrics(labels=clustering.labels_, target_values=target_values, title="Agglomerative n_clusters=2 (x,y)")

    for avg_length in range(5, 36, 5):
        flattenned_clustering(avg_length=avg_length, include_distances=False)
        df = pd.read_hdf(path_or_buf=f"handball.h5", key=f"dv0_p6-norm{avg_length}")
