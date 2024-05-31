import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN, AgglomerativeClustering

from handy.datasets import centroid_handball_possession, flattened_handball_possessions, ConfigFilter


def cluster_explorer(n_clusters, X, targets, label="attack"):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(X)
    buckets = kmeans.labels_
    data = X[["team_x_centroid", "team_y_centroid"]]
    data.insert(2, 'clazz', buckets, True)
    import seaborn as sns
    sns.scatterplot(data=data, x='team_x_centroid', y='team_y_centroid', hue="clazz")
    plt.savefig(f"kmc{n_clusters}_{label}-inspect.png")
    plt.clf()
    targets.insert(2, 'clazz', buckets, True)
    targets.to_excel(f"kmc{n_clusters}_{label}.xlsx")


def dbcluster_explorer(n_clusters, X, targets, label="attack"):
    clusters = HDBSCAN(min_cluster_size=2).fit(X)
    buckets = clusters.labels_
    data = X[["team_x_centroid", "team_y_centroid"]]
    data.insert(2, 'clazz', buckets, True)
    import seaborn as sns
    sns.scatterplot(data=data, x='team_x_centroid', y='team_y_centroid', hue="clazz")
    plt.savefig(f"hdbc{n_clusters}_{label}-inspect.png")
    plt.clf()
    targets.insert(2, 'clazz', buckets, True)
    targets.to_excel(f"hdbc{n_clusters}_{label}.xlsx")


def aglcluster_explorer(n_clusters, X, targets, label="attack"):
    clusters = AgglomerativeClustering(n_clusters=n_clusters).fit(X)
    buckets = clusters.labels_
    data = X[["team_x_centroid", "team_y_centroid"]]
    data.insert(2, 'clazz', buckets, True)
    import seaborn as sns
    sns.scatterplot(data=data, x='team_x_centroid', y='team_y_centroid', hue="clazz")
    plt.savefig(f"aglc{n_clusters}_{label}-inspect.png")
    plt.clf()
    targets.insert(2, 'clazz', buckets, True)
    targets.to_excel(f"aglc{n_clusters}_{label}.xlsx")


if __name__ == '__main__':
    configFilter = ConfigFilter()
    configFilter.include_distance = True
    configFilter.include_vel = False
    configFilter.include_acl = False
    configFilter.include_metadata = True
    configFilter.include_faults = True

    data, classes = centroid_handball_possession(target_attribute=["GAME", "possession", "game_phases"],
                                                 filter=configFilter)

    #data, classes = flattened_handball_possessions(target_attribute=["GAME", "possession", "game_phases"],
    #                                               length=20,
    #                                               include_metadata=False)

    phase = "AT"
    df = data.join(classes)
    poss = df[df["game_phases"] == phase]
    print(poss.shape)

    targets = poss[["GAME", "possession"]]
    X = poss.drop(columns=["GAME", "possession", "game_phases"])
    column_labels = ['team_x_centroid', 'team_y_centroid']
    for i in range(6):
        column_labels += [f"p{i}_x_centroid", f"p{i}_y_centroid"]
    X = X[column_labels]
    # for i in range(10):
    #    n_clusters = i + 1
    n_clusters = 3
    cluster_explorer(X=X, n_clusters=n_clusters, label=phase, targets=targets)
