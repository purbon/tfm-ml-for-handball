import os
from pprint import pprint

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.core.dtypes.common import is_numeric_dtype
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def describe_num_column(df, column):
    val = df[column]
    stats = {
        "column": column,
        "min": val.min(),
        "max": val.max(),
        "std": val.std(),
        "mean": val.mean(),
        "unique": val.nunique(),
        "count": val.count(),
    }
    return stats


def describe_str_column(df, column):
    val = df[column]
    size = val.count()

    stats = {
        "column": column,
        "unique": val.nunique(),
        "size": size,
    }
    for key, value in df.groupby(column).size().to_dict().items():
        stats[f"{key}_SIZE"] = value
        stats[f"{key}_frac"] = value / size
    return stats


def remove_columns(df):
    columns = ["GAME", "Unnamed: 14", "POS. B", "POS. B.1",
               "possession", "active_players", "active_sensors",
               "sequence_label", "live_possession_duration_in_sec"]
    for i in range(6):
        columns.append(f"player_{i}")
    return df.drop(columns=columns)


def charts_to(df, path):
    attrs = ["game_phases", "tactical_situation", "time_in_seconds", "score_diff", "HALF", "TIME_PER_HALF_IN_SECONDS"]
    for attr in attrs:
        create_chart(df=df, path=f"{path}/desc/f1-s-{attr}.png", x=attr)
        create_chart(df=df, path=f"{path}/desc/half/f1-s-{attr}.png", x=attr, hue="HALF")

    create_chart(df=df, path=f"{path}/f1-game_phases.png", x="game_phases")
    attrs = ["offense_type", "throw_zone", "organized_game"]
    for attr in attrs:
        attr_df = df.groupby(["possession", attr]).last()
        create_chart(df=attr_df, path=f"dumps/desc/f1-{attr}.png", x=attr)


def create_chart(df, path, x, hue=None):
    fig, axes = plt.subplots(figsize=(15, 8))
    if hue is None:
        sns.histplot(data=df, x=x, stat="count", ax=axes)
    else:
        sns.histplot(data=df, x=x, hue=hue, multiple="stack", ax=axes)

    fig.tight_layout()
    plt.savefig(path)
    plt.clf()



def encode_the_df(df):
    df = df.copy()
    for column in df.columns:
        le = LabelEncoder()
        if not is_numeric_dtype(df[column]):
            df[column] = le.fit_transform(df[column])
    return df

def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    return(res)

def one_hot_encoded_df(df):
    oh_df = df.copy()
    columns = oh_df.columns
    for column in columns:
        if not is_numeric_dtype(oh_df[column]):
            oh_df = encode_and_bind(original_dataframe=oh_df, feature_to_encode=column)
            oh_df.drop(columns=[column], inplace=True)
    return oh_df


def save_corr_matrix(corr_mat, filepath, title="Correlation heatmap", figsize=None):
    if figsize is None:
        plt.figure(figsize=(40, 30))
    else:
        plt.figure(figsize=(figsize, figsize / 2))

    heatmap = sns.heatmap(
        corr_mat,
        vmin=corr_mat.values.min(),
        vmax=1,
        square=True,
        linewidths=0.1,
        annot_kws={"size": 45 / np.sqrt(len(corr_mat))},
        annot=True)
    heatmap.set_title(title, fontdict={'fontsize': 12}, pad=2)
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0.01)
    plt.clf()


def plot_column_to(df, column, path, kde=False):
    filename = f"{path}/hists/{column}-hist.png"
    plt.figure(figsize=(15, 8))
    sns.histplot(data=df, x=column, kde=kde)
    plt.savefig(filename)
    plt.clf()


def describe_dataframe(df, dump=False):
    describe_stats = pd.DataFrame(columns=["min", "max", "std", "mean", "unique", "count"])
    categoricals = []
    categories = []
    for column in df.columns:
        val = df[column]
        kde = False
        if is_numeric_dtype(val):
            stats = describe_num_column(df=df, column=column)
            kde = True
        else:
            stats = describe_str_column(df=df, column=column)
        if dump:
            plot_column_to(df=df, column=column, path="dumps/desc", kde=kde)


        cat_series = pd.Series(stats)
        ser_df = pd.DataFrame([cat_series], columns=cat_series.index)
        categoricals.append(ser_df)
        categories.append(column)
        series = pd.Series(val.describe())
        ser_df = pd.DataFrame([series], columns=series.index)
        describe_stats = pd.concat([describe_stats, ser_df])
    return describe_stats, categoricals, categories


if __name__ == '__main__':
    key = "dv0_p6" # dv0_p6 pos
    df = pd.read_hdf(path_or_buf=f"handball.h5", key=key)
    grouped_df = df.groupby(["GAME", "possession"]).last()
    try:
        df = remove_columns(df=df)
    except:
        pass

    os.makedirs("dumps/desc/cats", exist_ok=True)
    os.makedirs("dumps/desc/hists", exist_ok=True)

    describe_stats, categoricals, categories = describe_dataframe(df=df, dump=False)
    describe_stats.reset_index().to_excel("dumps/desc/descriptions.xlsx")

    i = 0
    for i in range(len(categories)):
        cat_df = categoricals[i]
        category_name = categories[i]
        cat_df.to_excel(f"dumps/desc/cats/{category_name}.xlsx")

    # charts_to(df=grouped_df, path="dumps")

    ## Encoded DFs and Log10 version

    offense_df = df[df["game_phases"] == "AT"]

    describe_stats, categoricals, categories = describe_dataframe(df=offense_df)
    os.makedirs("dumps/desc/at", exist_ok=True)
    describe_stats.reset_index().to_excel("dumps/desc/at/descriptions.xlsx")

    i = 0
    for i in range(len(categories)):
        cat_df = categoricals[i]
        category_name = categories[i]
        cat_df.to_excel(f"dumps/desc/at/{category_name}.xlsx")

    encoded_df = encode_the_df(df=df)
    onehot_encoded_df = one_hot_encoded_df(df=df)
    encoded_offense_df = encode_the_df(df=offense_df)

    log10_df = encoded_df.copy()
    avoid_columns = ["passive_alert", "PEN_count", "TM_count", "FK_count", "Prev_TM", "Post_TM", "goal"]

    for column in log10_df.columns:
        if not avoid_columns.__contains__(column):
            log10_df[column] = np.log10(log10_df[column])

    ## correlation matrix
    save_corr_matrix(corr_mat=onehot_encoded_df.corr(numeric_only=True),
                     filepath=f"dumps/desc/poss_corr_heatmap-onehot.png",
                     title=f"Full correlation matrix")

    encoded_df.drop(columns=["Post_TM"], inplace=True)
    save_corr_matrix(corr_mat=encoded_df.corr(numeric_only=True),
                     filepath=f"dumps/desc/poss_corr_heatmap.png",
                     title=f"Full correlation matrix")

    save_corr_matrix(corr_mat=log10_df.corr(numeric_only=True),
                     filepath=f"dumps/desc/poss_corr_heatmap-log10.png",
                     title=f"Full correlation matrix")

    encoded_offense_df.drop(columns=["game_phases", "Post_TM"], inplace=True)
    save_corr_matrix(corr_mat=encoded_offense_df.corr(numeric_only=True),
                     filepath=f"dumps/desc/poss_offense_corr_heatmap.png",
                     title=f"Offense correlation matrix")

    defense_df = df[df["game_phases"] == "DEF"]

    describe_stats, categoricals, categories = describe_dataframe(df=defense_df)
    os.makedirs("dumps/desc/def", exist_ok=True)
    describe_stats.reset_index().to_excel("dumps/desc/def/descriptions.xlsx")

    i = 0
    for i in range(len(categories)):
        cat_df = categoricals[i]
        category_name = categories[i]
        cat_df.to_excel(f"dumps/desc/def/{category_name}.xlsx")