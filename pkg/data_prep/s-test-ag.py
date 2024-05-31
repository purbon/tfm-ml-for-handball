import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from handy.analyzer import data_collector
from handy.configs import ExperimentResults
from experiment_keras_kf import get_knn_experiment_configs

if __name__ == "__main__":
    experiment_type = "keras_kf"
    kfold_strat = "group"
    target_attribute = "possession_result"

    base_dir = ExperimentResults(type=experiment_type).get_group(target_attribute=target_attribute,
                                                                 kfold_strat=kfold_strat)
    use_averages = False
    df = pd.DataFrame()
    stats = data_collector(base_dir=base_dir)
    collections = {}
    configs = {}
    for stat in stats:
        run_id = stat.path_metadata['run_id']
        algorithm_id = stat.algorithm
        config_id = stat.config["config"]["id"]
        if experiment_type == "keras_kf":
            if stat.config["config"]["kfoldstrategy"] != kfold_strat:
                continue
        if collections.get(config_id) is None:
            collections[config_id] = {}

        if collections[config_id].get(algorithm_id) is None:
            collections[config_id][algorithm_id] = {}
        if collections[config_id][algorithm_id].get(run_id) is None:
            collections[config_id][algorithm_id][run_id] = []

        collections[config_id][algorithm_id][run_id].append(stat)

    n_algos = ["adaboost", "bernoulli", "hist", "knn", "randomforest"]
    if experiment_type == "keras_kf":
        n_algos = ["lstm", "transformer", "dense", "lstm2"]

    f_view = {}
    n_configs_range = sorted(collections.keys())
    for n_config in n_configs_range:
        if f_view.get(n_config) is None:
            f_view[n_config] = {}
        algorithm_keys = collections[n_config].keys()
        for algorithm_key in algorithm_keys:
            f_view[n_config][algorithm_key] = np.zeros(0)
        for algorithm_key in algorithm_keys:
            run_ids = collections[n_config][algorithm_key].keys()
            run_ids = sorted(run_ids, key=lambda x: int(x))
            for run_id in run_ids:
                my_runs = collections[n_config][algorithm_key][run_id]
                for my_run in my_runs:
                    f1_metric = my_run.result.metrics["f1-score"]
                    if use_averages:
                        f1_metric = np.average(f1_metric)
                        f_view[n_config][algorithm_key] = np.append(f_view[n_config][algorithm_key], f1_metric)
                    else:
                        f_view[n_config][algorithm_key] = np.concatenate((f_view[n_config][algorithm_key], f1_metric))

    df = pd.DataFrame()
    n_configs_range = sorted(collections.keys())
   # config_id, <list of algorithms>
    for n_config in n_configs_range:
        sample_collections = f_view[n_config]
        for a_algorithm, a_sample in sample_collections.items():
            for b_algorithm, b_sample in sample_collections.items():
                if a_algorithm == b_algorithm:
                    continue
                t_stat, p_value = ttest_ind(a_sample, b_sample)
                data = {
                    "config_id": n_config,
                    "Method A": a_algorithm,
                    "Method B": b_algorithm,
                    "t-stat": t_stat,
                    "p.value": p_value
                }
                new_row = pd.Series(data)
                df = pd.concat([df, new_row.to_frame().T], ignore_index=True)

    df['null_hipotesis'] = df.apply(lambda row: row["p.value"] < 0.05, axis=1)
    avg_label = "avg" if use_averages else "raw"

    df = df[df["null_hipotesis"] == True]
    df.to_excel(f"t-stats-ac-{avg_label}_{target_attribute}-{kfold_strat}.xlsx", index=False)
    df.to_latex(f"t-stats-ac-{avg_label}_{target_attribute}-{kfold_strat}.tex", index=False)

    configs = get_knn_experiment_configs()

    config_ids = df["config_id"].unique()
    print(config_ids)
    for config_id in config_ids:
        config = configs.__getitem__(config_id-1)
        print(config)