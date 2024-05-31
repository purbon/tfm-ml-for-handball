import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, f_oneway, tukey_hsd

from handy.analyzer import data_collector
from handy.configs import ExperimentResults

if __name__ == "__main__":
    experiment_type = "keras_kf"
    kfold_strat = "group"
    target_attribute = "organized_game"
    base_dir = ExperimentResults(type=experiment_type).get_group(target_attribute=target_attribute,
                                                                 kfold_strat=kfold_strat)
    use_averages = True
    df = pd.DataFrame()
    stats = data_collector(base_dir=base_dir)
    collections = {}
    configs = {}
    for stat in stats:
        run_id = stat.path_metadata['run_id']
        config_id = stat.algorithm
        if experiment_type == "keras_kf":
            if stat.config["config"]["kfoldstrategy"] != kfold_strat:
                continue
        if collections.get(config_id) is None:
            collections[config_id] = {}
        if collections[config_id].get(run_id) is None:
            collections[config_id][run_id] = []
        collections[config_id][run_id].append(stat)

    n_configs = [ "adaboost", "bernoulli", "hist", "knn", "randomforest"]
    if experiment_type == "keras_kf":
        n_configs = ["lstm", "transformer", "dense", "lstm2"]
    f_view = {}
    for n_config in n_configs:
        if f_view.get(n_config) is None:
            f_view[n_config] = np.zeros(0)
        config_keys = collections[n_config].keys()
        config_keys = sorted(config_keys, key=lambda x: int(x))
        for run_id in config_keys:
            my_runs = collections[n_config][run_id]
            for my_run in my_runs:
                f1_metric = my_run.result.metrics["f1-score"]
                if use_averages:
                    f1_metric = np.average(f1_metric)
                    f_view[n_config] = np.append(f_view[n_config], f1_metric)
                else:
                    f_view[n_config] = np.concatenate((f_view[n_config], f1_metric))

    df = pd.DataFrame()

    for i_config in n_configs:
        a_sample = f_view[i_config]
        for j_config in n_configs:
            if i_config == j_config:
                continue
            b_sample = f_view[j_config]
            t_stat, p_value = ttest_ind(a_sample, b_sample)
            data = {
                "method a": i_config,
                "method b": j_config,
                "t.stat": t_stat,
                "p.value": p_value
            }
            new_row = pd.Series(data)
            df = pd.concat([df, new_row.to_frame().T], ignore_index=True)

    df['null_hipotesis'] = df.apply(lambda row: row["p.value"] < 0.05, axis=1)
    avg_label = "avg" if use_averages else "raw"
    df.to_excel(f"t-stats-a-{target_attribute}-{kfold_strat}_{avg_label}.xlsx", index=False)
    df.to_latex(f"t-stats-a-{target_attribute}-{kfold_strat}_{avg_label}.tex", index=False)

    ## Anova
    n_configs = ["lstm", "transformer", "dense", "lstm2"]

    if experiment_type == "ekflod":
        statistic, pvalue = f_oneway(f_view['adaboost'],
                                     f_view['bernoulli'],
                                     f_view['hist'],
                                     f_view["knn"],
                                     f_view["randomforest"])
    else:
        statistic, pvalue = f_oneway(f_view["lstm"],
                                      f_view["transformer"],
                                      f_view["dense"],
                                      f_view["lstm2"])
    print(statistic)
    print(pvalue)

    if experiment_type == "ekflod":
        res = tukey_hsd(f_view['adaboost'],
                        f_view['bernoulli'],
                        f_view['hist'],
                        f_view["knn"],
                        f_view["randomforest"])
    else:
        res = tukey_hsd(f_view["lstm"], f_view["transformer"], f_view["dense"], f_view["lstm2"])
    print(res)