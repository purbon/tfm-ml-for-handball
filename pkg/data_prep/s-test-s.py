import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, f_oneway, tukey_hsd

from handy.analyzer import data_collector
from handy.configs import ExperimentResults

if __name__ == "__main__":
    target_attribute = "organized_game"
    kfold_strat = "group"
    experiment_type = "ekflod"

    base_dir = ExperimentResults(type=experiment_type).get_group(target_attribute=target_attribute,
                                             kfold_strat=kfold_strat)
    use_averages = False
    df = pd.DataFrame()
    stats = data_collector(base_dir=base_dir)
    collections = {}
    configs = {}
    for stat in stats:
        run_id = stat.path_metadata['run_id']
        if experiment_type == "keras_kf":
            if stat.config["config"]["kfoldstrategy"] != kfold_strat:
                continue
        if experiment_type == "ekflod":
            config_id = stat.config["extra_params"]["mode"]
        else:
            config_id = stat.config["config"]["do_augment"]
        if collections.get(config_id) is None:
            collections[config_id] = {}
        if collections[config_id].get(run_id) is None:
            collections[config_id][run_id] = []
        collections[config_id][run_id].append(stat)

    n_configs = sorted(collections.keys())
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
                "Method A": i_config,
                "Method B": j_config,
                "t-stat": t_stat,
                "p.value": p_value
            }
            new_row = pd.Series(data)
            df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
    df['null_hipotesis'] = df.apply(lambda row: row["p.value"] < 0.05, axis=1)

    avg_label = "avg" if use_averages else "raw"
    df.to_excel(f"t-stats-s-{avg_label}_{target_attribute}-{kfold_strat}.xlsx", index=False)
    df.to_latex(f"t-stats-s-{avg_label}_{target_attribute}-{kfold_strat}.tex", index=False)

    ## Anova

    #statistic, pvalue = f_oneway(f_view['centroids'], f_view['flat'], f_view['ls'])
    #print(statistic)
    #print(pvalue)

    #res = tukey_hsd(f_view['centroids'], f_view['flat'], f_view['ls'])
    #print(res)