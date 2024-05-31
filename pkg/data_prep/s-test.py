import os

import pandas as pd
from scipy.stats import ttest_ind

from handy.analyzer import data_collector
from handy.configs import ExperimentResults
from experiment_keras_kf import get_knn_experiment_configs

if __name__ == "__main__":
    # possession_result organized_game
    target_attribute = "possession_result"
    experiment_type = "keras_kf"
    kfold_strat = "group"
    base_dir = ExperimentResults(type=experiment_type).get_group(target_attribute=target_attribute,
                                                                 kfold_strat=kfold_strat)
    df = pd.DataFrame()

    stats = data_collector(base_dir=base_dir)
    collections = {}
    configs = {}
    run_ids = []
    for stat in stats:
        run_id = stat.path_metadata['run_id']
        lsm_mode = stat.config.get("extra_params", {}).get("mode", None)
        if experiment_type == "keras_kf":
            if stat.config["config"]["kfoldstrategy"] != kfold_strat:
                continue

        run_ids.append(int(run_id))
        if collections.get(run_id) is None:
            collections[run_id] = {}
            configs[run_id] = {}

        algorithm = stat.algorithm
        collections[run_id][algorithm] = stat.result
        configs[run_id] = stat.config
    run_ids = list(set(run_ids))
    for a_run_id in run_ids:
        one_run_key = str(a_run_id)
        one_run = collections[one_run_key]
        one_config = configs[one_run_key]
        one_config_id = one_config["config"]["id"]
        one_mode = one_config.get("extra_params", {}).get("mode", None)

        print(f"Dealing with {one_run_key}")
        for an_algorithm, algo_run1 in one_run.items():
            sample1 = algo_run1.metrics["f1-score"]

            for another_run_id in run_ids:
                #if another_run_id < a_run_id:
                #    continue

                another_run_key = str(another_run_id)
                another_run = collections[another_run_key]
                another_config = configs[another_run_key]
                another_config_id = another_config["config"]["id"]
                another_mode = another_config.get("extra_params", {}).get("mode", None)

                for another_algorithm, algo_run2 in another_run.items():
                    if an_algorithm == another_algorithm and one_run_key == another_run_key:
                        continue
                    sample2 = algo_run2.metrics["f1-score"]
                    t_stat, p_value = ttest_ind(sample1, sample2)

                    data = {
                        'a_run_id': one_run_key,
                        'another_run_id': another_run_key,
                        'a_config_id': one_config_id,
                        'another_config_id': another_config_id,
                        'one_mode': one_mode,
                        'another_mode': another_mode,
                        'an_algorithm': an_algorithm,
                        'another_algorithm': another_algorithm,
                        't_stat': t_stat,
                        'p_value': p_value
                    }
                    new_row = pd.Series(data)
                    df = pd.concat([df, new_row.to_frame().T], ignore_index=True)

    df.to_excel(f"t-stats-{target_attribute}-{kfold_strat}.xlsx", index=False)

    ## dump configs

    configs = get_knn_experiment_configs()
    for config in configs:
        if not run_ids.__contains__(config.id):
            continue
        print(config)