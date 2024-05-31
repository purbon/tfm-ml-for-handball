import json
import re
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, f_oneway, tukey_hsd

from handy.configs import ExperimentResults
from handy.stats import parse_train_test_file, explode_path_configs, Result


def data_collector(base_dir, results=None):
    if results is None:
        results = []
    for file_or_dir in listdir(base_dir):
        path = join(base_dir, file_or_dir)
        if isfile(path):  # it is a file to be processed
            config_data = None
            with open(join(base_dir, "config.json"), "r") as file:
                config_data = json.load(file)

            if path.endswith("-train_and_test.txt") and not path.endswith("config.json"):
                content_stats = parse_train_test_file(file_path=path)
                algorithm = re.split("\-", path)[0].split("/")[-1]
                path_metadata = explode_path_configs(path=path)

                result = Result(path=path,
                                result=content_stats,
                                config=config_data,
                                algorithm=algorithm,
                                path_metadata=path_metadata)

                results.append(result)
        else:
            data_collector(base_dir=path, results=results)
    return results


def compile_a_collection_for(target_attribute, sampling, kfold_strategy):
    base_dir = ExperimentResults().get_group(target_attribute=target_attribute,
                                             sampling_strat=sampling,
                                             kfold_strat=kfold_strategy)
    stats = data_collector(base_dir=base_dir)
    collections = {}
    for stat in stats:
        run_id = int(stat.path_metadata['run_id'])
        if collections.get(run_id) is None:
            collections[run_id] = []
        collections[run_id].append(stat)
    return collections


if __name__ == "__main__":
    df = pd.DataFrame()
    # stratified group
    # organized_game possesion_result

    target_attribute = "organized_game"
    kfold_strat = "stratified"

    up_collection = compile_a_collection_for(target_attribute=target_attribute,
                                             kfold_strategy=kfold_strat,
                                             sampling="upsample")

    none_collection = compile_a_collection_for(target_attribute=target_attribute,
                                               kfold_strategy=kfold_strat,
                                               sampling="none")

    n_configs = ["none", "upsample"]
    f_view = {}
    for n_config in n_configs:
        if f_view.get(n_config) is None:
            f_view[n_config] = np.zeros(0)
        collections = up_collection if n_config == "upsample" else none_collection
        for run_id in range(1, 71, 1):
            my_runs = collections[run_id]
            for my_run in my_runs:
                f1_metric = my_run.result.metrics["f1-score"]
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
                "i_config": i_config,
                "j_config": j_config,
                "t_stat": t_stat,
                "p_value": p_value
            }
            new_row = pd.Series(data)
            df = pd.concat([df, new_row.to_frame().T], ignore_index=True)

    df.to_excel(f"t-stats-sampl_{target_attribute}-{kfold_strat}.xlsx", index=False)

    ## Anova

    statistic, pvalue = f_oneway(f_view['none'], f_view['upsample'])
    print(statistic)
    print(pvalue)

    res = tukey_hsd(f_view['none'], f_view['upsample'])
    print(res)
