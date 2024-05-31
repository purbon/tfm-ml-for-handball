import json
import re
from functools import partial
from os import listdir
from os.path import isfile, join
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


class Result:
    def __init__(self, path, result, config=None):
        self.path = path
        self.config = config
        self.result = result


def accumulate_dirs(base_dir, func, results=None, ):
    if results is None:
        results = []
    for file_or_dir in listdir(base_dir):
        path = join(base_dir, file_or_dir)
        if isfile(path):  # it is a file to be processed
            config_data = None
            with open(join(base_dir, "config.json"), "r") as file:
                config_data = json.load(file)

            if path.endswith("avg-score.txt") and not path.endswith("config.json"):
                contents = Path(path).read_text().strip()
                data = float(re.split('\s+', contents)[-1].strip())
                if func(data=data):
                    result = Result(path=path, result=data, config=config_data)
                    results.append(result)
        else:
            accumulate_dirs(path, func, results)
    return results


def filter_result(data, val=0.5):
    return data > val


def explode_path_configs(path):
    fields = re.split("/", path)
    return {
        'algorithm': fields[-1].replace("-avg-score.txt", ""),
        'target_attribute': fields[3],
        'pre_processing': fields[4]
    }


if __name__ == '__main__':

    df = pd.DataFrame()

    base_dir = "experiments/ekflod/AT-0/"
    funct = partial(filter_result, val=0.0)
    results = accumulate_dirs(base_dir=base_dir, func=funct)
    results = sorted(results, key=lambda x: -1 * x.result)
    kfold_strategy = "group"  # stratified group

    for result in results:
        config = result.config
        if kfold_strategy is not None and config["extra_params"]["kfold_strategy"] != kfold_strategy:
            continue
        method = config["extra_params"]["mode"]
        has_sequence = config["extra_params"]["include_sequence"]
        print(f"{result.path} {config['config']['id']} {method} {has_sequence} {result.result}")
        path_configs = explode_path_configs(path=result.path)
        data = {'latent_space_method': method,
                'algorithm': path_configs['algorithm'],
                'config_id': config['config']['id'],
                'score': result.result,
                'has_sequences': has_sequence,
                'kfold_strategy': config["extra_params"]["kfold_strategy"],
                'target_attribute': path_configs['target_attribute'],
                'pre_processing': path_configs['pre_processing']}
        new_row = pd.Series(data)
        df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
    print(len(results))
    df.to_excel(f"kfold-{kfold_strategy}.xlsx")

    objectives = ["organized_game", "possession_result"]
    for objective in objectives:
        filtered_df = df[df["target_attribute"] == objective]
        colors = {'adaboost': 'red', 'randomforest': 'blue', 'hist-lgbm': 'green', 'knn': 'cyan', 'bernoulli': 'orange'}
        (sns.histplot(data=filtered_df, x="score", hue="algorithm", multiple="stack", palette=colors)
         .set_title(label=f"[{objective}] score histogram by algorithm"))
        plt.savefig(f"kfold-{kfold_strategy}-{objective}-hist_alg.png")
        plt.clf()
        colors = {'ls': 'red', 'centroids': 'blue', 'flat': 'orange'}
        (sns.histplot(data=filtered_df, x="score", hue="latent_space_method", multiple="stack", palette=colors)
         .set_title(label=f"[{objective}] score histogram by embedding method"))
        plt.savefig(f"kfold-{kfold_strategy}-{objective}-hist_lsm.png")
        plt.clf()
        colors = {'G1': 'brown', 'G2': 'green'}
        (sns.histplot(data=filtered_df, x="score", bins=5, hue="config_id", stat="count", multiple="dodge")
         .set_title(label=f"[{objective}] score histogram by config ID"))
        plt.savefig(f"kfold-{kfold_strategy}-{objective}-hist_config.png")
        plt.clf()
        colors = {'upsample': 'orange', 'none': 'blue'}
        (sns.histplot(data=filtered_df, x="algorithm", y="score", hue="pre_processing", palette=colors)
         .set_title(label=f"[{objective}] score histogram by pre processing algorithm"))
        plt.savefig(f"kfold-{kfold_strategy}-{objective}-hist_prepro.png")
        plt.clf()
        (sns.histplot(data=filtered_df, x="score", hue="has_sequences", multiple="stack")
         .set_title(label=f"[{objective}] score histogram by use of sequences in possessions"))
        plt.savefig(f"kfold-{kfold_strategy}-{objective}-hist_seq.png")
        plt.clf()
