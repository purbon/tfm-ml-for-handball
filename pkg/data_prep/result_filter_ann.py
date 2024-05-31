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
    def __init__(self, path, result, precision=None, recall=None, auc=None, config=None):
        self.path = path
        self.config = config
        self.result = result
        self.precision = precision
        self.recall = recall
        self.auc = auc


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
                fields = re.split("\s+", contents)
                # precision 7, recall 8, auc 9, f1 6
                f1 = float(fields[6].replace("f1=", "").replace(",", ""))
                precision = float(fields[7].replace("precision=", "").replace(",", ""))
                recall = float(fields[8].replace("recall=", "").replace(",", ""))
                auc = float(fields[9].replace("auc=", "").replace(",", ""))
                if func(data=f1):
                    result = Result(path=path,
                                    result=f1, precision=precision, recall=recall, auc=auc,
                                    config=config_data)
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

    base_dir = "experiments/keras_kf/AT/"
    funct = partial(filter_result, val=0.5)
    results = accumulate_dirs(base_dir=base_dir, func=funct)
    results = sorted(results, key=lambda x: -1 * x.result)
    kfold_strategy = "group"  # "stratified" # stratified group

    for result in results:
        config = result.config
        if kfold_strategy is not None and config["config"]["kfoldstrategy"] != kfold_strategy:
            continue
        print(f"{result.path} {config['config']['id']} {result.result}")
        path_configs = explode_path_configs(path=result.path)
        data = {'algorithm': path_configs['algorithm'],
                'config_id': config['config']['id'],
                'f1': result.result,
                'precision': result.precision,
                'recall': result.recall,
                'auc': result.auc,
                'kfold_strategy': config["config"]["kfoldstrategy"],
                'target_attribute': path_configs['target_attribute'],
                'pre_processing': path_configs['pre_processing'],
                'do_augment': config["config"]["do_augment"]}
        new_row = pd.Series(data)
        df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
    print(len(results))
    df.to_excel(f"keras-{kfold_strategy}.xlsx")

    objectives = ["organized_game", "possession_result"]
    for objective in objectives:
        filtered_df = df[df["target_attribute"] == objective]
        metrics = ["f1", "precision", "recall", "auc"]
        colors = {'dense': 'blue', 'lstm': 'orange', 'lstm2': 'green', 'transformer': 'red'}
        for metric in metrics:
            (sns.histplot(data=filtered_df, x=metric, hue="algorithm", multiple="stack", palette=colors)
             .set_title(label=f"[{objective}] score histogram by {metric} metric"))
            plt.savefig(f"keras-{kfold_strategy}-{objective}-hist_alg-{metric}.png")
            plt.clf()

        colors = {'upsample': 'blue', 'undersample': 'orange', 'none': 'green'}
        (sns.histplot(data=filtered_df, x="algorithm", y="f1", hue="pre_processing", palette=colors)
         .set_title(label=f"[{objective}] score histogram by pre processing algorithm"))
        plt.savefig(f"keras-{kfold_strategy}-{objective}-hist_prepro.png")
        plt.clf()
        (sns.histplot(data=filtered_df, x="algorithm", y="f1", hue="do_augment")
         .set_title(label=f"[{objective}] score histogram by data augmentation"))
        plt.savefig(f"keras-{kfold_strategy}-{objective}-hist_augment.png")
        plt.clf()
