import re

import numpy as np


class kFoldStats:

    def __init__(self, model=""):
        self.metrics = {}
        self.model = model

    def register_metric(self, name, size=10):
        self.metrics[name] = np.zeros(size, np.float32)

    def set_value(self, name, fold_id, value):
        np.put(self.metrics[name], fold_id, value)


class Result:
    def __init__(self, path, result, algorithm=None, path_metadata=None, config=None):
        self.path = path
        self.config = config
        self.result = result
        self.algorithm = algorithm
        self.path_metadata = path_metadata

    def __str__(self):
        return f"{self.algorithm}-{self.path_metadata['run_id']}"


def explode_path_configs(path):
    fields = re.split("/", path)
    return {
        'target_attribute': fields[3],
        'pre_processing': fields[4],
        'run_id': fields[-2]
    }


# Model adaboost Precision 0.6282828282828283 Recall 0.6 F1-score 0.6083333333333333 FoldId=6
def parse_train_test_file(file_path):
    stats = kFoldStats()
    stats.register_metric(name="precision")
    stats.register_metric(name="recall")
    stats.register_metric(name="f1-score")
    model = None
    regular_expression = "Model\s([\w|-]+)\sPrecision\s(\d\.\d+)\sRecall\s(\d\.\d+)\sF1-score\s(\d\.\d+)\sFoldId=(\d+)"
    with open(file_path) as file:
        for line in file:
            x = re.search(regular_expression, line)
            if x:
                if model is None:
                    model = x.group(1)
                fold_id = int(x.group(5))
                stats.set_value("precision", value=float(x.group(2)), fold_id=fold_id)
                stats.set_value("recall", value=float(x.group(3)), fold_id=fold_id)
                stats.set_value("f1-score", value=float(x.group(4)), fold_id=fold_id)
    stats.model = model
    return stats
