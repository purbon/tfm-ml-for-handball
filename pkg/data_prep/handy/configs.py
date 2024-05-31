import os

from handy.datasets import ConfigFilter


def methods_configs():
    configs = {}
    configs["knn"] = {
        'params': {'n_neighbors': [5, 7, 10, 15, 20, 25],
                   'weights': ['distance', 'uniform'],
                   'p': [1, 2, 3]},
        'name': 'knn'
    }
    configs["randomforest"] = {
        'params': {'n_estimators': [100, 150, 200, 300, 400],
                   'criterion': ["gini", "entropy", "log_loss"],
                   'max_depth': [None, 2, 3, 6]},
        'name': 'randomforest'
    }
    # configs["gradientboosting"] = {
    #    "params": {'n_estimators': [100, 150, 200, 300, 400, 500],
    #               'max_depth': [None, 2, 3, 6],
    #               'learning_rate': [0.5, 0.1, 0.05, 0.01, 0.001]},
    #    "name": "gradientboosting"
    # }
    configs["adaboost"] = {
        "params": {'n_estimators': [50, 100, 150, 200, 300, 400],
                   'algorithm': ["SAMME"],
                   'learning_rate': [0.5, 0.1, 0.05, 0.01, 0.001]},
        "name": "adaboost"
    }
    configs["hist-lgbm"] = {
        "params": {'learning_rate': [0.5, 0.1, 0.05, 0.01, 0.001]},
        "name": "hist-lgbm"
    }
    configs["bernoulli"] = {
        "params": {},
        "name": "bernoulli"
    }
    return configs


def get_experimental_configs():
    configs = []
    configFilter = ConfigFilter(_id=1,
                                include_centroids=True,
                                include_distance=True)
    configs.append(configFilter)
    configFilter = ConfigFilter(_id=2,
                                include_centroids=True,
                                include_distance=True,
                                include_metadata=True)
    configs.append(configFilter)
    configFilter = ConfigFilter(_id=3,
                                include_centroids=True,
                                include_distance=True,
                                include_metadata=True,
                                include_sequences=True)
    configs.append(configFilter)
    configFilter = ConfigFilter(_id=4,
                                include_centroids=True,
                                include_distance=True,
                                include_metadata=True,
                                include_sequences=True,
                                include_prev_possession_results=True)
    configs.append(configFilter)
    configFilter = ConfigFilter(_id=5,
                                include_centroids=True,
                                include_distance=True,
                                include_metadata=True,
                                include_sequences=True,
                                include_vel=True,
                                include_acl=True,
                                include_prev_possession_results=True,
                                include_faults=True,
                                include_breaks=True)
    configs.append(configFilter)
    configFilter = ConfigFilter(_id=6,
                                include_centroids=True,
                                include_distance=True,
                                include_sequences=True)
    configs.append(configFilter)
    configFilter = ConfigFilter(_id=7,
                                include_centroids=True,
                                include_distance=True,
                                include_sequences=True,
                                include_prev_possession_results=True)
    configs.append(configFilter)
    return configs


class ExperimentResults:

    def __init__(self, type="ekflod"):
        self.type = type

    #   self.paths = {
    #       'organized_game': {
    #           'none': {
    #               'stratified': "experiments/ekflod/AT/organized_game/none/1706872386.2240021",
    #               'group': "experiments/ekflod/AT/organized_game/none/1706992066.711571"
    #           },
    #           'upsample': {
    #               'stratified': "experiments/ekflod/AT/organized_game/upsample/1706872386.2240021",
    #               'group': "experiments/ekflod/AT/organized_game/upsample/1706992066.711571"
    #           }
    #       },
    #       'possession_result': {
    #           'none': {
    #               'stratified': "experiments/ekflod/AT/possession_result/none/1706872572.343076",
    #               'group': "experiments/ekflod/AT/possession_result/none/1706992077.176169"
    #           },
    #           'upsample': {
    #               'stratified': "experiments/ekflod/AT/possession_result/upsample/1706872572.343076",
    #               'group': "experiments/ekflod/AT/possession_result/upsample/1706992077.176169"
    #           }
    #       }
    #   }

    def get_group(self, target_attribute, sampling_strat="upsample", kfold_strat="stratified"):
        if self.type == "ekflod":
            phase = "AT-0"
        else:
            phase = "AT"
        root_path = os.path.join(os.getcwd(), "experiments", self.type, phase, target_attribute, sampling_strat)
        dirs = sorted(os.listdir(root_path), reverse=True)
        if self.type == "ekflod":
            dir_id = 0 if kfold_strat == "stratified" else 1
            file_path = os.path.join(root_path, dirs[dir_id])
        elif self.type == "keras_kf":
            file_path = os.path.join(root_path, dirs[0])
        else:
            raise Exception("Wrong experiment type")
        return file_path
