import time
from multiprocessing import Process

from handy.configs import get_experimental_configs
from handy.experimenter import run_experiments


def run(class_actions,
        modes,
        lse_options,
        target_attribute,
        phase="AT",
        n_splits=5,
        current_time=time.time(),
        train_test_split_method="split",
        kfold_strategy="stratified"):
    series_lengths = [25, 30]
    for class_action in class_actions:
        experiment_id = 1
        for configFilter in get_experimental_configs():
            if "centroids" in modes:
                for sequence_id  in range(2):
                    run_experiments(config=configFilter,
                                    _id=experiment_id,
                                    target_attribute=target_attribute,
                                    train_test_split_method=train_test_split_method,
                                    phase=phase,
                                    n_splits=n_splits,
                                    include_sequence=(sequence_id % 2 == 0),
                                    current_time=current_time,
                                    class_action=class_action,
                                    kfold_strategy=kfold_strategy,
                                    mode="centroids")
                    experiment_id += 1
            if "flat" in modes:
                for series_length in series_lengths:
                    for sequence_id in range(2):
                        run_experiments(config=configFilter,
                                        _id=experiment_id,
                                        target_attribute=target_attribute,
                                        train_test_split_method=train_test_split_method,
                                        phase=phase,
                                        n_splits=n_splits,
                                        include_sequence=(sequence_id % 2 == 0),
                                        timesteps=series_length,
                                        current_time=current_time,
                                        class_action=class_action,
                                        kfold_strategy=kfold_strategy,
                                        mode="flat")
                        experiment_id += 1

        for configFilter in get_experimental_configs():
            for lse_option in lse_options:
                run_experiments(config=configFilter,
                                _id=experiment_id,
                                target_attribute=target_attribute,
                                train_test_split_method=train_test_split_method,
                                phase=phase,
                                n_splits=n_splits,
                                current_time=current_time,
                                class_action=class_action,
                                lse_size=lse_option[0],
                                timesteps=lse_option[1],
                                kfold_strategy=kfold_strategy,
                                mode="ls")
                experiment_id += 1


if __name__ == '__main__':
    train_test_split_method = 'ekflod'  # kfold, split, ekflod
    n_splits = 10  # game_phases possession_result organized_game offense_type time_in_seconds throw_zone

    #target_attribute = "organized_game"
    #kfold_strategy = "stratified"
    phase = "AT"
    current_time = time.time()

    class_actions = ["upsample", "none"]
    modes = ["centroids", "flat", "ls"]
    lse_options = [(64, 25), (64, 30), (128, 25), (128, 30)]

    procs = []
    clazzes = ["organized_game", "possession_result"]

    for clazz in clazzes:
        for kfold_strategy in ["stratified", "group"]:
            if kfold_strategy == "group":
                n_splits = 9
            else:
                n_splits = 10
            p1 = Process(target=run, args=(class_actions, modes, lse_options, clazz, phase,
                                           n_splits, time.time(), train_test_split_method,
                                           kfold_strategy))
            procs.append(p1)
            p1.start()

    print(f"Working {len(procs)} in parallel....")

    for proc in procs:
        proc.join()

    #run(class_actions=class_actions,
    #    target_attribute=target_attribute,
    #    phase=phase,
    #    current_time=current_time,
    #    modes=modes,
    #    n_splits=n_splits,
    #    train_test_split_method=train_test_split_method,
    #    kfold_strategy=kfold_strategy,
    #    lse_options=lse_options)
