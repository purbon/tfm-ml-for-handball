import os
import os
import time
from multiprocessing import Process

import keras
from keras_tuner.src.backend.io import tf
from sklearn.model_selection import StratifiedKFold, GroupKFold

from data_pre.embeddings import raw_handball_possessions
from handy.datasets import KerasConfig
from handy.experimenter import config_dump, up_down_sample
from handy.models.ann import get_ann_model


def do_ann_experiment(config,
                      target_class,
                      model_type,
                      normalizer,
                      output_path="./"):
    callbacks = [keras.callbacks.EarlyStopping(patience=10, monitor="loss", restore_best_weights=True)]

    X, y, G = raw_handball_possessions(target_class=target_class,
                                       timesteps=config.timesteps,
                                       game_phase="AT",
                                       augment=config.do_augment,
                                       normalizer=normalizer)

    k_folder = StratifiedKFold(n_splits=config.n_splits, shuffle=True)
    if config.kfold_strategy != "stratified":
        k_folder = GroupKFold(n_splits=config.n_splits)

    fold_id = 0
    acc_scores = {"auc": 0, "precision": 0, "recall": 0, "f1": 0}

    print(f"ModelType={model_type} target_class={target_class}")
    for train_index, test_index in k_folder.split(X=X, y=y, groups=G):
        print(f"FoldId = {fold_id}")

        X_train = X[train_index, :]
        y_train = y[train_index, :]
        X_test = X[test_index, :]
        y_test = y[test_index, :]

        if normalizer:
            X_train, y_train = up_down_sample(X=X_train, y=y_train, class_action=config.class_action)

        model = get_ann_model(model_type=model_type, timesteps=config.timesteps, n_fields=12)
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            optimizer='adam',
            # optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-4),
            metrics=["binary_accuracy",
                     tf.keras.metrics.AUC(from_logits=False),
                     tf.keras.metrics.Precision(),
                     tf.keras.metrics.Recall()],
        )
        # model.summary()

        model.fit(
            X_train,
            y_train,
            epochs=config.epochs,
            batch_size=config.batch_size,
            callbacks=callbacks,
            verbose=0
        )

        score = model.evaluate(X_test,
                               y_test,
                               batch_size=config.batch_size,
                               verbose=0)
        print(model.metrics_names)
        print(score)

        precision_num = score[3]
        recall_num = score[4]
        if (precision_num + recall_num) == 0:
            f1_num = 0
        else:
            f1_num = 2 * (precision_num * recall_num) / (precision_num + recall_num)
        auc_num = score[2]

        acc_scores["auc"] += auc_num
        acc_scores["precision"] += precision_num
        acc_scores["recall"] += recall_num
        acc_scores["f1"] += f1_num

        with open(f"{output_path}/{model_type}-train_and_test.txt", 'a+') as f:
            print(f"Model {model_type} Precision {precision_num} Recall {recall_num} F1-score {f1_num} "
                  f"FoldId={fold_id} Auc={auc_num}",
                  file=f)
        fold_id += 1

    avg_auc = acc_scores["auc"] / fold_id
    avg_prec = acc_scores["precision"] / fold_id
    avg_rec = acc_scores["recall"] / fold_id
    avg_f1 = acc_scores["f1"] / fold_id

    with open(f"{output_path}/{model_type}-avg-score.txt", 'a+') as f:
        print(f"Avg score for {fold_id} folds is f1={avg_f1}, precision={avg_prec}, recall={avg_rec}, auc={avg_auc}",
              file=f)


def get_knn_experiment_configs():
    configs = []
    epochs = 50
    config_id = 1
    batch_sizes = [32, 64]
    timesteps = [20, 30, 35, 40, 45]
    for kfold_strategy in ["stratified", "group"]:
        n_splits = 10 if kfold_strategy == "stratified" else 9
        for batch_size in batch_sizes:
            for timestep in timesteps:
                for do_augment in [False, True]:
                    config = KerasConfig(id=config_id,
                                         kfold_strategy=kfold_strategy,
                                         n_splits=n_splits,
                                         timesteps=timestep,
                                         epochs=epochs,
                                         batch_size=batch_size,
                                         do_augment=do_augment)
                    configs.append(config)
                    config_id += 1
    return configs


def run_experiment(target_class,
                   do_augment,
                   current_time,
                   phase="AT"
                   ):
    _experiment_id = 1

    for config in get_knn_experiment_configs():
        class_action = do_augment
        output_path = f"experiments/keras_kf/{phase}/{target_class}/{class_action}/{current_time}/{_experiment_id}"
        os.makedirs(name=output_path, exist_ok=True)

        extra_params = {
            "_id": _experiment_id,
            "target_attribute": target_class,
            "phase": phase,
            "class_action": class_action
        }

        config_dump(output_path=output_path, config=config.to_dict(), extra_params=extra_params)
        config.class_action = class_action
        models = ["dense", "lstm", "lstm2", "transformer"]
        for model_type in models:
            normalizer = (model_type != "lstm2" and model_type != "transformer")
            if not normalizer:
                if class_action == "upsample":
                    config.do_augment = True
                elif class_action == "undersample":
                    continue
                elif class_action == "none":
                    config.do_augment = False
            try:
                do_ann_experiment(config=config,
                                  target_class=target_class,
                                  model_type=model_type,
                                  normalizer=normalizer,
                                  output_path=output_path)
            except:
                print(f"Something went wrong on experiment {_experiment_id} using #{model_type}")
        _experiment_id += 1


if __name__ == "__main__":
    # model_type = "dense"  # lstm dense lstm2 transformer
    # organized_game possession_result

    clazzes = ["organized_game", "possession_result"]
    resamplings = ["undersample", "upsample", "none"]
    procs = []

    for clazz in clazzes:
        for resampling in resamplings:
            p1 = Process(target=run_experiment, args=(clazz, resampling, time.time()))
            procs.append(p1)
            p1.start()

    print(f"Working {len(procs)} in parallel....")

    for proc in procs:
        proc.join()

    # run_experiment(target_class="organized_game",
    #               current_time=time.time(),
    #               phase="AT",
    #               do_augment=True)

    # model = get_dense_model()
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

    # model.fit(trainGenerator,
    #          epochs=epochs,
    #          batch_size=batch_size,
    #          validation_data=valGenerator)

    # score = model.evaluate(testGenerator,
    #                       batch_size=batch_size,
    #                       verbose=0)
    # print(model.metrics_names)
    # print(score)
