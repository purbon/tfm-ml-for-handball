import json
from os import listdir
from os.path import isfile, join


class Result:
    def __init__(self, path, result, config=None):
        self.path = path
        self.config = config
        self.result = result


def accumulate_dirs(base_dir, func, results=None,):
    if results is None:
        results = []
    for file_or_dir in listdir(base_dir):
        path = join(base_dir, file_or_dir)
        if isfile(path):  # it is a file to be processed
            config_data = None
            with open(join(base_dir, "config.json"), "r") as file:
                config_data = json.load(file)

            if path.endswith(".json") and not path.endswith("-scores.json") and not path.endswith("config.json"):
                data = None
                with open(path, "r") as file:
                    data = json.load(file)
                if func(data=data):
                    result = Result(path=path, result=data, config=config_data)
                    results.append(result)
        else:
            accumulate_dirs(path, func, results)
    return results


def filter_avg_result(data, val=0.85):
    return data["weighted avg"]["precision"] >= val and data["weighted avg"]["recall"] >= val


def filter_class_result(data, val=0.75):
    zero = data["0"]["precision"] > val and data["0"]["recall"] > val
    one = data["1"]["precision"] > val and data["1"]["recall"] > val
    return zero and one


if __name__ == '__main__':
    base_dir = "experiments/gen/"
    results = accumulate_dirs(base_dir=base_dir, func=filter_class_result)
    results = sorted(results, key=lambda x: x.result["weighted avg"]["precision"] + x.result["weighted avg"]["recall"])
    for result in results:
        config = result.config
        if True:  # config["config"]["id"] == 1:
            print(f"{result.path} {config['config']['id']}")
            # print(result.result)
            # print()
    print(len(results))
