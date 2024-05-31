import json
import re
from os import listdir
from os.path import isfile, join

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
                algorithm = re.split("/", path)[-1].split("-")[0]
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


class AnalyzerCollector:

    def __init__(self):
        pass
