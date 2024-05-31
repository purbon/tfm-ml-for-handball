import unittest

from handy.stats import parse_train_test_file


class StatsTestCase(unittest.TestCase):
    def test_parse_kfold_test_file(self):
        path = "../tests/resources/hist-lgbm-train_and_test.txt"
        kfold_stats = parse_train_test_file(file_path=path)
        print(kfold_stats)
        for key in ["precision", "recall", "f1-score"]:
            self.assertEqual(10, len(kfold_stats.metrics[key]))
        self.assertEqual("hist-lgbm", kfold_stats.model)


if __name__ == '__main__':
    unittest.main()
