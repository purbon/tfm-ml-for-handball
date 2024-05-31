import unittest

import pandas as pd

from data_io.meta import Schema
from data_pre.embeddings import DataGameGenerator


class KerasGeneratorTestCase(unittest.TestCase):
    def test_first_batch_gen(self):
        df = pd.read_hdf(path_or_buf=f"../handball.h5", key="pos")

        df.drop(columns=["player_0", "player_1", "player_2", "player_3", "player_4", "player_5"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        df = df[df[Schema.ACTIVE_PLAYERS_COLUMN] == 6]

        trainGenerator = DataGameGenerator(df=df, timesteps=70, start=0, end=2)

        x, y = trainGenerator.__getitem__(0)
        print(x.shape)
        print(y.shape)
        self.assertEqual(70, x.shape[1])
        self.assertEqual(70, y.shape[1])

if __name__ == '__main__':
    unittest.main()
