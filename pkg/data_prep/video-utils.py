import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

if __name__ == '__main__':
    agg_key = "sequence_label"  # possession sequence_label
    for i in range(1, 11, 1):
        game = f"game{i}"
        df = pd.read_hdf(path_or_buf=f"full-game.h5", key=game)
        table_df = df.reset_index().groupby(agg_key).agg({'WHOLE_GAME': [np.min, np.max]})["WHOLE_GAME"]
        table_df["min"] = table_df["min"].map(lambda t: t)
        table_df["max"] = table_df["max"].map(lambda t: t)
        table_df.to_csv(f's{game}.csv')
