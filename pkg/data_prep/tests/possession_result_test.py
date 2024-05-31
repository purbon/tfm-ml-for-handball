import unittest

import numpy as np
import pandas as pd


class PossessionResultTestCase(unittest.TestCase):

    def test_prev_poss_grab(self):
        df = pd.read_excel(io="../tests/resources/poss-file-pen.xlsx")
        df = self.add_prev_poss_status(df=df)
        print(df)

    def add_prev_poss_status(self, df):
        first_poss_df = df.groupby("possession", as_index=True).head(n=1)
        last_poss_df = df.groupby("possession", as_index=True).tail(n=1)

        fdf = first_poss_df.apply(lambda x: self.group_entries_for(x=x), axis=1, result_type="reduce")
        edf = last_poss_df.apply(lambda x: self.group_entries_for(x=x, is_first=False), axis=1, result_type="reduce")
        fdf.set_index("possession", inplace=True)
        edf.set_index("possession", inplace=True)

        mdf = fdf.merge(edf, on="possession")
        mdf.drop(columns=["end_x", "start_y"], inplace=True)
        mdf.rename(columns={ "start_x": "start", "end_y": "end"}, inplace=True)

        retrieved_key = "possession_result"
        last_poss_df.set_index("possession", inplace=True)

        for index, row in df.iterrows():
            current_possession = row["possession"]
            if isinstance(current_possession, str) and current_possession != "":
                possession_index = mdf.index.get_loc(current_possession)
                for i in range(5):
                    i_limit = i + 1
                    prev_possession = None if possession_index < i_limit else mdf.index.tolist()[possession_index-i_limit]
                    if prev_possession is not None:
                        prev_val = last_poss_df.loc[prev_possession][retrieved_key]
                    else:
                        prev_val = ""
                    df.at[index, f"prev{i_limit}_{retrieved_key}"] = prev_val
        return df

    def group_entries_for(self, x, is_first=True):
        start_value = x.name if is_first else 0
        end_value = 0 if is_first else x.name
        d = {"possession": x["possession"], "start": start_value, "end": end_value}
        return pd.Series(data=d)
