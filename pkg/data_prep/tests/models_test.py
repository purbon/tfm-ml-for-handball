import os
import unittest

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, splprep, splev

from data_io.meta import Schema
from handy.models.centroids import CentroidsModel
import random
import matplotlib.pyplot as plt

from handy.models.kinetics import Kinetics
from handy.models.normalizer import Normalizer


def generate_frames(n_frames, n_players, noice=False):
    frames = []
    for frame_id in range(n_frames):
        player_layout = []
        take_out = random.randrange(0, n_players - 1)
        for cid in range(n_players):
            if noice and take_out == cid:
                player_layout += [np.nan, np.nan]
            else:
                player_layout += [30 - frame_id, 3 + (3 * cid)]
        frames.append(player_layout)
    return np.array(frames)


class CentroidsTestCase(unittest.TestCase):

    def test_six_player_test(self):
        columns = []
        n_players = 6
        n_frames = 10
        for cid in range(n_players):
            columns += [f"player_{cid}_x", f"player_{cid}_y"]
        frames = generate_frames(n_frames=n_frames, n_players=n_players)

        df = pd.DataFrame(frames, columns=columns)

        cm = CentroidsModel()
        cm.fit(possession_array=df)
        self.assertListEqual(list1=[25.5, 10.5], list2=cm.team_centroid)
        self.assertListEqual(list1=[[25.5, 3.0], [25.5, 6.0], [25.5, 9.0], [25.5, 12.0], [25.5, 15.0], [25.5, 18.0]],
                             list2=cm.players_centroid)
        self.assertListEqual(list1=[7.5, 4.5, 1.5, 1.5, 4.5, 7.5], list2=cm.players_distances)

    @unittest.skip("skipping")
    def test_random_5_players(self):
        columns = []
        n_players = 6
        n_frames = 1
        for cid in range(n_players):
            columns += [f"player_{cid}_x", f"player_{cid}_y"]
        frames = generate_frames(n_frames=n_frames, n_players=6, noice=True)

        df = pd.DataFrame(frames, columns=columns)

        cm = CentroidsModel()
        cm.fit(possession_array=df)
        print(cm.team_centroid)

    def test_kinetics_test_6_players(self):
        fixture_file = os.path.join(os.path.dirname(__file__),
                                    'resources',
                                    'handball.xlsx')
        df = pd.read_excel(fixture_file)

        kn = Kinetics()
        kn.fit(poss_array=df, number_of_players=6)

        self.assertListEqual(list1=[16.075552246948146, 18.34072801588001, 23.895439073568955,
                                    18.208327106897332, 27.262496003488433, 26.166267724474224],
                             list2=kn.cumdist)

        self.assertListEqual(list1=[0.6642467590251026, 0.8081513738071094, 0.780665724763187,
                                    0.8209621836459616, 0.978793436625876, 0.9642796164919453],
                             list2=kn.avg_velocity)

    def test_kinetics_test_6_players_size1(self):
        fixture_file = os.path.join(os.path.dirname(__file__),
                                    'resources',
                                    'handball-s1.xlsx')
        df = pd.read_excel(fixture_file)

        kn = Kinetics()
        kn.fit(poss_array=df, number_of_players=6)
        zero_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assertListEqual(list1=zero_list, list2=kn.cumdist)
        self.assertListEqual(list1=zero_list, list2=kn.avg_velocity)

    def test_kinetics_test_6_players2(self):
        fixture_file = os.path.join(os.path.dirname(__file__),
                                    'resources',
                                    'handball.xlsx')
        df = pd.read_excel(fixture_file)
        px = df["player_2_x"].values
        py = df["player_2_y"].values
        tx = range(len(px))

        # cs = CubicSpline(x=px, y=py)
        # ey = [cs(epx) for epx in px]

        tck, u = splprep([px, py])
        ey = splev(u, tck)

        f, ax = plt.subplots(1)
        ax.plot(px, py, color="r", label="real")
        ax.plot(ey[0], ey[1], color="b", label="interpolated", linestyle='dashed')
        ax.legend()

        plt.grid()
        plt.show()

    def test_velocity(self):
        ids = range(34)
        data = [0., 2.62585057, 3.24251299, 3.40138696, 3.49351464, 3.79558911, 3.36876639, 2.61301336, 1.86520984,
                1.20652628, 1.01980706, 0.32879261, 1.05339831, 0.22751501, 0.17954397, 0.87815906, 0.33270921,
                1.13807533, 1.28729979, 1.1984293, 1.25129096, 2.23678431, 0.25041477, 0.4365783, 0.62706453,
                0.68386645, 0.68666878, 0.50762969, 0.24844399, 0.54351349, 0.95693457, 1.11812893, 0.67590041,
                0.22601213]

        cs = CubicSpline(x=ids, y=data)
        x_range = np.linspace(ids[0], ids[-1], 10)
        print(cs(x_range))
        print("****")
        print(cs(34))  # at second 34
        derivate_2 = cs.derivative(nu=1)
        print(derivate_2(34))
        v34 = np.array(data).sum() / 34
        print(v34)


class NormalizerTestCase(unittest.TestCase):

    def test_six_players(self):
        fixture_file = os.path.join(os.path.dirname(__file__),
                                    'resources',
                                    'handball-s5.xlsx')
        df = pd.read_excel(fixture_file)

        nm = Normalizer()
        ndf = nm.fit(poss_array=df, cutoff_size=7, number_of_players=6)
        self.assertEquals(7, ndf.shape[0])

    def test_six_players_size1(self):
        fixture_file = os.path.join(os.path.dirname(__file__),
                                    'resources',
                                    'handball-s1.xlsx')
        df = pd.read_excel(fixture_file)

        nm = Normalizer()
        ndf = nm.fit(poss_array=df, cutoff_size=7, number_of_players=6)
        self.assertEquals(7, ndf.shape[0])