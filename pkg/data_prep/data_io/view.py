from itertools import cycle

import numpy as np
from floodlight import XY
from floodlight.vis.pitches import plot_handball_pitch
from floodlight.vis.positions import plot_trajectories, plot_positions
from matplotlib import pyplot as plt, cm
from skimage.draw import set_color
from skimage.io import imsave
from PIL import Image

class HandballPlot:

    def __init__(self):
        self.ax = None
        self.cycol = cycle('bgrcmky')

    def handball_plot(self, title, figsize=(16, 9)):
        self.ax = plt.subplots(figsize=figsize)[1]
        if title != "":
            self.ax.set_title(title)
        plot_handball_pitch(xlim=(0, 40), ylim=(0, 20),
                            unit='m',
                            color_scheme='standard',
                            show_axis_ticks=False, ax=self.ax)
        return self

    def add_trajectories(self, x, y, label):
        xy_pos = np.column_stack((x, y))
        plot_trajectories(xy=XY(xy_pos), start_frame=0, end_frame=x.size, ball=False,
                          ax=self.ax,
                          color=next(self.cycol),
                          label=label)
        return self

    def plot_positions(self, x, y, label=""):
        xy_pos = np.column_stack((x, y))
        plot_positions(xy=XY(xy_pos), frame=0, ball=False, ax=self.ax, color=next(self.cycol), label=label)
        return self

    def add_legend(self):
        self.ax.legend(bbox_to_anchor=(1, 1))
        return self

    def view(self):
        plt.show()

    def save(self, chart_path):
        plt.tight_layout()
        plt.savefig(chart_path, format='png')
        plt.clf()


class SimpleHandballPlot:

    def __init__(self):
        self.img = None
        self.cycol = cycle('bgrcmk')
        self.colormap = {
            'b' : (0, 0, 255, 255),
            'g' : (0, 255, 0, 255),
            'r' : (255, 0, 0, 255),
            'c' : (0, 255, 255, 255),
            'm' : (255, 0, 255, 255),
            'k' : (0, 0, 0, 255),
        }

    def handball_plot(self, title, figsize=(16, 9)):
        self.img = Image.new('RGB', (48, 32))
        return self

    def add_trajectories(self, x, y, label):
        color = self.colormap[next(self.cycol)]
        x = x.astype(np.uint8)
        y = y.astype(np.uint8)
        xy_pos = np.column_stack((x, y))
        for xy in xy_pos:
            if xy[0] <= 40 and xy[1] <= 20:
                self.img.putpixel(xy, color)
        return self

    def view(self):
        pass

    def save(self, chart_path):
        self.img.save(chart_path, "PNG")