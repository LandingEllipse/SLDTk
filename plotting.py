import numpy as np
import matplotlib.pyplot as plt


class Plotter(object):
    def __init__(self, out_path):
        self.out_path = out_path
        self.fig, self.ax = plt.subplots()

        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.xaxis.set(ticks=np.arange(0., 1.1, 0.2))
        self.ax.yaxis.set(ticks=np.arange(0., 1.1, 0.2))
        self.ax.set_xlabel("Relative Intensity")
        self.ax.set_ylabel("Distance From Centre")
        self.fig.subplots_adjust(top=0.92)
        self.fig.suptitle("Intensity Profile", size=18)  # TODO: find a better title

    def plot_intensity_profile(self, profile):
        y = profile / profile.max()
        x = np.linspace(0., 1., num=len(profile))
        self.ax.scatter(x, y, s=4, c='b')
        # TODO: add legend

    def plot_model(self, model):
        x = np.linspace(0., 1., num=200)
        y = model.evaluate(x, relative=True)
        self.ax.plot(x, y, linewidth=2, c='r')
        # TODO: add coefficients
        # TODO: add legend

    def show(self):
        plt.show()  # FIXME: self.fig.show() shows empty plot

    def save(self, out_path=None, dpi=160):
        if not out_path:
            out_path = self.out_path
        self.fig.savefig(out_path, dpi=dpi)

