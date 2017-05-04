import numpy as np
import matplotlib.pyplot as plt


class Plotter(object):
    def __init__(self, img_name, out_path):
        self.out_path = out_path
        self.fig, self.ax = plt.subplots()

        self.ax.set_xlim(0, 1.025)
        self.ax.set_ylim(0, 1.025)
        self.ax.xaxis.set(ticks=np.arange(0., 1.1, 0.2))
        self.ax.yaxis.set(ticks=np.arange(0., 1.1, 0.2))
        self.ax.set_xlabel("Relative Distance From Centre")
        self.ax.set_ylabel("Relative Intensity")
        self.fig.subplots_adjust(top=0.92)
        self.extra_artists = list()  # Added to extra artists during render to prevent cutoff
        self.extra_artists.append(self.fig.suptitle("{} - Intensity Profile".format(img_name), size=14))

    def plot_intensity_profile(self, profile):
        y = profile / profile[0]
        x = np.linspace(0., 1., num=len(profile))
        self.ax.scatter(x, y, s=3, c='b', label=r"$\mathtt{Intensity\ \ profile}$", zorder=1)
        # TODO: add legend

    def plot_model(self, model):
        x = np.linspace(0., 1., num=400)
        y = model.evaluate(x, relative=True)
        label = r"$\mathtt{{Fitted: \ \ \ \ \ \ a_0={:.2f}, a_1={:.2f}, a_2={:.2f}}}$".format(*model.coefs)
        self.ax.plot(x, y, linewidth=2, c='r', label=label, zorder=3)

    # TODO: change implementation to something sensible
    def plot_expected_model(self, model, coefs):
        x = np.linspace(0., 1., num=400)
        y = model.evaluate(x, relative=True, coefs=coefs)
        label = r"$\mathtt{{Expected: a_0={:.2f}, a_1={:.2f}, a_2={:.2f}}}$".format(*coefs)
        self.ax.plot(x, y, linewidth=2, c='g', label=label, linestyle=":", zorder=2)

    def show(self):
        plt.show()  # FIXME: self.fig.show() shows empty plot

    def save(self, out_path=None, dpi=160):
        if not out_path:
            out_path = self.out_path
        self.extra_artists.append(self.ax.legend(loc='lower left', bbox_to_anchor=(0, 0), fontsize=11))
        self.fig.savefig(out_path, dpi=dpi, bbox_extra_artists=self.extra_artists, bbox_inches='tight')

