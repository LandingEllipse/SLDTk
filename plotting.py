import numpy as np
import matplotlib.pyplot as plt


class Plotter(object):
    """
    """
    def __init__(self, img_name, out_path):
        """
        Args:
            img_name: 
            out_path: 
        """
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
        self.ax.grid(which="major", axis="y", linestyle='--', linewidth=1, zorder=1)

    def plot_intensity_profile(self, profile):
        """Normalizes an intensity profile and scatter-plots it.
        
        The values of the profile are normalized about the first (center) element and plotted as a fraction of this. The
        user should therefore ensure that the first element of the profile is representative of the center intensity.
        All datapoints in the profile are assumed to be linearly spaced from the sun center to its limb.
        
        Args:
            profile (np.ndarray): 1-D intensity profile to plot. 
        """
        y = profile / profile[0]
        x = np.linspace(0., 1., num=len(profile))
        self.ax.scatter(x, y, s=3, c='b', label=r"$\mathtt{Intensity\ \ profile}$", zorder=10)

    def plot_intensity_profile_2(self, profile):  # FIXME: report temp
        y = profile / profile[0]
        x = np.linspace(0., 1., num=len(profile))
        self.ax.scatter(x, y, s=3, c='r', label=r"$\mathtt{Radial\ \ mean}$", zorder=9, marker='^')

    def plot_model(self, name, model, zorder=2, color='r', linestyle='-'):
        """
        Args:
            name:
            model: 

        """
        x = np.linspace(0., 1., num=700)
        y = model.eval(x)
        coefs = ["a_{}={:.2f}".format(i, c) for i, c in enumerate(model.coefs)]
        label = r"$\mathtt{{{}: \ {}}}$".format(name, ', '.join(coefs))
        self.ax.plot(x, y, linewidth=2, c=color, label=label, zorder=10+zorder, linestyle=linestyle)

    def show(self):
        plt.show()  # FIXME: self.fig.show() shows empty plot

    def save(self, out_path=None, dpi=160):
        if not out_path:
            out_path = self.out_path
        self.extra_artists.append(self.ax.legend(loc='lower left', bbox_to_anchor=(0, 0), fontsize=11))
        self.fig.savefig(out_path, dpi=dpi, bbox_extra_artists=self.extra_artists, bbox_inches='tight')

