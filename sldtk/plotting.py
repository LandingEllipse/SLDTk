import numpy as np
import matplotlib.pyplot as plt


class Plotter(object):
    """Plot intensity profiles and models for solar disks.

    Parameters
    ----------
    img_name : str
        Title used for the graph.
    out_path : str
        Default filepath for the graph.

    """
    def __init__(self, img_name, out_path):
        self.out_path = out_path
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.ax.set_aspect('equal')
        self.ax.set_xlim(0, 1.025)
        self.ax.set_ylim(0, 1.025)  # Can be overridden by `plot_profile`.
        self.ax.xaxis.set(ticks=np.arange(0., 1.1, 0.1))
        self.ax.yaxis.set(ticks=np.arange(0., 1.1, 0.1))
        self.ax.set_xlabel("Relative Distance From Centre")
        self.ax.set_ylabel("Relative Intensity")
        self.fig.subplots_adjust(top=0.94)
        self.extra_artists = list()
        self.extra_artists.append(self.fig.suptitle(
            "{} - Intensity Profile".format(img_name), size=14))
        self.ax.grid(which="major", axis="both", linestyle='--', linewidth=1,
                     zorder=1)

    def plot_profile(self, profile, label="Profile", color='b', zorder=1):
        """Normalize an intensity profile and scatter-plot it.

        Parameters
        ----------
        profile : numpy.ndarray
            1-D intensity profile to plot.
        label : optional
            Custom plot label forwarded to matplotlib.pyplot.scatter.
        color : optional
            Custom plot color forwarded to matplotlib.pyplot.scatter.
        zorder : optional
            Z-order of this profile plot (will be shifted up by 10 from given).

        Notes
        -----
        The values of the profile are normalized about the first
        (center) element and plotted as a fraction of this. The
        user should therefore ensure that the first element of the
        profile is representative of the center intensity.
        All data points in the profile are assumed to be linearly
        spaced from the sun center to its limb.

        """
        y = profile / profile[0]
        x = np.linspace(0., 1., num=len(profile))

        # Extend axis limit if needed to show max value.
        desired_axis_height = y.max() + y.max()*0.025
        if desired_axis_height > self.ax.get_ylim()[1]:
            self.ax.set_ylim((0, desired_axis_height))

        self.ax.scatter(x, y, s=3, c=color,
                        label=label,
                        zorder=10+zorder)

    def plot_model(self, name, model, zorder=1, color='r', linestyle='-'):
        """Add a plot of a model to the figure.

        Parameters
        ----------
        name : str
            Name of the model for the figure legend.
        model : limb_model.LimbModel
            Model used to evaluate intensities across the disk.
        zorder : int, optional
            Z-order of this model plot (will be shifted up by 20 from given).
        color : optional
            Custom line colour forwarded to matplotlib.pyplot.plot.
        linestyle : optional
            Custom line style forwarded to matplotlib.pyplot.plot.

        """
        x = np.linspace(0., 1., num=700)
        y = model.eval(x)
        label = r"{}: ${{{}}}$".format(name, model.coefs_str())
        self.ax.plot(x, y, linewidth=2, c=color, label=label, zorder=20+zorder,
                     linestyle=linestyle)

    def show(self):
        self.extra_artists.append(self.ax.legend(loc='lower left',
                                                 bbox_to_anchor=(0, 0),
                                                 fontsize=11))
        plt.show()

    def save(self, out_path=None, dpi=160):
        """Save the graph with legends for the individual plots.

        Parameters
        ----------
        out_path : str, optional
            Can be used to override the class' default assigned output path.
        dpi : int, optional
            Dots per inch forwarded to matplotlib.pyplot.savefig.

        """
        if out_path is None:
            out_path = self.out_path
        self.extra_artists.append(self.ax.legend(loc='lower left',
                                                 bbox_to_anchor=(0, 0),
                                                 fontsize=11))
        self.fig.savefig(out_path, dpi=dpi,
                         bbox_extra_artists=self.extra_artists,
                         bbox_inches='tight')

