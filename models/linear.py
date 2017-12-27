"""
TODO
----
Using scipy.stats.linregress a p-value can be obtained directly, which
would arguably be pretty  useful for actual evaluation (e.g. display on plot).

"""
import numpy as np

from .limb_model import LimbModel


class Linear(LimbModel):
    def __init__(self):
        self._coefs = None
        self._i_0 = None

    def fit(self, intensity_profile, params=None):
        """Fit a line to an intensity profile by linear regression.

        Parameters
        ----------
        intensity_profile : numpy.ndarray
            Single dimension intensity profile to fit the polynomial to.
        params : dict, optional
            The linear model takes no parameters.

        Notes
        -----
        Normalization is done by `_i_0`, the first element of the intensity
        profile, rather than by the max value. The choice is rather
        arbitrary and this might change in the future.

        """

        self._i_0 = intensity_profile[0]
        r = len(intensity_profile)
        y_norm = intensity_profile / self._i_0
        x_norm = np.linspace(0., 1., num=r)
        A = np.vstack((x_norm, np.ones(len(x_norm)))).T

        self._coefs = np.linalg.lstsq(A, y_norm)[0]

    def eval(self, x, absolute=False):
        """Evaluate the line at a relative distance from center.

        Parameters
        ----------
        x : float
            Relative radial distance from center of the disk at which to
            evaluate the polynomial.
        absolute : bool, optional
            Whether to return a relative (False) or absolute (True)
            brightness value for distance `x`.

        Returns
        -------
        float
            Intensity (relative or absolute depending on `absolute`) at `x`.

        Raises
        ------
        RuntimeError
            If absolute intensity is requested but a center intensity has
            not been set.
        """
        if absolute and self._i_0 is None:
            raise RuntimeError("Absolute intensity evaluation requested but "
                               "no center intensity has been set.")

        i = self._coefs[0]*x + self._coefs[1]
        if absolute:
            i = i * self._i_0
        return i

    @property
    def coefs(self):
        return self._coefs

    @coefs.setter
    def coefs(self, coefs):
        self._coefs = coefs

    def coefs_str(self):
        """Generate the string representation of the model's coefficients."""
        if self._coefs is not None:
            return "m={:.2f}, c={:.2f}".format(*self._coefs)
        else:
            return None

