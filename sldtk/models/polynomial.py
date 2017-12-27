import math
import numpy as np
import numpy.polynomial.polynomial as poly

from .limb_model import LimbModel

DEFAULT_ORDER = 2


class Polynomial(LimbModel):
    """Model a limb darkening profile as a polynomial in :math:`\cos{\psi}`.

    Attributes
    ----------
    coefs : tuple of numbers
        Coefficients for the polynomial, either explicitly set or derived by
        `fit`.

    """
    def __init__(self):
        self._coefs = None
        self._i_0 = None

    def fit(self, intensity_profile, params=None):
        """Fit the polynomial to an intensity profile.

        If "degree" is not specified in `params`, the order of the
        polynomial will default to 2.

        Parameters
        ----------
        intensity_profile : numpy.ndarray
            Single dimension intensity profile to fit the polynomial to.
        params : int, optional
            Degree of the fitted polynomial.

        Notes
        -----
        The polynomial is anchored to the first element of `intensity_profile`,
        and it is therefore recommended to ensure that its value is
        representative of the center intensity.

        TODO
        ----
        Reconsider assumptions associated with int casting of degree (params).

        """

        if params is not None:
            degree = int(params)
        else:
            degree = DEFAULT_ORDER

        self._i_0 = intensity_profile[0]
        r = len(intensity_profile)
        y_normalized = intensity_profile / self._i_0
        x_normalized = np.linspace(0., 1., num=r)
        x_cos_psi = np.sqrt(1 - x_normalized**2)

        weights = np.ones(len(intensity_profile))
        weights[0] = 1e5

        self._coefs = poly.polyfit(x_cos_psi, y_normalized,
                                   w=weights, deg=degree)

    def eval(self, x, absolute=False):
        """Evaluate the polynomial at a relative distance from center.

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

        cos_psi = self._dist_to_cos_psi(x)
        i = poly.polyval(cos_psi, self._coefs)
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
            coefs = ["a_{}={:.2f}".format(i, c) for i, c in enumerate(self._coefs)]
            return ', '.join(coefs)
        else:
            return None

    @staticmethod
    def _dist_to_cos_psi(x):
        if isinstance(x, np.ndarray):
            if (x > 1).any() or (x < 0).any():
                raise ValueError("relative distance is out of bounds.")
            return np.sqrt(1 - x**2)
        elif isinstance(x, float):
            if x > 1 or x < 0:
                raise ValueError("{} is out of bounds.".format(x))
            return math.sqrt(1 - x**2)
        else:
            raise TypeError("Unable to evaluate {}.".format(type(x)))
