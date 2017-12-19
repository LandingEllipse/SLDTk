import math
import numpy as np
import numpy.polynomial.polynomial as poly

from .limb_model import LimbModel

DEFAULT_ORDER = 2


class Polynomial(LimbModel):

    def __init__(self):
        self._coefs = None
        self._i_0 = None

    def fit(self, intensity_profile, params=None):

        if type(params) is dict and 'degree' in params:
            degree = params['degree']
        else:
            degree = DEFAULT_ORDER

        self._i_0 = intensity_profile[0]
        r = len(intensity_profile)
        y_normalized = intensity_profile / self._i_0
        x_normalized = np.linspace(0., 1., num=r)
        x_cos_psi = np.sqrt(1 - x_normalized**2)

        weights = np.ones(len(intensity_profile))
        weights[0] = 1e5

        self._coefs = poly.polyfit(x_cos_psi, y_normalized, w=weights, deg=degree)

    def eval(self, x, absolute=False):
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

    @staticmethod
    def _dist_to_cos_psi(x):
        if isinstance(x, np.ndarray):
            # if (x > 1).any() or (x < 0).any():
            #     raise ValueError("relative distance is out of bounds.")
            return np.sqrt(1 - x**2)
        elif type(x) is float:
            # if x > 1 or x < 0:
            #     raise ValueError("{} is out of bounds.".format(x))
            return math.sqrt(1 - x**2)
        else:
            raise TypeError("unable to evaluate {}".format(type(x)))
