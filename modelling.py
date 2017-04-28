import math
import numpy.polynomial.polynomial as poly
import numpy as np


def generate_model(intensity_profile):
    """
    Uses the approach in [1] to model limb darkening as a 2nd order polynomial in cos(psi).
    The approximation that the relative distance `x_normalized` defines sin(psi) is used to derive cos(psi).
    The transformation from x to cos(psi) is done in order to evaluate the derived coefficients against established standards.
    
    [1] Cox, Arthur N. (ed) (2000). Allen's Astrophysical Quantities (14th ed.)
    """
    i_0 = intensity_profile.max()
    r = len(intensity_profile)
    y_nomalized = intensity_profile / i_0  # TODO: probably an ok approximation but consider constraining fit to unity at x=0
    x_normalized = np.linspace(0., 1., num=r)
    x_cos_psi = np.sqrt(1 - x_normalized**2)

    coefs = poly.polyfit(x_cos_psi, y_nomalized, deg=2)

    return Model(coefs, i_0)


class Model(object):
    def __init__(self, coefs, i_0):
        self.coefs = coefs
        self.i_0 = i_0

    def evaluate(self, x, relative=False):
        cos_psi = self.dist_to_cos_psi(x)
        i = poly.polyval(cos_psi, self.coefs)
        if not relative:
            i = i * self.i_0
        return i

    @staticmethod
    def dist_to_cos_psi(x):
        if type(x) is float:
            if x > 1 or x < 0:
                raise ValueError("{} is out of bounds.".format(x))
            return math.sqrt(1 - x**2)
        elif type(x) is np.ndarray:
            if (x > 1).any() or (x < 0).any():
                raise ValueError("relative distance is out of bounds.")
            return np.sqrt(1 - x**2)
        else:
            raise TypeError("unable to evaluate {}".format(type(x)))
