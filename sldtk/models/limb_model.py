import abc


class LimbModel(metaclass=abc.ABCMeta):
    """Baseclass serving as poor man's interface for all limb models."""

    @abc.abstractmethod
    def fit(self, intensity_profile, params=None):
        pass

    @abc.abstractmethod
    def eval(self, x, absolute=False):
        pass

    @property
    @abc.abstractmethod
    def coefs(self):
        pass

    @coefs.setter
    @abc.abstractmethod
    def coefs(self, coefs):
        pass

    @abc.abstractmethod
    def coefs_str(self):
        pass