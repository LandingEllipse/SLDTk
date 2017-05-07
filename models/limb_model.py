import abc


class LimbModel(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def fit(self, intensity_profile, config=None):
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
