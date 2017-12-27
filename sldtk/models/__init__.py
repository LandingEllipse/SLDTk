from .limb_model import LimbModel
from .polynomial import Polynomial
from .linear import Linear

models = {
    "polynomial": Polynomial,
    "linear": Linear,
}

reference_models = {
    "poly-550": [Polynomial, "550nm", (0.3, 0.93, -0.23)],
}