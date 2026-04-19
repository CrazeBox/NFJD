from .classifier import MultiTaskClassifier
from .small_regressor import LargeRegressor, MediumRegressor, MODEL_REGISTRY, SmallRegressor
from .lenet_mtl import LeNetMTL
from .river_flow_mlp import RiverFlowMLP

__all__ = ["SmallRegressor", "MediumRegressor", "LargeRegressor", "MODEL_REGISTRY", "MultiTaskClassifier", "LeNetMTL", "RiverFlowMLP"]
