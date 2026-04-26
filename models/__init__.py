from .classifier import MultiTaskClassifier
from .small_regressor import LargeRegressor, MediumRegressor, MODEL_REGISTRY, SmallRegressor
from .basic_cnn_mtl import BasicCNNMTL
from .lenet_mtl import LeNetMTL
from .river_flow_mlp import RiverFlowMLP
from .celeba_cnn import CelebaCNN

__all__ = ["SmallRegressor", "MediumRegressor", "LargeRegressor", "MODEL_REGISTRY", "MultiTaskClassifier", "BasicCNNMTL", "LeNetMTL", "RiverFlowMLP", "CelebaCNN"]
