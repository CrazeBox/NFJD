from .classifier import MultiTaskClassifier
from .small_regressor import LargeRegressor, MediumRegressor, MODEL_REGISTRY, SmallRegressor
from .basic_cnn_mtl import BasicCNNMTL
from .cifar_resnet import CIFARResNet18MTL
from .femnist_cnn import FEMNISTCNN
from .lenet_mtl import LeNetMTL
from .river_flow_mlp import RiverFlowMLP
from .celeba_cnn import CelebaCNN
from .text_classifier import MeanPooledTextClassifier

__all__ = ["SmallRegressor", "MediumRegressor", "LargeRegressor", "MODEL_REGISTRY", "MultiTaskClassifier", "BasicCNNMTL", "CIFARResNet18MTL", "FEMNISTCNN", "LeNetMTL", "RiverFlowMLP", "CelebaCNN", "MeanPooledTextClassifier"]
