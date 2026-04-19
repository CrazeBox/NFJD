from .synthetic import (
    FederatedRegressionData,
    make_high_conflict_federated_regression,
    make_synthetic_federated_regression,
)
from .classification import FederatedData, make_federated_classification
from .multimnist import make_multimnist
from .river_flow import make_river_flow

__all__ = [
    "FederatedRegressionData",
    "make_synthetic_federated_regression",
    "make_high_conflict_federated_regression",
    "FederatedData",
    "make_federated_classification",
    "make_multimnist",
    "make_river_flow",
]
