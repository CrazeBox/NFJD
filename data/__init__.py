from .synthetic import (
    FederatedRegressionData,
    make_high_conflict_federated_regression,
    make_synthetic_federated_regression,
)
from .classification import FederatedData, make_federated_classification
from .image_classification import make_federated_image_classification
from .federated_vision import (
    VisionFederatedData,
    make_cifar10_dirichlet,
    make_cifar10_fedmgda_paper_shards,
    make_femnist_writers,
)
from .multimnist import make_multimnist
from .river_flow import make_river_flow
from .celeba import make_celeba

__all__ = [
    "FederatedRegressionData",
    "make_synthetic_federated_regression",
    "make_high_conflict_federated_regression",
    "FederatedData",
    "make_federated_classification",
    "make_federated_image_classification",
    "VisionFederatedData",
    "make_cifar10_dirichlet",
    "make_cifar10_fedmgda_paper_shards",
    "make_femnist_writers",
    "make_multimnist",
    "make_river_flow",
    "make_celeba",
]
