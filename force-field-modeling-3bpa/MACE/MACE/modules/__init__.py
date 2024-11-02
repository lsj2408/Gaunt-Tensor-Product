from typing import Callable, Dict, Type

from .blocks import (
    AtomicEnergiesBlock,
    RadialEmbeddingBlock,
    LinearNodeEmbeddingBlock,
    NonLinearBlock,
    InteractionBlock,
    LinearReadoutBlock,
    ProductBasisBlock,
    EquivariantProductBasisBlock,
    ComplexAgnosticResidualInteractionBlock,
    NonLinearReadoutBlock,
    AgnosticNonlinearInteractionBlock,
    RealAgnosticInteractionBlock,
    RealAgnosticNonLinearResidualInteractionBlock,
    RealAgnosticResidualInteractionBlock,
    RealElementDependentResidualInteractionBlock,
    ResidualElementDependentInteractionBlock,
    AgnosticResidualNonlinearInteractionBlock,
    LinearDipoleReadoutBlock,
    NonLinearDipoleReadoutBlock,
)
from .loss import EnergyForcesLoss, ACELoss, WeightedEnergyForcesLoss
from .models import (
    InvariantMACE,
    ScaleShiftNonLinearBodyOrderedModel,
    NonLinearBodyOrderedModel,
    RealNonLinearBodyOrderedModel,
    RealScaleShiftNonLinearBodyOrderedModel,
    EquivariantNonLinearBodyOrderedModel,
    EquivariantRealScaleShiftNonLinearBodyOrderedModel,
    BOTNet,
    ScaleShiftBOTNet,
)

from .radial import BesselBasis, PolynomialCutoff
from .utils import (
    compute_mean_std_atomic_inter_energy,
    compute_mean_rms_energy_forces,
    compute_avg_num_neighbors,
)


interaction_classes: Dict[str, Type[InteractionBlock]] = {
    "AgnosticNonlinearInteractionBlock": AgnosticNonlinearInteractionBlock,
    "ComplexAgnosticResidualInteractionBlock": ComplexAgnosticResidualInteractionBlock,
    "ResidualElementDependentInteractionBlock": ResidualElementDependentInteractionBlock,
    "AgnosticResidualNonlinearInteractionBlock": AgnosticResidualNonlinearInteractionBlock,
    "RealAgnosticResidualInteractionBlock": RealAgnosticResidualInteractionBlock,
    "RealAgnosticInteractionBlock": RealAgnosticInteractionBlock,
    "RealElementDependentResidualInteractionBlock": RealElementDependentResidualInteractionBlock,
    "RealAgnosticNonLinearResidualInteractionBlock": RealAgnosticNonLinearResidualInteractionBlock,
}

scaling_classes: Dict[str, Type[Callable]] = {
    "std_scaling": compute_mean_std_atomic_inter_energy,
    "rms_forces_scaling": compute_mean_rms_energy_forces,
}

__all__ = [
    "AtomicEnergiesBlock",
    "RadialEmbeddingBlock",
    "LinearNodeEmbeddingBlock",
    "NonLinearBlock",
    "PolynomialCutoff",
    "LinearReadoutBlock",
    "AtomicBaseBlock",
    "ProductBasisBlock",
    "EquivariantProductBasisBlock",
    "BesselBasis",
    "EnergyForcesLoss",
    "ACELoss",
    "WeightedEnergyForcesLoss",
    "interaction_classes",
    "InteractionBlock",
    "InvariantMACE",
    "compute_mean_std_atomic_inter_energy",
    "compute_num_avg_neighbors",
    "NonLinearReadoutBlock",
    "NonLinearBodyOrderedModel",
    "ScaleShiftNonLinearBodyOrderedModel",
    "RealNonLinearBodyOrderedModel",
    "RealScaleShiftNonLinearBodyOrderedModel",
    "LinearDipoleReadoutBlock",
    "NonLinearDipoleReadoutBlock",
    "ScaleShiftNonLinearBodyOrderedModel",
    "EquivariantNonLinearBodyOrderedModel",
    "EquivariantRealScaleShiftNonLinearBodyOrderedModel",
    "NonLinearBodyOrderedModel",
    "BOTNet",
    "ScaleShiftBOTNet",
    "compute_avg_num_neighbors",
]
