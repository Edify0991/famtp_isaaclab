"""Prior networks and expert data utilities."""

from .expert_buffer import ExpertBuffer, ExpertSegment

try:
    from .amp_discriminator import AmpDiscriminator
    from .part_discriminators import PART_NAMES, PartwiseConfig, PartwiseRawDiscriminators
except ModuleNotFoundError:  # pragma: no cover - optional torch dependency
    AmpDiscriminator = None
    PART_NAMES = []
    PartwiseConfig = None
    PartwiseRawDiscriminators = None

__all__ = [
    "AmpDiscriminator",
    "ExpertBuffer",
    "ExpertSegment",
    "PART_NAMES",
    "PartwiseConfig",
    "PartwiseRawDiscriminators",
]
