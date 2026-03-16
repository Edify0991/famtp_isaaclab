"""Prior networks and expert data utilities."""

from .expert_buffer import ExpertBuffer, ExpertSegment

try:
    from .amp_discriminator import AmpDiscriminator
    from .coupling import GlobalCouplingScorer
    from .latent_part_discriminators import LatentPartDiscriminators
    from .manifold_encoders import ManifoldEncoderCfg, MultiPartManifoldEncoder, SharedPartManifoldEncoder
    from .part_discriminators import PART_NAMES, PartwiseConfig, PartwiseRawDiscriminators
except ModuleNotFoundError:  # pragma: no cover - optional torch dependency
    AmpDiscriminator = None
    LatentPartDiscriminators = None
    GlobalCouplingScorer = None
    ManifoldEncoderCfg = None
    MultiPartManifoldEncoder = None
    SharedPartManifoldEncoder = None
    PART_NAMES = []
    PartwiseConfig = None
    PartwiseRawDiscriminators = None

__all__ = [
    "AmpDiscriminator",
    "ExpertBuffer",
    "ExpertSegment",
    "LatentPartDiscriminators",
    "GlobalCouplingScorer",
    "ManifoldEncoderCfg",
    "SharedPartManifoldEncoder",
    "MultiPartManifoldEncoder",
    "PART_NAMES",
    "PartwiseConfig",
    "PartwiseRawDiscriminators",
]
