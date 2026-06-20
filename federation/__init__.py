"""System 5 — federated layer (abstract signatures + sharing bus)."""

from .federation_bus import FederatedInstance, FederationBus, MatchResult
from .signature_extractor import Signature, SignatureExtractor

__all__ = [
    "SignatureExtractor",
    "Signature",
    "FederationBus",
    "FederatedInstance",
    "MatchResult",
]
