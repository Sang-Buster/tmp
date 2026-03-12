"""
Multi-Vehicle Algorithm Framework

Contains formation control, path planning, jamming response, MAVLink protocol,
spoofing attacks, and cryptographic authentication for multi-vehicle coordination.
"""

from .base import (
    FormationState,
    JammingZone,
    MultiVehicleController,
    VehicleCommand,
)
from .controller import UnifiedController
from .crypto_auth import CryptoAuth, get_crypto_auth, reset_crypto_auth
from .formation import FORMATION_TYPES, FormationGenerator
from .jamming_response import JAMMING_STRATEGIES, JammingResponse
from .mavlink import MAVLinkBus, MAVLinkMessage, MessageType, get_mavlink_bus, reset_mavlink_bus
from .path_planning import PATH_ALGORITHMS, PathPlanner
from .spoofing import SpoofType, SpoofingEngine, SpoofingZone, get_spoofing_engine, reset_spoofing_engine
from .v2v_channel import V2VChannelModel, ChannelParams, get_channel_model, reset_channel_model

__all__ = [
    # Base classes
    "MultiVehicleController",
    "JammingZone",
    "VehicleCommand",
    "FormationState",
    # Controller
    "UnifiedController",
    # Formation
    "FormationGenerator",
    "FORMATION_TYPES",
    # Path Planning
    "PathPlanner",
    "PATH_ALGORITHMS",
    # Jamming
    "JammingResponse",
    "JAMMING_STRATEGIES",
    # MAVLink Protocol
    "MAVLinkBus",
    "MAVLinkMessage",
    "MessageType",
    "get_mavlink_bus",
    "reset_mavlink_bus",
    # Spoofing
    "SpoofingZone",
    "SpoofType",
    "SpoofingEngine",
    "get_spoofing_engine",
    "reset_spoofing_engine",
    # Crypto Auth
    "CryptoAuth",
    "get_crypto_auth",
    "reset_crypto_auth",
    # V2V Channel Model
    "V2VChannelModel",
    "ChannelParams",
    "get_channel_model",
    "reset_channel_model",
]
