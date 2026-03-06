"""
Spoofing attack engine for multi-vehicle simulation.

Implements three attack types that manipulate MAVLink messages:
  1. Phantom Attack - injects fake agent messages
  2. Position Falsification - corrupts real agent positions with random offsets
  3. Coordinate Attack - systematically shifts all affected positions by a vector
"""
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .mavlink import MAVLinkMessage, MessageType


class SpoofType(str, Enum):
    PHANTOM = "phantom"
    POSITION_FALSIFICATION = "position_falsification"
    COORDINATE = "coordinate"


SPOOF_COLORS = {
    SpoofType.PHANTOM: "#9333ea",
    SpoofType.POSITION_FALSIFICATION: "#ec4899",
    SpoofType.COORDINATE: "#7c3aed",
}

SPOOF_PARAMS = {
    SpoofType.PHANTOM: {"phantom_count": 2},
    SpoofType.POSITION_FALSIFICATION: {"magnitude": 8.0},
    SpoofType.COORDINATE: {"attack_vector": [10.0, 10.0, 0.0]},
}


@dataclass
class SpoofingZone:
    """Spatial zone where spoofing attacks are active."""
    id: str
    center: list[float]
    radius: float
    active: bool = True
    spoof_type: SpoofType = SpoofType.PHANTOM
    phantom_count: int = 2
    falsification_magnitude: float = 8.0
    coordinate_vector: list[float] = field(default_factory=lambda: [10.0, 10.0, 0.0])

    def contains(self, position: list[float]) -> bool:
        dx = position[0] - self.center[0]
        dy = position[1] - self.center[1]
        dz = position[2] - self.center[2] if len(position) > 2 else 0.0
        return math.sqrt(dx * dx + dy * dy + dz * dz) <= self.radius

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "center": self.center,
            "radius": self.radius,
            "active": self.active,
            "spoof_type": self.spoof_type.value,
            "phantom_count": self.phantom_count,
            "falsification_magnitude": self.falsification_magnitude,
            "coordinate_vector": self.coordinate_vector,
        }


class SpoofingEngine:
    """
    Processes MAVLink messages through active spoofing zones.
    Called each tick between broadcast and crypto verification.
    """

    def __init__(self):
        self._phantom_sequence: dict[str, int] = {}
        # Stable phantom positions per zone (regenerated when zone changes)
        self._phantom_cache: dict[str, list[dict]] = {}

    def process(
        self,
        messages: list[MAVLinkMessage],
        spoofing_zones: list[SpoofingZone],
        agent_positions: dict[str, list[float]],
    ) -> list[MAVLinkMessage]:
        """
        Apply all active spoofing zones to the message stream.
        Returns modified message list with injected/altered messages.
        """
        result = list(messages)

        for zone in spoofing_zones:
            if not zone.active:
                continue

            if zone.spoof_type == SpoofType.PHANTOM:
                result += self._generate_phantom_messages(zone)
            elif zone.spoof_type == SpoofType.POSITION_FALSIFICATION:
                result = self._falsify_positions(result, zone, agent_positions)
            elif zone.spoof_type == SpoofType.COORDINATE:
                result = self._coordinate_attack(result, zone, agent_positions)

        return result

    def _generate_phantom_messages(self, zone: SpoofingZone) -> list[MAVLinkMessage]:
        """Inject fake agent position messages from non-existent phantom agents."""
        phantoms = []
        zone_id = zone.id

        # Generate stable phantom positions (slowly drifting)
        if zone_id not in self._phantom_cache or len(self._phantom_cache[zone_id]) != zone.phantom_count:
            self._phantom_cache[zone_id] = []
            for i in range(zone.phantom_count):
                angle = 2 * math.pi * i / zone.phantom_count
                r = zone.radius * 0.6
                self._phantom_cache[zone_id].append({
                    "base_angle": angle,
                    "base_r": r,
                    "z_offset": random.uniform(-2, 2),
                })

        for i in range(zone.phantom_count):
            phantom_id = f"phantom_{zone_id}_{i+1}"
            cache = self._phantom_cache[zone_id][i]

            # Slowly orbit around zone center with small perturbation
            t = time.time()
            angle = cache["base_angle"] + t * 0.1
            r = cache["base_r"] + math.sin(t * 0.5 + i) * 2.0
            pos = [
                zone.center[0] + r * math.cos(angle),
                zone.center[1] + r * math.sin(angle),
                zone.center[2] + cache["z_offset"] if len(zone.center) > 2 else cache["z_offset"],
            ]

            seq = self._phantom_sequence.get(phantom_id, 0)
            self._phantom_sequence[phantom_id] = seq + 1

            msg = MAVLinkMessage(
                msg_type=MessageType.GLOBAL_POSITION_INT,
                sender_id=phantom_id,
                sequence=seq,
                timestamp=time.time(),
                payload={
                    "position": pos,
                    "velocity": [random.uniform(-0.5, 0.5) for _ in range(3)],
                    "heading": angle,
                },
                is_spoofed=True,
            )
            phantoms.append(msg)

        return phantoms

    def _falsify_positions(
        self,
        messages: list[MAVLinkMessage],
        zone: SpoofingZone,
        agent_positions: dict[str, list[float]],
    ) -> list[MAVLinkMessage]:
        """Add random offsets to position messages for agents within the zone."""
        result = []
        for msg in messages:
            if (msg.msg_type == MessageType.GLOBAL_POSITION_INT
                    and not msg.is_spoofed
                    and msg.sender_id in agent_positions):
                true_pos = agent_positions[msg.sender_id]
                if zone.contains(true_pos):
                    spoofed = msg.clone()
                    mag = zone.falsification_magnitude
                    # Deterministic-ish offset per agent per zone (changes slowly)
                    seed = hash(f"{msg.sender_id}_{zone.id}_{int(time.time() / 2.0)}")
                    rng = random.Random(seed)
                    offset = [
                        rng.gauss(0, mag),
                        rng.gauss(0, mag),
                        rng.gauss(0, mag * 0.3),
                    ]
                    spoofed.payload["position"] = [
                        true_pos[i] + offset[i] for i in range(3)
                    ]
                    spoofed.is_spoofed = True
                    spoofed.signature = None
                    result.append(spoofed)
                    continue
            result.append(msg)
        return result

    def _coordinate_attack(
        self,
        messages: list[MAVLinkMessage],
        zone: SpoofingZone,
        agent_positions: dict[str, list[float]],
    ) -> list[MAVLinkMessage]:
        """
        Shift all affected agents' reported positions by the same vector.
        The formation appears internally consistent but is translated in space,
        causing the swarm to navigate toward a false location.
        """
        result = []
        vec = zone.coordinate_vector
        for msg in messages:
            if (msg.msg_type == MessageType.GLOBAL_POSITION_INT
                    and not msg.is_spoofed
                    and msg.sender_id in agent_positions):
                true_pos = agent_positions[msg.sender_id]
                if zone.contains(true_pos):
                    spoofed = msg.clone()
                    spoofed.payload["position"] = [
                        true_pos[i] + vec[i] for i in range(3)
                    ]
                    spoofed.is_spoofed = True
                    spoofed.signature = None
                    result.append(spoofed)
                    continue
            result.append(msg)
        return result

    def get_phantom_positions(self, spoofing_zones: list[SpoofingZone]) -> dict[str, list[float]]:
        """Return current phantom agent positions for visualization."""
        positions = {}
        for zone in spoofing_zones:
            if not zone.active or zone.spoof_type != SpoofType.PHANTOM:
                continue
            zone_id = zone.id
            # Auto-populate cache if not yet initialized (before simulation starts)
            if zone_id not in self._phantom_cache or len(self._phantom_cache[zone_id]) != zone.phantom_count:
                self._phantom_cache[zone_id] = []
                for i in range(zone.phantom_count):
                    angle = 2 * math.pi * i / zone.phantom_count
                    r = zone.radius * 0.6
                    self._phantom_cache[zone_id].append({
                        "base_angle": angle,
                        "base_r": r,
                        "z_offset": random.uniform(-2, 2),
                    })
            for i, cache in enumerate(self._phantom_cache[zone_id]):
                phantom_id = f"phantom_{zone_id}_{i+1}"
                t = time.time()
                angle = cache["base_angle"] + t * 0.1
                r = cache["base_r"] + math.sin(t * 0.5 + i) * 2.0
                positions[phantom_id] = [
                    zone.center[0] + r * math.cos(angle),
                    zone.center[1] + r * math.sin(angle),
                    zone.center[2] + cache["z_offset"] if len(zone.center) > 2 else cache["z_offset"],
                ]
        return positions

    def reset(self):
        self._phantom_sequence.clear()
        self._phantom_cache.clear()


# Singleton
_spoofing_engine: Optional[SpoofingEngine] = None


def get_spoofing_engine() -> SpoofingEngine:
    global _spoofing_engine
    if _spoofing_engine is None:
        _spoofing_engine = SpoofingEngine()
    return _spoofing_engine


def reset_spoofing_engine():
    global _spoofing_engine
    if _spoofing_engine is not None:
        _spoofing_engine.reset()
    _spoofing_engine = None
