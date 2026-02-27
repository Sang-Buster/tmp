"""
MAVLink v2-inspired communication protocol for multi-vehicle simulation.

Provides a message bus that mediates agent-to-agent communication, enabling
spoofing attack injection and cryptographic authentication verification.
"""
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class MessageType(IntEnum):
    HEARTBEAT = 0
    GLOBAL_POSITION_INT = 33
    ATTITUDE = 30
    FORMATION_STATUS = 200
    COMMAND_LONG = 76


@dataclass
class MAVLinkMessage:
    msg_type: MessageType
    sender_id: str
    sequence: int
    timestamp: float
    payload: dict
    signature: Optional[bytes] = None
    is_spoofed: bool = False

    def clone(self) -> "MAVLinkMessage":
        return MAVLinkMessage(
            msg_type=self.msg_type,
            sender_id=self.sender_id,
            sequence=self.sequence,
            timestamp=self.timestamp,
            payload=dict(self.payload),
            signature=self.signature,
            is_spoofed=self.is_spoofed,
        )


@dataclass
class ProtocolStats:
    messages_sent: int = 0
    messages_received: int = 0
    messages_dropped: int = 0
    messages_spoofed_injected: int = 0
    messages_crypto_rejected: int = 0
    messages_crypto_accepted: int = 0
    packet_loss_rate: float = 0.0
    rejection_log: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "messages_dropped": self.messages_dropped,
            "messages_spoofed_injected": self.messages_spoofed_injected,
            "messages_crypto_rejected": self.messages_crypto_rejected,
            "messages_crypto_accepted": self.messages_crypto_accepted,
            "packet_loss_rate": round(self.packet_loss_rate, 4),
            "rejection_log": self.rejection_log[-20:],
        }

    def reset(self):
        self.messages_sent = 0
        self.messages_received = 0
        self.messages_dropped = 0
        self.messages_spoofed_injected = 0
        self.messages_crypto_rejected = 0
        self.messages_crypto_accepted = 0
        self.packet_loss_rate = 0.0
        self.rejection_log.clear()


class MAVLinkBus:
    """
    Central message bus simulating MAVLink v2 communication between agents.

    Each simulation tick:
      1. All agents broadcast their state as MAVLink messages (signed if crypto enabled)
      2. Spoofing engine may inject/modify messages
      3. Crypto auth filters invalid signatures (when enabled)
      4. Controller reads perceived positions from surviving messages
    """

    def __init__(self, packet_loss_base: float = 0.02):
        self._sequence_counters: dict[str, int] = defaultdict(int)
        self._message_queue: list[MAVLinkMessage] = []
        self._packet_loss_base = packet_loss_base
        self.stats = ProtocolStats()

        # Perceived state after full pipeline (spoofing + crypto filtering)
        self._perceived_positions: dict[str, list[float]] = {}
        self._perceived_velocities: dict[str, list[float]] = {}
        self._phantom_ids: set[str] = set()

        # Position falsification tracking (true -> spoofed offsets)
        self._falsification_offsets: dict[str, list[float]] = {}

    def broadcast(self, agent_id: str, position: list[float],
                  velocity: list[float] = None, heading: float = 0.0) -> MAVLinkMessage:
        """Create and queue a position message from an agent."""
        seq = self._sequence_counters[agent_id]
        self._sequence_counters[agent_id] = seq + 1

        msg = MAVLinkMessage(
            msg_type=MessageType.GLOBAL_POSITION_INT,
            sender_id=agent_id,
            sequence=seq,
            timestamp=time.time(),
            payload={
                "position": list(position),
                "velocity": list(velocity or [0, 0, 0]),
                "heading": heading,
            },
        )
        self._message_queue.append(msg)
        self.stats.messages_sent += 1
        return msg

    def inject(self, msg: MAVLinkMessage):
        """Inject a spoofed message into the bus."""
        msg.is_spoofed = True
        self._message_queue.append(msg)
        self.stats.messages_spoofed_injected += 1

    def apply_packet_loss(self, comm_qualities: dict[str, float] = None):
        """
        Drop messages probabilistically based on communication quality.
        Higher jamming => higher packet loss.
        """
        surviving = []
        for msg in self._message_queue:
            base_loss = self._packet_loss_base
            if comm_qualities and msg.sender_id in comm_qualities:
                quality = comm_qualities[msg.sender_id]
                loss_prob = base_loss + (1.0 - quality) * 0.3
            else:
                loss_prob = base_loss

            if random.random() > loss_prob:
                surviving.append(msg)
            else:
                self.stats.messages_dropped += 1

        total = len(self._message_queue)
        dropped = total - len(surviving)
        self.stats.packet_loss_rate = dropped / total if total > 0 else 0.0
        self._message_queue = surviving

    def get_messages(self) -> list[MAVLinkMessage]:
        """Return current message queue (for crypto auth filtering)."""
        return self._message_queue

    def set_messages(self, messages: list[MAVLinkMessage]):
        """Replace message queue (after crypto auth filtering)."""
        self._message_queue = messages

    def build_perceived_state(self, ground_truth_agents: dict):
        """
        Build perceived positions from surviving messages.
        Falls back to ground truth for agents with no messages.
        """
        self._perceived_positions.clear()
        self._perceived_velocities.clear()
        self._phantom_ids.clear()
        self._falsification_offsets.clear()

        seen_senders = set()
        for msg in self._message_queue:
            if msg.msg_type == MessageType.GLOBAL_POSITION_INT:
                sid = msg.sender_id
                self._perceived_positions[sid] = msg.payload["position"]
                self._perceived_velocities[sid] = msg.payload.get("velocity", [0, 0, 0])
                seen_senders.add(sid)
                self.stats.messages_received += 1

                if msg.is_spoofed and sid.startswith("phantom_"):
                    self._phantom_ids.add(sid)

                # Track falsification offsets
                if msg.is_spoofed and sid in ground_truth_agents:
                    true_pos = ground_truth_agents[sid].position
                    spoofed_pos = msg.payload["position"]
                    self._falsification_offsets[sid] = [
                        spoofed_pos[i] - true_pos[i] for i in range(3)
                    ]

        # Fall back to ground truth for agents with no messages
        for aid, agent in ground_truth_agents.items():
            if aid not in seen_senders:
                self._perceived_positions[aid] = list(agent.position)
                self._perceived_velocities[aid] = list(agent.velocity)

    def get_perceived_positions(self, agent_ids: list[str]) -> dict[str, list[float]]:
        """Return perceived positions for requested agents (including phantoms)."""
        result = {}
        for aid in agent_ids:
            if aid in self._perceived_positions:
                result[aid] = self._perceived_positions[aid]
        # Include phantom agents
        for pid in self._phantom_ids:
            if pid in self._perceived_positions:
                result[pid] = self._perceived_positions[pid]
        return result

    def get_all_perceived_agent_ids(self, real_agent_ids: list[str]) -> list[str]:
        """Return real agent IDs plus any phantom IDs from spoofing."""
        all_ids = list(real_agent_ids)
        for pid in self._phantom_ids:
            if pid not in all_ids:
                all_ids.append(pid)
        return all_ids

    def get_phantom_ids(self) -> set[str]:
        return set(self._phantom_ids)

    def get_falsification_offsets(self) -> dict[str, list[float]]:
        return dict(self._falsification_offsets)

    def clear_queue(self):
        """Clear message queue for next tick."""
        self._message_queue.clear()

    def reset(self):
        """Full reset."""
        self._message_queue.clear()
        self._sequence_counters.clear()
        self._perceived_positions.clear()
        self._perceived_velocities.clear()
        self._phantom_ids.clear()
        self._falsification_offsets.clear()
        self.stats.reset()


# Singleton
_mavlink_bus: Optional[MAVLinkBus] = None


def get_mavlink_bus() -> MAVLinkBus:
    global _mavlink_bus
    if _mavlink_bus is None:
        _mavlink_bus = MAVLinkBus()
    return _mavlink_bus


def reset_mavlink_bus():
    global _mavlink_bus
    if _mavlink_bus is not None:
        _mavlink_bus.reset()
    _mavlink_bus = None
