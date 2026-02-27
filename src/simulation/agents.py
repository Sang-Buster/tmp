"""
Agent state management for the vehicle simulation.
"""
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from ..config import (
    HIGH_COMM_QUAL,
    LOW_COMM_QUAL,
    MISSION_END,
    NUM_AGENTS,
    X_RANGE,
    Y_RANGE,
    Z_RANGE,
    get_agent_ids,
    get_initial_agent_positions,
)


@dataclass
class AgentState:
    """State of a single agent/vehicle."""
    agent_id: str
    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    heading: float = 0.0  # radians

    # Jamming state
    jammed: bool = False
    communication_quality: float = HIGH_COMM_QUAL

    # Target/command state
    llm_target: Optional[list[float]] = None

    # Formation info
    formation_role: str = "follower"  # leader, wingman, follower
    neighbors: list[str] = field(default_factory=list)
    formation_error: float = 0.0

    # Mission info
    path: list[list[float]] = field(default_factory=list)
    path_index: int = 0

    # MAVLink / spoofing state
    is_phantom: bool = False
    crypto_verified: bool = True

    # Timestamps
    last_update: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        # Calculate mission info
        pos = self.position
        dest = list(MISSION_END)
        distance_to_goal = math.sqrt(
            (pos[0] - dest[0])**2 +
            (pos[1] - dest[1])**2 +
            (pos[2] - dest[2])**2
        )

        speed = math.sqrt(sum(v**2 for v in self.velocity))
        eta = distance_to_goal / speed if speed > 0.01 else float('inf')

        # Helper to convert numpy types to Python native types
        def to_native(val):
            if hasattr(val, 'item'):  # numpy scalar
                return val.item()
            return val

        return {
            "agent_id": str(self.agent_id),
            "position": [float(p) for p in self.position],
            "velocity": [float(v) for v in self.velocity],
            "heading": float(self.heading),
            "heading_degrees": float(math.degrees(self.heading)),
            "speed": float(speed),
            # Jamming - ensure bool conversion
            "jammed": bool(to_native(self.jammed)),
            "communication_quality": float(to_native(self.communication_quality)),
            # Target
            "llm_target": [float(t) for t in self.llm_target] if self.llm_target else None,
            # Formation
            "formation_role": self.formation_role,
            "neighbors": list(self.neighbors) if self.neighbors else [],
            "formation_error": float(to_native(self.formation_error)),
            # Mission
            "distance_to_goal": float(distance_to_goal),
            "eta": float(eta) if eta != float('inf') else None,
            "path_points": int(len(self.path)),
            # MAVLink / spoofing
            "is_phantom": self.is_phantom,
            "crypto_verified": self.crypto_verified,
            # Meta
            "last_update": str(self.last_update),
        }

    def update_position(self, new_pos: list[float], is_jammed: bool = None):
        """Update agent position and calculate velocity/heading."""
        # Calculate velocity from position change
        if hasattr(self, '_prev_pos'):
            dt = 0.1  # Assume 100ms updates
            self.velocity = [
                (new_pos[i] - self._prev_pos[i]) / dt
                for i in range(3)
            ]

            # Calculate heading from horizontal velocity
            vx, vy = self.velocity[0], self.velocity[1]
            if abs(vx) > 0.01 or abs(vy) > 0.01:
                self.heading = math.atan2(vy, vx)

        self._prev_pos = self.position.copy()
        self.position = new_pos

        if is_jammed is not None:
            # Ensure Python bool, not numpy.bool_
            self.jammed = bool(is_jammed) if hasattr(is_jammed, 'item') else bool(is_jammed)
            self.communication_quality = LOW_COMM_QUAL if self.jammed else HIGH_COMM_QUAL

        self.last_update = datetime.now().isoformat()

    def set_formation_info(
        self,
        role: str = None,
        neighbors: list[str] = None,
        error: float = None,
    ):
        """Update formation-related info."""
        if role:
            self.formation_role = role
        if neighbors is not None:
            self.neighbors = neighbors
        if error is not None:
            self.formation_error = error

    def set_path(self, path: list[list[float]]):
        """Set planned path."""
        self.path = path
        self.path_index = 0

    def get_next_waypoint(self) -> Optional[list[float]]:
        """Get next waypoint on path."""
        if self.path_index < len(self.path):
            return self.path[self.path_index]
        return None

    def advance_waypoint(self):
        """Move to next waypoint."""
        if self.path_index < len(self.path):
            self.path_index += 1


def init_agents(num_agents: int = NUM_AGENTS, positions: list = None) -> dict[str, AgentState]:
    """
    Initialize all agents with configured starting positions.
    
    Args:
        num_agents: Number of agents to create
        positions: Optional list of [x, y, z] positions. If None, uses config.
        
    Returns:
        Dictionary mapping agent_id to AgentState
    """
    agents = {}
    agent_ids = get_agent_ids(num_agents)

    # Get positions from config if not provided
    if positions is None:
        positions = get_initial_agent_positions(num_agents)

    print(f"[SIM] Initializing {num_agents} agents...")

    for idx, agent_id in enumerate(agent_ids):
        # Get position (either from provided list or config)
        if idx < len(positions):
            start_pos = list(positions[idx])
        else:
            # Fallback to random if not enough positions provided
            start_pos = [
                random.uniform(X_RANGE[0], X_RANGE[1]),
                random.uniform(Y_RANGE[0], Y_RANGE[1]),
                random.uniform(Z_RANGE[0], min(5.0, Z_RANGE[1])),
            ]

        # No role assignment by default - communication-aware is distributed
        # Roles are only assigned for geometric formations by the controller
        agents[agent_id] = AgentState(
            agent_id=agent_id,
            position=start_pos,
            formation_role=None,  # Distributed control - no hierarchy
        )
        agents[agent_id]._prev_pos = start_pos.copy()

        print(f"[SIM] {agent_id} at ({start_pos[0]:.2f}, {start_pos[1]:.2f}, {start_pos[2]:.2f})")

    print(f"[SIM] All {num_agents} agents initialized")
    return agents


def move_agent_towards_target(
    agent: AgentState,
    target: list[float],
    max_step: float = 1.0
) -> list[float]:
    """
    Move agent one step towards target position.
    
    Args:
        agent: Current agent state
        target: Target position [x, y, z]
        max_step: Maximum distance to move per step
        
    Returns:
        New position [x, y, z]
    """
    current = agent.position

    # Calculate direction vector
    dx = target[0] - current[0]
    dy = target[1] - current[1]
    dz = target[2] - current[2]

    # Distance to target
    distance = math.sqrt(dx**2 + dy**2 + dz**2)

    if distance <= max_step:
        # Close enough, snap to target
        return target.copy()

    # Move max_step distance towards target
    ratio = max_step / distance
    new_pos = [
        current[0] + dx * ratio,
        current[1] + dy * ratio,
        current[2] + dz * ratio,
    ]

    # Clamp to boundaries
    new_pos[0] = max(X_RANGE[0], min(X_RANGE[1], new_pos[0]))
    new_pos[1] = max(Y_RANGE[0], min(Y_RANGE[1], new_pos[1]))
    new_pos[2] = max(Z_RANGE[0], min(Z_RANGE[1], new_pos[2]))

    return new_pos


def calculate_distance(pos1: list[float], pos2: list[float]) -> float:
    """Calculate 3D distance between two positions."""
    return math.sqrt(sum((a - b)**2 for a, b in zip(pos1, pos2)))
