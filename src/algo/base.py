"""
Base classes and types for multi-vehicle control algorithms.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class ObstacleType(Enum):
    """
    Types of obstacles/jamming zones with different behaviors.
    
    - PHYSICAL: Hard obstacle that must be completely avoided (gray)
    - LOW_JAM: Low-power jamming with mild comm degradation (yellow)
    - HIGH_JAM: High-power jamming with severe comm degradation (red)
    """
    PHYSICAL = "physical"  # Gray - hard block, no comm model
    LOW_JAM = "low_jam"    # Yellow - paper's model (mild degradation)
    HIGH_JAM = "high_jam"  # Red - severe degradation


# Default parameters by obstacle type
# Physical: no degradation (D=1.0 always), just hard blocking
# Low-power: paper's model (kappa=2.0, D_base=0.3, D_min=0.1)
# High-power: severe model (kappa=2.5, D_base=0.1, D_min=0.01)
OBSTACLE_PARAMS = {
    ObstacleType.PHYSICAL: {"kappa_j": 1.0, "d_base": 1.0, "d_min": 1.0},
    ObstacleType.LOW_JAM: {"kappa_j": 2.0, "d_base": 0.3, "d_min": 0.1},
    ObstacleType.HIGH_JAM: {"kappa_j": 2.5, "d_base": 0.1, "d_min": 0.01},
}


@dataclass
class JammingZone:
    """
    Represents a spherical obstacle/jamming zone with type-based behavior.
    
    Obstacle Types:
    - PHYSICAL: Hard obstacle, must path around completely (no comm model)
    - LOW_JAM: Low-power jamming with paper's degradation model
    - HIGH_JAM: High-power jamming with severe degradation
    
    The jamming model follows the paper's formulation:
    - r_jam = κ_J × r_obs (jamming extends beyond physical radius)
    - d_pen = penetration depth into jamming field
    - D_final = degradation factor applied to communication quality
    
    Reference: Section 3.4 "Jamming-Induced Communication Degradation"
    """
    id: str
    center: list[float]  # [x, y, z] - 3D center
    radius: float  # Physical radius r_obs
    intensity: float = 1.0  # 0.0 to 1.0
    active: bool = True
    obstacle_type: ObstacleType = ObstacleType.LOW_JAM  # Default for backward compat
    # Paper's jamming parameters (set by __post_init__ based on type, or override manually)
    kappa_j: float = None  # Jamming radius multiplier κ_J (r_jam = κ_J × r_obs)
    d_base: float = None   # Base degradation factor at edge of jamming field D_base
    d_min: float = None    # Minimum degradation floor D_min
    
    def __post_init__(self):
        """Set default parameters based on obstacle type if not explicitly provided."""
        params = OBSTACLE_PARAMS.get(self.obstacle_type, OBSTACLE_PARAMS[ObstacleType.LOW_JAM])
        if self.kappa_j is None:
            self.kappa_j = params["kappa_j"]
        if self.d_base is None:
            self.d_base = params["d_base"]
        if self.d_min is None:
            self.d_min = params["d_min"]

    @property
    def jamming_radius(self) -> float:
        """
        Calculate jamming radius: r_jam = κ_J × r_obs
        
        The jamming effect extends beyond the physical obstacle radius.
        With κ_J = 2.0, jamming extends to 2× the physical radius.
        """
        return self.kappa_j * self.radius

    def contains(self, position: list[float]) -> bool:
        """Check if a position is inside the physical jamming zone."""
        pos = np.array(position[:3])
        center = np.array(self.center)
        distance = np.linalg.norm(pos - center)
        return distance <= self.radius

    def is_in_jamming_field(self, position: list[float]) -> bool:
        """Check if a position is within the jamming field (includes extended radius)."""
        pos = np.array(position[:3])
        center = np.array(self.center)
        distance = np.linalg.norm(pos - center)
        return distance <= self.jamming_radius

    def get_penetration_depth(self, position: list[float]) -> float:
        """
        Calculate penetration depth d_pen_i from paper (3D).
        
        Formula: d_pen_i = 1 - max(0, (d_c_i - r_obs) / (r_jam - r_obs))
        
        Where:
        - d_c_i = 3D Euclidean distance from agent i to jamming center
        - r_obs = physical radius of jamming source
        - r_jam = κ_J × r_obs = extended jamming radius
        
        Returns:
        - 0.0 if outside jamming field (d_c >= r_jam)
        - 1.0 if at or inside physical obstacle (d_c <= r_obs)
        - Value between 0-1 in the transition zone
        """
        pos = np.array(position[:3])
        center = np.array(self.center)
        d_c = np.linalg.norm(pos - center)  # 3D Euclidean distance
        
        r_obs = self.radius
        r_jam = self.jamming_radius
        
        if d_c >= r_jam:
            return 0.0  # Outside jamming field
        if d_c <= r_obs:
            return 1.0  # Deep inside physical obstacle
        
        # d_pen = 1 - (d_c - r_obs) / (r_jam - r_obs)
        return 1.0 - (d_c - r_obs) / (r_jam - r_obs)

    def get_degradation_factor(self, position: list[float]) -> float:
        """
        Calculate communication degradation factor D_final_i from paper.
        
        Formula: D_final_i = max(D_min, D_base + (1 - D_base) × (1 - d_pen_i))
        
        Where:
        - D_base = base degradation at edge of jamming field (e.g., 0.3)
        - D_min = minimum floor to ensure communication isn't totally lost (e.g., 0.1)
        - d_pen_i = penetration depth
        
        Returns:
        - 1.0 if outside jamming field (no degradation)
        - D_min to 1.0 inside jamming field (degraded communication)
        - Always 1.0 for PHYSICAL obstacles (no comm degradation, just blocking)
        
        The degraded communication quality is: φ'_ij = φ(r_ij) × D_i × D_j
        """
        # Physical obstacles don't affect communication - they're just hard blockers
        if self.obstacle_type == ObstacleType.PHYSICAL:
            return 1.0
        
        d_pen = self.get_penetration_depth(position)
        
        if d_pen == 0:
            return 1.0  # No degradation outside jamming field
        
        # D_final = max(D_min, D_base + (1 - D_base)(1 - d_pen))
        d_final = self.d_base + (1.0 - self.d_base) * (1.0 - d_pen)
        return max(self.d_min, d_final)

    def get_jamming_level(self, position: list[float]) -> float:
        """
        Get jamming intensity at position (legacy method for compatibility).
        
        Returns value between 0.0 (no jamming) and 1.0 (maximum jamming).
        This is essentially 1 - degradation_factor.
        """
        return 1.0 - self.get_degradation_factor(position)

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "center": self.center,
            "radius": self.radius,
            "intensity": self.intensity,
            "active": self.active,
            "obstacle_type": self.obstacle_type.value,
            "kappa_j": self.kappa_j,
            "d_base": self.d_base,
            "d_min": self.d_min,
            "jamming_radius": self.jamming_radius,
        }


@dataclass
class VehicleCommand:
    """Command for a single vehicle."""
    agent_id: str
    target_position: Optional[list[float]] = None  # [x, y, z]
    velocity: Optional[list[float]] = None  # [vx, vy, vz]
    heading: Optional[float] = None  # radians

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "target_position": self.target_position,
            "velocity": self.velocity,
            "heading": self.heading,
        }


class FormationRole(Enum):
    """Role of vehicle in formation."""
    LEADER = "leader"
    FOLLOWER = "follower"
    WINGMAN = "wingman"


@dataclass
class FormationState:
    """Current state of the formation."""
    converged: bool = False
    formation_error: float = 0.0
    average_comm_quality: float = 0.0
    average_neighbor_distance: float = 0.0
    roles: dict = field(default_factory=dict)  # agent_id -> role
    neighbors: dict = field(default_factory=dict)  # agent_id -> list of neighbor ids

    def to_dict(self) -> dict:
        return {
            "converged": self.converged,
            "formation_error": self.formation_error,
            "average_comm_quality": self.average_comm_quality,
            "average_neighbor_distance": self.average_neighbor_distance,
            "roles": {k: v.value if isinstance(v, FormationRole) else v for k, v in self.roles.items()},
            "neighbors": self.neighbors,
        }


class MultiVehicleController(ABC):
    """
    Abstract base class for multi-vehicle control algorithms.
    
    Implementations should handle:
    - Formation control (maintain shape while moving)
    - Path planning (avoid obstacles, reach destination)
    - Jamming response (detect, avoid, recover)
    """

    @abstractmethod
    def compute_commands(
        self,
        agents: dict,  # agent_id -> AgentState
        destination: tuple[float, float, float],
        jamming_zones: list[JammingZone],
        dt: float = 0.1,
    ) -> dict[str, VehicleCommand]:
        """
        Compute movement commands for all agents.
        
        Args:
            agents: Current state of all agents
            destination: Target destination [x, y, z]
            jamming_zones: List of active jamming zones
            dt: Time step
            
        Returns:
            Dictionary mapping agent_id to VehicleCommand
        """
        pass

    @abstractmethod
    def get_formation_state(self) -> FormationState:
        """Get current formation status."""
        pass

    @abstractmethod
    def get_path_for_agent(self, agent_id: str) -> Optional[list[list[float]]]:
        """Get planned path for an agent (list of waypoints)."""
        pass

    def update_config(self, **kwargs):
        """Update controller configuration."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


# Utility functions adapted from utils_3d.py
def calculate_distance(agent_i: np.ndarray, agent_j: np.ndarray) -> float:
    """Calculate 3D distance between two agents."""
    return float(np.linalg.norm(np.array(agent_i) - np.array(agent_j)))


def calculate_aij(alpha: float, delta: float, rij: float, r0: float, v: float) -> float:
    """Calculate communication quality in antenna far-field."""
    return float(np.exp(-alpha * (2**delta - 1) * (rij / r0) ** v))


def calculate_gij(rij: float, r0: float) -> float:
    """Calculate communication quality in antenna near-field."""
    return float(rij / np.sqrt(rij**2 + r0**2))


def calculate_rho_ij(beta: float, v: float, rij: float, r0: float) -> float:
    """Calculate the derivative of phi_ij."""
    numerator = (-beta * v * rij ** (v + 2) - beta * v * (r0**2) * (rij**v) + r0 ** (v + 2))
    denominator = np.sqrt((rij**2 + r0**2) ** 3)
    return float(numerator * np.exp(-beta * (rij / r0) ** v) / denominator)


def calculate_Jn(comm_matrix: np.ndarray, neighbor_matrix: np.ndarray, PT: float) -> float:
    """Calculate average communication performance indicator."""
    total_comm = 0.0
    total_neighbors = 0
    n = comm_matrix.shape[0]

    for i in range(n):
        for j in range(n):
            if i != j and neighbor_matrix[i, j] > PT:
                total_comm += comm_matrix[i, j]
                total_neighbors += 1

    return total_comm / total_neighbors if total_neighbors > 0 else 0.0


def calculate_rn(dist_matrix: np.ndarray, neighbor_matrix: np.ndarray, PT: float) -> float:
    """Calculate average neighboring distance performance indicator."""
    total_dist = 0.0
    total_neighbors = 0
    n = dist_matrix.shape[0]

    for i in range(n):
        for j in range(n):
            if i != j and neighbor_matrix[i, j] > PT:
                total_dist += dist_matrix[i, j]
                total_neighbors += 1

    return total_dist / total_neighbors if total_neighbors > 0 else 0.0
