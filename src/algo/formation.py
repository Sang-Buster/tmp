"""
Formation patterns and generation for multi-vehicle systems.
"""
import numpy as np

from .base import FormationRole

# Available formation types
# "communication_aware" is the default and handled by the controller directly
FORMATION_TYPES = [
    "communication_aware",  # Default: uses communication quality metrics
    "v_formation",
    "line",
    "circle",
    "wedge",
    "column",
    "diamond",
]


class FormationGenerator:
    """
    Generate formation positions for agents.
    
    Supports various formation patterns for multi-vehicle coordination.
    """

    def __init__(
        self,
        formation_type: str = "v_formation",
        spacing: float = 5.0,
        leader_index: int = 0,
    ):
        """
        Initialize formation generator.
        
        Args:
            formation_type: Type of formation (v_formation, line, circle, etc.)
            spacing: Distance between agents in formation
            leader_index: Index of the leader agent
        """
        self.formation_type = formation_type
        self.spacing = spacing
        self.leader_index = leader_index

        self._generators = {
            "v_formation": self._v_formation,
            "line": self._line_formation,
            "circle": self._circle_formation,
            "wedge": self._wedge_formation,
            "column": self._column_formation,
            "diamond": self._diamond_formation,
        }

    def generate_offsets(
        self,
        num_agents: int,
        heading: float = 0.0,
    ) -> dict[int, np.ndarray]:
        """
        Generate position offsets for each agent relative to formation center.
        
        Args:
            num_agents: Number of agents
            heading: Formation heading in radians
            
        Returns:
            Dictionary mapping agent index to offset [x, y, z]
        """
        generator = self._generators.get(self.formation_type, self._v_formation)
        offsets = generator(num_agents)

        # Rotate offsets by heading
        if heading != 0:
            cos_h = np.cos(heading)
            sin_h = np.sin(heading)
            for idx in offsets:
                x, y, z = offsets[idx]
                offsets[idx] = np.array([
                    x * cos_h - y * sin_h,
                    x * sin_h + y * cos_h,
                    z
                ])

        return offsets

    def get_target_positions(
        self,
        center: np.ndarray,
        num_agents: int,
        heading: float = 0.0,
    ) -> dict[int, np.ndarray]:
        """
        Get absolute target positions for formation.
        
        Args:
            center: Formation center position [x, y, z]
            num_agents: Number of agents
            heading: Formation heading in radians
            
        Returns:
            Dictionary mapping agent index to target position
        """
        offsets = self.generate_offsets(num_agents, heading)
        center = np.array(center)

        return {idx: center + offset for idx, offset in offsets.items()}

    def get_roles(self, num_agents: int) -> dict[int, FormationRole]:
        """Get role assignments for agents."""
        roles = {}
        for i in range(num_agents):
            if i == self.leader_index:
                roles[i] = FormationRole.LEADER
            elif i < 3:  # First few agents are wingmen
                roles[i] = FormationRole.WINGMAN
            else:
                roles[i] = FormationRole.FOLLOWER
        return roles

    def _v_formation(self, n: int) -> dict[int, np.ndarray]:
        """V-formation (like flying geese)."""
        offsets = {}
        offsets[0] = np.array([0.0, 0.0, 0.0])  # Leader at center

        for i in range(1, n):
            side = 1 if i % 2 == 1 else -1
            row = (i + 1) // 2
            offsets[i] = np.array([
                -row * self.spacing * 0.7,  # Behind leader
                side * row * self.spacing,  # Left or right
                0.0
            ])

        return offsets

    def _line_formation(self, n: int) -> dict[int, np.ndarray]:
        """Line formation (side by side)."""
        offsets = {}
        center_offset = (n - 1) / 2

        for i in range(n):
            offsets[i] = np.array([
                0.0,
                (i - center_offset) * self.spacing,
                0.0
            ])

        return offsets

    def _circle_formation(self, n: int) -> dict[int, np.ndarray]:
        """Circle formation."""
        offsets = {}
        radius = self.spacing * n / (2 * np.pi) if n > 1 else 0

        for i in range(n):
            angle = 2 * np.pi * i / n
            offsets[i] = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                0.0
            ])

        return offsets

    def _wedge_formation(self, n: int) -> dict[int, np.ndarray]:
        """Wedge/arrow formation."""
        offsets = {}
        offsets[0] = np.array([0.0, 0.0, 0.0])  # Leader at front

        for i in range(1, n):
            side = 1 if i % 2 == 1 else -1
            row = (i + 1) // 2
            offsets[i] = np.array([
                -row * self.spacing,  # Behind
                side * row * self.spacing * 0.5,  # Spread
                0.0
            ])

        return offsets

    def _column_formation(self, n: int) -> dict[int, np.ndarray]:
        """Column formation (single file)."""
        offsets = {}

        for i in range(n):
            offsets[i] = np.array([
                -i * self.spacing,
                0.0,
                0.0
            ])

        return offsets

    def _diamond_formation(self, n: int) -> dict[int, np.ndarray]:
        """Diamond formation."""
        offsets = {}

        if n >= 1:
            offsets[0] = np.array([self.spacing, 0.0, 0.0])  # Front
        if n >= 2:
            offsets[1] = np.array([0.0, self.spacing, 0.0])  # Right
        if n >= 3:
            offsets[2] = np.array([0.0, -self.spacing, 0.0])  # Left
        if n >= 4:
            offsets[3] = np.array([-self.spacing, 0.0, 0.0])  # Back

        # Additional agents in outer diamond
        for i in range(4, n):
            angle = 2 * np.pi * (i - 4) / (n - 4) if n > 4 else 0
            offsets[i] = np.array([
                2 * self.spacing * np.cos(angle),
                2 * self.spacing * np.sin(angle),
                0.0
            ])

        return offsets

    def set_formation_type(self, formation_type: str):
        """Change formation type."""
        if formation_type in self._generators:
            self.formation_type = formation_type
        else:
            raise ValueError(f"Unknown formation: {formation_type}. Available: {list(self._generators.keys())}")
