"""
Jamming detection and response based on obstacle types.

Replaces the old strategy-based system with type-based handlers:
- PHYSICAL: Hard obstacle, strong repulsion (should never enter)
- LOW_JAM: Proceed with caution, reduced speed, maintain formation
- HIGH_JAM: Strong avoidance; if trapped, fallback + fast traverse
"""
from typing import Optional

import numpy as np

from .base import JammingZone, ObstacleType

# Legacy - kept for backward compatibility but no longer used for dispatch
JAMMING_STRATEGIES = [
    "avoidance",
    "penetration",
    "scatter",
    "fallback",
]


class JammingResponse:
    """
    Handle obstacle/jamming detection and type-based response.
    
    Provides different behaviors based on obstacle type:
    - PHYSICAL: Hard block - strong repulsion force (must path around)
    - LOW_JAM: Caution mode - reduced speed, maintain formation
    - HIGH_JAM: Emergency mode - strong avoidance; fallback + fast traverse if trapped
    
    Key difference from old system: Always computes avoidance forces PROACTIVELY,
    not just reactively when already inside a zone.
    """

    def __init__(
        self,
        strategy: str = "avoidance",  # Legacy param, ignored
        safety_margin: float = 3.0,
    ):
        """
        Initialize jamming response handler.
        
        Args:
            strategy: Legacy parameter (ignored, behavior determined by obstacle type)
            safety_margin: Extra margin to maintain from obstacles
        """
        self.strategy = strategy  # Kept for backward compatibility
        self.safety_margin = safety_margin

        # Track last safe positions (for fallback behavior)
        self.last_safe_positions: dict[str, np.ndarray] = {}

    def is_inside_jamming_field(
        self,
        agent_id: str,
        position: np.ndarray,
        jamming_zones: list[JammingZone],
    ) -> tuple[bool, float, Optional[JammingZone]]:
        """
        Check if agent is currently INSIDE a jamming field (for reactive response).
        
        NOTE: This does NOT give prior knowledge of jamming zones!
        It only checks the agent's CURRENT position - the agent only knows
        it's being jammed when comm quality drops below PT.
        
        Args:
            agent_id: Agent identifier
            position: Agent position [x, y, z]
            jamming_zones: List of obstacles/jamming zones
            
        Returns:
            Tuple of (is_inside, effect_level, closest_zone)
        """
        pos = np.array(position)
        max_effect = 0.0
        closest_zone = None
        closest_dist = float('inf')

        for zone in jamming_zones:
            if not zone.active:
                continue

            center = np.array(zone.center)
            dist = np.linalg.norm(pos - center)

            # Check if INSIDE zone's jamming field (not just nearby)
            if zone.is_in_jamming_field(position):
                # Use penetration depth as effect level
                effect = zone.get_penetration_depth(position)
                if effect > max_effect:
                    max_effect = effect

                if dist < closest_dist:
                    closest_dist = dist
                    closest_zone = zone

        # Update last safe position (where not significantly affected)
        if max_effect < 0.1:
            self.last_safe_positions[agent_id] = pos.copy()

        is_inside = max_effect > 0.05
        return is_inside, max_effect, closest_zone

    def compute_response(
        self,
        agent_id: str,
        position: np.ndarray,
        velocity: np.ndarray,
        goal: np.ndarray,
        jamming_zones: list[JammingZone],
    ) -> np.ndarray:
        """
        Compute response velocity based on obstacle types.
        
        PROACTIVE: Always computes avoidance forces for nearby obstacles,
        even before entering them.
        
        Args:
            agent_id: Agent identifier
            position: Current position [x, y, z]
            velocity: Current velocity [vx, vy, vz]
            goal: Target goal position
            jamming_zones: Active obstacles/jamming zones
            
        Returns:
            Modified velocity vector
        """
        pos = np.array(position)
        vel = np.array(velocity)
        goal = np.array(goal)

        # Accumulate response from all nearby zones
        total_response = np.zeros(3)
        speed = np.linalg.norm(vel)
        if speed < 0.01:
            speed = 1.0  # Default speed if stationary

        # Track if we're deeply inside any zone (for emergency response)
        is_trapped = False
        trapped_zone = None

        for zone in jamming_zones:
            if not zone.active:
                continue

            center = np.array(zone.center)
            dist = np.linalg.norm(pos - center)
            
            # Compute response based on obstacle type
            if zone.obstacle_type == ObstacleType.PHYSICAL:
                response = self._handle_physical(pos, vel, goal, zone, dist)
            elif zone.obstacle_type == ObstacleType.LOW_JAM:
                response = self._handle_low_jam(pos, vel, goal, zone, dist)
            elif zone.obstacle_type == ObstacleType.HIGH_JAM:
                response, trapped = self._handle_high_jam(agent_id, pos, vel, goal, zone, dist)
                if trapped:
                    is_trapped = True
                    trapped_zone = zone
            else:
                # Default to low_jam behavior
                response = self._handle_low_jam(pos, vel, goal, zone, dist)
            
            total_response += response

        # If trapped in high-power jamming, use emergency escape
        if is_trapped and trapped_zone is not None:
            return self._emergency_escape(agent_id, pos, vel, goal, trapped_zone)

        # Combine with original velocity
        if np.linalg.norm(total_response) > 0.01:
            # Blend: more avoidance weight when stronger response
            avoidance_strength = min(1.0, np.linalg.norm(total_response) / 3.0)
            result = vel * (1.0 - avoidance_strength * 0.7) + total_response
            
            # Normalize to maintain reasonable speed
            if np.linalg.norm(result) > 0:
                result = result / np.linalg.norm(result) * speed
            return result
        
        return vel

    def _handle_physical(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        goal: np.ndarray,
        zone: JammingZone,
        dist: float,
    ) -> np.ndarray:
        """
        Handle PHYSICAL obstacle: Strong repulsion, must not enter.
        
        Computes strong repulsive force that increases sharply near obstacle.
        """
        center = np.array(zone.center)
        to_agent = pos - center
        
        # Influence range for physical obstacles (use physical radius, not jamming radius)
        influence_range = zone.radius + self.safety_margin + 10.0
        
        if dist >= influence_range or dist < 0.01:
            return np.zeros(3)
        
        # Strong exponential repulsion for physical obstacles
        # Strength increases sharply as agent approaches
        normalized_dist = dist / influence_range
        strength = 5.0 * np.exp(-3.0 * normalized_dist)  # Strong exponential
        
        # Extra penalty if very close
        if dist < zone.radius + self.safety_margin:
            strength *= 3.0  # Triple strength when in danger zone
        
        direction = to_agent / dist
        return direction * strength

    def _handle_low_jam(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        goal: np.ndarray,
        zone: JammingZone,
        dist: float,
    ) -> np.ndarray:
        """
        Handle LOW_JAM: Proceed with caution, reduced speed.
        
        Soft preference to avoid, but can pass through if needed.
        Speed is reduced inside the zone.
        """
        center = np.array(zone.center)
        to_agent = pos - center
        
        # Influence range includes jamming radius
        influence_range = zone.jamming_radius + self.safety_margin
        
        if dist >= influence_range or dist < 0.01:
            return np.zeros(3)
        
        # Moderate repulsion - prefers to avoid but not strongly
        normalized_dist = dist / influence_range
        strength = 1.5 * (1.0 - normalized_dist)  # Linear falloff
        
        # If inside jamming field, reduce speed toward goal
        if zone.is_in_jamming_field(pos.tolist()):
            # Slow down to 50% when inside low-power jamming
            to_goal = goal - pos
            if np.linalg.norm(to_goal) > 0:
                # Reduce forward momentum
                strength *= 0.5
        
        direction = to_agent / dist
        return direction * strength

    def _handle_high_jam(
        self,
        agent_id: str,
        pos: np.ndarray,
        vel: np.ndarray,
        goal: np.ndarray,
        zone: JammingZone,
        dist: float,
    ) -> tuple[np.ndarray, bool]:
        """
        Handle HIGH_JAM: Strong avoidance, emergency traverse if trapped.
        
        Returns:
            Tuple of (response_vector, is_trapped)
        """
        center = np.array(zone.center)
        to_agent = pos - center
        
        # Influence range for high-power jamming
        influence_range = zone.jamming_radius + self.safety_margin + 5.0
        
        # Check if trapped (deep inside the zone)
        is_trapped = dist < zone.radius * 0.8
        
        if dist >= influence_range or dist < 0.01:
            return np.zeros(3), False
        
        # Strong repulsion - similar to physical but slightly weaker
        normalized_dist = dist / influence_range
        strength = 4.0 * np.exp(-2.5 * normalized_dist)
        
        # Extra penalty when inside jamming field
        if zone.is_in_jamming_field(pos.tolist()):
            strength *= 2.0
        
        direction = to_agent / dist
        return direction * strength, is_trapped

    def _emergency_escape(
        self,
        agent_id: str,
        pos: np.ndarray,
        vel: np.ndarray,
        goal: np.ndarray,
        zone: JammingZone,
    ) -> np.ndarray:
        """
        Emergency escape from high-power jamming: fallback + fast traverse.
        
        If agent has a safe position, move toward it.
        Otherwise, move toward goal at 2x speed to minimize time in zone.
        """
        speed = np.linalg.norm(vel)
        if speed < 0.01:
            speed = 1.0

        # Try to fallback to last safe position
        if agent_id in self.last_safe_positions:
            safe_pos = self.last_safe_positions[agent_id]
            to_safe = safe_pos - pos
            dist_to_safe = np.linalg.norm(to_safe)
            
            if dist_to_safe > 1.0:
                # Move toward safe position at 1.5x speed
                direction = to_safe / dist_to_safe
                return direction * speed * 1.5

        # No safe position - push through toward goal at 2x speed
        to_goal = goal - pos
        dist_to_goal = np.linalg.norm(to_goal)
        
        if dist_to_goal > 0:
            direction = to_goal / dist_to_goal
            return direction * speed * 2.0  # Double speed to minimize exposure
        
        # Last resort: move away from zone center
        center = np.array(zone.center)
        away = pos - center
        if np.linalg.norm(away) > 0.01:
            return (away / np.linalg.norm(away)) * speed * 2.0
        
        return vel

    def compute_proactive_avoidance(
        self,
        position: np.ndarray,
        jamming_zones: list[JammingZone],
    ) -> np.ndarray:
        """
        Compute proactive avoidance force for path planning integration.
        
        This can be called by the controller to get avoidance forces
        even when not using the full compute_response method.
        
        Args:
            position: Current position [x, y, z]
            jamming_zones: Active obstacles/jamming zones
            
        Returns:
            Avoidance force vector
        """
        pos = np.array(position)
        total_avoidance = np.zeros(3)

        for zone in jamming_zones:
            if not zone.active:
                continue

            center = np.array(zone.center)
            to_agent = pos - center
            dist = np.linalg.norm(to_agent)
            
            if dist < 0.01:
                continue
            
            # Determine influence range based on type
            if zone.obstacle_type == ObstacleType.PHYSICAL:
                influence_range = zone.radius + self.safety_margin + 10.0
                strength_mult = 5.0
            elif zone.obstacle_type == ObstacleType.HIGH_JAM:
                influence_range = zone.jamming_radius + self.safety_margin + 5.0
                strength_mult = 4.0
            else:  # LOW_JAM
                influence_range = zone.jamming_radius + self.safety_margin
                strength_mult = 1.5
            
            if dist < influence_range:
                normalized_dist = dist / influence_range
                strength = strength_mult * (1.0 - normalized_dist)
                direction = to_agent / dist
                total_avoidance += direction * strength

        return total_avoidance

    def reset(self):
        """Reset jamming response state."""
        self.last_safe_positions.clear()
