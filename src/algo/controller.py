"""
Unified Controller for Multi-Vehicle Coordination.

Uses communication-aware formation control as the default formation algorithm,
integrating path planning and jamming response.

The formation control uses communication quality metrics (aij, gij, rho_ij) 
to maintain optimal inter-agent distances.

Supports behavior-based obstacle avoidance as well as grid-based path planning 
algorithms from pathfinding3d.
"""
from typing import Optional

import numpy as np

from .base import (
    FormationRole,
    FormationState,
    JammingZone,
    MultiVehicleController,
    ObstacleType,
    VehicleCommand,
    calculate_aij,
    calculate_distance,
    calculate_gij,
    calculate_Jn,
    calculate_rho_ij,
    calculate_rn,
)
from .formation import FORMATION_TYPES, FormationGenerator
from .jamming_response import JammingResponse
from .path_planning import PATH_ALGORITHMS, PathPlanner

# Try to import config parameters
try:
    from ..config import (
        DEFAULT_FORMATION,
        get_behavior_params,
        get_formation_params,
    )
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False


class UnifiedController(MultiVehicleController):
    """
    Unified controller handling:
    - Communication-aware formation control (default) - maintains optimal distances
      based on communication quality metrics
    - Path planning (avoid obstacles, reach destination)
    - Jamming response (detect, avoid, recover)
    
    The communication-aware formation uses:
    - aij: Communication quality in antenna far-field
    - gij: Communication quality in antenna near-field  
    - rho_ij: Derivative of phi_ij for gradient-based control
    - Formation converges when Jn (avg comm quality) stabilizes
    """

    def __init__(
        self,
        formation_type: str = None,  # Default from config or "communication_aware"
        path_algorithm: str = None,  # Default from config or "direct"
        # Communication parameters for formation control
        alpha: float = None,
        delta: float = None,
        r0: float = None,
        v: float = None,
        PT: float = None,
        # Behavior-based control parameters
        attraction_magnitude: float = None,
        distance_threshold: float = None,
        avoidance_magnitude: float = None,
        buffer_zone: float = None,
        wall_follow_zone: float = None,
        # Geometric formation params (for non-comm-aware modes)
        spacing: float = 5.0,
    ):
        """
        Initialize unified controller.
        
        Args:
            formation_type: "communication_aware" (default) or geometric types
            path_algorithm: "direct" (default), "astar", etc.
            alpha: Antenna characteristic parameter
            delta: Required data rate
            r0: Reference distance
            v: Path loss exponent
            PT: Reception probability threshold
            attraction_magnitude: Destination attraction strength
            distance_threshold: Distance scaling threshold
            avoidance_magnitude: Obstacle avoidance strength
            buffer_zone: Obstacle buffer zone size
            wall_follow_zone: Wall following zone size
            spacing: Spacing for geometric formations
        """
        # Load defaults from config if available
        if CONFIG_AVAILABLE:
            form_params = get_formation_params()
            behav_params = get_behavior_params()

            formation_type = formation_type or DEFAULT_FORMATION
            path_algorithm = path_algorithm or "direct"  # Default to direct

            alpha = alpha if alpha is not None else form_params["alpha"]
            delta = delta if delta is not None else form_params["delta"]
            r0 = r0 if r0 is not None else form_params["r0"]
            v = v if v is not None else form_params["v"]
            PT = PT if PT is not None else form_params["PT"]

            attraction_magnitude = attraction_magnitude if attraction_magnitude is not None else behav_params["attraction_magnitude"]
            distance_threshold = distance_threshold if distance_threshold is not None else behav_params["distance_threshold"]
            avoidance_magnitude = avoidance_magnitude if avoidance_magnitude is not None else behav_params["avoidance_magnitude"]
            buffer_zone = buffer_zone if buffer_zone is not None else behav_params["buffer_zone"]
            wall_follow_zone = wall_follow_zone if wall_follow_zone is not None else behav_params["wall_follow_zone"]
        else:
            # Hardcoded defaults
            formation_type = formation_type or "communication_aware"
            path_algorithm = path_algorithm or "direct"
            alpha = alpha if alpha is not None else 1e-5
            delta = delta if delta is not None else 2.0
            r0 = r0 if r0 is not None else 5.0
            v = v if v is not None else 3.0
            PT = PT if PT is not None else 0.94
            attraction_magnitude = attraction_magnitude if attraction_magnitude is not None else 0.7
            distance_threshold = distance_threshold if distance_threshold is not None else 1.0
            avoidance_magnitude = avoidance_magnitude if avoidance_magnitude is not None else 3.5
            buffer_zone = buffer_zone if buffer_zone is not None else 8.0
            wall_follow_zone = wall_follow_zone if wall_follow_zone is not None else 4.0

        self.formation_type = formation_type
        self.path_algorithm = path_algorithm

        # Communication parameters
        self.alpha = alpha
        self.delta = delta
        self.r0 = r0
        self.v = v
        self.PT = PT
        self.beta = alpha * (2**delta - 1)

        # Behavior-based control parameters
        self.attraction_magnitude = attraction_magnitude
        self.distance_threshold = distance_threshold
        self.avoidance_magnitude = avoidance_magnitude
        self.buffer_zone = buffer_zone
        self.wall_follow_zone = wall_follow_zone

        # Initialize sub-controllers
        self.geometric_formation = FormationGenerator(
            formation_type if formation_type != "communication_aware" else "v_formation",
            spacing
        )

        # Get bounds from config if available
        try:
            from ..config import X_RANGE, Y_RANGE, Z_RANGE
            # Keep z bounds at 0 or above - vehicles shouldn't go underground
            bounds_min = [X_RANGE[0], Y_RANGE[0], max(0, Z_RANGE[0])]
            bounds_max = [X_RANGE[1], Y_RANGE[1], Z_RANGE[1]]
        except ImportError:
            bounds_min = [-50, -50, 0]
            bounds_max = [50, 150, 50]
        
        # Store bounds for position clamping
        self.bounds_min = np.array(bounds_min)
        self.bounds_max = np.array(bounds_max)

        self.path_planner = PathPlanner(
            algorithm=path_algorithm if path_algorithm != "direct" else "astar",
            voxel_size=2.0,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
        )
        self.jamming_handler = JammingResponse()

        # State tracking
        self.formation_converged = False
        self.convergence_threshold = 20  # iterations of stable Jn
        self.Jn_history: list[float] = []
        self.rn_history: list[float] = []
        
        # Visualization state
        self._last_avoidance_vectors: dict = {}

        # Matrices for formation computation
        self._comm_matrix: Optional[np.ndarray] = None
        self._dist_matrix: Optional[np.ndarray] = None
        self._neighbor_matrix: Optional[np.ndarray] = None

        # Cached formation state
        self._formation_state = FormationState()

        # Agent paths
        self._agent_paths: dict[str, list[list[float]]] = {}

        # Per-agent communication quality (average phi_rij)
        self._agent_comm_quality: dict[str, float] = {}

        # Comm quality tracking for jamming detection via degradation
        # (Realistic: agents detect jamming when comm quality < PT)
        self._comm_quality_history: dict[str, list[float]] = {}
        self._baseline_comm_quality: dict[str, float] = {}
        self._baseline_samples: int = 10  # Samples to establish baseline before jamming detection

        # Control inputs (accumulated per step)
        self._control_inputs: Optional[np.ndarray] = None

        # Per-agent discovered jamming zones (realistic: agents don't know zones until detected)
        # Maps agent_id -> list of JammingZone objects they've discovered
        self._discovered_obstacles: dict[str, list[JammingZone]] = {}
        # Maps agent_id -> set of zone IDs they've discovered (for quick lookup)
        self._known_zone_ids: dict[str, set[str]] = {}
        # Track which positions have been marked as obstacles (avoid duplicates)
        self._discovered_obstacle_positions: dict[str, list[tuple]] = {}

        # Deadlock detection - track swarm center history
        self._swarm_center_history: list[np.ndarray] = []
        self._deadlock_boost: float = 1.0  # Multiplier for destination control when stuck
        self._deadlock_check_window: int = 50  # Check last N steps
        self._deadlock_threshold: float = 2.0  # If moved less than this, consider stuck

        print(f"[Controller] Initialized: formation={formation_type}, path={path_algorithm}")
        if formation_type == "communication_aware":
            print(f"[Controller] Comm-aware params: alpha={alpha}, delta={delta}, r0={r0}, v={v}, PT={PT}")

    def compute_commands(
        self,
        agents: dict,  # agent_id -> AgentState
        destination: tuple[float, float, float],
        jamming_zones: list[JammingZone],
        dt: float = 0.1,
        perceived_positions: dict[str, list[float]] = None,
    ) -> dict[str, VehicleCommand]:
        """
        Compute movement commands for all agents using communication-aware formation control.
        
        The algorithm:
        1. Compute pairwise communication quality metrics
        2. Apply formation control: rho_ij * eij for each neighbor pair
        3. After convergence, add destination control with jamming avoidance
        
        Args:
            agents: Current state of all agents
            destination: Target destination [x, y, z]
            jamming_zones: List of active jamming zones
            dt: Time step
            perceived_positions: MAVLink-mediated positions (may include spoofed data).
                                 When None, uses ground-truth positions from agents dict.
            
        Returns:
            Dictionary mapping agent_id to VehicleCommand
        """
        agent_ids = list(agents.keys())
        n = len(agent_ids)

        if n == 0:
            return {}

        # Use MAVLink-perceived positions if available, otherwise ground truth
        if perceived_positions is not None:
            positions = np.array([
                perceived_positions.get(aid, agents[aid].position if hasattr(agents[aid], 'position')
                                        else agents[aid].get('position', [0, 0, 0]))
                for aid in agent_ids
            ])
        else:
            positions = np.array([
                agents[aid].position if hasattr(agents[aid], 'position')
                else agents[aid].get('position', [0, 0, 0])
                for aid in agent_ids
            ])

        # =====================================================================
        # OBSTACLE VISIBILITY: Physical obstacles are visible, jamming is invisible
        # =====================================================================
        # Physical obstacles can be seen and avoided proactively
        # Jamming zones (LOW_JAM, HIGH_JAM) are invisible until discovered via comm degradation
        visible_obstacles = [z for z in jamming_zones if z.active and z.obstacle_type == ObstacleType.PHYSICAL]
        invisible_jamming = [z for z in jamming_zones if z.active and z.obstacle_type != ObstacleType.PHYSICAL]

        # Initialize matrices
        self._comm_matrix = np.zeros((n, n))
        self._dist_matrix = np.zeros((n, n))
        self._neighbor_matrix = np.zeros((n, n))

        # Reset control inputs
        control_inputs = np.zeros((n, 3))

        # =====================================================================
        # COMMUNICATION-AWARE FORMATION CONTROL
        # =====================================================================
        if self.formation_type == "communication_aware":
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue

                    # Calculate distance
                    rij = calculate_distance(positions[i], positions[j])

                    # Calculate communication quality metrics
                    aij = calculate_aij(self.alpha, self.delta, rij, self.r0, self.v)
                    gij = calculate_gij(rij, self.r0)

                    # Calculate rho_ij (derivative) only if neighbor
                    if aij >= self.PT:
                        rho_ij = calculate_rho_ij(self.beta, self.v, rij, self.r0)
                    else:
                        rho_ij = 0

                    # Direction vector
                    qi = positions[i]
                    qj = positions[j]
                    if rij > 0:
                        eij = (qi - qj) / np.sqrt(rij)
                    else:
                        eij = np.zeros(3)

                    # Apply jamming degradation to communication quality (paper's model)
                    # φ'_ij = φ(r_ij) × D_i × D_j
                    # Where D_i and D_j are degradation factors from jamming zones
                    D_i = 1.0
                    D_j = 1.0
                    for zone in jamming_zones:
                        if zone.active:
                            D_i *= zone.get_degradation_factor(positions[i].tolist())
                            D_j *= zone.get_degradation_factor(positions[j].tolist())
                    
                    # Record matrices with degraded communication quality
                    phi_rij = gij * aij * D_i * D_j  # Degraded by jamming
                    self._comm_matrix[i, j] = phi_rij
                    self._dist_matrix[i, j] = rij
                    self._neighbor_matrix[i, j] = aij * D_i * D_j  # Also degrade neighbor metric

                    # Formation control input: rho_ij * eij
                    # Before convergence: prioritize formation
                    # After convergence: significantly reduce formation weight to allow destination progress
                    if self.formation_converged:
                        formation_weight = 0.15  # Much reduced after convergence
                    else:
                        formation_weight = 1.0

                    control_inputs[i] += formation_weight * rho_ij * eij
        else:
            # Use geometric formation (v_formation, line, circle, etc.)
            # Calculate formation center (centroid of all agents)
            current_center = np.mean(positions, axis=0)
            
            # Direction to destination for heading calculation
            dest_array = np.array(destination)
            to_dest = dest_array - current_center
            dist_to_dest = np.linalg.norm(to_dest)
            
            if dist_to_dest > 0.1:
                # Normalize direction
                direction = to_dest / dist_to_dest
                formation_heading = float(np.arctan2(to_dest[1], to_dest[0]))
                
                # Move formation center toward destination
                # The target center advances toward destination each step
                move_speed = min(self.v * dt * 10, dist_to_dest)  # Move at reasonable speed
                target_center = current_center + direction * move_speed
            else:
                formation_heading = 0.0
                target_center = current_center
            
            # Get target positions around the moving target center
            # Args order: center, num_agents, heading
            target_positions = self.geometric_formation.get_target_positions(
                target_center.tolist(), n, formation_heading
            )
            
            # Apply formation control - move each agent toward its target position
            formation_gain = 1.2  # Control gain for formation keeping
            
            for i in range(n):
                target_pos = np.array(target_positions.get(i, positions[i]))
                error = target_pos - positions[i]
                control_inputs[i] += formation_gain * error
            
            # Also calculate communication matrices for monitoring
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    rij = calculate_distance(positions[i], positions[j])
                    aij = calculate_aij(self.alpha, self.delta, rij, self.r0, self.v)
                    gij = calculate_gij(rij, self.r0)

                    # Apply jamming degradation to communication quality (paper's model)
                    D_i = 1.0
                    D_j = 1.0
                    for zone in jamming_zones:
                        if zone.active:
                            D_i *= zone.get_degradation_factor(positions[i].tolist())
                            D_j *= zone.get_degradation_factor(positions[j].tolist())

                    self._comm_matrix[i, j] = gij * aij * D_i * D_j  # Degraded by jamming
                    self._dist_matrix[i, j] = rij
                    self._neighbor_matrix[i, j] = aij * D_i * D_j  # Also degrade neighbor metric

        # Calculate performance indicators
        Jn = calculate_Jn(self._comm_matrix, self._neighbor_matrix, self.PT)
        rn = calculate_rn(self._dist_matrix, self._neighbor_matrix, self.PT)

        self.Jn_history.append(Jn)
        self.rn_history.append(rn)

        # Check convergence
        self._check_convergence()

        # =====================================================================
        # PATH PLANNING (compute paths for grid-based algorithms)
        # =====================================================================
        # Path planning only starts AFTER formation has converged
        # During formation phase, agents focus on communication-aware positioning
        # 
        # OPTIMIZATION: Use SWARM CENTER for A* instead of individual agents
        # This reduces computation from N * A* to just 1 * A*
        if self.path_algorithm in ["astar", "theta_star", "bi_astar", "dijkstra", "bfs", "greedy", "msp"]:
            if not self.formation_converged:
                # Formation phase - no path planning yet, ensure no stale paths shown
                self.path_planner.clear_all_paths()
            else:
                # Formation converged - now do path planning toward destination
                dest = np.array(destination)
                
                # Calculate SWARM CENTER - single path for the whole swarm
                swarm_center = np.mean(positions, axis=0)
                
                # Collect obstacles for path planning:
                # 1. Physical obstacles (always visible/known)
                # 2. Discovered jamming zones (from comm degradation detection)
                obstacles_for_planning = list(visible_obstacles)  # Physical always known
                all_known_ids = set(z.id for z in visible_obstacles)
                
                # Add discovered jamming zones from all agents (swarm knowledge sharing)
                for agent_id in agent_ids:
                    for zone in self._discovered_obstacles.get(agent_id, []):
                        if zone.id not in all_known_ids:
                            obstacles_for_planning.append(zone)
                            all_known_ids.add(zone.id)
                
                # Check if swarm center path needs planning
                existing_path = self.path_planner.paths.get("swarm_center")
                needs_replan = existing_path is None or len(existing_path) == 0
                
                if needs_replan:
                    n_physical = len(visible_obstacles)
                    n_discovered = len(obstacles_for_planning) - n_physical
                    print(f"[Controller] SWARM CENTER needs path planning, avoiding {n_physical} physical + {n_discovered} discovered obstacles")
                    # Plan ONE path from swarm center to destination
                    path = self.path_planner.plan_path(
                        start=swarm_center,
                        goal=dest,
                        agent_id="swarm_center",
                        jamming_zones=obstacles_for_planning
                    )
                    if path:
                        print(f"[Controller] Planned swarm center path: {len(path)} waypoints")
                    else:
                        print("[Controller] FAILED to plan swarm center path")

        # =====================================================================
        # DESTINATION CONTROL (after formation converges)
        # =====================================================================
        # Clear avoidance vectors from previous frame
        self._last_avoidance_vectors = {}
        
        # Compute avoidance vectors even before convergence for visualization
        # This helps show obstacle avoidance behavior during the entire simulation
        if True:  # Always compute avoidance for visualization
            dest = np.array(destination)

            for i in range(n):
                agent_id = agent_ids[i]
                pos = positions[i]

                # Check if agent is currently INSIDE a jamming field (for reactive response)
                # This does NOT give prior knowledge - just checks current position
                is_jammed = False
                for jz in invisible_jamming:
                    if jz.is_in_jamming_field(pos.tolist()):
                        is_jammed = True
                        break

                # Get ALL discovered jamming zones from the swarm (knowledge sharing)
                # When ANY agent discovers jamming via comm quality < PT, ALL agents know
                # This enables proactive avoidance by non-jammed vehicles
                all_discovered_jamming = []
                discovered_ids = set()
                for other_agent_id in agent_ids:
                    for zone in self._discovered_obstacles.get(other_agent_id, []):
                        if zone.id not in discovered_ids:
                            all_discovered_jamming.append(zone)
                            discovered_ids.add(zone.id)
                
                # Combine obstacles the agent knows about:
                # 1. Physical obstacles (always visible/known)
                # 2. ALL discovered jamming zones from swarm (shared knowledge)
                known_obstacles = list(visible_obstacles) + all_discovered_jamming

                # =========================================================
                # WAYPOINT-FOLLOWING FOR PATH PLANNING ALGORITHMS
                # =========================================================
                # When using A*/grid-based path planning, follow waypoints instead of going directly to destination
                # This ensures vehicles actually follow the planned path that avoids obstacles
                effective_target = dest  # Default: final destination
                
                if self.path_algorithm in ["astar", "theta_star", "bi_astar", "dijkstra", "bfs", "greedy", "msp"]:
                    # Get swarm center position
                    swarm_center = np.mean(positions, axis=0)
                    
                    # Get next waypoint from planned path
                    next_waypoint = self.path_planner.get_next_waypoint(
                        swarm_center, "swarm_center", threshold=5.0
                    )
                    
                    if next_waypoint is not None:
                        # Use waypoint as target instead of final destination
                        effective_target = np.array(next_waypoint)
                    # else: no path or reached end, fall back to final destination

                # Destination control with obstacle/jamming avoidance
                # Only avoid obstacles the agent KNOWS about (physical + discovered jamming)
                dest_control, avoidance_vec = self._compute_destination_control_with_avoidance(
                    pos, effective_target, known_obstacles, is_jammed
                )

                # Apply jamming response if jammed
                if is_jammed:
                    dest_control = self.jamming_handler.compute_response(
                        agent_id, pos, dest_control, effective_target, known_obstacles
                    )

                # Only apply destination control after convergence
                if self.formation_converged:
                    control_inputs[i] += dest_control
                
                # Store NET control vector for visualization (only when avoidance is active)
                # This shows the actual direction the vehicle will move, not just avoidance
                if avoidance_vec is not None and np.linalg.norm(avoidance_vec) > 0.01:
                    net_control = control_inputs[i]
                    net_magnitude = np.linalg.norm(net_control)
                    if net_magnitude > 0.01:
                        self._last_avoidance_vectors[agent_id] = {
                            'direction': (net_control / (net_magnitude + 1e-6)).tolist(),
                            'magnitude': float(net_magnitude),
                        }

        # Update formation state
        self._update_formation_state(agent_ids, positions)

        # =====================================================================
        # DEADLOCK DETECTION AND BOOST
        # =====================================================================
        # Track swarm center to detect if swarm is stuck
        swarm_center = np.mean(positions, axis=0)
        self._swarm_center_history.append(swarm_center.copy())
        
        # Keep only recent history
        if len(self._swarm_center_history) > self._deadlock_check_window * 2:
            self._swarm_center_history = self._swarm_center_history[-self._deadlock_check_window:]
        
        # Check for deadlock after formation converges
        if self.formation_converged and len(self._swarm_center_history) >= self._deadlock_check_window:
            # Compare current position to position N steps ago
            old_center = self._swarm_center_history[-self._deadlock_check_window]
            movement = np.linalg.norm(swarm_center - old_center)
            
            if movement < self._deadlock_threshold:
                # STUCK! Boost destination control
                self._deadlock_boost = min(3.0, self._deadlock_boost + 0.1)  # Gradual increase
                print(f"[Controller] DEADLOCK DETECTED: moved only {movement:.2f} units in {self._deadlock_check_window} steps, boost={self._deadlock_boost:.2f}")
                
                # Apply boost to all control inputs toward destination
                dest = np.array(destination)
                for i in range(n):
                    to_dest = dest - positions[i]
                    dist_to_dest = np.linalg.norm(to_dest)
                    if dist_to_dest > 1.0:
                        boost_control = (to_dest / dist_to_dest) * self.attraction_magnitude * (self._deadlock_boost - 1.0)
                        control_inputs[i] += boost_control
            else:
                # Moving fine, reduce boost
                self._deadlock_boost = max(1.0, self._deadlock_boost - 0.05)

        # =====================================================================
        # JAMMING DETECTION VIA COMMUNICATION QUALITY < PT (REALISTIC)
        # =====================================================================
        # Agents detect jamming when comm quality drops below PT (same threshold as LLM assistance)
        # This is the ONLY way to discover invisible jamming zones - no prior knowledge!
        # Use GLOBAL tracking to avoid duplicate obstacle detection from multiple agents
        if not hasattr(self, '_global_discovered_positions'):
            self._global_discovered_positions = set()
        
        for i, agent_id in enumerate(agent_ids):
            current = self._agent_comm_quality.get(agent_id, 1.0)
            
            # Only detect after baseline is established (first N samples)
            history = self._comm_quality_history.get(agent_id, [])
            if len(history) <= self._baseline_samples:
                continue
            
            # Detect jamming when comm quality < PT (same threshold as LLM assistance)
            # This unifies detection - both path replanning AND LLM assistance trigger at same point
            if current > 0 and current < self.PT:
                # Round position to LARGER grid (20 units) to reduce duplicates
                # This means obstacles within 20 units of each other are considered the same
                grid_size = 20
                pos_rounded = (round(positions[i][0] / grid_size) * grid_size,
                               round(positions[i][1] / grid_size) * grid_size,
                               round(positions[i][2] / grid_size) * grid_size)
                
                # Check GLOBAL set - if any agent already detected this area, skip
                # Limit total discovered obstacles to prevent runaway detection
                max_obstacles = 10
                total_obstacles = sum(len(obs) for obs in self._discovered_obstacles.values())
                
                if pos_rounded not in self._global_discovered_positions and total_obstacles < max_obstacles:
                    print(f"[Controller] {agent_id} DETECTED JAMMING via comm quality < PT: "
                          f"quality={current:.3f} (PT={self.PT:.3f})")
                    
                    # Mark a SMALL obstacle at detected position
                    # The agent doesn't know the exact zone size/center, so we use a conservative estimate
                    # IMPORTANT: Use small radius to avoid blocking paths - the obstacle is just a "waypoint" 
                    # indicating "jamming detected here", not the full zone
                    import time
                    estimated_obstacle = JammingZone(
                        id=f"discovered_{agent_id}_{int(time.time()*1000)}",
                        center=positions[i].tolist(),  # 3D position where comm dropped
                        radius=5.0,  # SMALL radius - just mark the detection point
                        intensity=1.0,
                        active=True,
                        obstacle_type=ObstacleType.LOW_JAM,  # Comm degradation implies jamming
                    )
                    
                    # Add to GLOBAL discovered positions
                    self._global_discovered_positions.add(pos_rounded)
                    
                    # Add to discovered obstacles for this agent
                    if agent_id not in self._discovered_obstacles:
                        self._discovered_obstacles[agent_id] = []
                    self._discovered_obstacles[agent_id].append(estimated_obstacle)
                    
                    if agent_id not in self._discovered_obstacle_positions:
                        self._discovered_obstacle_positions[agent_id] = []
                    self._discovered_obstacle_positions[agent_id].append(pos_rounded)
                    
                    # Add to known zone IDs
                    if agent_id not in self._known_zone_ids:
                        self._known_zone_ids[agent_id] = set()
                    self._known_zone_ids[agent_id].add(estimated_obstacle.id)
                    
                    # Trigger path replan by clearing swarm center path
                    self.path_planner.clear_path("swarm_center")
                    print(f"[Controller] Marked obstacle at {pos_rounded}, triggering path replan (total: {total_obstacles + 1})")

        # Create commands
        commands = {}
        max_control_magnitude = 10.0  # Limit control input magnitude to prevent runaway
        
        for i, agent_id in enumerate(agent_ids):
            # Clamp control inputs to prevent extreme values
            vel = control_inputs[i]
            vel_magnitude = np.linalg.norm(vel)
            if vel_magnitude > max_control_magnitude:
                vel = vel * (max_control_magnitude / vel_magnitude)
                print(f"[Controller] WARNING: Clamped control for {agent_id} from {vel_magnitude:.1f} to {max_control_magnitude}")
            
            # Calculate new target position
            new_pos = positions[i] + vel * dt
            
            # Clamp position to world bounds
            new_pos = np.clip(new_pos, self.bounds_min, self.bounds_max)

            # Calculate heading from velocity
            heading = float(np.arctan2(vel[1], vel[0])) if np.linalg.norm(vel[:2]) > 0.01 else 0.0

            commands[agent_id] = VehicleCommand(
                agent_id=agent_id,
                target_position=new_pos.tolist(),
                velocity=vel.tolist(),
                heading=heading,
            )

            # Store full path history (trail)
            if agent_id not in self._agent_paths:
                self._agent_paths[agent_id] = []
            self._agent_paths[agent_id].append(new_pos.tolist())
            
            # Cap at 2000 points max to prevent memory issues
            if len(self._agent_paths[agent_id]) > 2000:
                self._agent_paths[agent_id] = self._agent_paths[agent_id][-2000:]

        return commands

    def _compute_destination_control(
        self,
        pos: np.ndarray,
        dest: np.ndarray,
        jamming_zones: list[JammingZone],
        is_jammed: bool,
    ) -> np.ndarray:
        """
        Compute destination-reaching control with obstacle/jamming zone avoidance.
        
        Combines destination control with obstacle avoidance.
        Uses behavior-based control when path_algorithm is "direct".
        """
        control = np.zeros(3)

        if self.path_algorithm == "direct":
            # Direct destination with behavior-based obstacle avoidance
            control = self._behavior_based_control(pos, dest, jamming_zones)
        else:
            # Simple destination control with avoidance
            control = self._simple_destination_control(pos, dest, jamming_zones)

        return control
    
    def _compute_destination_control_with_avoidance(
        self,
        pos: np.ndarray,
        dest: np.ndarray,
        jamming_zones: list[JammingZone],
        is_jammed: bool = False,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Compute destination control and return both control vector and avoidance component.
        
        Returns:
            Tuple of (control_vector, avoidance_vector)
        """
        control = np.zeros(3)
        avoidance = np.zeros(3)

        if self.path_algorithm == "direct":
            # Direct destination with behavior-based obstacle avoidance
            control, avoidance = self._behavior_based_control_with_avoidance(pos, dest, jamming_zones)
        else:
            # Simple destination control with avoidance
            control, avoidance = self._simple_destination_control_with_avoidance(pos, dest, jamming_zones)

        return control, avoidance

    def _behavior_based_control(
        self,
        pos: np.ndarray,
        dest: np.ndarray,
        jamming_zones: list[JammingZone],
    ) -> np.ndarray:
        """
        Behavior-based obstacle avoidance for direct path planning.
        
        Behavior depends on obstacle type:
        - physical: Hard avoid (must path around completely)
        - high_jam: Strong avoid (large buffer zone)
        - low_jam: Cautious proceed (normal buffer zone)
        """
        control = np.zeros(3)
        has_obstacle_influence = False

        # Check for obstacle/jamming zone influence
        for zone in jamming_zones:
            if not zone.active:
                continue

            obstacle_pos = np.array(zone.center)
            # Use jamming radius for jamming types, physical radius for physical
            obstacle_type = getattr(zone, 'obstacle_type', ObstacleType.LOW_JAM)
            if obstacle_type == ObstacleType.PHYSICAL:
                effective_radius = zone.radius
            else:
                effective_radius = zone.jamming_radius

            # Calculate distance to obstacle center
            dist_to_center = np.linalg.norm(pos - obstacle_pos)

            # Buffer zones based on obstacle type
            if obstacle_type == ObstacleType.PHYSICAL:
                # Physical obstacles: must hard avoid
                buffer_zone = effective_radius + self.buffer_zone * 2.0
                wall_follow_zone_dist = effective_radius + self.wall_follow_zone * 2.0
            elif obstacle_type == ObstacleType.HIGH_JAM:
                # High-power jamming: strong avoid
                buffer_zone = effective_radius + self.buffer_zone * 1.5
                wall_follow_zone_dist = effective_radius + self.wall_follow_zone * 1.5
            else:  # LOW_JAM (default)
                # Low-power jamming: cautious proceed
                buffer_zone = effective_radius + self.buffer_zone
                wall_follow_zone_dist = effective_radius + self.wall_follow_zone

            if dist_to_center < buffer_zone:
                has_obstacle_influence = True

                if dist_to_center < wall_follow_zone_dist:
                    # Strong avoidance when very close
                    avoidance = self._obstacle_avoidance_3d(pos, obstacle_pos, effective_radius)
                    control += avoidance
                    # Minimal destination control when very close to obstacle
                    control += self._destination_control_3d(pos, dest, weight=0.2)
                else:
                    # Wall following when in outer buffer zone
                    if dist_to_center > 0:
                        wall_normal = (pos - obstacle_pos) / dist_to_center
                        wall_pos = obstacle_pos + wall_normal * effective_radius
                        wall_follow = self._wall_following_3d(pos, dest, wall_pos, wall_normal)
                        control += wall_follow
                    # Reduced destination control during wall following
                    control += self._destination_control_3d(pos, dest, weight=0.3)

        # If not influenced by any obstacle, apply normal destination control
        if not has_obstacle_influence:
            control += self._destination_control_3d(pos, dest, weight=1.0)

        return control
    
    def _behavior_based_control_with_avoidance(
        self,
        pos: np.ndarray,
        dest: np.ndarray,
        jamming_zones: list[JammingZone],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Behavior-based control that also returns the avoidance component.
        
        Returns:
            Tuple of (total_control, avoidance_component)
        """
        control = np.zeros(3)
        total_avoidance = np.zeros(3)
        has_obstacle_influence = False

        # Check for obstacle/jamming zone influence
        for zone in jamming_zones:
            if not zone.active:
                continue

            obstacle_pos = np.array(zone.center)
            obstacle_type = getattr(zone, 'obstacle_type', ObstacleType.LOW_JAM)
            
            # Use jamming radius for jamming types, physical radius for physical
            if obstacle_type == ObstacleType.PHYSICAL:
                effective_radius = zone.radius
            else:
                effective_radius = zone.jamming_radius
            
            dist_to_center = np.linalg.norm(pos - obstacle_pos)

            # Buffer zones based on obstacle type
            if obstacle_type == ObstacleType.PHYSICAL:
                buffer_zone = effective_radius + self.buffer_zone * 2.0
                wall_follow_zone_dist = effective_radius + self.wall_follow_zone * 2.0
            elif obstacle_type == ObstacleType.HIGH_JAM:
                buffer_zone = effective_radius + self.buffer_zone * 1.5
                wall_follow_zone_dist = effective_radius + self.wall_follow_zone * 1.5
            else:  # LOW_JAM
                buffer_zone = effective_radius + self.buffer_zone
                wall_follow_zone_dist = effective_radius + self.wall_follow_zone

            if dist_to_center < buffer_zone:
                has_obstacle_influence = True

                if dist_to_center < wall_follow_zone_dist:
                    avoidance = self._obstacle_avoidance_3d(pos, obstacle_pos, effective_radius)
                    control += avoidance
                    total_avoidance += avoidance
                    # INCREASED: Ensure meaningful destination control even near obstacles
                    control += self._destination_control_3d(pos, dest, weight=0.5)
                else:
                    if dist_to_center > 0:
                        wall_normal = (pos - obstacle_pos) / dist_to_center
                        wall_pos = obstacle_pos + wall_normal * effective_radius
                        wall_follow = self._wall_following_3d(pos, dest, wall_pos, wall_normal)
                        control += wall_follow
                        total_avoidance += wall_follow * 0.5
                    # INCREASED: Better destination pull in buffer zone
                    control += self._destination_control_3d(pos, dest, weight=0.6)

        # ALWAYS add minimum destination control to prevent deadlock
        # This ensures forward progress even when surrounded by obstacles
        min_dest_control = self._destination_control_3d(pos, dest, weight=0.3)
        control += min_dest_control
        
        if not has_obstacle_influence:
            # No obstacles - strong destination control
            control += self._destination_control_3d(pos, dest, weight=0.7)

        return control, total_avoidance
    
    def _simple_destination_control_with_avoidance(
        self,
        pos: np.ndarray,
        dest: np.ndarray,
        jamming_zones: list[JammingZone],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simple destination control that also returns the avoidance component.
        
        Returns:
            Tuple of (total_control, avoidance_component)
        """
        am = self.attraction_magnitude
        bm = self.distance_threshold

        # Direction to destination
        dest_vector = dest - pos
        dist_to_dest = np.linalg.norm(dest_vector)

        if dist_to_dest < 0.1:
            return np.zeros(3), np.zeros(3)

        dest_direction = dest_vector / dist_to_dest
        if dist_to_dest > bm:
            control_param = am
        else:
            control_param = am * (dist_to_dest / bm)

        dest_control = dest_direction * control_param

        # Add jamming zone avoidance
        avoidance = np.zeros(3)

        for zone in jamming_zones:
            if not zone.active:
                continue

            center = np.array(zone.center)
            to_agent = pos - center
            dist = np.linalg.norm(to_agent)
            
            # Use obstacle type to determine buffer multiplier
            obstacle_type = getattr(zone, 'obstacle_type', ObstacleType.LOW_JAM)
            if obstacle_type == ObstacleType.PHYSICAL:
                buffer_mult = 2.0
                effective_radius = zone.radius
            elif obstacle_type == ObstacleType.HIGH_JAM:
                buffer_mult = 1.5
                effective_radius = zone.jamming_radius
            else:  # LOW_JAM
                buffer_mult = 1.0
                effective_radius = zone.jamming_radius

            buffer = effective_radius + self.buffer_zone * buffer_mult
            wall_follow = effective_radius + self.wall_follow_zone * buffer_mult

            if dist < buffer and dist > 0:
                if dist < wall_follow:
                    avoidance_strength = self.avoidance_magnitude * np.exp(-0.3 * (dist - effective_radius))
                    avoidance += (to_agent / dist) * avoidance_strength
                    # INCREASED: Don't reduce dest_control as much to prevent deadlock
                    dest_control *= 0.5
                else:
                    avoidance_strength = 2.0 * (1.0 - (dist - wall_follow) / (buffer - wall_follow))
                    avoidance += (to_agent / dist) * avoidance_strength
                    dest_control *= 0.6

        # ALWAYS maintain minimum destination pull to prevent deadlock
        dest_norm = np.linalg.norm(dest - pos)
        if dest_norm > 0.1:
            min_dest_pull = (dest - pos) / dest_norm * self.attraction_magnitude * 0.3
            dest_control = np.maximum(np.abs(dest_control), np.abs(min_dest_pull)) * np.sign(dest_control + min_dest_pull + 1e-6)
        
        return dest_control + avoidance, avoidance

    def _destination_control_3d(
        self,
        pos: np.ndarray,
        dest: np.ndarray,
        weight: float = 1.0,
    ) -> np.ndarray:
        """
        Destination-reaching control input - attracts agent toward goal.
        """
        am = self.attraction_magnitude
        bm = self.distance_threshold

        dest_vector = dest - pos
        dist_to_dest = np.linalg.norm(dest_vector)

        if dist_to_dest < 0.1:
            return np.zeros(3)

        dest_direction = dest_vector / dist_to_dest

        # Scale control input based on distance
        if dist_to_dest > bm:
            control_param = am
        else:
            control_param = am * (dist_to_dest / bm)

        return weight * dest_direction * control_param

    def _obstacle_avoidance_3d(
        self,
        pos: np.ndarray,
        obstacle_pos: np.ndarray,
        obstacle_radius: float,
    ) -> np.ndarray:
        """
        Obstacle avoidance control - repels agent from obstacles/jamming zones.
        """
        ao = self.avoidance_magnitude
        bo = self.buffer_zone

        # Vector away from obstacle
        obstacle_vector = pos - obstacle_pos
        dist_to_obstacle = np.linalg.norm(obstacle_vector)

        if dist_to_obstacle < (obstacle_radius + bo) and dist_to_obstacle > 0:
            avoidance_direction = obstacle_vector / dist_to_obstacle

            # Exponential scaling for aggressive close-range avoidance
            proximity_factor = np.exp(-0.3 * (dist_to_obstacle - obstacle_radius))
            control_param = ao * proximity_factor * (1 + 1 / (dist_to_obstacle - obstacle_radius + 0.1))

            return avoidance_direction * control_param

        return np.zeros(3)

    def _wall_following_3d(
        self,
        pos: np.ndarray,
        dest: np.ndarray,
        wall_pos: np.ndarray,
        wall_normal: np.ndarray,
    ) -> np.ndarray:
        """
        Wall-following control - guides agent around obstacle boundaries.
        """
        af = 2.0  # Wall following force
        df = 12.0  # Desired distance from wall

        # Calculate perpendicular distance to wall
        distance_to_wall = np.dot(pos - wall_pos, wall_normal)

        # Create tangent direction using cross product
        if abs(wall_normal[2]) < 0.9:
            reference = np.array([0, 0, 1])
        else:
            reference = np.array([1, 0, 0])

        tangent1 = np.cross(wall_normal, reference)
        tangent1 = tangent1 / (np.linalg.norm(tangent1) + 1e-6)

        # Get direction to destination
        to_dest = dest - pos
        to_dest_norm = np.linalg.norm(to_dest)
        if to_dest_norm > 0:
            to_dest = to_dest / to_dest_norm

        # Project destination direction onto tangent plane
        tangent_dest = to_dest - np.dot(to_dest, wall_normal) * wall_normal
        tangent_dest_norm = np.linalg.norm(tangent_dest)
        if tangent_dest_norm > 0:
            tangent_direction = tangent_dest / tangent_dest_norm
        else:
            tangent_direction = tangent1

        # Wall following behavior
        if abs(distance_to_wall) > df:
            # Correction when too close or too far from wall
            correction = -np.sign(distance_to_wall) * wall_normal
            control = af * (0.4 * tangent_direction + 0.6 * correction)
        else:
            # Wall following when at good distance
            control = 1.2 * af * tangent_direction

        return control

    def _simple_destination_control(
        self,
        pos: np.ndarray,
        dest: np.ndarray,
        jamming_zones: list[JammingZone],
    ) -> np.ndarray:
        """
        Simple destination control with basic avoidance (non-behavior mode).
        Uses obstacle type to determine avoidance behavior.
        """
        am = self.attraction_magnitude
        bm = self.distance_threshold

        # Direction to destination
        dest_vector = dest - pos
        dist_to_dest = np.linalg.norm(dest_vector)

        if dist_to_dest < 0.1:
            return np.zeros(3)

        dest_direction = dest_vector / dist_to_dest

        # Scale control based on distance
        if dist_to_dest > bm:
            control_param = am
        else:
            control_param = am * (dist_to_dest / bm)

        dest_control = dest_direction * control_param

        # Add jamming zone avoidance based on obstacle type
        avoidance = np.zeros(3)
            
        for zone in jamming_zones:
            if not zone.active:
                continue

            center = np.array(zone.center)
            to_agent = pos - center
            dist = np.linalg.norm(to_agent)
            
            # Determine buffer multiplier based on obstacle type
            obstacle_type = getattr(zone, 'obstacle_type', ObstacleType.LOW_JAM)
            if obstacle_type == ObstacleType.PHYSICAL:
                buffer_mult = 2.0
                effective_radius = zone.radius
            elif obstacle_type == ObstacleType.HIGH_JAM:
                buffer_mult = 1.5
                effective_radius = zone.jamming_radius
            else:  # LOW_JAM
                buffer_mult = 1.0
                effective_radius = zone.jamming_radius

            buffer = effective_radius + self.buffer_zone * buffer_mult
            wall_follow = effective_radius + self.wall_follow_zone * buffer_mult

            if dist < buffer and dist > 0:
                if dist < wall_follow:
                    avoidance_strength = self.avoidance_magnitude * np.exp(-0.3 * (dist - effective_radius))
                    avoidance += (to_agent / dist) * avoidance_strength
                    # INCREASED: Don't reduce dest_control as much to prevent deadlock
                    dest_control *= 0.5
                else:
                    avoidance_strength = 2.0 * (1.0 - (dist - wall_follow) / (buffer - wall_follow))
                    avoidance += (to_agent / dist) * avoidance_strength
                    dest_control *= 0.6

        # ALWAYS maintain minimum destination pull to prevent deadlock
        dest_norm = np.linalg.norm(dest - pos)
        if dest_norm > 0.1:
            min_dest_pull = (dest - pos) / dest_norm * self.attraction_magnitude * 0.3
            dest_control += min_dest_pull
        
        return dest_control + avoidance

    def _check_convergence(self):
        """Check if formation has converged based on Jn stabilizing."""
        if len(self.Jn_history) < self.convergence_threshold:
            return

        recent = self.Jn_history[-self.convergence_threshold:]

        # Check if Jn has stabilized (all values within small tolerance)
        rounded = [round(x, 4) for x in recent]
        if len(set(rounded)) == 1:
            if not self.formation_converged:
                print(f"[Controller] Formation converged! Jn={recent[-1]:.4f}")
                print("[Controller] Now moving toward destination...")
                self.formation_converged = True

    def _update_formation_state(self, agent_ids: list[str], positions: np.ndarray):
        """Update cached formation state."""
        n = len(agent_ids)

        # Determine neighbors and communication quality
        roles = {}
        neighbors = {}
        agent_comm_quality = {}  # Per-agent average phi_rij

        for i, aid in enumerate(agent_ids):
            # Find neighbors (agents with aij > PT)
            agent_neighbors = [
                agent_ids[j] for j in range(n)
                if i != j and self._neighbor_matrix[i, j] > self.PT
            ]
            neighbors[aid] = agent_neighbors

            # Calculate per-agent average phi_rij (communication quality)
            # phi_rij = gij * aij is stored in _comm_matrix
            neighbor_count = 0
            phi_sum = 0.0
            for j in range(n):
                if i != j and self._neighbor_matrix[i, j] > self.PT:
                    phi_sum += self._comm_matrix[i, j]
                    neighbor_count += 1

            if neighbor_count > 0:
                agent_comm_quality[aid] = phi_sum / neighbor_count
            else:
                agent_comm_quality[aid] = 0.0

            # Communication-aware formation is DISTRIBUTED - no leader/follower roles
            # Only geometric formations have hierarchical roles
            if self.formation_type == "communication_aware":
                roles[aid] = None  # No role in distributed control
            else:
                # Assign roles for geometric formations
                if i == 0:
                    roles[aid] = FormationRole.LEADER
                elif len(agent_neighbors) >= 3:
                    roles[aid] = FormationRole.WINGMAN
                else:
                    roles[aid] = FormationRole.FOLLOWER

        # Store per-agent comm quality for API access
        self._agent_comm_quality = agent_comm_quality

        # Track comm quality history and establish baseline for jamming detection
        # (Realistic: agents detect jamming through comm quality drop)
        for aid in agent_ids:
            current_quality = agent_comm_quality.get(aid, 0.0)
            
            # Initialize history if needed
            if aid not in self._comm_quality_history:
                self._comm_quality_history[aid] = []
            
            # Add current quality to history
            self._comm_quality_history[aid].append(current_quality)
            
            # Keep history bounded (last 100 samples)
            if len(self._comm_quality_history[aid]) > 100:
                self._comm_quality_history[aid] = self._comm_quality_history[aid][-100:]
            
            # Establish baseline from first N samples (before hitting jamming)
            history = self._comm_quality_history[aid]
            if len(history) <= self._baseline_samples:
                # Still building baseline - use mean of current history
                self._baseline_comm_quality[aid] = sum(history) / len(history) if history else 0.97

        # Calculate formation error (deviation from optimal comm quality)
        if n > 1:
            error = 1.0 - (self.Jn_history[-1] if self.Jn_history else 0.0)
        else:
            error = 0.0

        self._formation_state = FormationState(
            converged=self.formation_converged,
            formation_error=float(error),
            average_comm_quality=float(self.Jn_history[-1]) if self.Jn_history else 0.0,
            average_neighbor_distance=float(self.rn_history[-1]) if self.rn_history else 0.0,
            roles={k: v for k, v in roles.items()},
            neighbors=neighbors,
        )

    def get_formation_state(self) -> FormationState:
        """Get current formation status."""
        return self._formation_state

    def get_agent_comm_quality(self, agent_id: str) -> float:
        """Get per-agent communication quality (average phi_rij)."""
        return self._agent_comm_quality.get(agent_id, 0.0)

    def get_all_agent_comm_quality(self) -> dict[str, float]:
        """Get communication quality for all agents."""
        return self._agent_comm_quality.copy()

    def get_path_for_agent(self, agent_id: str) -> Optional[list[list[float]]]:
        """Get recorded path for an agent."""
        return self._agent_paths.get(agent_id)

    def get_all_traveled_paths(self, short: bool = False) -> dict[str, list[list[float]]]:
        """
        Get all traveled paths (trail history) for visualization.
        
        Args:
            short: If True, return only last 100 points (short trail)
        """
        if short:
            return {
                agent_id: path[-100:] if len(path) > 100 else path
                for agent_id, path in self._agent_paths.items()
            }
        return self._agent_paths.copy()

    def get_planned_waypoints(self) -> dict[str, list[list[float]]]:
        """
        Get planned waypoints from path planner.
        Returns dict with "swarm_center" -> list of [x, y, z] waypoints.
        (Swarm center path is shared by all agents for efficiency)
        """
        return self.path_planner.get_all_paths()

    def get_communication_links(self) -> list[dict]:
        """
        Get all communication links between agents.
        
        Shows both strong links (aij >= PT) and weak/degraded links.

        Returns:
            List of dicts with: {
                "from": agent_id,
                "to": agent_id,
                "quality": phi_rij (0-1),
                "distance": rij,
                "strong": bool (True if aij >= PT)
            }
        """
        links = []

        # #region agent log
        import json as _json
        from datetime import datetime as _dt
        _log_path = "/home/singsong/Downloads/cars_demo_13/sim/.cursor/debug.log"
        _log_entry = {
            "location": "controller.py:get_communication_links:entry",
            "message": "Getting communication links",
            "data": {
                "neighbor_matrix_is_none": self._neighbor_matrix is None,
                "comm_matrix_is_none": self._comm_matrix is None,
                "neighbor_matrix_shape": list(self._neighbor_matrix.shape) if self._neighbor_matrix is not None else None,
            },
            "timestamp": _dt.now().timestamp() * 1000,
            "sessionId": "debug-session",
            "hypothesisId": "A,B"
        }
        try:
            with open(_log_path, "a") as _f:
                _f.write(_json.dumps(_log_entry) + "\n")
        except Exception:
            pass
        # #endregion

        if self._neighbor_matrix is None or self._comm_matrix is None:
            return links

        n = self._neighbor_matrix.shape[0]
        agent_ids = list(self._formation_state.neighbors.keys()) if self._formation_state.neighbors else []

        if len(agent_ids) != n:
            # Fallback to generic IDs
            agent_ids = [f"agent{i+1}" for i in range(n)]

        # Only add each link once (i < j to avoid duplicates)
        for i in range(n):
            for j in range(i + 1, n):
                quality = float(self._comm_matrix[i, j]) if self._comm_matrix is not None else 0
                aij = float(self._neighbor_matrix[i, j])
                
                # Show links if there's any meaningful communication (quality > 0.3)
                # or if agents are close enough to potentially communicate
                if quality > 0.3 or aij > 0.5:
                    links.append({
                        "from": agent_ids[i],
                        "to": agent_ids[j],
                        "quality": quality,
                        "distance": float(self._dist_matrix[i, j]) if self._dist_matrix is not None else 0,
                        "strong": aij >= self.PT,  # True if above perception threshold
                    })

        # #region agent log
        _log_entry2 = {
            "location": "controller.py:get_communication_links:exit",
            "message": "Returning communication links",
            "data": {
                "links_count": len(links),
                "sample_qualities": [link["quality"] for link in links[:3]] if links else [],
                "sample_aij": [link["strong"] for link in links[:3]] if links else [],
            },
            "timestamp": _dt.now().timestamp() * 1000,
            "sessionId": "debug-session",
            "hypothesisId": "A,B"
        }
        try:
            with open(_log_path, "a") as _f:
                _f.write(_json.dumps(_log_entry2) + "\n")
        except Exception:
            pass
        # #endregion

        return links

    def get_visualization_data(self) -> dict:
        """
        Get all visualization data for the frontend.
        
        Returns:
            Dict with communication_links, waypoints, formation_state, avoidance_vectors
        """
        return {
            "communication_links": self.get_communication_links(),
            "waypoints": self.get_planned_waypoints(),
            "avoidance_vectors": self.get_avoidance_vectors(),
            "formation": {
                "converged": self.formation_converged,
                "type": self.formation_type,
                "path_algorithm": self.path_algorithm,
                "Jn": self.Jn_history[-1] if self.Jn_history else 0,
                "rn": self.rn_history[-1] if self.rn_history else 0,
            }
        }
    
    def get_avoidance_vectors(self) -> dict:
        """
        Get current avoidance vectors for each agent.
        
        Returns:
            Dict mapping agent_id to avoidance vector info
        """
        result = self._last_avoidance_vectors if hasattr(self, '_last_avoidance_vectors') else {}
        
        # #region agent log
        import json as _json
        from datetime import datetime as _dt
        _log_path = "/home/singsong/Downloads/cars_demo_13/sim/.cursor/debug.log"
        _log_entry = {
            "location": "controller.py:get_avoidance_vectors",
            "message": "Getting avoidance vectors",
            "data": {
                "avoidance_vectors_count": len(result),
                "agent_ids_with_avoidance": list(result.keys()),
                "formation_converged": self.formation_converged,
            },
            "timestamp": _dt.now().timestamp() * 1000,
            "sessionId": "debug-session",
            "hypothesisId": "E"
        }
        try:
            with open(_log_path, "a") as _f:
                _f.write(_json.dumps(_log_entry) + "\n")
        except Exception:
            pass
        # #endregion
        
        return result

    def reset(self):
        """Reset controller state."""
        self.formation_converged = False
        self.Jn_history.clear()
        self.rn_history.clear()
        self._agent_paths.clear()
        self.path_planner.clear_all_paths()
        self.jamming_handler.reset()
        # Reset per-agent discovered obstacles (realistic: agents forget discovered zones on reset)
        self._discovered_obstacles.clear()
        self._known_zone_ids.clear()
        self._discovered_obstacle_positions.clear()
        # Reset global discovered positions
        if hasattr(self, '_global_discovered_positions'):
            self._global_discovered_positions.clear()
        # Reset comm quality tracking for jamming detection
        self._comm_quality_history.clear()
        self._baseline_comm_quality.clear()
        # Reset deadlock detection
        self._swarm_center_history.clear()
        self._deadlock_boost = 1.0
        print("[Controller] Reset")

    def get_discovered_obstacles(self) -> list[dict]:
        """
        Get all discovered obstacles for visualization (minimap).
        
        Returns:
            List of obstacle dicts with center, radius, discovered_by
        """
        obstacles = []
        for agent_id, agent_obstacles in self._discovered_obstacles.items():
            for obs in agent_obstacles:
                obstacles.append({
                    "id": obs.id,
                    "center": obs.center,
                    "radius": obs.radius,
                    "discovered_by": agent_id,
                })
        return obstacles

    def set_formation_type(self, formation_type: str):
        """Change formation type."""
        valid_types = ["communication_aware"] + FORMATION_TYPES
        if formation_type in valid_types:
            self.formation_type = formation_type
            if formation_type != "communication_aware":
                self.geometric_formation.set_formation_type(formation_type)
            print(f"[Controller] Formation changed to: {formation_type}")

    def set_path_algorithm(self, algorithm: str):
        """Change path planning algorithm."""
        if algorithm in PATH_ALGORITHMS:
            self.path_algorithm = algorithm
            self.path_planner.set_algorithm(algorithm)
            print(f"[Controller] Path algorithm changed to: {algorithm}")

# Singleton controller instance
_controller: Optional[UnifiedController] = None


def get_controller() -> UnifiedController:
    """Get or create controller instance."""
    global _controller
    if _controller is None:
        _controller = UnifiedController()
    return _controller


def reset_controller():
    """Reset the global controller."""
    global _controller
    if _controller:
        _controller.reset()
    _controller = None
