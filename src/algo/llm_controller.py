"""
LLM Assistance Controller for autonomous vehicle control.

When enabled, this controller monitors agent communication quality and provides
LLM-guided control when communication degrades below the perception threshold (PT).

The LLM uses historical telemetry data and knowledge of jamming zones to compute
optimal evasion vectors that help agents:
1. Restore communication quality
2. Continue progressing toward the destination
3. Avoid jamming zones
"""
import json
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import numpy as np

# Import config
try:
    from ..config import (
        LLM_MODEL,
        LLM_TIMEOUT,
        MISSION_END,
        OLLAMA_HOST,
        PT,
        chat_with_retry,
        get_ollama_client,
    )
    from ..rag import get_telemetry_history
except ImportError:
    # Fallback for standalone testing
    PT = 0.94
    MISSION_END = (35, 150, 30)
    LLM_MODEL = "llama3.2:3b-instruct-q4_K_M"
    LLM_TIMEOUT = 30
    OLLAMA_HOST = "http://localhost:11434"


@dataclass
class LLMGuidance:
    """Guidance output from LLM."""
    agent_id: str
    direction: list[float]  # [dx, dy, dz] normalized direction vector
    speed: float  # Recommended speed multiplier
    reasoning: str  # LLM's explanation
    timestamp: str
    expires_at: float  # Time when this guidance expires


class LLMAssistanceController:
    """
    Controller that provides LLM-guided assistance when agent communication
    quality falls below the perception threshold (PT).
    
    Uses async queue pattern for non-blocking LLM requests.
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize the LLM assistance controller.
        
        Args:
            enabled: Whether LLM assistance is enabled by default
        """
        self.enabled = enabled
        self.pt_threshold = PT
        
        # LLM client
        self._client = None
        self._model = LLM_MODEL
        
        # Async request handling
        self._request_queue = queue.Queue()
        self._result_queue = queue.Queue()
        self._pending_requests: set[str] = set()
        self._worker_thread: Optional[threading.Thread] = None
        
        # Active guidance cache
        self._active_guidance: dict[str, LLMGuidance] = {}
        self._guidance_lifetime = 5.0  # Seconds before guidance expires
        
        # Rate limiting
        self._last_request_time: dict[str, float] = {}
        self._min_request_interval = 2.0  # Minimum seconds between requests per agent
        
        # User command blocking - agents with active user commands are blocked from auto-LLM
        self._user_blocked_agents: set[str] = set()
        
        # Logging
        self._log_history: list[dict] = []
        
        print(f"[LLMAssist] Initialized (enabled={enabled}, PT={self.pt_threshold})")
    
    @property
    def client(self):
        """Lazy-load Ollama client."""
        if self._client is None:
            try:
                self._client = get_ollama_client()
            except Exception as e:
                print(f"[LLMAssist] Failed to get Ollama client: {e}")
        return self._client
    
    def set_enabled(self, enabled: bool):
        """Enable or disable LLM assistance."""
        self.enabled = enabled
        print(f"[LLMAssist] {'Enabled' if enabled else 'Disabled'}")
        
        if not enabled:
            # Clear active guidance when disabled
            self._active_guidance.clear()
    
    def block_agent(self, agent_id: str):
        """
        Block an agent from receiving auto-LLM assistance.
        Called when user command is set for an agent.
        """
        self._user_blocked_agents.add(agent_id)
        # Also clear any active guidance for this agent
        if agent_id in self._active_guidance:
            del self._active_guidance[agent_id]
        # Remove from pending requests
        self._pending_requests.discard(agent_id)
        print(f"[LLMAssist] Blocked auto-assistance for {agent_id} (user command)")
    
    def unblock_agent(self, agent_id: str):
        """
        Unblock an agent, allowing auto-LLM assistance to resume.
        Called when user command completes.
        """
        self._user_blocked_agents.discard(agent_id)
        print(f"[LLMAssist] Unblocked auto-assistance for {agent_id} (user command completed)")
    
    def is_blocked(self, agent_id: str) -> bool:
        """Check if an agent is blocked from auto-LLM assistance."""
        return agent_id in self._user_blocked_agents
    
    def check_agents_needing_assistance(
        self,
        agents: dict[str, Any]
    ) -> list[str]:
        """
        Check which agents need LLM assistance based on communication quality.
        
        Args:
            agents: Dict of agent_id -> agent state
            
        Returns:
            List of agent IDs with comm_quality < PT (excluding blocked agents)
        """
        if not self.enabled:
            return []
        
        needing_assistance = []
        
        for agent_id, agent in agents.items():
            # Skip agents blocked by user commands
            if agent_id in self._user_blocked_agents:
                continue
            
            # Get communication quality
            if hasattr(agent, 'communication_quality'):
                comm_quality = agent.communication_quality
            elif isinstance(agent, dict):
                comm_quality = agent.get('communication_quality', 1.0)
            else:
                continue
            
            # Check if below threshold
            if comm_quality < self.pt_threshold:
                needing_assistance.append(agent_id)
        
        return needing_assistance
    
    def request_guidance(
        self,
        agent_id: str,
        agent_state: Any,
        destination: tuple[float, float, float],
        jamming_zones: list[Any],
        discovered_obstacles: list[Any] = None,
    ):
        """
        Request LLM guidance for an agent (non-blocking).
        
        If LLM is slow/unavailable, immediately uses fallback guidance.
        
        Args:
            agent_id: ID of agent needing assistance
            agent_state: Current state of the agent
            destination: Mission destination coordinates
            jamming_zones: List of active jamming zones
            discovered_obstacles: List of obstacles discovered by the swarm
        """
        if not self.enabled:
            return
        
        # Skip agents blocked by user commands
        if agent_id in self._user_blocked_agents:
            return
        
        # Rate limiting
        current_time = time.time()
        last_request = self._last_request_time.get(agent_id, 0)
        if current_time - last_request < self._min_request_interval:
            return
        
        # Get agent position for fallback
        if hasattr(agent_state, 'position'):
            position = agent_state.position
        elif isinstance(agent_state, dict):
            position = agent_state.get('position', [0, 0, 0])
        else:
            return
        
        # Combine jamming zones with discovered obstacles
        all_zones = []
        for zone in jamming_zones:
            if hasattr(zone, 'center'):
                all_zones.append({'center': zone.center, 'radius': zone.radius})
            elif isinstance(zone, dict):
                all_zones.append({'center': zone.get('center', [0, 0, 0]), 'radius': zone.get('radius', 10)})
        
        # Add discovered obstacles (these have been found by the swarm)
        if discovered_obstacles:
            for obs in discovered_obstacles:
                if hasattr(obs, 'center'):
                    all_zones.append({'center': obs.center, 'radius': obs.radius})
                elif isinstance(obs, dict):
                    all_zones.append({'center': obs.get('center', [0, 0, 0]), 'radius': obs.get('radius', 15)})
        
        # IMMEDIATE FALLBACK: If there's no active guidance and agent is jammed,
        # provide instant deterministic guidance while waiting for LLM
        if agent_id not in self._active_guidance or time.time() >= self._active_guidance[agent_id].expires_at:
            fallback = self._fallback_guidance(agent_id, position, destination, all_zones)
            self._active_guidance[agent_id] = fallback
            print(f"[LLMAssist] INSTANT fallback for {agent_id}: {fallback.reasoning}")
        
        # Don't duplicate pending requests
        if agent_id in self._pending_requests:
            return
        
        # Start worker thread if not running
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._worker_thread = threading.Thread(
                target=self._request_worker,
                daemon=True
            )
            self._worker_thread.start()
        
        # Queue the request
        request = {
            'agent_id': agent_id,
            'agent_state': agent_state,
            'destination': destination,
            'jamming_zones': all_zones,  # Combined zones
            'timestamp': current_time,
        }
        
        self._request_queue.put(request)
        self._pending_requests.add(agent_id)
        self._last_request_time[agent_id] = current_time
        
        print(f"[LLMAssist] Queued LLM request for {agent_id} (fallback active meanwhile)")
    
    def _request_worker(self):
        """Background worker that processes LLM requests."""
        while True:
            try:
                # Get request with timeout
                request = self._request_queue.get(timeout=1.0)
                
                agent_id = request['agent_id']
                
                try:
                    guidance = self._compute_guidance_sync(request)
                    if guidance:
                        self._result_queue.put(guidance)
                except Exception as e:
                    print(f"[LLMAssist] Error computing guidance for {agent_id}: {e}")
                finally:
                    self._pending_requests.discard(agent_id)
                    self._request_queue.task_done()
                    
            except queue.Empty:
                # No requests, check if we should exit
                if self._request_queue.empty():
                    continue
    
    def _compute_guidance_sync(self, request: dict) -> Optional[LLMGuidance]:
        """
        Compute LLM guidance synchronously (runs in worker thread).
        
        Args:
            request: Request dict with agent info
            
        Returns:
            LLMGuidance object or None if failed
        """
        agent_id = request['agent_id']
        agent_state = request['agent_state']
        destination = request['destination']
        jamming_zones = request['jamming_zones']
        
        # Get agent position
        if hasattr(agent_state, 'position'):
            position = agent_state.position
            comm_quality = getattr(agent_state, 'communication_quality', 0.5)
        elif isinstance(agent_state, dict):
            position = agent_state.get('position', [0, 0, 0])
            comm_quality = agent_state.get('communication_quality', 0.5)
        else:
            return None
        
        # Get historical trajectory from Qdrant
        try:
            history = get_telemetry_history(agent_id, limit=10)
            trajectory = [h.get('position', [0, 0, 0]) for h in history]
        except Exception:
            trajectory = []
        
        # Format jamming zones
        zones_info = []
        for zone in jamming_zones:
            if hasattr(zone, 'center'):
                zones_info.append({
                    'center': zone.center,
                    'radius': zone.radius,
                })
            elif isinstance(zone, dict):
                zones_info.append({
                    'center': zone.get('center', [0, 0, 0]),
                    'radius': zone.get('radius', 10),
                })
        
        # Build prompt
        prompt = self._build_prompt(
            agent_id=agent_id,
            position=position,
            comm_quality=comm_quality,
            destination=destination,
            jamming_zones=zones_info,
            trajectory=trajectory,
        )
        
        # Query LLM
        try:
            if self.client is None:
                return self._fallback_guidance(agent_id, position, destination, zones_info)
            
            response = chat_with_retry(
                self.client,
                self._model,
                messages=[{"role": "user", "content": prompt}],
            )
            
            if response:
                content = response.get("message", {}).get("content", "")
                guidance = self._parse_llm_response(agent_id, content)
                
                if guidance:
                    self._log_guidance(agent_id, prompt, content, guidance)
                    return guidance
                    
        except Exception as e:
            print(f"[LLMAssist] LLM query failed for {agent_id}: {e}")
        
        # Fallback to simple avoidance
        return self._fallback_guidance(agent_id, position, destination, zones_info)
    
    def _build_prompt(
        self,
        agent_id: str,
        position: list[float],
        comm_quality: float,
        destination: tuple[float, float, float],
        jamming_zones: list[dict],
        trajectory: list[list[float]],
    ) -> str:
        """Build the LLM prompt for guidance request with pre-computed escape directions."""
        pos_arr = np.array(position)
        dest_arr = np.array(destination)
        to_dest = dest_arr - pos_arr
        dist_to_dest = np.linalg.norm(to_dest)
        
        # Format trajectory
        traj_str = ""
        if trajectory:
            traj_points = [f"({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})" for p in trajectory[-5:]]
            traj_str = f"\nRECENT TRAJECTORY: {' -> '.join(traj_points)}"
        
        # Analyze jamming zones and pre-compute escape directions
        zones_str = ""
        escape_recommendations = []
        
        if jamming_zones:
            zone_parts = []
            for i, z in enumerate(jamming_zones):
                c = np.array(z['center'])
                r = z['radius']
                jamming_radius = r * 2.0  # κ_J = 2
                
                # Calculate relationship to this zone
                to_zone = c - pos_arr
                dist_to_zone_center = np.linalg.norm(to_zone)
                dist_to_zone_edge = dist_to_zone_center - jamming_radius
                
                # Check if agent is INSIDE this jamming zone
                if dist_to_zone_center < jamming_radius:
                    penetration = 1.0 - (dist_to_zone_center / jamming_radius)
                    severity = "severe" if penetration > 0.5 else "moderate" if penetration > 0.2 else "mild"
                    
                    zone_parts.append(
                        f"Zone {i+1}: center=({c[0]:.0f}, {c[1]:.0f}, {c[2]:.0f}), "
                        f"radius={jamming_radius:.0f}, AGENT INSIDE ({severity}, {penetration:.0%} deep)"
                    )
                    
                    # PRE-COMPUTE ESCAPE DIRECTION (tangent escape)
                    if dist_to_zone_center > 0.1:
                        from_zone = pos_arr - c
                        from_zone_norm = from_zone / (np.linalg.norm(from_zone) + 1e-6)
                        to_dest_norm = to_dest / (np.linalg.norm(to_dest) + 1e-6) if dist_to_dest > 0.1 else np.array([0, 1, 0])
                        
                        # Tangent direction (perpendicular to from_zone, toward destination)
                        cross1 = np.cross(from_zone_norm, to_dest_norm)
                        if np.linalg.norm(cross1) > 0.01:
                            tangent = np.cross(cross1, from_zone_norm)
                            tangent = tangent / (np.linalg.norm(tangent) + 1e-6)
                            # Choose direction that points more toward destination
                            if np.dot(tangent, to_dest_norm) < 0:
                                tangent = -tangent
                        else:
                            tangent = from_zone_norm  # Parallel case, just move away
                        
                        # Add push-out component for deep penetration
                        push_weight = min(0.5, penetration * 0.8)
                        escape_dir = (1 - push_weight) * tangent + push_weight * from_zone_norm
                        escape_dir = escape_dir / (np.linalg.norm(escape_dir) + 1e-6)
                        
                        escape_recommendations.append(
                            f"RECOMMENDED ESCAPE from Zone {i+1}: direction=[{escape_dir[0]:.2f}, {escape_dir[1]:.2f}, {escape_dir[2]:.2f}], "
                            f"move {int(max(5, penetration * 15))} units to exit"
                        )
                else:
                    zone_parts.append(
                        f"Zone {i+1}: center=({c[0]:.0f}, {c[1]:.0f}, {c[2]:.0f}), "
                        f"radius={jamming_radius:.0f}, {-dist_to_zone_edge:.0f} units away"
                    )
            
            zones_str = "\nJAMMING ZONES:\n" + "\n".join(zone_parts)
        
        escape_str = ""
        if escape_recommendations:
            escape_str = "\n\n*** PRE-COMPUTED ESCAPE (USE THIS!) ***\n" + "\n".join(escape_recommendations)
        
        return f"""You are a tactical advisor for autonomous vehicle navigation.

SITUATION:
- Agent {agent_id} at ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})
- Destination at ({destination[0]:.1f}, {destination[1]:.1f}, {destination[2]:.1f}) - {dist_to_dest:.1f} units away
- Communication quality: {comm_quality:.2f} (DEGRADED - need to escape jamming)
{zones_str}
{escape_str}
{traj_str}

IMPORTANT: Use the PRE-COMPUTED ESCAPE direction above! It has been calculated to:
1. Move AROUND the jamming zone (tangent escape)
2. Progress toward destination
3. Exit the jamming field efficiently

If the pre-computed escape direction is provided, USE IT DIRECTLY in your response.

Respond with ONLY valid JSON:
{{"direction": [dx, dy, dz], "speed": 0.8, "reasoning": "brief explanation"}}

JSON:"""
    
    def _parse_llm_response(
        self,
        agent_id: str,
        content: str
    ) -> Optional[LLMGuidance]:
        """Parse LLM response into guidance object."""
        try:
            # Clean markdown if present
            text = content.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(
                    line for line in lines 
                    if not line.startswith("```")
                )
            
            # Find JSON in response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = text[start:end]
                data = json.loads(json_str)
                
                direction = data.get("direction", [0, 1, 0])
                speed = float(data.get("speed", 0.5))
                reasoning = data.get("reasoning", "No explanation provided")
                
                # Normalize direction
                direction = np.array(direction, dtype=float)
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                else:
                    direction = np.array([0, 1, 0])
                
                # Clamp speed
                speed = max(0.1, min(1.0, speed))
                
                return LLMGuidance(
                    agent_id=agent_id,
                    direction=direction.tolist(),
                    speed=speed,
                    reasoning=reasoning,
                    timestamp=datetime.now().isoformat(),
                    expires_at=time.time() + self._guidance_lifetime,
                )
                
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"[LLMAssist] Failed to parse LLM response: {e}")
        
        return None
    
    def _fallback_guidance(
        self,
        agent_id: str,
        position: list[float],
        destination: tuple[float, float, float],
        jamming_zones: list[dict],
    ) -> LLMGuidance:
        """
        Compute deterministic fallback guidance without LLM.
        
        Strategy: AGGRESSIVE TANGENT ESCAPE with PUSH OUT when deep inside jamming.
        - Deep inside jamming (>50% penetration): strong push OUT + tangent
        - Near edge (<50% penetration): pure tangent escape toward destination
        """
        pos = np.array(position)
        dest = np.array(destination)
        
        # Direction to destination
        to_dest = dest - pos
        dist_to_dest = np.linalg.norm(to_dest)
        if dist_to_dest > 0:
            to_dest_norm = to_dest / dist_to_dest
        else:
            to_dest_norm = np.array([0, 1, 0])
        
        # Find the most influential jamming zone
        best_zone = None
        best_influence = 0.0
        
        for zone in jamming_zones:
            zone_center = np.array(zone['center'])
            zone_radius = zone['radius']
            
            to_zone = zone_center - pos
            dist_to_zone = np.linalg.norm(to_zone)
            
            # Calculate influence based on how deep inside the jamming field
            jamming_radius = zone_radius * 2  # Approximate κ_J = 2
            if dist_to_zone < jamming_radius:
                # Influence is stronger when deeper inside
                influence = 1.0 - (dist_to_zone / jamming_radius)
                if influence > best_influence:
                    best_influence = influence
                    best_zone = zone
        
        if best_zone is not None and best_influence > 0.1:
            zone_center = np.array(best_zone['center'])
            zone_radius = best_zone['radius']
            jamming_radius = zone_radius * 2.0
            
            # Vector from zone center to agent
            from_zone = pos - zone_center
            dist_from_zone = np.linalg.norm(from_zone)
            
            if dist_from_zone > 0.1:
                from_zone_norm = from_zone / dist_from_zone
                
                # AGGRESSIVE STRATEGY based on penetration depth
                if best_influence > 0.5:
                    # DEEP INSIDE (>50% penetration): PUSH OUT is primary
                    # Move directly outward from zone center with some tangent to continue progress
                    # Distance to edge: jamming_radius - dist_from_zone
                    
                    # Strong push-out direction with some tangent to avoid going through center
                    cross1 = np.cross(from_zone_norm, to_dest_norm)
                    if np.linalg.norm(cross1) > 0.01:
                        tangent = np.cross(cross1, from_zone_norm)
                        tangent = tangent / (np.linalg.norm(tangent) + 1e-6)
                        if np.dot(tangent, to_dest_norm) < 0:
                            tangent = -tangent
                    else:
                        tangent = np.array([0, 1, 0]) if from_zone_norm[1] < 0.5 else np.array([1, 0, 0])
                    
                    # 70% push out, 30% tangent toward destination
                    direction = 0.70 * from_zone_norm + 0.30 * tangent
                    reasoning = f"EMERGENCY EXIT: {best_influence:.0%} deep, pushing OUT"
                else:
                    # NEAR EDGE (<50% penetration): TANGENT ESCAPE is primary
                    cross1 = np.cross(from_zone_norm, to_dest_norm)
                    if np.linalg.norm(cross1) > 0.01:
                        tangent = np.cross(cross1, from_zone_norm)
                        tangent = tangent / (np.linalg.norm(tangent) + 1e-6)
                        if np.dot(tangent, to_dest_norm) < 0:
                            tangent = -tangent
                    else:
                        tangent = from_zone_norm
                    
                    # Blend based on influence: more tangent when deeper
                    escape_weight = min(0.7, best_influence * 1.4)
                    dest_weight = 1.0 - escape_weight
                    
                    # Add push-out component scaled by influence
                    push_out = from_zone_norm * best_influence * 0.4
                    
                    direction = escape_weight * tangent + dest_weight * to_dest_norm + push_out
                    reasoning = f"Tangent escape: {best_influence:.0%} influence, orbiting out"
            else:
                # At zone center - move toward destination
                direction = to_dest_norm
                reasoning = "At jamming center: heading to destination"
        else:
            # Not in jamming - go straight to destination
            direction = to_dest_norm
            reasoning = "Clear path: heading to destination"
        
        # Normalize
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        
        # Speed: faster when deeper in jamming for more decisive escape
        speed = 0.8 + best_influence * 0.2 if best_influence > 0 else 0.8
        
        return LLMGuidance(
            agent_id=agent_id,
            direction=direction.tolist(),
            speed=min(1.0, speed),
            reasoning=reasoning,
            timestamp=datetime.now().isoformat(),
            expires_at=time.time() + self._guidance_lifetime,
        )
    
    def get_guidance(self, agent_id: str) -> Optional[LLMGuidance]:
        """
        Get active guidance for an agent.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Active LLMGuidance or None if no valid guidance
        """
        # Process any pending results
        self._process_result_queue()
        
        # Check for active guidance
        guidance = self._active_guidance.get(agent_id)
        if guidance and time.time() < guidance.expires_at:
            return guidance
        
        # Guidance expired or doesn't exist
        if agent_id in self._active_guidance:
            del self._active_guidance[agent_id]
        
        return None
    
    def _process_result_queue(self):
        """Process completed guidance requests from result queue."""
        while True:
            try:
                guidance = self._result_queue.get_nowait()
                self._active_guidance[guidance.agent_id] = guidance
                print(f"[LLMAssist] New guidance for {guidance.agent_id}: {guidance.reasoning}")
                self._result_queue.task_done()
            except queue.Empty:
                break
    
    def apply_guidance(
        self,
        agent_id: str,
        base_control: np.ndarray,
        guidance: LLMGuidance,
        comm_quality: float = 1.0,
    ) -> np.ndarray:
        """
        Apply LLM guidance with ADAPTIVE weighting based on jamming severity.
        
        When communication quality is very low (deep in jamming), LLM gets MORE weight
        to help escape. When near the edge of jamming, path planning is more dominant.
        
        Control hierarchy:
        1. Human MCP Commands (highest) - blocks LLM auto-assistance
        2. Path Planning + Formation Control (adaptive weight based on comm quality)
        3. LLM Auto-Assistance (adaptive weight - higher when deeper in jamming)
        
        Args:
            agent_id: Agent ID
            base_control: Base control vector from path planning + formation
            guidance: LLM guidance to apply
            comm_quality: Current communication quality (0-1, lower = more jammed)
            
        Returns:
            Blended control vector with adaptive weighting
        """
        if guidance is None:
            return base_control
        
        # Convert guidance direction to control vector
        llm_control = np.array(guidance.direction) * guidance.speed
        
        # ADAPTIVE WEIGHTING based on communication quality
        # - comm_quality close to PT (0.94): mostly path planning (80/20)
        # - comm_quality very low (~0.5): equal weight (50/50) 
        # - comm_quality near 0: LLM dominant (30/70)
        
        # Calculate jamming severity (0 = near PT threshold, 1 = comm_quality near 0)
        severity = max(0.0, min(1.0, 1.0 - comm_quality / self.pt_threshold))
        
        # LLM weight increases with severity: 20% -> 70%
        llm_weight = 0.20 + severity * 0.50
        path_weight = 1.0 - llm_weight
        
        blended = path_weight * base_control + llm_weight * llm_control
        
        print(f"[LLMAssist] {agent_id} weights: path={path_weight:.0%}, llm={llm_weight:.0%} (comm={comm_quality:.2f})")
        
        return blended
    
    def _log_guidance(
        self,
        agent_id: str,
        prompt: str,
        response: str,
        guidance: LLMGuidance,
    ):
        """Log guidance for debugging."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'agent_id': agent_id,
            'prompt_preview': prompt[:200] + "...",
            'response_preview': response[:200] + "...",
            'direction': guidance.direction,
            'speed': guidance.speed,
            'reasoning': guidance.reasoning,
        }
        
        self._log_history.append(entry)
        
        # Keep history manageable
        if len(self._log_history) > 100:
            self._log_history = self._log_history[-100:]
    
    def get_status(self) -> dict:
        """Get controller status."""
        return {
            'enabled': self.enabled,
            'pt_threshold': self.pt_threshold,
            'active_guidance_count': len(self._active_guidance),
            'pending_requests': list(self._pending_requests),
            'log_entries': len(self._log_history),
        }
    
    def get_active_guidance_for_visualization(self) -> list[dict]:
        """
        Get all active guidance for 3D visualization.
        
        Returns:
            List of dicts with agent_id, direction, speed, reasoning for active guidance
        """
        # Process any pending results first
        self._process_result_queue()
        
        current_time = time.time()
        active = []
        expired_count = 0
        
        for agent_id, guidance in list(self._active_guidance.items()):
            if current_time < guidance.expires_at:
                active.append({
                    'agent_id': agent_id,
                    'direction': guidance.direction,
                    'speed': guidance.speed,
                    'reasoning': guidance.reasoning,
                    'timestamp': guidance.timestamp,
                    'expires_in': guidance.expires_at - current_time,
                })
            else:
                # Clean up expired guidance
                del self._active_guidance[agent_id]
                expired_count += 1
        
        # #region agent log
        import json as _json
        from datetime import datetime as _dt
        _log_path = "/home/singsong/Downloads/cars_demo_13/sim/.cursor/debug.log"
        _log_entry = {
            "location": "llm_controller.py:get_active_guidance_for_visualization",
            "message": "Returning LLM guidance for visualization",
            "data": {
                "enabled": self.enabled,
                "active_guidance_count": len(active),
                "expired_count": expired_count,
                "pending_requests": list(self._pending_requests),
                "active_agent_ids": [g['agent_id'] for g in active],
            },
            "timestamp": _dt.now().timestamp() * 1000,
            "sessionId": "debug-session",
            "hypothesisId": "D"
        }
        try:
            with open(_log_path, "a") as _f:
                _f.write(_json.dumps(_log_entry) + "\n")
        except Exception:
            pass
        # #endregion
        
        return active
    
    def get_recent_activity(self, limit: int = 10) -> list[dict]:
        """
        Get recent LLM guidance activity for display in chat panel.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of recent guidance log entries
        """
        return self._log_history[-limit:]
    
    def get_current_context(self, agents: dict = None, jamming_zones: list = None, spoofing_zones: list = None) -> dict:
        """
        Get the current context data that would be sent to LLM.
        
        Args:
            agents: Current agent states (optional)
            jamming_zones: Current jamming zones (optional)
            spoofing_zones: Current spoofing zones (optional)
            
        Returns:
            Dict with context information for display
        """
        context = {
            'enabled': self.enabled,
            'pt_threshold': self.pt_threshold,
            'agents_being_assisted': [],
            'active_guidance': [],
            'last_prompts': [],
            'jamming_zones': [],
            'spoofing_zones': [],
        }
        
        # Add agents needing assistance
        if agents:
            for agent_id, agent in agents.items():
                if hasattr(agent, 'communication_quality'):
                    comm_quality = agent.communication_quality
                elif isinstance(agent, dict):
                    comm_quality = agent.get('communication_quality', 1.0)
                else:
                    continue
                
                if comm_quality < self.pt_threshold:
                    pos = agent.position if hasattr(agent, 'position') else agent.get('position', [0, 0, 0])
                    context['agents_being_assisted'].append({
                        'agent_id': agent_id,
                        'communication_quality': float(comm_quality),
                        'position': [float(p) for p in pos],
                    })
        
        # Add active guidance
        context['active_guidance'] = self.get_active_guidance_for_visualization()
        
        # Add jamming zones info
        if jamming_zones:
            for zone in jamming_zones:
                if hasattr(zone, 'center'):
                    context['jamming_zones'].append({
                        'id': zone.id if hasattr(zone, 'id') else 'unknown',
                        'center': zone.center,
                        'radius': zone.radius,
                    })
                elif isinstance(zone, dict):
                    context['jamming_zones'].append({
                        'id': zone.get('id', 'unknown'),
                        'center': zone.get('center', [0, 0, 0]),
                        'radius': zone.get('radius', 10),
                    })
        
        # Add spoofing zones info
        if spoofing_zones:
            for zone in spoofing_zones:
                if hasattr(zone, 'center'):
                    context['spoofing_zones'].append({
                        'id': zone.id if hasattr(zone, 'id') else 'unknown',
                        'center': zone.center,
                        'radius': zone.radius,
                        'spoof_type': zone.spoof_type.value if hasattr(zone, 'spoof_type') else 'unknown',
                        'active': zone.active if hasattr(zone, 'active') else True,
                    })
                elif isinstance(zone, dict):
                    context['spoofing_zones'].append({
                        'id': zone.get('id', 'unknown'),
                        'center': zone.get('center', [0, 0, 0]),
                        'radius': zone.get('radius', 10),
                        'spoof_type': zone.get('spoof_type', 'unknown'),
                        'active': zone.get('active', True),
                    })

        # Add recent prompts (last 3)
        for entry in self._log_history[-3:]:
            context['last_prompts'].append({
                'agent_id': entry.get('agent_id'),
                'timestamp': entry.get('timestamp'),
                'prompt_preview': entry.get('prompt_preview', '')[:300],
                'reasoning': entry.get('reasoning', ''),
            })
        
        return context


# Global instance
_llm_controller: Optional[LLMAssistanceController] = None


def get_llm_controller() -> LLMAssistanceController:
    """Get or create the global LLM assistance controller."""
    global _llm_controller
    if _llm_controller is None:
        _llm_controller = LLMAssistanceController(enabled=True)
    return _llm_controller


def reset_llm_controller():
    """Reset the global LLM controller."""
    global _llm_controller
    _llm_controller = None
