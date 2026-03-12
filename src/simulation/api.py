"""
Simulation API - FastAPI endpoints for vehicle simulation state.
"""
import asyncio
import uuid
from datetime import datetime
from typing import Any, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ..algo.base import OBSTACLE_PARAMS, JammingZone, ObstacleType
from ..algo.controller import get_controller, reset_controller
from ..algo.crypto_auth import get_crypto_auth, reset_crypto_auth
from ..algo.formation import FORMATION_TYPES
from ..algo.jamming_response import JAMMING_STRATEGIES
from ..algo.llm_controller import get_llm_controller
from ..algo.mavlink import get_mavlink_bus, reset_mavlink_bus
from ..algo.path_planning import PATH_ALGORITHMS
from ..algo.spoofing import SpoofingZone, SpoofType, get_spoofing_engine, reset_spoofing_engine
from ..algo.v2v_channel import reset_channel_model
from ..config import (
    CRYPTO_AUTH_ENABLED,
    DEFAULT_SPOOF_TYPE,
    LLM_ASSISTANCE_ENABLED,
    MAVLINK_ENABLED,
    MISSION_END,
    NUM_AGENTS,
    PHANTOM_COUNT,
    POSITION_FALSIFICATION_MAGNITUDE,
    COORDINATE_ATTACK_VECTOR,
    X_RANGE,
    Y_RANGE,
    Z_RANGE,
    get_initial_obstacles,
    get_initial_spoofing_zones,
    print_config,
)
from .agents import AgentState, init_agents

# Create FastAPI app
app = FastAPI(title="Vehicle Simulation API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory state
agent_states: dict[str, AgentState] = {}
llm_targets: dict[str, list[float]] = {}
jamming_zones: dict[str, JammingZone] = {}

# Simulation control state
simulation_running: bool = False
simulation_task: Optional[asyncio.Task] = None

# LLM assistance state (from config, defaults to True)
llm_assistance_enabled: bool = LLM_ASSISTANCE_ENABLED

# Default obstacle type for new obstacles (from UI selection)
default_obstacle_type: str = "low_jam"

# Spoofing zones state
spoofing_zones: dict[str, SpoofingZone] = {}
default_spoof_type: str = "phantom"

# MAVLink / Crypto state
mavlink_enabled: bool = MAVLINK_ENABLED
crypto_auth_enabled: bool = CRYPTO_AUTH_ENABLED

# Simulation results tracking
simulation_results: dict = {
    "start_time": None,
    "end_time": None,
    "duration_seconds": 0,
    "completed": False,
    "destination_reached": False,
    "steps": 0,
    "Jn_history": [],
    "rn_history": [],
    "timestamps": [],
    "avg_Jn": 0,
    "avg_rn": 0,
    "final_Jn": 0,
    "final_rn": 0,
    "agent_count": 0,
    "jammed_count_history": [],
    "avg_comm_quality_history": [],
    "avg_traveled_path": 0,  # Average path length traveled by all agents
    "per_agent_path_lengths": {},  # Path length per agent
}


@app.on_event("startup")
async def startup():
    """Initialize agents and obstacles on startup."""
    global agent_states, jamming_zones

    # Print configuration
    print_config()

    # Initialize agents
    agent_states = init_agents(NUM_AGENTS)

    # Initialize obstacles from config
    # Format: (x, y, z, radius, type) where type is "physical", "low_jam", or "high_jam"
    initial_obstacles = get_initial_obstacles()
    for i, obs in enumerate(initial_obstacles):
        zone_id = f"obstacle_{i+1}"
        # Parse obstacle type (5th element, defaults to "low_jam")
        type_str = obs[4] if len(obs) > 4 else "low_jam"
        try:
            obstacle_type = ObstacleType(type_str)
        except ValueError:
            print(f"[SIM] Unknown obstacle type '{type_str}', defaulting to low_jam")
            obstacle_type = ObstacleType.LOW_JAM
        
        jamming_zones[zone_id] = JammingZone(
            id=zone_id,
            center=[obs[0], obs[1], obs[2] if len(obs) > 2 else 0],
            radius=obs[3] if len(obs) > 3 else 5.0,
            intensity=1.0,
            active=True,
            obstacle_type=obstacle_type,
        )
        print(f"[SIM] Loaded {obstacle_type.value} obstacle {zone_id}: center=({obs[0]}, {obs[1]}, {obs[2] if len(obs) > 2 else 0}), radius={obs[3] if len(obs) > 3 else 5.0}")

    if initial_obstacles:
        print(f"[SIM] Loaded {len(initial_obstacles)} initial obstacles")

    # Initialize spoofing zones from config
    initial_spoof_zones = get_initial_spoofing_zones()
    for i, sz in enumerate(initial_spoof_zones):
        zone_id = f"zone_{i+1}"
        spoof_type_str = sz[4] if len(sz) > 4 else DEFAULT_SPOOF_TYPE
        try:
            spoof_type = SpoofType(spoof_type_str)
        except ValueError:
            print(f"[SIM] Unknown spoof type '{spoof_type_str}', defaulting to phantom")
            spoof_type = SpoofType.PHANTOM

        spoofing_zones[zone_id] = SpoofingZone(
            id=zone_id,
            center=[sz[0], sz[1], sz[2] if len(sz) > 2 else 0],
            radius=sz[3] if len(sz) > 3 else 10.0,
            active=True,
            spoof_type=spoof_type,
            phantom_count=PHANTOM_COUNT,
            falsification_magnitude=POSITION_FALSIFICATION_MAGNITUDE,
            coordinate_vector=list(COORDINATE_ATTACK_VECTOR),
        )
        print(f"[SIM] Loaded {spoof_type.value} spoofing zone {zone_id}: center=({sz[0]}, {sz[1]}, {sz[2] if len(sz) > 2 else 0}), radius={sz[3] if len(sz) > 3 else 10.0}")

    if initial_spoof_zones:
        print(f"[SIM] Loaded {len(initial_spoof_zones)} initial spoofing zones")

    # Initialize LLM assistance controller with config state
    llm_controller = get_llm_controller()
    llm_controller.set_enabled(llm_assistance_enabled)
    print(f"[SIM] LLM Assistance: {'enabled' if llm_assistance_enabled else 'disabled'}")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "simulation",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/status")
async def get_status():
    """Get simulation status."""
    return {
        "running": True,
        "agent_count": len(agent_states),
        "boundaries": {
            "x_range": X_RANGE,
            "y_range": Y_RANGE,
            "z_range": Z_RANGE,
            "mission_end": MISSION_END,
        },
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/agents")
async def get_agents():
    """Get all agent states with per-agent communication quality from controller."""
    controller = get_controller()
    agent_comm = controller.get_all_agent_comm_quality()

    result = {}
    for agent_id, state in agent_states.items():
        agent_data = state.to_dict()
        if agent_id in agent_comm:
            agent_data["communication_quality"] = agent_comm[agent_id]
        result[agent_id] = agent_data

    # Include phantom agents from MAVLink bus
    bus = get_mavlink_bus()
    bus_phantom_ids = bus.get_phantom_ids()
    for pid in bus_phantom_ids:
        if pid in bus._perceived_positions:
            pos = bus._perceived_positions[pid]
            result[pid] = {
                "agent_id": pid,
                "position": pos,
                "velocity": [0, 0, 0],
                "heading": 0,
                "heading_degrees": 0,
                "speed": 0,
                "jammed": False,
                "communication_quality": 1.0,
                "llm_target": None,
                "formation_role": None,
                "neighbors": [],
                "formation_error": 0,
                "distance_to_goal": 0,
                "eta": None,
                "path_points": 0,
                "is_phantom": True,
                "crypto_verified": False,
                "last_update": datetime.now().isoformat(),
            }

    # Fallback: if bus has no phantoms yet (pre-simulation), use SpoofingEngine
    if not bus_phantom_ids and spoofing_zones:
        engine = get_spoofing_engine()
        phantom_positions = engine.get_phantom_positions(list(spoofing_zones.values()))
        for pid, pos in phantom_positions.items():
            if pid not in result:
                result[pid] = {
                    "agent_id": pid,
                    "position": pos,
                    "velocity": [0, 0, 0],
                    "heading": 0,
                    "heading_degrees": 0,
                    "speed": 0,
                    "jammed": False,
                    "communication_quality": 1.0,
                    "llm_target": None,
                    "formation_role": None,
                    "neighbors": [],
                    "formation_error": 0,
                    "distance_to_goal": 0,
                    "eta": None,
                    "path_points": 0,
                    "is_phantom": True,
                    "crypto_verified": False,
                    "last_update": datetime.now().isoformat(),
                }

    return {
        "agents": result,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    """Get specific agent state."""
    if agent_id not in agent_states:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    return agent_states[agent_id].to_dict()


@app.put("/agents/{agent_id}")
async def update_agent(agent_id: str, state: dict[str, Any]):
    """Update agent state (for simulation updates)."""
    if agent_id not in agent_states:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    agent = agent_states[agent_id]

    if "position" in state:
        agent.position = state["position"]
    if "jammed" in state:
        agent.jammed = state["jammed"]
    if "communication_quality" in state:
        agent.communication_quality = state["communication_quality"]

    agent.last_update = datetime.now().isoformat()

    return agent.to_dict()


@app.post("/agents")
async def create_agent(data: dict[str, Any]):
    """
    Create a new agent at specified coordinates.
    
    Args:
        data: {
            "x": float,
            "y": float,
            "z": float (optional, defaults to 0)
        }
    
    Returns:
        New agent data
    """
    from .agents import AgentState
    
    x = data.get("x")
    y = data.get("y")
    z = data.get("z", 0.0)
    
    # Validate coordinates
    if x is None or y is None:
        raise HTTPException(status_code=400, detail="Missing 'x' or 'y' coordinates")
    
    # Validate coordinates are within bounds
    if not (X_RANGE[0] <= x <= X_RANGE[1]):
        raise HTTPException(
            status_code=400,
            detail=f"X coordinate {x} outside boundaries {X_RANGE}"
        )
    if not (Y_RANGE[0] <= y <= Y_RANGE[1]):
        raise HTTPException(
            status_code=400,
            detail=f"Y coordinate {y} outside boundaries {Y_RANGE}"
        )
    if not (Z_RANGE[0] <= z <= Z_RANGE[1]):
        raise HTTPException(
            status_code=400,
            detail=f"Z coordinate {z} outside boundaries {Z_RANGE}"
        )
    
    # Generate next agent ID
    existing_nums = []
    for aid in agent_states.keys():
        if aid.startswith("agent"):
            try:
                num = int(aid.replace("agent", ""))
                existing_nums.append(num)
            except ValueError:
                pass
    
    next_num = max(existing_nums, default=0) + 1
    agent_id = f"agent{next_num}"
    
    # Create new agent
    new_agent = AgentState(
        agent_id=agent_id,
        position=[float(x), float(y), float(z)],
        formation_role=None,
    )
    new_agent._prev_pos = [float(x), float(y), float(z)]
    
    agent_states[agent_id] = new_agent

    # Register crypto key so the agent can communicate when crypto is enabled
    if mavlink_enabled:
        crypto = get_crypto_auth()
        crypto.add_agent_key(agent_id)

    print(f"[API] Created new agent {agent_id} at ({x}, {y}, {z})")

    return {
        "success": True,
        "agent": new_agent.to_dict(),
        "message": f"Created {agent_id} at ({x:.1f}, {y:.1f}, {z:.1f})"
    }


@app.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str):
    """
    Remove an agent from the simulation.
    
    Args:
        agent_id: Agent identifier (e.g., "agent1")
    
    Returns:
        Success status
    """
    if agent_id not in agent_states:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    # Remove from agent_states
    del agent_states[agent_id]

    # Remove from llm_targets if present
    if agent_id in llm_targets:
        del llm_targets[agent_id]

    # Remove crypto key so it's not left in the bus
    if mavlink_enabled:
        crypto = get_crypto_auth()
        crypto.remove_agent_key(agent_id)

    # Clean up path planner data
    controller = get_controller()
    if controller and hasattr(controller, 'path_planner'):
        controller.path_planner.clear_path(agent_id)

    # Clean up LLM controller guidance
    llm_controller = get_llm_controller()
    if llm_controller and agent_id in llm_controller._active_guidance:
        del llm_controller._active_guidance[agent_id]
    
    print(f"[API] Deleted agent {agent_id}")
    
    return {
        "success": True,
        "message": f"Deleted {agent_id}"
    }


@app.post("/move_agent")
async def move_agent(command: dict[str, Any]):
    """
    LLM-commanded agent movement endpoint.
    
    Args:
        command: {
            "agent": "agent1",
            "x": 5.0,
            "y": 10.0,
            "z": 2.0 (optional)
        }
    
    Returns:
        Status with agent information
    """
    agent_id = command.get("agent")
    x = command.get("x")
    y = command.get("y")
    z = command.get("z", 0.0)

    # Validate inputs
    if not agent_id:
        raise HTTPException(status_code=400, detail="Missing 'agent' field")

    if x is None or y is None:
        raise HTTPException(status_code=400, detail="Missing 'x' or 'y' coordinates")

    if agent_id not in agent_states:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    # Validate coordinates are within bounds
    if not (X_RANGE[0] <= x <= X_RANGE[1]):
        raise HTTPException(
            status_code=400,
            detail=f"X coordinate {x} outside boundaries {X_RANGE}"
        )
    if not (Y_RANGE[0] <= y <= Y_RANGE[1]):
        raise HTTPException(
            status_code=400,
            detail=f"Y coordinate {y} outside boundaries {Y_RANGE}"
        )
    if not (Z_RANGE[0] <= z <= Z_RANGE[1]):
        raise HTTPException(
            status_code=400,
            detail=f"Z coordinate {z} outside boundaries {Z_RANGE}"
        )

    # Store LLM target
    target = [float(x), float(y), float(z)]
    agent_states[agent_id].llm_target = target
    llm_targets[agent_id] = target

    # Block auto-LLM assistance for this agent - user command takes priority
    llm_controller = get_llm_controller()
    if llm_controller:
        llm_controller.block_agent(agent_id)

    current_pos = agent_states[agent_id].position
    is_jammed = bool(agent_states[agent_id].jammed)
    comm_quality = float(agent_states[agent_id].communication_quality)

    print(f"[SIM API] LLM commanded {agent_id} to move to ({x}, {y}, {z})")
    print(f"[SIM API]   Current position: {current_pos}")
    print(f"[SIM API]   Jammed: {is_jammed}, Comm: {comm_quality}")

    return {
        "success": True,
        "message": f"LLM command accepted: {agent_id} will move to ({x}, {y}, {z})",
        "agent": agent_id,
        "target": target,
        "current_position": [float(p) for p in current_pos],
        "jammed": is_jammed,
        "communication_quality": comm_quality,
    }


@app.get("/llm_targets")
async def get_llm_targets():
    """Get all active LLM-commanded targets."""
    return {"targets": llm_targets}


@app.delete("/llm_targets/{agent_id}")
async def clear_llm_target(agent_id: str):
    """Clear LLM target for an agent."""
    if agent_id in llm_targets:
        del llm_targets[agent_id]

    if agent_id in agent_states:
        agent_states[agent_id].llm_target = None

    return {"success": True, "message": f"Cleared LLM target for {agent_id}"}


@app.post("/llm_targets/clear_all")
async def clear_all_llm_targets():
    """Clear all LLM targets."""
    llm_targets.clear()

    for agent in agent_states.values():
        agent.llm_target = None

    return {"success": True, "message": "Cleared all LLM targets"}


@app.post("/simulate_step")
async def simulate_step():
    """
    Advance simulation by one step.
    Moves agents towards their LLM targets.
    """
    from .agents import move_agent_towards_target

    moved_agents = []
    llm_controller = get_llm_controller()

    for agent_id, agent in agent_states.items():
        if agent.llm_target:
            new_pos = move_agent_towards_target(agent, agent.llm_target, max_step=1.0)
            agent.update_position(new_pos)

            # Check if reached target
            dist = sum((a - b) ** 2 for a, b in zip(new_pos, agent.llm_target)) ** 0.5
            reached = dist < 0.1
            if reached:
                agent.llm_target = None
                if agent_id in llm_targets:
                    del llm_targets[agent_id]
                # Unblock auto-LLM assistance
                if llm_controller:
                    llm_controller.unblock_agent(agent_id)

            moved_agents.append({
                "agent_id": agent_id,
                "position": new_pos,
                "reached_target": reached,
            })

    return {
        "success": True,
        "moved_agents": moved_agents,
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================================
# JAMMING ZONE ENDPOINTS
# ============================================================================

@app.get("/jamming_zones")
async def get_jamming_zones():
    """Get all jamming zones."""
    zones_list = [zone.to_dict() for zone in jamming_zones.values()]
    
    # #region agent log
    import json as _json
    _log_path = "/home/singsong/Downloads/cars_demo_13/sim/.cursor/debug.log"
    _log_entry = {
        "location": "api.py:get_jamming_zones",
        "message": "Jamming zones API returning data",
        "data": {
            "zones_count": len(zones_list),
            "zone_ids": [z.get("id") for z in zones_list],
            "zone_active_states": [z.get("active") for z in zones_list],
        },
        "timestamp": datetime.now().timestamp() * 1000,
        "sessionId": "debug-session",
        "hypothesisId": "C"
    }
    try:
        with open(_log_path, "a") as _f:
            _f.write(_json.dumps(_log_entry) + "\n")
    except Exception:
        pass
    # #endregion
    
    return {
        "zones": zones_list,
        "count": len(jamming_zones),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/jamming_zones/{zone_id}")
async def get_jamming_zone(zone_id: str):
    """Get specific jamming zone."""
    if zone_id not in jamming_zones:
        raise HTTPException(status_code=404, detail=f"Jamming zone {zone_id} not found")
    return jamming_zones[zone_id].to_dict()


@app.post("/jamming_zones")
async def create_jamming_zone(zone_data: dict[str, Any]):
    """
    Create a new jamming zone.
    
    Args:
        zone_data: {
            "center": [x, y, z],
            "radius": float,
            "intensity": float (optional, 0.0-1.0, default 1.0),
            "active": bool (optional, default True),
            "obstacle_type": str (optional, "physical"|"low_jam"|"high_jam", uses UI selection)
        }
    """
    global default_obstacle_type
    
    center = zone_data.get("center")
    radius = zone_data.get("radius")

    if not center or len(center) < 3:
        raise HTTPException(status_code=400, detail="Missing or invalid 'center' [x, y, z]")
    if radius is None or radius <= 0:
        raise HTTPException(status_code=400, detail="Missing or invalid 'radius'")

    zone_id = zone_data.get("id") or f"zone_{uuid.uuid4().hex[:8]}"
    
    # Parse obstacle type (uses UI-selected default if not specified)
    type_str = zone_data.get("obstacle_type", default_obstacle_type)
    if type_str == "none":
        return {"success": False, "message": "Obstacle type 'none' selected -- no zone created"}
    try:
        obstacle_type = ObstacleType(type_str)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid obstacle_type '{type_str}'. Must be 'physical', 'low_jam', or 'high_jam'")

    zone = JammingZone(
        id=zone_id,
        center=[float(c) for c in center],
        radius=float(radius),
        intensity=float(zone_data.get("intensity", 1.0)),
        active=zone_data.get("active", True),
        obstacle_type=obstacle_type,
    )

    jamming_zones[zone_id] = zone

    # Update agent jamming status
    _update_agent_jamming_status()

    print(f"[SIM API] Created {obstacle_type.value} zone {zone_id} at {center} with radius {radius}")

    return {
        "success": True,
        "zone": zone.to_dict(),
        "message": f"Created jamming zone {zone_id}",
    }


@app.put("/jamming_zones/{zone_id}")
async def update_jamming_zone(zone_id: str, zone_data: dict[str, Any]):
    """Update an existing jamming zone."""
    if zone_id not in jamming_zones:
        raise HTTPException(status_code=404, detail=f"Jamming zone {zone_id} not found")

    zone = jamming_zones[zone_id]

    if "center" in zone_data:
        zone.center = [float(c) for c in zone_data["center"]]
    if "radius" in zone_data:
        zone.radius = float(zone_data["radius"])
    if "intensity" in zone_data:
        zone.intensity = float(zone_data["intensity"])
    if "active" in zone_data:
        zone.active = bool(zone_data["active"])
    
    # Handle obstacle_type changes - update type and recalculate jamming parameters
    if "obstacle_type" in zone_data:
        type_str = zone_data["obstacle_type"]
        try:
            new_type = ObstacleType(type_str)
            zone.obstacle_type = new_type
            # Recalculate jamming parameters based on new type
            params = OBSTACLE_PARAMS.get(new_type, OBSTACLE_PARAMS[ObstacleType.LOW_JAM])
            zone.kappa_j = params["kappa_j"]
            zone.d_base = params["d_base"]
            zone.d_min = params["d_min"]
            print(f"[SIM API] Zone {zone_id} type changed to {new_type.value}")
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid obstacle_type '{type_str}'. Must be 'physical', 'low_jam', or 'high_jam'")

    # Update agent jamming status
    _update_agent_jamming_status()

    return {
        "success": True,
        "zone": zone.to_dict(),
        "message": f"Updated jamming zone {zone_id}",
    }


@app.delete("/jamming_zones/{zone_id}")
async def delete_jamming_zone(zone_id: str):
    """Delete a jamming zone."""
    if zone_id not in jamming_zones:
        raise HTTPException(status_code=404, detail=f"Jamming zone {zone_id} not found")

    del jamming_zones[zone_id]

    # Update agent jamming status
    _update_agent_jamming_status()

    print(f"[SIM API] Deleted jamming zone {zone_id}")

    return {"success": True, "message": f"Deleted jamming zone {zone_id}"}


@app.delete("/jamming_zones")
async def clear_all_jamming_zones():
    """Delete all jamming zones."""
    count = len(jamming_zones)
    jamming_zones.clear()

    # Update agent jamming status
    _update_agent_jamming_status()

    return {"success": True, "message": f"Deleted {count} jamming zones"}


def _update_agent_jamming_status():
    """Update jamming status for all agents based on current zones."""
    zones_list = list(jamming_zones.values())

    for agent in agent_states.values():
        max_jamming = 0.0
        for zone in zones_list:
            if zone.active:
                level = zone.get_jamming_level(agent.position)
                max_jamming = max(max_jamming, float(level))

        # Ensure Python native bool and float, not numpy types
        agent.jammed = bool(max_jamming > 0.1)
        agent.communication_quality = float(1.0 - max_jamming * 0.8)  # Reduce comm quality


# ============================================================================
# SPOOFING ZONE ENDPOINTS
# ============================================================================

@app.get("/spoofing_zones")
async def get_spoofing_zones():
    """Get all spoofing zones."""
    zones_list = [zone.to_dict() for zone in spoofing_zones.values()]
    # Include phantom agent positions for visualization
    engine = get_spoofing_engine()
    phantom_positions = engine.get_phantom_positions(list(spoofing_zones.values()))

    return {
        "zones": zones_list,
        "count": len(spoofing_zones),
        "phantom_agents": phantom_positions,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/spoofing_zones")
async def create_spoofing_zone(zone_data: dict[str, Any]):
    """
    Create a new spoofing zone.

    Args:
        zone_data: {
            "center": [x, y, z],
            "radius": float,
            "spoof_type": "phantom" | "position_falsification" | "coordinate",
            "phantom_count": int (optional),
            "falsification_magnitude": float (optional),
            "coordinate_vector": [x, y, z] (optional)
        }
    """
    center = zone_data.get("center")
    radius = zone_data.get("radius")

    if not center or len(center) < 3:
        raise HTTPException(status_code=400, detail="Missing or invalid 'center' [x, y, z]")
    if radius is None or radius <= 0:
        raise HTTPException(status_code=400, detail="Missing or invalid 'radius'")

    # Sequential naming: zone_1, zone_2, ...
    existing_nums = []
    for zid in spoofing_zones:
        if zid.startswith("zone_"):
            try:
                existing_nums.append(int(zid.split("_")[1]))
            except (ValueError, IndexError):
                pass
    next_num = max(existing_nums, default=0) + 1
    zone_id = zone_data.get("id") or f"zone_{next_num}"

    type_str = zone_data.get("spoof_type", default_spoof_type)
    try:
        spoof_type = SpoofType(type_str)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid spoof_type '{type_str}'. Must be 'phantom', 'position_falsification', or 'coordinate'"
        )

    zone = SpoofingZone(
        id=zone_id,
        center=[float(c) for c in center],
        radius=float(radius),
        active=zone_data.get("active", True),
        spoof_type=spoof_type,
        phantom_count=int(zone_data.get("phantom_count", 2)),
        falsification_magnitude=float(zone_data.get("falsification_magnitude", 8.0)),
        coordinate_vector=[float(v) for v in zone_data.get("coordinate_vector", [10.0, 10.0, 0.0])],
    )
    spoofing_zones[zone_id] = zone
    print(f"[SIM API] Created {spoof_type.value} spoofing zone {zone_id} at {center} r={radius}")

    return {
        "success": True,
        "zone": zone.to_dict(),
        "message": f"Created spoofing zone {zone_id}",
    }


@app.delete("/spoofing_zones/{zone_id}")
async def delete_spoofing_zone(zone_id: str):
    """Delete a spoofing zone."""
    if zone_id not in spoofing_zones:
        raise HTTPException(status_code=404, detail=f"Spoofing zone {zone_id} not found")
    del spoofing_zones[zone_id]
    print(f"[SIM API] Deleted spoofing zone {zone_id}")
    return {"success": True, "message": f"Deleted spoofing zone {zone_id}"}


@app.delete("/spoofing_zones")
async def clear_all_spoofing_zones():
    """Delete all spoofing zones."""
    count = len(spoofing_zones)
    spoofing_zones.clear()
    get_spoofing_engine().reset()
    return {"success": True, "message": f"Deleted {count} spoofing zones"}


# ============================================================================
# MAVLINK / CRYPTO AUTH ENDPOINTS
# ============================================================================

@app.get("/simulation/crypto_auth")
async def get_crypto_auth_state():
    """Get crypto auth status and stats."""
    crypto = get_crypto_auth()
    return {
        "enabled": crypto.enabled,
        "mavlink_enabled": mavlink_enabled,
        "status": crypto.get_status(),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/simulation/crypto_auth")
async def set_crypto_auth_state(data: dict[str, Any]):
    """
    Toggle cryptographic authentication and optionally set algorithm.

    Args:
        data: {
            "enabled": bool,
            "algorithm": "hmac_sha256" | "chacha20_poly1305" | "aes_256_ctr" (optional)
        }
    """
    global crypto_auth_enabled
    enabled = data.get("enabled", False)
    crypto_auth_enabled = enabled

    crypto = get_crypto_auth()
    crypto.enabled = enabled

    # Set algorithm if provided
    if "algorithm" in data:
        crypto.set_algorithm(data["algorithm"])
        print(f"[SIM API] Crypto algorithm set to: {crypto.algorithm.value}")

    # Generate keys for current agents if enabling for the first time
    if enabled and not crypto.has_key(next(iter(agent_states), "")):
        crypto.generate_keys(list(agent_states.keys()))

    algo_label = crypto.algorithm.value
    print(f"[SIM API] Crypto auth {'enabled' if enabled else 'disabled'} ({algo_label})")
    return {
        "success": True,
        "enabled": enabled,
        "algorithm": algo_label,
        "message": f"Crypto auth {'enabled' if enabled else 'disabled'} ({algo_label})",
    }


@app.get("/simulation/v2v_channel")
async def get_v2v_channel_state():
    """Get V2V channel model status and link state details."""
    from ..algo.v2v_channel import get_channel_model
    ctrl = get_controller()
    model = get_channel_model()
    link_states = model.get_link_states()

    links_summary = []
    for (i, j), ls in link_states.items():
        links_summary.append({
            "pair": [i, j],
            "link_type": ls.link_type.value,
            "path_loss_db": round(ls.path_loss_db, 1),
            "shadow_fading_db": round(ls.shadow_fading_db, 1),
            "snr_db": round(ls.snr_db, 1),
            "quality": round(ls.quality, 4),
        })

    return {
        "enabled": ctrl.use_v2v_channel,
        "link_count": len(links_summary),
        "links": links_summary,
        "params": {
            "tx_power": model.params.tx_power,
            "freq_ghz": model.params.freq_ghz,
            "n_los": model.params.n_los,
            "n_nlosv": model.params.n_nlosv,
            "n_nloso": model.params.n_nloso,
            "vehicle_loss_db": model.params.vehicle_loss_db,
            "shadow_fading": model.params.enable_shadow_fading,
            "small_scale_fading": model.params.enable_small_scale_fading,
        },
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/simulation/v2v_channel")
async def set_v2v_channel_state(data: dict[str, Any]):
    """Toggle V2V channel model and/or update parameters."""
    ctrl = get_controller()
    from ..algo.v2v_channel import get_channel_model
    model = get_channel_model()

    if "enabled" in data:
        ctrl.use_v2v_channel = bool(data["enabled"])

    if "params" in data:
        p = data["params"]
        for key, val in p.items():
            if hasattr(model.params, key):
                setattr(model.params, key, type(getattr(model.params, key))(val))

    return {
        "success": True,
        "enabled": ctrl.use_v2v_channel,
        "message": f"V2V channel model {'enabled' if ctrl.use_v2v_channel else 'disabled'}",
    }


@app.get("/protocol_stats")
async def get_protocol_stats():
    """Get MAVLink protocol statistics."""
    bus = get_mavlink_bus()
    crypto = get_crypto_auth()
    engine = get_spoofing_engine()

    return {
        "mavlink_enabled": mavlink_enabled,
        "crypto_auth_enabled": crypto.enabled,
        "crypto_algorithm": crypto.algorithm.value,
        "mavlink": bus.stats.to_dict(),
        "crypto": crypto.stats.to_dict(),
        "spoofing_zones_active": sum(1 for z in spoofing_zones.values() if z.active),
        "phantom_agents": list(bus.get_phantom_ids()),
        "falsification_offsets": bus.get_falsification_offsets(),
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================================
# SIMULATION CONTROL ENDPOINTS
# ============================================================================

@app.get("/simulation/config")
async def get_simulation_config():
    """Get available simulation configuration options."""
    return {
        "formations": FORMATION_TYPES,
        "path_algorithms": PATH_ALGORITHMS,
        "jamming_strategies": JAMMING_STRATEGIES,
        "current": {
            "running": simulation_running,
        },
    }


@app.post("/simulation/start")
async def start_simulation(config: dict[str, Any], background_tasks: BackgroundTasks):
    """
    Start autonomous simulation with configured algorithm.

    Args:
        config: {
            "formation": "v_formation",
            "path_algorithm": "astar",
            "default_obstacle_type": "low_jam",
            "destination": [10, 10, 0] (optional)
        }
    """
    global simulation_running, simulation_task, simulation_results, default_obstacle_type, crypto_auth_enabled

    if simulation_running:
        return {"success": False, "message": "Simulation already running"}

    # Configure controller
    controller = get_controller()

    # Initialize MAVLink + crypto auth
    if mavlink_enabled:
        bus = get_mavlink_bus()
        bus.reset()
        crypto = get_crypto_auth()
        if "crypto_auth" in config:
            crypto_auth_enabled = config["crypto_auth"]
            crypto.enabled = crypto_auth_enabled
        if "crypto_algorithm" in config:
            crypto.set_algorithm(config["crypto_algorithm"])
        if crypto.enabled:
            crypto.generate_keys(list(agent_states.keys()))
            print(f"[SIM API] Crypto auth enabled ({crypto.algorithm.value}), keys generated for {len(agent_states)} agents")
        get_spoofing_engine()  # ensure initialized

    if "formation" in config:
        controller.set_formation_type(config["formation"])
    if "path_algorithm" in config:
        controller.set_path_algorithm(config["path_algorithm"])
    if "default_obstacle_type" in config:
        default_obstacle_type = config["default_obstacle_type"]
        print(f"[SIM API] Default obstacle type set to: {default_obstacle_type}")

    destination = config.get("destination", list(MISSION_END))

    # Initialize results tracking
    simulation_results = {
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "duration_seconds": 0,
        "completed": False,
        "destination_reached": False,
        "steps": 0,
        "Jn_history": [],
        "rn_history": [],
        "timestamps": [],
        "avg_Jn": 0,
        "avg_rn": 0,
        "final_Jn": 0,
        "final_rn": 0,
        "agent_count": len(agent_states),
        "jammed_count_history": [],
        "avg_comm_quality_history": [],
        "avg_traveled_path": 0,
        "per_agent_path_lengths": {},
        "config": {
            "formation": controller.formation_type,
            "path_algorithm": controller.path_algorithm,
            "default_obstacle_type": default_obstacle_type,
            "destination": destination,
        },
    }

    simulation_running = True

    # Start simulation loop in background
    background_tasks.add_task(run_simulation_loop, destination)

    print(f"[SIM API] Simulation started with config: {config}")

    return {
        "success": True,
        "message": "Simulation started",
        "config": {
            "formation": controller.formation_type,
            "path_algorithm": controller.path_algorithm,
            "default_obstacle_type": default_obstacle_type,
            "destination": destination,
        },
    }


@app.post("/simulation/algorithm")
async def update_algorithm(config: dict[str, Any]):
    """
    Update formation / path algorithm / obstacle type at any time (before or during simulation).

    Args:
        config: {
            "formation": "v_formation",          (optional)
            "path_algorithm": "astar",            (optional)
            "default_obstacle_type": "low_jam"    (optional)
        }
    """
    global default_obstacle_type
    controller = get_controller()

    changed = {}
    if "formation" in config:
        controller.set_formation_type(config["formation"])
        changed["formation"] = config["formation"]
    if "path_algorithm" in config:
        controller.set_path_algorithm(config["path_algorithm"])
        changed["path_algorithm"] = config["path_algorithm"]
    if "default_obstacle_type" in config:
        default_obstacle_type = config["default_obstacle_type"]
        changed["default_obstacle_type"] = default_obstacle_type

    print(f"[SIM API] Algorithm updated mid-sim: {changed}")
    return {"success": True, "changed": changed}


@app.post("/simulation/stop")
async def stop_simulation():
    """Stop autonomous simulation."""
    global simulation_running

    simulation_running = False

    print("[SIM API] Simulation stopped")

    return {"success": True, "message": "Simulation stopped"}


@app.post("/simulation/reset")
async def reset_simulation():
    """Reset simulation to initial state, reloading zones from config."""
    global simulation_running, agent_states

    simulation_running = False

    # Reset agents
    agent_states = init_agents(NUM_AGENTS)
    llm_targets.clear()

    # Reset controller
    reset_controller()

    # Reset MAVLink, spoofing, crypto, channel model
    reset_mavlink_bus()
    reset_spoofing_engine()
    reset_crypto_auth()
    reset_channel_model()

    # Reload jamming zones from config
    jamming_zones.clear()
    initial_obstacles = get_initial_obstacles()
    for i, obs in enumerate(initial_obstacles):
        zone_id = f"obstacle_{i+1}"
        type_str = obs[4] if len(obs) > 4 else "low_jam"
        try:
            obstacle_type = ObstacleType(type_str)
        except ValueError:
            obstacle_type = ObstacleType.LOW_JAM
        jamming_zones[zone_id] = JammingZone(
            id=zone_id,
            center=[obs[0], obs[1], obs[2] if len(obs) > 2 else 0],
            radius=obs[3] if len(obs) > 3 else 5.0,
            intensity=1.0,
            active=True,
            obstacle_type=obstacle_type,
        )
    _update_agent_jamming_status()

    # Reload spoofing zones from config
    spoofing_zones.clear()
    initial_spoof_zones = get_initial_spoofing_zones()
    for i, sz in enumerate(initial_spoof_zones):
        zone_id = f"zone_{i+1}"
        spoof_type_str = sz[4] if len(sz) > 4 else DEFAULT_SPOOF_TYPE
        try:
            spoof_type = SpoofType(spoof_type_str)
        except ValueError:
            spoof_type = SpoofType.PHANTOM
        spoofing_zones[zone_id] = SpoofingZone(
            id=zone_id,
            center=[sz[0], sz[1], sz[2] if len(sz) > 2 else 0],
            radius=sz[3] if len(sz) > 3 else 10.0,
            active=True,
            spoof_type=spoof_type,
            phantom_count=PHANTOM_COUNT,
            falsification_magnitude=POSITION_FALSIFICATION_MAGNITUDE,
            coordinate_vector=list(COORDINATE_ATTACK_VECTOR),
        )

    print(f"[SIM API] Simulation reset (reloaded {len(jamming_zones)} jamming + {len(spoofing_zones)} spoofing zones)")

    return {"success": True, "message": "Simulation reset"}


@app.get("/simulation/state")
async def get_simulation_state():
    """Get current simulation state including formation metrics."""
    controller = get_controller()
    formation_state = controller.get_formation_state()
    llm_controller = get_llm_controller()

    return {
        "running": simulation_running,
        "llm_assistance_enabled": llm_assistance_enabled,
        "llm_assistance_status": llm_controller.get_status(),
        "formation": formation_state.to_dict(),
        "agents": {aid: agent.to_dict() for aid, agent in agent_states.items()},
        "jamming_zones": [z.to_dict() for z in jamming_zones.values()],
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================================
# LLM ASSISTANCE ENDPOINTS
# ============================================================================

@app.get("/simulation/llm_assistance")
async def get_llm_assistance_state():
    """Get LLM assistance state."""
    llm_controller = get_llm_controller()
    
    return {
        "enabled": llm_assistance_enabled,
        "status": llm_controller.get_status(),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/simulation/llm_assistance")
async def set_llm_assistance_state(data: dict[str, Any]):
    """
    Enable or disable LLM assistance.
    
    Args:
        data: {"enabled": bool}
    """
    global llm_assistance_enabled
    
    enabled = data.get("enabled", True)
    llm_assistance_enabled = enabled
    
    # Update the controller
    llm_controller = get_llm_controller()
    llm_controller.set_enabled(enabled)
    
    print(f"[SIM API] LLM assistance {'enabled' if enabled else 'disabled'}")
    
    return {
        "success": True,
        "enabled": llm_assistance_enabled,
        "message": f"LLM assistance {'enabled' if enabled else 'disabled'}",
    }


@app.get("/llm_activity")
async def get_llm_activity(limit: int = 10):
    """
    Get recent LLM guidance activity for display in chat panel.
    
    Args:
        limit: Maximum number of entries to return (default 10)
        
    Returns:
        List of recent LLM guidance events with reasoning
    """
    llm_controller = get_llm_controller()
    
    return {
        "enabled": llm_assistance_enabled,
        "activity": llm_controller.get_recent_activity(limit=limit),
        "active_guidance": llm_controller.get_active_guidance_for_visualization(),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/llm_context")
async def get_llm_context():
    """
    Get current LLM context data for display in context panel.
    
    Returns:
        Current context being monitored by LLM assistance
    """
    llm_controller = get_llm_controller()
    zones_list = list(jamming_zones.values())
    
    spoof_zones_list = list(spoofing_zones.values())

    context = llm_controller.get_current_context(
        agents=agent_states,
        jamming_zones=zones_list,
        spoofing_zones=spoof_zones_list,
    )
    
    context["llm_assistance_enabled"] = llm_assistance_enabled
    context["timestamp"] = datetime.now().isoformat()
    
    return context


@app.get("/visualization")
async def get_visualization_data(trail_length: str = "short"):
    """
    Get visualization data for communication links and planned waypoints.
    
    Args:
        trail_length: "short" (last 100 points) or "all" (full history)
    
    Returns:
        - communication_links: List of links between agents (aij >= PT)
        - waypoints: Dict of agent_id -> list of planned waypoints
        - formation: Current formation info
        - llm_guidance: List of active LLM guidance with direction vectors
        - traveled_paths: Dict of agent_id -> list of traveled positions
    """
    controller = get_controller()
    llm_controller = get_llm_controller()
    vis_data = controller.get_visualization_data()

    # Also include agent positions for the frontend
    vis_data["agent_positions"] = {
        aid: agent.position for aid, agent in agent_states.items()
    }
    
    # Add LLM guidance data for visualization
    vis_data["llm_guidance"] = llm_controller.get_active_guidance_for_visualization()
    
    # Add chat-commanded LLM targets for trajectory visualization
    vis_data["llm_targets"] = {
        aid: target for aid, target in llm_targets.items()
    }
    
    # Add traveled paths (trail history) for visualization
    short_trail = trail_length == "short"
    vis_data["traveled_paths"] = controller.get_all_traveled_paths(short=short_trail)
    
    # Add discovered obstacles for minimap visualization
    vis_data["discovered_obstacles"] = controller.get_discovered_obstacles()

    # MAVLink / spoofing visualization data
    bus = get_mavlink_bus()
    bus_phantoms = {pid: pos for pid, pos in bus.get_perceived_positions([]).items()
                    if pid.startswith("phantom_")}
    # When simulation is idle the MAVLink bus has no perceived positions yet.
    # Fall back to the spoofing engine's static phantom positions so they are
    # always visible in the 3D scene regardless of whether the sim is running.
    vis_data["phantom_agents"] = bus_phantoms or get_spoofing_engine().get_phantom_positions(
        list(spoofing_zones.values())
    )
    vis_data["falsification_offsets"] = bus.get_falsification_offsets()
    vis_data["crypto_auth_enabled"] = get_crypto_auth().enabled
    
    vis_data["timestamp"] = datetime.now().isoformat()

    # Log waypoints count for debugging
    waypoints = vis_data.get("waypoints", {})
    if waypoints:
        waypoint_summary = {k: len(v) for k, v in waypoints.items()}
        print(f"[VIS API] Returning waypoints: {waypoint_summary}")

    return vis_data


async def run_simulation_loop(destination: list[float]):
    """Background task for simulation loop."""
    global simulation_running, simulation_results

    import numpy as np
    
    controller = get_controller()
    llm_controller = get_llm_controller()
    step_count = 0

    # MAVLink pipeline references
    bus = get_mavlink_bus() if mavlink_enabled else None
    crypto = get_crypto_auth() if mavlink_enabled else None
    spoof_engine = get_spoofing_engine() if mavlink_enabled else None

    while simulation_running:
        step_count += 1
        # Refresh zones each tick so add/remove during simulation takes effect
        zones_list = list(jamming_zones.values())

        # ================================================================
        # MAVLINK PIPELINE: broadcast -> spoofing -> crypto -> perceive
        # ================================================================
        perceived_positions = None
        if bus is not None:
            bus.clear_queue()

            # 1. Broadcast: each real agent sends its position
            for aid, agent in agent_states.items():
                msg = bus.broadcast(aid, agent.position, agent.velocity, agent.heading)
                if crypto and crypto.enabled:
                    crypto.sign_message(msg)

            # 2. Spoofing: inject/modify messages through active zones
            spoof_list = [z for z in spoofing_zones.values() if z.active]
            if spoof_list and spoof_engine:
                agent_pos = {aid: list(a.position) for aid, a in agent_states.items()}
                spoofed_messages = spoof_engine.process(bus.get_messages(), spoof_list, agent_pos)
                bus.set_messages(spoofed_messages)

            # 3. Packet loss (tied to existing comm quality / jamming)
            comm_quals = {aid: float(a.communication_quality) for aid, a in agent_states.items()}
            bus.apply_packet_loss(comm_quals)

            # 4. Crypto auth: filter invalid signatures
            if crypto:
                filtered = crypto.filter_messages(bus.get_messages())
                bus.set_messages(filtered)

            # 5. Build perceived state
            bus.build_perceived_state(agent_states)
            perceived_positions = bus.get_perceived_positions(list(agent_states.keys()))

        # Compute commands from formation controller
        commands = controller.compute_commands(
            agents=agent_states,
            destination=tuple(destination),
            jamming_zones=zones_list,
            dt=0.1,
            perceived_positions=perceived_positions,
        )

        # LLM Assistance: Check if any agents need help
        if llm_assistance_enabled:
            # Check which agents have degraded communication
            agents_needing_help = llm_controller.check_agents_needing_assistance(
                agent_states
            )
            
            # Get discovered obstacles from controller (swarm knowledge sharing)
            # This gives the LLM knowledge of obstacles discovered by ANY agent
            discovered_obs = controller.get_discovered_obstacles() if hasattr(controller, 'get_discovered_obstacles') else []
            
            for agent_id in agents_needing_help:
                # Skip agents with user-commanded targets (human priority)
                # User chat commands should always override auto LLM assistance
                if agent_id in llm_targets:
                    continue
                
                agent = agent_states[agent_id]
                
                # Request LLM guidance (non-blocking)
                # Pass discovered obstacles so LLM knows exact obstacle locations
                llm_controller.request_guidance(
                    agent_id=agent_id,
                    agent_state=agent,
                    destination=tuple(destination),
                    jamming_zones=zones_list,
                    discovered_obstacles=discovered_obs,
                )
                
                # Check for active guidance
                guidance = llm_controller.get_guidance(agent_id)
                
                if guidance and agent_id in commands:
                    cmd = commands[agent_id]
                    if cmd.target_position:
                        # Get base control vector
                        current_pos = np.array(agent.position)
                        target_pos = np.array(cmd.target_position)
                        base_control = target_pos - current_pos
                        
                        # Apply LLM guidance with adaptive weighting based on comm quality
                        blended_control = llm_controller.apply_guidance(
                            agent_id, base_control, guidance, agent.communication_quality
                        )
                        
                        # Clamp control magnitude to prevent extreme movements
                        control_magnitude = np.linalg.norm(blended_control)
                        max_llm_control = 2.0  # Max units per step
                        if control_magnitude > max_llm_control:
                            blended_control = blended_control * (max_llm_control / control_magnitude)
                        
                        # Update target position with blended control
                        new_target = current_pos + blended_control
                        
                        # Clamp to world bounds
                        new_target = np.clip(new_target, 
                                           controller.bounds_min, 
                                           controller.bounds_max)
                        cmd.target_position = new_target.tolist()
                        
                        print(f"[LLM] Applied guidance to {agent_id}: {guidance.reasoning[:50]}...")

        # Apply commands - ensure Python native types
        for agent_id, cmd in commands.items():
            if agent_id in agent_states and cmd.target_position:
                agent = agent_states[agent_id]
                
                # USER COMMAND PRIORITY: If agent has user-commanded llm_target,
                # move toward that target instead of formation control target
                if agent_id in llm_targets:
                    from .agents import move_agent_towards_target
                    user_target = llm_targets[agent_id]
                    new_pos = move_agent_towards_target(agent, user_target, max_step=1.0)
                    agent.position = [float(p) for p in new_pos]
                    agent.velocity = [0.0, 0.0, 0.0]  # Reset velocity
                    agent.last_update = datetime.now().isoformat()
                    
                    # Check if reached user-commanded target
                    dist = sum((a - b) ** 2 for a, b in zip(new_pos, user_target)) ** 0.5
                    if dist < 1.0:  # Reached target
                        print(f"[SIM] {agent_id} reached user-commanded target {user_target}")
                        print(f"[SIM] {agent_id} resuming formation control and path planning")
                        agent.llm_target = None
                        del llm_targets[agent_id]
                        # Unblock auto-LLM assistance - agent resumes normal control
                        llm_controller.unblock_agent(agent_id)
                    continue
                
                # Normal formation control
                # Convert numpy arrays/values to Python native types
                agent.position = [float(p) for p in cmd.target_position]
                agent.velocity = [float(v) for v in (cmd.velocity or [0, 0, 0])]
                agent.heading = float(cmd.heading or 0.0)
                agent.last_update = datetime.now().isoformat()

        # Update jamming status
        _update_agent_jamming_status()

        # Track metrics for results
        jammed_count = sum(1 for a in agent_states.values() if a.jammed)
        avg_comm = sum(a.communication_quality for a in agent_states.values()) / max(1, len(agent_states))
        
        # Get Jn and rn from controller's history
        current_Jn = controller.Jn_history[-1] if controller.Jn_history else 0.0
        current_rn = controller.rn_history[-1] if controller.rn_history else 0.0
        
        simulation_results["steps"] = step_count
        simulation_results["Jn_history"].append(float(current_Jn))
        simulation_results["rn_history"].append(float(current_rn))
        simulation_results["timestamps"].append(datetime.now().isoformat())
        simulation_results["jammed_count_history"].append(jammed_count)
        simulation_results["avg_comm_quality_history"].append(float(avg_comm))

        # Check if reached destination
        if controller.formation_converged:
            positions = [agent.position for agent in agent_states.values()]
            center = np.mean(positions, axis=0)
            dist = np.linalg.norm(center - np.array(destination))

            if dist < 1.0:
                print("[SIM API] Simulation complete - destination reached!")
                simulation_results["destination_reached"] = True
                simulation_running = False
                break

        await asyncio.sleep(0.1)  # 100ms update rate
    
    # Finalize results when simulation ends
    _finalize_simulation_results()


def _finalize_simulation_results():
    """Compute final metrics when simulation ends."""
    import numpy as np
    
    global simulation_results
    
    simulation_results["end_time"] = datetime.now().isoformat()
    simulation_results["completed"] = True
    
    # Calculate duration
    if simulation_results["start_time"]:
        start = datetime.fromisoformat(simulation_results["start_time"])
        end = datetime.fromisoformat(simulation_results["end_time"])
        simulation_results["duration_seconds"] = (end - start).total_seconds()
    
    # Calculate averages
    if simulation_results["Jn_history"]:
        simulation_results["avg_Jn"] = sum(simulation_results["Jn_history"]) / len(simulation_results["Jn_history"])
        simulation_results["final_Jn"] = simulation_results["Jn_history"][-1]
    
    if simulation_results["rn_history"]:
        simulation_results["avg_rn"] = sum(simulation_results["rn_history"]) / len(simulation_results["rn_history"])
        simulation_results["final_rn"] = simulation_results["rn_history"][-1]
    
    # Calculate average traveled path from controller's path history
    controller = get_controller()
    if hasattr(controller, '_agent_paths'):
        path_lengths = {}
        total_path_length = 0.0
        num_agents = 0
        
        for agent_id, path in controller._agent_paths.items():
            if len(path) >= 2:
                # Calculate total path length for this agent
                agent_path_length = 0.0
                for i in range(1, len(path)):
                    p1 = np.array(path[i-1])
                    p2 = np.array(path[i])
                    agent_path_length += np.linalg.norm(p2 - p1)
                
                path_lengths[agent_id] = round(agent_path_length, 2)
                total_path_length += agent_path_length
                num_agents += 1
        
        if num_agents > 0:
            simulation_results["avg_traveled_path"] = round(total_path_length / num_agents, 2)
            simulation_results["per_agent_path_lengths"] = path_lengths
            print(f"[SIM API] Path lengths: avg={simulation_results['avg_traveled_path']:.2f}, per-agent={path_lengths}")
    
    print(f"[SIM API] Results finalized: {simulation_results['steps']} steps, {simulation_results['duration_seconds']:.1f}s")


@app.get("/simulation/results")
async def get_simulation_results():
    """Get simulation results including Jn/rn history."""
    return {
        **simulation_results,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/simulation/results/download")
async def download_simulation_results(format: str = "json"):
    """
    Download simulation results in specified format.
    
    Args:
        format: "json" or "csv"
    """
    if format == "csv":
        # Build CSV content
        lines = ["timestamp,step,Jn,rn,jammed_count,avg_comm_quality"]
        
        for i, ts in enumerate(simulation_results.get("timestamps", [])):
            Jn = simulation_results["Jn_history"][i] if i < len(simulation_results["Jn_history"]) else 0
            rn = simulation_results["rn_history"][i] if i < len(simulation_results["rn_history"]) else 0
            jammed = simulation_results["jammed_count_history"][i] if i < len(simulation_results["jammed_count_history"]) else 0
            comm = simulation_results["avg_comm_quality_history"][i] if i < len(simulation_results["avg_comm_quality_history"]) else 0
            
            lines.append(f"{ts},{i+1},{Jn:.6f},{rn:.6f},{jammed},{comm:.4f}")
        
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(
            content="\n".join(lines),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=simulation_results.csv"}
        )
    else:
        # JSON format
        return {
            **simulation_results,
            "download_timestamp": datetime.now().isoformat(),
        }


# For running standalone
if __name__ == "__main__":
    import uvicorn

    from ..config import SIM_API_PORT

    print("=" * 60)
    print("Starting Simulation API")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=SIM_API_PORT)
