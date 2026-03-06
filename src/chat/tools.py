"""
MCP-style tool definitions for the chat LLM.

Each tool has a schema (name, description, parameters) and an execute function.
The LLM selects tools from the registry, and the tool runner executes them
and feeds results back for multi-step reasoning.
"""
from typing import Any

import httpx

from ..config import SIMULATION_API_URL
from ..rag import add_log

# ============================================================================
# TOOL REGISTRY
# ============================================================================

TOOL_SCHEMAS = [
    {
        "name": "move_agent",
        "description": "Move a vehicle agent to specific 3D coordinates. Use when the user wants to relocate, move, send, or navigate an agent.",
        "parameters": {
            "agent": {"type": "string", "description": "Agent ID e.g. 'agent1'"},
            "x": {"type": "number", "description": "X coordinate"},
            "y": {"type": "number", "description": "Y coordinate"},
            "z": {"type": "number", "description": "Z coordinate (default 0)", "default": 0},
        },
        "required": ["agent", "x", "y"],
    },
    {
        "name": "get_agent_status",
        "description": "Get the current status of one or all agents including position, jamming state, communication quality, and formation info.",
        "parameters": {
            "agent": {"type": "string", "description": "Agent ID (omit for all agents)", "default": None},
        },
        "required": [],
    },
    {
        "name": "get_simulation_status",
        "description": "Get overall simulation status including whether it's running, formation state, and metrics.",
        "parameters": {},
        "required": [],
    },
    {
        "name": "add_agent",
        "description": "Create a new agent at the specified coordinates.",
        "parameters": {
            "x": {"type": "number", "description": "X coordinate"},
            "y": {"type": "number", "description": "Y coordinate"},
            "z": {"type": "number", "description": "Z coordinate (default 0)", "default": 0},
        },
        "required": ["x", "y"],
    },
    {
        "name": "remove_agent",
        "description": "Remove an agent from the simulation.",
        "parameters": {
            "agent": {"type": "string", "description": "Agent ID e.g. 'agent3'"},
        },
        "required": ["agent"],
    },
    {
        "name": "add_spoofing_zone",
        "description": "Create a spoofing attack zone. Types: 'phantom' (injects ghost agents), 'position_falsification' (corrupts positions), 'coordinate' (systematic shift).",
        "parameters": {
            "x": {"type": "number", "description": "Center X coordinate"},
            "y": {"type": "number", "description": "Center Y coordinate"},
            "z": {"type": "number", "description": "Center Z coordinate", "default": 10},
            "radius": {"type": "number", "description": "Zone radius", "default": 15},
            "spoof_type": {"type": "string", "description": "'phantom', 'position_falsification', or 'coordinate'", "default": "phantom"},
        },
        "required": ["x", "y"],
    },
    {
        "name": "toggle_crypto_auth",
        "description": "Enable or disable cryptographic authentication on MAVLink messages. When enabled, spoofing attacks are detected and rejected.",
        "parameters": {
            "enabled": {"type": "boolean", "description": "True to enable, False to disable"},
            "algorithm": {"type": "string", "description": "'hmac_sha256', 'chacha20_poly1305', or 'aes_256_ctr'", "default": "hmac_sha256"},
        },
        "required": ["enabled"],
    },
    {
        "name": "get_protocol_stats",
        "description": "Get MAVLink protocol statistics: messages sent/received/dropped, spoofing injection count, crypto rejections, and timing data.",
        "parameters": {},
        "required": [],
    },
    {
        "name": "delete_spoofing_zone",
        "description": "Remove a spoofing attack zone by ID. Use when the user wants to deactivate or clear a specific spoofing zone.",
        "parameters": {
            "zone_id": {"type": "string", "description": "Zone ID e.g. 'zone_1'"},
        },
        "required": ["zone_id"],
    },
    {
        "name": "add_jamming_zone",
        "description": "Create a jamming zone that degrades agent communication quality. Types: 'physical' (impenetrable obstacle), 'low_jam' (mild communication interference), 'high_jam' (severe jamming that nearly disables comms).",
        "parameters": {
            "x": {"type": "number", "description": "Center X coordinate"},
            "y": {"type": "number", "description": "Center Y coordinate"},
            "z": {"type": "number", "description": "Center Z coordinate", "default": 10},
            "radius": {"type": "number", "description": "Zone radius", "default": 15},
            "jam_type": {"type": "string", "description": "'physical', 'low_jam', or 'high_jam'", "default": "low_jam"},
        },
        "required": ["x", "y"],
    },
    {
        "name": "delete_jamming_zone",
        "description": "Remove a jamming/obstacle zone by ID.",
        "parameters": {
            "zone_id": {"type": "string", "description": "Zone ID e.g. 'obstacle_1'"},
        },
        "required": ["zone_id"],
    },
    {
        "name": "start_simulation",
        "description": "Start the autonomous simulation. Vehicles navigate toward the mission destination using the specified formation and path algorithm.",
        "parameters": {
            "formation": {"type": "string", "description": "Formation type: 'communication_aware', 'v_formation', 'line', 'circle', 'wedge', 'column', 'diamond'", "default": "communication_aware"},
            "path_algorithm": {"type": "string", "description": "Path algorithm: 'astar', 'direct', 'theta_star', 'dijkstra', 'bfs', 'greedy'", "default": "astar"},
        },
        "required": [],
    },
    {
        "name": "stop_simulation",
        "description": "Stop the running simulation. All vehicles freeze in their current positions.",
        "parameters": {},
        "required": [],
    },
    {
        "name": "reset_simulation",
        "description": "Reset simulation to initial state. Agents return to starting positions, spoofing zones are cleared, and MAVLink/crypto state is reset.",
        "parameters": {},
        "required": [],
    },
    {
        "name": "set_formation",
        "description": "Change the swarm formation type. Can be applied while simulation is running.",
        "parameters": {
            "formation": {"type": "string", "description": "Formation type: 'communication_aware', 'v_formation', 'line', 'circle', 'wedge', 'column', 'diamond'"},
        },
        "required": ["formation"],
    },
    {
        "name": "get_telemetry_history",
        "description": "Get recent position and state history for an agent from the telemetry database. Useful for tracking trajectory, checking when an agent was jammed, or analyzing movement patterns.",
        "parameters": {
            "agent_id": {"type": "string", "description": "Agent ID e.g. 'agent1'"},
            "limit": {"type": "integer", "description": "Number of history entries to return (default 10)", "default": 10},
        },
        "required": ["agent_id"],
    },
]


def get_tool_schemas_text() -> str:
    """Format tool schemas for inclusion in LLM prompt."""
    lines = []
    for tool in TOOL_SCHEMAS:
        params_desc = []
        for pname, pinfo in tool.get("parameters", {}).items():
            req = "(required)" if pname in tool.get("required", []) else "(optional)"
            params_desc.append(f"    - {pname}: {pinfo['description']} {req}")
        params_str = "\n".join(params_desc) if params_desc else "    (none)"
        lines.append(f"  {tool['name']}: {tool['description']}\n  Parameters:\n{params_str}")
    return "\n\n".join(lines)


# ============================================================================
# TOOL EXECUTION
# ============================================================================

async def execute_tool(name: str, args: dict) -> dict[str, Any]:
    """Execute a tool by name with the given arguments."""
    executor = TOOL_EXECUTORS.get(name)
    if not executor:
        return {"success": False, "error": f"Unknown tool: {name}"}
    try:
        return await executor(**args)
    except Exception as e:
        return {"success": False, "error": str(e)}


async def move_agent(agent: str, x: float, y: float, z: float = 0.0) -> dict[str, Any]:
    """Move an agent to specific coordinates."""
    print(f"[TOOL] move_agent({agent}, {x}, {y}, {z})")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{SIMULATION_API_URL}/move_agent",
                json={"agent": agent, "x": x, "y": y, "z": z},
                timeout=5.0,
            )

            if response.status_code == 200:
                result = response.json()
                add_log(
                    f"Moving agent {agent} to ({x}, {y}, {z})",
                    metadata={"agent_id": agent, "target": [x, y, z], "jammed": result.get("jammed", False)},
                    source="mcp",
                    message_type="command",
                )
                return {
                    "success": True,
                    "message": f"Moving {agent} to ({x}, {y}, {z})" + (
                        f" (agent is jammed, comm={result.get('communication_quality', 0):.1f})"
                        if result.get("jammed") else ""
                    ),
                    "current_position": result.get("current_position"),
                    "jammed": result.get("jammed", False),
                }
            else:
                return {"success": False, "error": response.text}
        except Exception as e:
            return {"success": False, "error": str(e)}


async def get_agent_status(agent: str = None) -> dict[str, Any]:
    """Get status of one or all agents."""
    async with httpx.AsyncClient() as client:
        try:
            if agent:
                response = await client.get(f"{SIMULATION_API_URL}/agents/{agent}", timeout=5.0)
            else:
                response = await client.get(f"{SIMULATION_API_URL}/agents", timeout=5.0)
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": response.text}
        except Exception as e:
            return {"success": False, "error": str(e)}


async def get_simulation_status() -> dict[str, Any]:
    """Get overall simulation status."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{SIMULATION_API_URL}/simulation/state", timeout=5.0)
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            return {"success": False, "error": response.text}
        except Exception as e:
            return {"success": False, "error": str(e)}


async def add_agent(x: float, y: float, z: float = 0.0) -> dict[str, Any]:
    """Create a new agent."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{SIMULATION_API_URL}/agents",
                json={"x": x, "y": y, "z": z},
                timeout=5.0,
            )
            if response.status_code == 200:
                result = response.json()
                return {"success": True, "message": result.get("message", "Agent created"), "agent": result.get("agent")}
            return {"success": False, "error": response.text}
        except Exception as e:
            return {"success": False, "error": str(e)}


async def remove_agent(agent: str) -> dict[str, Any]:
    """Remove an agent."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.delete(f"{SIMULATION_API_URL}/agents/{agent}", timeout=5.0)
            if response.status_code == 200:
                return {"success": True, "message": f"Removed {agent}"}
            return {"success": False, "error": response.text}
        except Exception as e:
            return {"success": False, "error": str(e)}


async def add_spoofing_zone(
    x: float, y: float, z: float = 10.0, radius: float = 15.0, spoof_type: str = "phantom"
) -> dict[str, Any]:
    """Create a spoofing zone."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{SIMULATION_API_URL}/spoofing_zones",
                json={"center": [x, y, z], "radius": radius, "spoof_type": spoof_type, "active": True},
                timeout=5.0,
            )
            if response.status_code == 200:
                result = response.json()
                return {"success": True, "message": f"Created {spoof_type} spoofing zone at ({x},{y},{z}) r={radius}"}
            return {"success": False, "error": response.text}
        except Exception as e:
            return {"success": False, "error": str(e)}


async def toggle_crypto_auth(enabled: bool, algorithm: str = "hmac_sha256") -> dict[str, Any]:
    """Toggle crypto auth."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{SIMULATION_API_URL}/simulation/crypto_auth",
                json={"enabled": enabled, "algorithm": algorithm},
                timeout=5.0,
            )
            if response.status_code == 200:
                result = response.json()
                return {"success": True, "message": result.get("message", "Crypto toggled")}
            return {"success": False, "error": response.text}
        except Exception as e:
            return {"success": False, "error": str(e)}


async def get_protocol_stats() -> dict[str, Any]:
    """Get MAVLink protocol stats."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{SIMULATION_API_URL}/protocol_stats", timeout=5.0)
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            return {"success": False, "error": response.text}
        except Exception as e:
            return {"success": False, "error": str(e)}


async def delete_spoofing_zone(zone_id: str) -> dict[str, Any]:
    """Remove a spoofing zone by ID."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.delete(
                f"{SIMULATION_API_URL}/spoofing_zones/{zone_id}", timeout=5.0
            )
            if response.status_code == 200:
                return {"success": True, "message": f"Deleted spoofing zone {zone_id}"}
            return {"success": False, "error": response.text}
        except Exception as e:
            return {"success": False, "error": str(e)}


async def add_jamming_zone(
    x: float, y: float, z: float = 10.0, radius: float = 15.0, jam_type: str = "low_jam"
) -> dict[str, Any]:
    """Create a jamming zone."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{SIMULATION_API_URL}/jamming_zones",
                json={"center": [x, y, z], "radius": radius, "obstacle_type": jam_type},
                timeout=5.0,
            )
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "message": f"Created {jam_type} jamming zone at ({x}, {y}, {z}) r={radius}",
                    "zone_id": result.get("zone", {}).get("id"),
                }
            return {"success": False, "error": response.text}
        except Exception as e:
            return {"success": False, "error": str(e)}


async def delete_jamming_zone(zone_id: str) -> dict[str, Any]:
    """Remove a jamming zone by ID."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.delete(
                f"{SIMULATION_API_URL}/jamming_zones/{zone_id}", timeout=5.0
            )
            if response.status_code == 200:
                return {"success": True, "message": f"Deleted jamming zone {zone_id}"}
            return {"success": False, "error": response.text}
        except Exception as e:
            return {"success": False, "error": str(e)}


async def start_simulation(
    formation: str = "communication_aware", path_algorithm: str = "astar"
) -> dict[str, Any]:
    """Start the autonomous simulation."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{SIMULATION_API_URL}/simulation/start",
                json={"formation": formation, "path_algorithm": path_algorithm},
                timeout=10.0,
            )
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "message": result.get("message", "Simulation started"),
                    "config": result.get("config"),
                }
            return {"success": False, "error": response.text}
        except Exception as e:
            return {"success": False, "error": str(e)}


async def stop_simulation() -> dict[str, Any]:
    """Stop the running simulation."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{SIMULATION_API_URL}/simulation/stop", timeout=5.0
            )
            if response.status_code == 200:
                return {"success": True, "message": "Simulation stopped"}
            return {"success": False, "error": response.text}
        except Exception as e:
            return {"success": False, "error": str(e)}


async def reset_simulation() -> dict[str, Any]:
    """Reset simulation to initial state."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{SIMULATION_API_URL}/simulation/reset", timeout=5.0
            )
            if response.status_code == 200:
                return {"success": True, "message": "Simulation reset to initial state"}
            return {"success": False, "error": response.text}
        except Exception as e:
            return {"success": False, "error": str(e)}


async def set_formation(formation: str) -> dict[str, Any]:
    """Change swarm formation type."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{SIMULATION_API_URL}/simulation/algorithm",
                json={"formation": formation},
                timeout=5.0,
            )
            if response.status_code == 200:
                return {"success": True, "message": f"Formation changed to {formation}"}
            return {"success": False, "error": response.text}
        except Exception as e:
            return {"success": False, "error": str(e)}


async def get_telemetry_history(agent_id: str, limit: int = 10) -> dict[str, Any]:
    """Get recent telemetry history for an agent from the database."""
    try:
        from ..rag import get_telemetry_history as _get_history
        history = _get_history(agent_id, limit=limit)
        return {
            "success": True,
            "agent_id": agent_id,
            "count": len(history),
            "history": history,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


TOOL_EXECUTORS = {
    "move_agent": move_agent,
    "get_agent_status": get_agent_status,
    "get_simulation_status": get_simulation_status,
    "add_agent": add_agent,
    "remove_agent": remove_agent,
    "add_spoofing_zone": add_spoofing_zone,
    "delete_spoofing_zone": delete_spoofing_zone,
    "add_jamming_zone": add_jamming_zone,
    "delete_jamming_zone": delete_jamming_zone,
    "toggle_crypto_auth": toggle_crypto_auth,
    "get_protocol_stats": get_protocol_stats,
    "start_simulation": start_simulation,
    "stop_simulation": stop_simulation,
    "reset_simulation": reset_simulation,
    "set_formation": set_formation,
    "get_telemetry_history": get_telemetry_history,
}
