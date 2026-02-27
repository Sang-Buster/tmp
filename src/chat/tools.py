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


TOOL_EXECUTORS = {
    "move_agent": move_agent,
    "get_agent_status": get_agent_status,
    "get_simulation_status": get_simulation_status,
    "add_agent": add_agent,
    "remove_agent": remove_agent,
    "add_spoofing_zone": add_spoofing_zone,
    "toggle_crypto_auth": toggle_crypto_auth,
    "get_protocol_stats": get_protocol_stats,
}
