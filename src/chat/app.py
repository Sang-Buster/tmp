"""
Chat API - FastAPI app for MCP chat interface.
"""
import asyncio
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..config import SIMULATION_API_URL, async_chat_with_retry, LLM_MODEL, test_ollama_connection
from ..rag import get_all_telemetry, get_logs
from .llm import answer_question
from .tools import move_agent

# Create FastAPI app
app = FastAPI(title="Vehicle Simulation Chat")

# Background task for LLM target processing
_llm_target_task = None

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates and static files
BASE_DIR = Path(__file__).parent.parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "static"))


async def llm_target_loop():
    """
    Background loop that moves agents toward their LLM targets when simulation is stopped.
    
    This enables the "move agent1 to 5, 5" chat commands to actually move agents
    even when the main simulation loop is not running.
    """
    while True:
        try:
            async with httpx.AsyncClient() as client:
                # Check if simulation is running
                state_response = await client.get(
                    f"{SIMULATION_API_URL}/simulation/state",
                    timeout=2.0
                )
                
                if state_response.status_code == 200:
                    state_data = state_response.json()
                    sim_running = state_data.get("running", False)
                    
                    if not sim_running:
                        # Simulation is stopped - check if any agents have LLM targets
                        agents_response = await client.get(
                            f"{SIMULATION_API_URL}/agents",
                            timeout=2.0
                        )
                        
                        if agents_response.status_code == 200:
                            agents_data = agents_response.json().get("agents", {})
                            has_targets = any(
                                agent.get("llm_target") is not None
                                for agent in agents_data.values()
                            )
                            
                            if has_targets:
                                # Process one simulation step to move agents
                                await client.post(
                                    f"{SIMULATION_API_URL}/simulate_step",
                                    timeout=2.0
                                )
        except Exception:
            # Silently ignore errors - simulation API might not be ready
            pass
        
        await asyncio.sleep(0.1)  # 100ms update rate


@app.on_event("startup")
async def startup():
    """Check services on startup."""
    global _llm_target_task
    
    print("[CHAT] Starting Chat API...")

    # Check LLM
    if test_ollama_connection(verbose=True):
        print("[CHAT] LLM connected")
    else:
        print("[CHAT] LLM not available - will retry on requests")
    
    # Start background task for LLM target processing
    _llm_target_task = asyncio.create_task(llm_target_loop())
    print("[CHAT] LLM target processing loop started")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve main dashboard HTML."""
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        return HTMLResponse(
            content=f"<html><body>Error loading dashboard: {e}</body></html>",
            status_code=500
        )


@app.get("/health")
async def health_check():
    """Check system health."""
    # Check simulation API
    sim_status = "offline"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{SIMULATION_API_URL}/", timeout=2.0)
            if response.status_code == 200:
                sim_status = "online"
    except Exception:
        pass

    # Check LLM
    llm_status = "ready" if test_ollama_connection(verbose=False) else "unavailable"

    return {
        "chat_api": "online",
        "simulation_api": sim_status,
        "llm": llm_status,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/chat")
async def chat(request: Request):
    """
    Main chat endpoint with MCP-style tool calling.

    The LLM has access to tools (move_agent, get_agent_status, add_spoofing_zone,
    toggle_crypto_auth, etc.) and can call them autonomously during the conversation.

    Simple move commands ("move agent1 to 5,5") are also handled via a fast regex
    path that skips the LLM for responsiveness.
    """
    try:
        data = await request.json()
        user_message = data.get("message", "").strip()

        if not user_message:
            return {"response": "Please enter a message."}

        print(f"\n[CHAT] Message: {user_message}")

        # Fast path: simple move commands bypass LLM for instant response
        if _is_move_command(user_message):
            result = await _handle_move_command(user_message)
            return {"response": result}

        # MCP tool-calling LLM agent handles everything else
        response = await answer_question(user_message)

        return {"response": response}

    except Exception as e:
        print(f"[CHAT] Error: {e}")
        import traceback
        traceback.print_exc()
        return {"response": f"Error: {str(e)}"}


def _is_move_command(message: str) -> bool:
    """Check if message is a move command."""
    msg = message.lower()
    # Check for move-related keywords
    move_keywords = ["move", "send", "relocate", "position", "go to", "navigate"]
    return any(kw in msg for kw in move_keywords) and ("agent" in msg or "vehicle" in msg)


async def _handle_move_command(message: str) -> str:
    """Parse and execute move command with LLM fallback for complex commands."""
    import re

    msg = message.lower()

    # Extract agent ID
    agent_match = re.search(r"(?:agent|vehicle)\s*(\d+)", msg)
    if not agent_match:
        return "Could not identify which agent to move. Please specify an agent (e.g., 'agent1')."

    agent_id = f"agent{agent_match.group(1)}"

    # Try simple coordinate parsing first (fast path)
    coord_match = re.search(r"to\s*\(?(-?\d+\.?\d*)[,\s]+(-?\d+\.?\d*)(?:[,\s]+(-?\d+\.?\d*))?\)?", msg)
    if coord_match:
        x = float(coord_match.group(1))
        y = float(coord_match.group(2))
        z = float(coord_match.group(3)) if coord_match.group(3) else 0.0
        result = await move_agent(agent_id, x, y, z)
        return result.get("message", "Move command sent.")

    # Complex command - use LLM to parse
    print(f"[CHAT] Using LLM to parse complex move command for {agent_id}")
    parsed = await _llm_parse_move_command(message, agent_id)
    
    if parsed:
        x, y, z = parsed["x"], parsed["y"], parsed["z"]
        result = await move_agent(agent_id, x, y, z)
        explanation = parsed.get("explanation", "")
        response = result.get("message", "Move command sent.")
        if explanation:
            response += f" ({explanation})"
        return response
    
    return "Could not understand the move target. Try 'move agent1 to 5, 5' or describe the destination."


async def _llm_parse_move_command(message: str, agent_id: str) -> dict | None:
    """
    Use LLM to parse complex move commands.
    
    Handles references like:
    - "previous location"
    - "starting position"
    - "near agent2"
    - "center of the map"
    - "away from jamming zone"
    """
    from ..rag import get_telemetry_history
    
    # Get agent history for context
    try:
        history = get_telemetry_history(agent_id, limit=20)
        trajectory = [(h.get("position", [0, 0, 0]), h.get("timestamp", "")) for h in history]
    except Exception:
        trajectory = []
    
    # Get current agent positions
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{SIMULATION_API_URL}/agents", timeout=5.0)
            agents_data = response.json().get("agents", {})
    except Exception:
        agents_data = {}
    
    # Build context for LLM
    agent_positions = {aid: a.get("position", [0, 0, 0]) for aid, a in agents_data.items()}
    current_pos = agent_positions.get(agent_id, [0, 0, 0])
    
    # Format trajectory
    traj_str = ""
    if trajectory:
        recent = trajectory[:5]
        traj_str = "Recent positions: " + " -> ".join([f"({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})" for p, _ in recent])
        if len(trajectory) > 0:
            oldest = trajectory[-1][0]
            traj_str += f"\nStarting position: ({oldest[0]:.1f}, {oldest[1]:.1f}, {oldest[2]:.1f})"
    
    # Format other agents
    other_agents = "\n".join([
        f"  {aid}: ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})"
        for aid, p in agent_positions.items() if aid != agent_id
    ])
    
    prompt = f"""Parse this vehicle movement command and extract the target coordinates.

COMMAND: "{message}"

CURRENT STATE:
- Target agent: {agent_id}
- Current position: ({current_pos[0]:.1f}, {current_pos[1]:.1f}, {current_pos[2]:.1f})
{traj_str}

OTHER AGENTS:
{other_agents if other_agents else "  (none)"}

MAP BOUNDS: X: -200 to 200, Y: -200 to 200, Z: 0 to 200
DESTINATION: (35, 150, 30)

TASK: Extract target coordinates. Handle references like:
- "previous location" = second most recent position in trajectory
- "starting position" = oldest position in trajectory
- "near agent2" = close to agent2's position
- "center" = (0, 0, 50)
- "origin" = (0, 0, 0)

Respond with ONLY valid JSON:
{{"x": 0.0, "y": 0.0, "z": 0.0, "explanation": "brief reason"}}

JSON:"""

    try:
        response = await async_chat_with_retry(
            LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        
        if response:
            content = response.get("message", {}).get("content", "")
            
            # Parse JSON from response
            import json
            text = content.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(line for line in lines if not line.startswith("```"))
            
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                return {
                    "x": float(data.get("x", 0)),
                    "y": float(data.get("y", 0)),
                    "z": float(data.get("z", 0)),
                    "explanation": data.get("explanation", ""),
                }
    except Exception as e:
        print(f"[CHAT] LLM parse error: {e}")
    
    return None


@app.get("/data/qdrant")
async def get_qdrant_data():
    """Get recent telemetry from Qdrant."""
    try:
        data = get_all_telemetry(limit=50)
        return {"data": data, "count": len(data)}
    except Exception as e:
        return {"data": [], "error": str(e)}


@app.get("/data/postgresql")
async def get_postgresql_data():
    """Get recent logs from PostgreSQL."""
    try:
        data = get_logs(limit=50)
        return {"data": data, "count": len(data)}
    except Exception as e:
        return {"data": [], "error": str(e)}


@app.get("/agents")
async def proxy_agents():
    """Proxy to simulation API for agents."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{SIMULATION_API_URL}/agents", timeout=5.0)
            return response.json()
    except Exception as e:
        return {"agents": {}, "error": str(e)}


@app.post("/agents")
async def proxy_create_agent(request: Request):
    """Create a new agent - proxy to simulation API."""
    try:
        data = await request.json()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{SIMULATION_API_URL}/agents",
                json=data,
                timeout=5.0
            )
            if response.status_code >= 400:
                return JSONResponse(
                    status_code=response.status_code,
                    content=response.json()
                )
            return response.json()
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)}
        )


@app.delete("/agents/{agent_id}")
async def proxy_delete_agent(agent_id: str):
    """Delete an agent - proxy to simulation API."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{SIMULATION_API_URL}/agents/{agent_id}",
                timeout=5.0
            )
            if response.status_code >= 400:
                return JSONResponse(
                    status_code=response.status_code,
                    content=response.json()
                )
            return response.json()
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)}
        )


@app.get("/visualization")
async def proxy_visualization(trail_length: str = "short"):
    """Get visualization data (communication links, waypoints, trails)."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{SIMULATION_API_URL}/visualization",
                params={"trail_length": trail_length},
                timeout=5.0
            )
            return response.json()
    except Exception as e:
        return {"communication_links": [], "waypoints": {}, "traveled_paths": {}, "error": str(e)}


# ============================================================================
# JAMMING ZONE PROXY ROUTES
# ============================================================================

@app.get("/jamming_zones")
async def proxy_jamming_zones():
    """Get all jamming zones from simulation API."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{SIMULATION_API_URL}/jamming_zones", timeout=5.0)
            return response.json()
    except Exception as e:
        return {"zones": [], "error": str(e)}


@app.post("/jamming_zones")
async def proxy_create_jamming_zone(request: Request):
    """Create a new jamming zone."""
    try:
        data = await request.json()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{SIMULATION_API_URL}/jamming_zones",
                json=data,
                timeout=5.0
            )
            return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.delete("/jamming_zones/{zone_id}")
async def proxy_delete_jamming_zone(zone_id: str):
    """Delete a jamming zone."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{SIMULATION_API_URL}/jamming_zones/{zone_id}",
                timeout=5.0
            )
            return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# SPOOFING ZONE PROXY ROUTES
# ============================================================================

@app.get("/spoofing_zones")
async def proxy_spoofing_zones():
    """Get all spoofing zones."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{SIMULATION_API_URL}/spoofing_zones", timeout=5.0)
            return response.json()
    except Exception as e:
        return {"zones": [], "count": 0, "error": str(e)}


@app.post("/spoofing_zones")
async def proxy_create_spoofing_zone(request: Request):
    """Create a spoofing zone."""
    try:
        data = await request.json()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{SIMULATION_API_URL}/spoofing_zones",
                json=data,
                timeout=5.0
            )
            return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.delete("/spoofing_zones/{zone_id}")
async def proxy_delete_spoofing_zone(zone_id: str):
    """Delete a spoofing zone."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{SIMULATION_API_URL}/spoofing_zones/{zone_id}",
                timeout=5.0
            )
            return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.delete("/spoofing_zones")
async def proxy_clear_spoofing_zones():
    """Clear all spoofing zones."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.delete(f"{SIMULATION_API_URL}/spoofing_zones", timeout=5.0)
            return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# MAVLINK / CRYPTO AUTH PROXY ROUTES
# ============================================================================

@app.get("/simulation/crypto_auth")
async def proxy_get_crypto_auth():
    """Get crypto auth status."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{SIMULATION_API_URL}/simulation/crypto_auth", timeout=5.0)
            return response.json()
    except Exception as e:
        return {"enabled": False, "error": str(e)}


@app.post("/simulation/crypto_auth")
async def proxy_set_crypto_auth(request: Request):
    """Toggle crypto auth."""
    try:
        data = await request.json()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{SIMULATION_API_URL}/simulation/crypto_auth",
                json=data,
                timeout=5.0
            )
            return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/protocol_stats")
async def proxy_protocol_stats():
    """Get MAVLink protocol statistics."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{SIMULATION_API_URL}/protocol_stats", timeout=5.0)
            return response.json()
    except Exception as e:
        return {"mavlink_enabled": False, "error": str(e)}


# ============================================================================
# SIMULATION CONTROL PROXY ROUTES
# ============================================================================

@app.get("/simulation/config")
async def proxy_simulation_config():
    """Get simulation configuration options."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{SIMULATION_API_URL}/simulation/config", timeout=5.0)
            return response.json()
    except Exception as e:
        return {"error": str(e)}


@app.post("/simulation/algorithm")
async def proxy_simulation_algorithm(request: Request):
    """Update formation / path algorithm / obstacle type mid-simulation."""
    try:
        data = await request.json()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{SIMULATION_API_URL}/simulation/algorithm",
                json=data,
                timeout=5.0
            )
            return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/simulation/start")
async def proxy_simulation_start(request: Request):
    """Start simulation."""
    try:
        data = await request.json()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{SIMULATION_API_URL}/simulation/start",
                json=data,
                timeout=5.0
            )
            return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/simulation/stop")
async def proxy_simulation_stop():
    """Stop simulation."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{SIMULATION_API_URL}/simulation/stop",
                timeout=5.0
            )
            return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/simulation/reset")
async def proxy_simulation_reset():
    """Reset simulation."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{SIMULATION_API_URL}/simulation/reset",
                timeout=5.0
            )
            return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/simulation/state")
async def proxy_simulation_state():
    """Get simulation state."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{SIMULATION_API_URL}/simulation/state", timeout=5.0)
            return response.json()
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# LLM ASSISTANCE PROXY ROUTES
# ============================================================================

@app.get("/simulation/llm_assistance")
async def proxy_get_llm_assistance():
    """Get LLM assistance state."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{SIMULATION_API_URL}/simulation/llm_assistance",
                timeout=5.0
            )
            return response.json()
    except Exception as e:
        return {"enabled": True, "error": str(e)}


@app.post("/simulation/llm_assistance")
async def proxy_set_llm_assistance(request: Request):
    """Set LLM assistance state."""
    try:
        data = await request.json()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{SIMULATION_API_URL}/simulation/llm_assistance",
                json=data,
                timeout=5.0
            )
            return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/llm_activity")
async def proxy_llm_activity(limit: int = 10):
    """Get recent LLM activity for chat panel."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{SIMULATION_API_URL}/llm_activity",
                params={"limit": limit},
                timeout=5.0
            )
            return response.json()
    except Exception as e:
        return {"activity": [], "error": str(e)}


@app.get("/llm_context")
async def proxy_llm_context():
    """Get LLM context data for context panel."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{SIMULATION_API_URL}/llm_context",
                timeout=5.0
            )
            return response.json()
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# SIMULATION RESULTS PROXY ROUTES
# ============================================================================

@app.get("/simulation/results")
async def proxy_simulation_results():
    """Get simulation results."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{SIMULATION_API_URL}/simulation/results",
                timeout=5.0
            )
            return response.json()
    except Exception as e:
        return {"error": str(e)}


@app.get("/simulation/results/download")
async def proxy_simulation_results_download(format: str = "json"):
    """Download simulation results."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{SIMULATION_API_URL}/simulation/results/download",
                params={"format": format},
                timeout=5.0
            )
            if format == "csv":
                from fastapi.responses import PlainTextResponse
                return PlainTextResponse(
                    content=response.text,
                    media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=simulation_results.csv"}
                )
            return response.json()
    except Exception as e:
        return {"error": str(e)}


# Mount static files after routes
@app.on_event("startup")
async def mount_static():
    """Mount static files."""
    static_dir = BASE_DIR / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# For running standalone
if __name__ == "__main__":
    import uvicorn

    from ..config import CHAT_API_PORT

    print("=" * 60)
    print("Starting Chat API")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=CHAT_API_PORT)
