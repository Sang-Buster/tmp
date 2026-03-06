"""
Centralized configuration for the vehicle simulation system.
All settings are configurable via environment variables.
"""
import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv

# Load .env file, fall back to .env.example if .env is missing
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    example_path = Path(__file__).parent.parent / ".env.example"
    if example_path.exists():
        load_dotenv(example_path)
        print("[Config] No .env found — using .env.example defaults. Copy it to .env and set OLLAMA_HOST for your setup.")

# =============================================================================
# OLLAMA / LLM CONFIGURATION
# =============================================================================

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b-instruct-q4_K_M")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

# Keep model loaded in GPU memory.
# -1 = never unload (recommended for HPC/tunneled setups)
# 0  = unload immediately after request
# Positive int = seconds to keep loaded
# String with unit = e.g. "30m", "1h"
_keep_alive_raw = os.getenv("LLM_KEEP_ALIVE", "-1")
try:
    LLM_KEEP_ALIVE: int | str = int(_keep_alive_raw)
except ValueError:
    LLM_KEEP_ALIVE = _keep_alive_raw  # e.g. "30m", "1h"

# LLM Assistance - auto-assists when agent comm quality drops below PT
# Set to "false" to disable by default
LLM_ASSISTANCE_ENABLED = os.getenv("LLM_ASSISTANCE_ENABLED", "true").lower() == "true"

# =============================================================================
# QDRANT CONFIGURATION (unified storage for telemetry and logs)
# =============================================================================

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
VECTOR_DIM = 384  # MiniLM embedding size

# =============================================================================
# SIMULATION BOUNDARIES
# =============================================================================

X_RANGE = (
    float(os.getenv("X_MIN", "-200")),
    float(os.getenv("X_MAX", "200")),
)
Y_RANGE = (
    float(os.getenv("Y_MIN", "-200")),
    float(os.getenv("Y_MAX", "200")),
)
Z_RANGE = (
    float(os.getenv("Z_MIN", "0")),
    float(os.getenv("Z_MAX", "200")),
)

# Mission endpoint / destination
MISSION_END = (
    float(os.getenv("MISSION_END_X", "35")),
    float(os.getenv("MISSION_END_Y", "150")),
    float(os.getenv("MISSION_END_Z", "30")),
)

# =============================================================================
# AGENT CONFIGURATION
# =============================================================================

NUM_AGENTS = int(os.getenv("NUM_AGENTS", "5"))

# Agent position mode: "random" or "manual"
AGENT_POSITION_MODE = os.getenv("AGENT_POSITION_MODE", "random")

# Manual agent positions (JSON array of [x, y, z] positions)
# Example: '[[-5, 14, 0], [-5, -19, 5], [0, 0, -5], [35, -4, 0], [68, 0, 5]]'
AGENT_POSITIONS_MANUAL = os.getenv("AGENT_POSITIONS_MANUAL", "")

# Movement parameters
MAX_MOVEMENT_PER_STEP = float(os.getenv("MAX_MOVEMENT_PER_STEP", "1.0"))
UPDATE_FREQ = float(os.getenv("UPDATE_FREQ", "10.0"))  # Hz

# Communication quality levels
HIGH_COMM_QUAL = 1.0
LOW_COMM_QUAL = 0.2

# =============================================================================
# OBSTACLE / JAMMING ZONE CONFIGURATION
# =============================================================================

NUM_OBSTACLES = int(os.getenv("NUM_OBSTACLES", "0"))

# Obstacle position mode: "random" or "manual"
OBSTACLE_POSITION_MODE = os.getenv("OBSTACLE_POSITION_MODE", "manual")

# Manual obstacle positions (JSON array of [x, y, z, radius] tuples)
# Example: '[[35, 75, 15, 10], [20, 50, 10, 15], [50, 100, 20, 12]]'
OBSTACLES_MANUAL = os.getenv("OBSTACLES_MANUAL", "")

# Random obstacle parameters
OBSTACLE_RADIUS_MIN = float(os.getenv("OBSTACLE_RADIUS_MIN", "3.0"))
OBSTACLE_RADIUS_MAX = float(os.getenv("OBSTACLE_RADIUS_MAX", "8.0"))

# =============================================================================
# FORMATION CONTROL PARAMETERS (Communication-Aware)
# =============================================================================

# Default formation type: "communication_aware", "v_formation", "line", etc.
DEFAULT_FORMATION = os.getenv("DEFAULT_FORMATION", "communication_aware")

# Communication quality parameters
ALPHA = float(os.getenv("ALPHA", "1e-5"))          # Antenna characteristic
DELTA = float(os.getenv("DELTA", "2.0"))           # Required data rate
R0 = float(os.getenv("R0", "5.0"))                 # Reference distance
V_PATH_LOSS = float(os.getenv("V_PATH_LOSS", "3.0"))  # Path loss exponent
PT = float(os.getenv("PT", "0.94"))                # Reception probability threshold

# Derived parameter
BETA = ALPHA * (2 ** DELTA - 1)

# Convergence parameters
CONVERGENCE_THRESHOLD = int(os.getenv("CONVERGENCE_THRESHOLD", "20"))

# =============================================================================
# PATH PLANNING CONFIGURATION
# =============================================================================

# Default algorithm: "direct", "astar", "theta_star", "dijkstra", etc.
DEFAULT_PATH_ALGORITHM = os.getenv("DEFAULT_PATH_ALGORITHM", "astar")

# Grid-based path planning parameters
VOXEL_SIZE = float(os.getenv("VOXEL_SIZE", "2.0"))
PATH_BOUNDS_MARGIN = float(os.getenv("PATH_BOUNDS_MARGIN", "50.0"))

# Behavior-based control parameters
ATTRACTION_MAGNITUDE = float(os.getenv("ATTRACTION_MAGNITUDE", "0.7"))
DISTANCE_THRESHOLD = float(os.getenv("DISTANCE_THRESHOLD", "1.0"))
AVOIDANCE_MAGNITUDE = float(os.getenv("AVOIDANCE_MAGNITUDE", "3.5"))
BUFFER_ZONE = float(os.getenv("BUFFER_ZONE", "8.0"))
WALL_FOLLOW_ZONE = float(os.getenv("WALL_FOLLOW_ZONE", "4.0"))

# =============================================================================
# OBSTACLE TYPE CONFIGURATION
# =============================================================================

# Default obstacle type for new obstacles: "physical", "low_jam", "high_jam"
DEFAULT_OBSTACLE_TYPE = os.getenv("DEFAULT_OBSTACLE_TYPE", "low_jam")

# Detection parameters
JAMMING_DETECTION_RADIUS = float(os.getenv("JAMMING_DETECTION_RADIUS", "15.0"))
JAMMING_SAFETY_MARGIN = float(os.getenv("JAMMING_SAFETY_MARGIN", "3.0"))

# =============================================================================
# MAVLINK PROTOCOL CONFIGURATION
# =============================================================================

MAVLINK_ENABLED = os.getenv("MAVLINK_ENABLED", "true").lower() == "true"
MAVLINK_PACKET_LOSS_BASE = float(os.getenv("MAVLINK_PACKET_LOSS_BASE", "0.02"))

# =============================================================================
# SPOOFING ATTACK CONFIGURATION
# =============================================================================

# Default spoofing type: "phantom", "position_falsification", "coordinate"
DEFAULT_SPOOF_TYPE = os.getenv("DEFAULT_SPOOF_TYPE", "phantom")

# Phantom attack: how many ghost agents per zone
PHANTOM_COUNT = int(os.getenv("PHANTOM_COUNT", "2"))

# Position falsification: magnitude of random offset (units)
POSITION_FALSIFICATION_MAGNITUDE = float(os.getenv("POSITION_FALSIFICATION_MAGNITUDE", "8.0"))

# Coordinate attack: systematic shift vector [x, y, z]
COORDINATE_ATTACK_VECTOR = json.loads(os.getenv("COORDINATE_ATTACK_VECTOR", "[10.0, 10.0, 0.0]"))

# Manual spoofing zones (JSON array of [x, y, z, radius, spoof_type])
SPOOFING_ZONES_MANUAL = os.getenv("SPOOFING_ZONES_MANUAL", "")

# =============================================================================
# CRYPTOGRAPHIC AUTHENTICATION CONFIGURATION
# =============================================================================

CRYPTO_AUTH_ENABLED = os.getenv("CRYPTO_AUTH_ENABLED", "false").lower() == "true"

# =============================================================================
# API CONFIGURATION
# =============================================================================

SIM_API_PORT = int(os.getenv("SIM_API_PORT", "5001"))
CHAT_API_PORT = int(os.getenv("CHAT_API_PORT", "5000"))
SIMULATION_API_URL = f"http://localhost:{SIM_API_PORT}"
CHAT_API_URL = f"http://localhost:{CHAT_API_PORT}"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_agent_ids(num_agents: int = NUM_AGENTS) -> list[str]:
    """Generate agent IDs."""
    return [f"agent{i+1}" for i in range(num_agents)]


def get_initial_agent_positions(num_agents: int = NUM_AGENTS) -> list[list[float]]:
    """
    Get initial agent positions based on configuration.
    
    Returns:
        List of [x, y, z] positions for each agent
    """
    # Try manual positions first
    if AGENT_POSITION_MODE == "manual" and AGENT_POSITIONS_MANUAL:
        try:
            positions = json.loads(AGENT_POSITIONS_MANUAL)
            if len(positions) >= num_agents:
                return [list(p) for p in positions[:num_agents]]
            else:
                print(f"[Config] Manual positions ({len(positions)}) < num_agents ({num_agents}), using random")
        except json.JSONDecodeError as e:
            print(f"[Config] Failed to parse AGENT_POSITIONS_MANUAL: {e}")

    # Generate random positions
    positions = []
    for _ in range(num_agents):
        pos = [
            random.uniform(X_RANGE[0], X_RANGE[1]),
            random.uniform(Y_RANGE[0], Y_RANGE[1]),
            random.uniform(Z_RANGE[0], min(5.0, Z_RANGE[1])),  # Start low
        ]
        positions.append(pos)

    return positions


def get_initial_obstacles() -> list[tuple]:
    """
    Get initial obstacles based on configuration.
    
    Supports two formats:
    - 4-parameter: [x, y, z, radius] - defaults to "low_jam" type
    - 5-parameter: [x, y, z, radius, type] - type is "physical", "low_jam", or "high_jam"
    
    Returns:
        List of (x, y, z, radius, type) tuples
    """
    # Try manual obstacles first
    if OBSTACLE_POSITION_MODE == "manual" and OBSTACLES_MANUAL:
        try:
            obstacles = json.loads(OBSTACLES_MANUAL)
            result = []
            for o in obstacles:
                if len(o) == 4:
                    # Backward compat: 4 params defaults to low_jam
                    result.append((o[0], o[1], o[2], o[3], "low_jam"))
                elif len(o) >= 5:
                    # 5 params: includes type
                    result.append((o[0], o[1], o[2], o[3], o[4]))
                else:
                    print(f"[Config] Invalid obstacle format: {o}")
            return result
        except json.JSONDecodeError as e:
            print(f"[Config] Failed to parse OBSTACLES_MANUAL: {e}")

    # Generate random obstacles if NUM_OBSTACLES > 0
    obstacles = []
    for _ in range(NUM_OBSTACLES):
        # Random position avoiding agent start area and destination
        x = random.uniform(X_RANGE[0] + 5, X_RANGE[1] - 5)
        y = random.uniform(Y_RANGE[0] + 5, Y_RANGE[1] - 5)
        z = random.uniform(Z_RANGE[0], Z_RANGE[1] / 2)
        radius = random.uniform(OBSTACLE_RADIUS_MIN, OBSTACLE_RADIUS_MAX)
        # Random obstacles default to low_jam
        obstacles.append((x, y, z, radius, "low_jam"))

    return obstacles


def get_initial_spoofing_zones() -> list[tuple]:
    """
    Get initial spoofing zones based on configuration.
    
    Supports two formats:
    - 4-parameter: [x, y, z, radius] - defaults to DEFAULT_SPOOF_TYPE
    - 5-parameter: [x, y, z, radius, spoof_type]
    
    Returns:
        List of (x, y, z, radius, spoof_type) tuples
    """
    if SPOOFING_ZONES_MANUAL:
        try:
            zones = json.loads(SPOOFING_ZONES_MANUAL)
            result = []
            for z in zones:
                if len(z) == 4:
                    result.append((z[0], z[1], z[2], z[3], DEFAULT_SPOOF_TYPE))
                elif len(z) >= 5:
                    result.append((z[0], z[1], z[2], z[3], z[4]))
                else:
                    print(f"[Config] Invalid spoofing zone format: {z}")
            return result
        except json.JSONDecodeError as e:
            print(f"[Config] Failed to parse SPOOFING_ZONES_MANUAL: {e}")
    return []


def get_formation_params() -> dict:
    """Get formation control parameters."""
    return {
        "alpha": ALPHA,
        "delta": DELTA,
        "r0": R0,
        "v": V_PATH_LOSS,
        "PT": PT,
        "beta": BETA,
        "convergence_threshold": CONVERGENCE_THRESHOLD,
    }


def get_behavior_params() -> dict:
    """Get behavior-based control parameters."""
    return {
        "attraction_magnitude": ATTRACTION_MAGNITUDE,
        "distance_threshold": DISTANCE_THRESHOLD,
        "avoidance_magnitude": AVOIDANCE_MAGNITUDE,
        "buffer_zone": BUFFER_ZONE,
        "wall_follow_zone": WALL_FOLLOW_ZONE,
    }


def get_path_planning_params() -> dict:
    """Get path planning parameters."""
    return {
        "voxel_size": VOXEL_SIZE,
        "bounds_margin": PATH_BOUNDS_MARGIN,
    }


def get_ollama_client():
    """Get configured Ollama client (sync, for startup checks only)."""
    import ollama
    return ollama.Client(host=OLLAMA_HOST)


def test_ollama_connection(verbose: bool = True) -> bool:
    """Test if Ollama is accessible (sync, for startup only)."""
    try:
        import httpx
        response = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=5.0)
        if response.status_code == 200:
            if verbose:
                models = response.json().get("models", [])
                model_names = [m.get("name", "unknown") for m in models]
                print(f"[LLM] Connected to {OLLAMA_HOST}")
                print(f"[LLM] Available models: {model_names}")
            return True
        return False
    except Exception as e:
        if verbose:
            print(f"[LLM] Connection failed: {e}")
        return False


async def async_chat_with_retry(model: str, messages: list, max_retries: int = LLM_MAX_RETRIES, timeout_secs: float | None = None) -> dict | None:
    """
    Async Ollama chat call via httpx - non-blocking, safe to call from FastAPI endpoints.
    Uses the Ollama REST API directly so the event loop is never blocked.

    timeout_secs overrides LLM_TIMEOUT for this call (useful for warmup with a longer budget).
    """
    import asyncio
    import httpx

    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": LLM_KEEP_ALIVE,
    }

    # Allow per-call timeout override (e.g. warmup needs longer budget than a chat request)
    read_timeout = float(timeout_secs) if timeout_secs is not None else float(LLM_TIMEOUT)
    timeout = httpx.Timeout(connect=5.0, read=read_timeout, write=10.0, pool=5.0)

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload)
                if response.status_code != 200:
                    print(f"[LLM] HTTP {response.status_code}: {response.text[:200]}")
                    raise httpx.HTTPStatusError("non-200", request=response.request, response=response)
                data = response.json()
                msg = data.get("message", {})
                if not msg or not msg.get("content"):
                    print(f"[LLM] Empty response from Ollama: {data}")
                    raise ValueError("Empty message content from Ollama")
                # Normalize to ollama SDK response shape: {"message": {"content": "..."}}
                return {"message": msg}
        except Exception as e:
            print(f"[LLM] Attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(min(2 ** attempt, 8))
    return None


def chat_with_retry(client, model: str, messages: list, max_retries: int = LLM_MAX_RETRIES):
    """
    Sync Ollama chat with retry (kept for backward compatibility with startup checks).
    Do NOT call this from async FastAPI endpoints - use async_chat_with_retry instead.
    """
    import time
    import httpx

    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"num_predict": 512},
        "keep_alive": LLM_KEEP_ALIVE,
    }

    for attempt in range(max_retries):
        try:
            response = httpx.post(url, json=payload, timeout=float(LLM_TIMEOUT))
            response.raise_for_status()
            data = response.json()
            return {"message": data.get("message", {"content": ""})}
        except Exception as e:
            print(f"[LLM] Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None


def print_config():
    """Print current configuration for debugging."""
    print("=" * 60)
    print("SIMULATION CONFIGURATION")
    print("=" * 60)
    print(f"Agents: {NUM_AGENTS} ({AGENT_POSITION_MODE} positions)")
    print(f"Obstacles: {NUM_OBSTACLES} ({OBSTACLE_POSITION_MODE} positions)")
    print(f"Boundaries: X={X_RANGE}, Y={Y_RANGE}, Z={Z_RANGE}")
    print(f"Destination: {MISSION_END}")
    print(f"Formation: {DEFAULT_FORMATION}")
    print(f"Path Algorithm: {DEFAULT_PATH_ALGORITHM}")
    print(f"Default Obstacle Type: {DEFAULT_OBSTACLE_TYPE}")
    print(f"Comm Params: alpha={ALPHA}, delta={DELTA}, r0={R0}, v={V_PATH_LOSS}, PT={PT}")
    print("=" * 60)
