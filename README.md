# 3D Multi-Vehicle Simulation

A comprehensive 3D multi-vehicle simulation system with LLM-powered chat control, vector database storage, and advanced multi-vehicle coordination algorithms.

## Features

### Core Simulation

- **3D Visualization**: Real-time Three.js visualization with labeled axes, interactive controls
- **Multi-Agent System**: Configurable number of vehicles with individual state tracking
- **Jamming Zones**: Create and manage spherical jamming zones that affect vehicle communication

### Algorithm Framework

- **Formation Control**: V-formation, line, circle, wedge, column, diamond patterns
- **Path Planning**: Potential field, A\*, Dijkstra, greedy, and direct algorithms
- **Jamming Response**: Avoidance, penetration, scatter, and fallback strategies

### LLM Integration

- **Natural Language Control**: Control vehicles via chat commands
- **Ollama Backend**: Configurable LLM model (default: llama3.2:3b)
- **RAG Support**: Unified semantic search across telemetry and logs

### Data Storage (Qdrant Only)

- **Telemetry Collection**: Agent positions, states, communication quality
- **Logs Collection**: Conversation history, commands, notifications
- **Simplified Architecture**: Single database for all vector storage and RAG retrieval

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Ollama (for LLM features)
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Setup

```bash
# Navigate to sim directory
cd sim

# Create and activate virtual environment with uv
uv venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
uv pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env as needed

# Start Qdrant vector database
docker compose up -d

# Start Ollama (if not running)
ollama serve  # In another terminal

# Start the simulation
python -m src.main
```

### Access Points

- **Dashboard**: http://localhost:5000
- **Simulation API**: http://localhost:5001
- **Qdrant UI**: http://localhost:6333/dashboard

## Project Structure

```
sim/
├── src/
│   ├── config.py              # Centralized configuration
│   ├── main.py                # Main entry point
│   ├── algo/                  # Algorithm framework
│   │   ├── base.py            # Base classes (JammingZone, VehicleCommand)
│   │   ├── controller.py      # Unified multi-vehicle controller
│   │   ├── formation.py       # Formation patterns
│   │   ├── path_planning.py   # Path planning algorithms
│   │   ├── path_planning_3d.py # 3D pathfinding with pathfinding3d library
│   │   ├── jamming_response.py # Jamming response strategies
│   │   └── utils_3d.py        # 3D utility functions
│   ├── simulation/
│   │   ├── api.py             # FastAPI simulation endpoints
│   │   └── agents.py          # Agent state management
│   ├── chat/
│   │   ├── app.py             # FastAPI chat/dashboard endpoints
│   │   ├── llm.py             # LLM agent
│   │   └── tools.py           # MCP tools
│   └── rag/
│       ├── qdrant.py          # Qdrant telemetry storage
│       └── postgresql.py      # PostgreSQL log storage
├── static/
│   ├── css/style.css
│   ├── index.html
│   └── js/
│       ├── app.js             # Main application logic
│       ├── scene3d.js         # Three.js 3D visualization
│       └── chat.js            # Chat interface
├── docker-compose.yml         # PostgreSQL + Qdrant
├── pyproject.toml             # Python dependencies
└── .env.example               # Environment template
```

## Configuration

All configuration is done via environment variables in `.env` file. Copy `.env.example` to `.env` and modify as needed.

### Key Configuration Options

#### Agent Configuration

```bash
NUM_AGENTS=5                    # Number of vehicles
AGENT_POSITION_MODE=random      # "random" or "manual"
# For manual positions (JSON array):
# AGENT_POSITIONS_MANUAL=[[-5, 14, 0], [-5, -19, 5], [0, 0, -5]]
```

#### Obstacle/Jamming Zone Configuration

```bash
NUM_OBSTACLES=0                 # Number of random obstacles
OBSTACLE_POSITION_MODE=manual   # "random" or "manual"
# For manual obstacles (JSON array of [x, y, z, radius]):
# OBSTACLES_MANUAL=[[35, 75, 15, 10], [20, 50, 10, 15]]
```

#### Algorithm Defaults

```bash
DEFAULT_FORMATION=communication_aware  # or v_formation, line, etc.
DEFAULT_PATH_ALGORITHM=astar           # or direct, dijkstra, theta_star, etc.
DEFAULT_OBSTACLE_TYPE=low_jam          # or physical, high_jam
```

#### Communication-Aware Formation Parameters

```bash
ALPHA=1e-5          # Antenna characteristic
DELTA=2.0           # Required data rate
R0=5.0              # Reference distance
V_PATH_LOSS=3.0     # Path loss exponent
PT=0.94             # Reception probability threshold
```

#### Behavior-Based Control Parameters

```bash
ATTRACTION_MAGNITUDE=0.7   # Destination attraction strength
AVOIDANCE_MAGNITUDE=3.5    # Obstacle avoidance strength
BUFFER_ZONE=8.0            # Obstacle buffer zone size
WALL_FOLLOW_ZONE=4.0       # Wall following zone size
```

### Full Environment Variables (`.env`)

```bash
# LLM Configuration
OLLAMA_HOST=http://localhost:11444
LLM_MODEL=llama3.2:3b-instruct-q4_K_M

# Database
DB_HOST=localhost
DB_PORT=5435

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Simulation
NUM_AGENTS=5
X_MIN=-10
X_MAX=10
Y_MIN=-10
Y_MAX=10
Z_MIN=0
Z_MAX=20
MISSION_END_X=10
MISSION_END_Y=10
MISSION_END_Z=0

# API Ports
SIM_API_PORT=5001
CHAT_API_PORT=5000
```

## API Endpoints

### Simulation API (port 5001)

| Endpoint              | Method | Description                 |
| --------------------- | ------ | --------------------------- |
| `/agents`             | GET    | Get all agent states        |
| `/agents/{id}`        | GET    | Get specific agent          |
| `/move_agent`         | POST   | LLM-commanded move          |
| `/jamming_zones`      | GET    | List jamming zones          |
| `/jamming_zones`      | POST   | Create jamming zone         |
| `/jamming_zones/{id}` | DELETE | Delete jamming zone         |
| `/simulation/start`   | POST   | Start autonomous simulation |
| `/simulation/stop`    | POST   | Stop simulation             |
| `/simulation/reset`   | POST   | Reset to initial state      |
| `/simulation/state`   | GET    | Get formation metrics       |

### Chat API (port 5000)

| Endpoint         | Method          | Description         |
| ---------------- | --------------- | ------------------- |
| `/`              | GET             | Dashboard HTML      |
| `/health`        | GET             | System health check |
| `/chat`          | POST            | Send chat message   |
| `/agents`        | GET             | Proxy to sim API    |
| `/jamming_zones` | GET/POST/DELETE | Proxy to sim API    |

## Usage

### Dashboard Controls

1. **3D Scene**

   - Left drag: Rotate view
   - Right drag: Pan
   - Scroll: Zoom
   - Click vehicle: Select for details

2. **Agent Panel**

   - Shows all agents with real-time status
   - Position, speed, heading, communication quality
   - Formation role and distance to goal

3. **Jamming Zones Panel**

   - List all active jamming zones
   - Add new zones with center coordinates and radius
   - Delete zones with the × button

4. **Algorithm Control**

   - Select formation type
   - Choose path planning algorithm
   - Set default obstacle type (physical, low_jam, high_jam)
   - Start/Stop/Reset simulation

5. **Chat Control**
   - Natural language queries: "Where is agent1?"
   - Direct commands: "Move agent1 to 5, 5"

### Chat Commands

```
# Move commands
move agent1 to 5, 5
move agent2 to (3, 7, 2)

# Status queries
where is agent1?
what is the status of all agents?
is any agent jammed?
```

## Algorithm Framework

### Formation Types

- `communication_aware` **(default)**: **Distributed** control algorithm with NO leader/follower hierarchy. Uses communication quality metrics (aij, gij, rho_ij) to maintain optimal inter-agent distances. All agents are equal.
- `v_formation`: Classic V-shape (like flying geese) - has leader/wingman/follower roles
- `line`: Side-by-side horizontal line
- `circle`: Circular arrangement
- `wedge`: Arrow/wedge shape
- `column`: Single file
- `diamond`: Diamond pattern

### Path Planning

- `direct` **(default)**: Direct destination with behavior-based obstacle avoidance - includes wall-following, exponential repulsion
- `astar`: A\* grid-based pathfinding (pathfinding3d) - optimal and efficient
- `theta_star`: Theta\* with line-of-sight optimization - smoother paths
- `dijkstra`: Dijkstra's shortest path - guaranteed optimal
- `bfs`: Breadth-First Search - unweighted shortest path
- `greedy`: Greedy Best-First Search - fast but not optimal
- `bi_astar`: Bidirectional A\* - searches from both ends
- `msp`: Minimum Spanning Tree - explores all reachable space

### Jamming Response

- `avoidance`: Route around jamming zones
- `penetration`: Speed through zones quickly
- `scatter`: Spread out formation
- `fallback`: Return to last safe position

### Communication-Aware Formation Control (Distributed)

The default formation algorithm is a **distributed control** system where all agents are equal (no leader/follower hierarchy). Each agent independently calculates control inputs based on communication quality with its neighbors:

**Parameters:**

- `alpha` (1e-5): Antenna characteristic parameter
- `delta` (2.0): Required application data rate
- `r0` (5.0): Reference distance
- `v` (3.0): Path loss exponent
- `PT` (0.94): Reception probability threshold

**Key Metrics:**

- `aij`: Communication quality in antenna far-field = exp(-α(2^δ-1)(rij/r0)^v)
- `gij`: Communication quality in antenna near-field = rij / √(rij² + r0²)
- `φij`: Combined quality = gij × aij
- `ρij`: Derivative of φij for gradient-based control
- `Jn`: Average communication performance indicator

**Algorithm:**

1. For each agent pair (i,j), compute communication quality metrics
2. Apply formation control: `control_i += ρij × eij` where `eij = (qi - qj) / √rij`
3. Formation converges when Jn stabilizes over 20 iterations
4. After convergence, destination control is added with jamming avoidance

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Adding New Formations

Edit `src/algo/formation.py`:

```python
def _my_formation(self, n: int) -> dict[int, np.ndarray]:
    """Custom formation pattern."""
    offsets = {}
    for i in range(n):
        # Calculate offset for each agent
        offsets[i] = np.array([x, y, z])
    return offsets
```

### Adding New Path Algorithms

Edit `src/algo/path_planning.py`:

```python
def _my_algorithm_path(self, start, goal, jamming_zones):
    """Custom path planning algorithm."""
    path = [start.copy()]
    # Compute path
    return path
```

## Troubleshooting

### Port Already in Use

```bash
# Check what's using the port
lsof -i :5001

# Stop existing Docker containers
docker compose down

# Restart
docker compose up -d
```

### LLM Not Responding

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Pull model if needed
ollama pull llama3.2:3b-instruct-q4_K_M
```

### Database Connection Failed

```bash
# Check Docker containers
docker ps

# View logs
docker compose logs postgres
docker compose logs qdrant
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser Dashboard                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │   3D Scene   │  │ Agent Panel  │  │   Chat Interface     │   │
│  │  (Three.js)  │  │              │  │                      │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  Chat API     │    │  Simulation   │    │    Ollama     │
│  (port 5000)  │◄──►│  API (5001)   │    │    (LLM)      │
└───────────────┘    └───────┬───────┘    └───────────────┘
        │                    │
        │            ┌───────┴───────┐
        │            │               │
        ▼            ▼               ▼
┌───────────────┐  ┌─────────┐  ┌─────────┐
│   PostgreSQL  │  │ Qdrant  │  │  Algo   │
│   (pgvector)  │  │         │  │Framework│
└───────────────┘  └─────────┘  └─────────┘
```

## License

MIT License
