"""
3D Path Planning Algorithms Module

Supports multiple path planning algorithms in 3D space:
- A* (AStar)
- Dijkstra
- Breadth-First Search (BFS)
- Greedy Best-First Search
- Theta* (direct line-of-sight optimization)
- Jump Point Search (JPS)
"""

import numpy as np
from pathfinding3d.core.diagonal_movement import DiagonalMovement
from pathfinding3d.core.grid import Grid
from pathfinding3d.finder.a_star import AStarFinder
from pathfinding3d.finder.best_first import BestFirst
from pathfinding3d.finder.bi_a_star import BiAStarFinder
from pathfinding3d.finder.breadth_first import BreadthFirstFinder
from pathfinding3d.finder.dijkstra import DijkstraFinder
from pathfinding3d.finder.msp import MinimumSpanningTree
from pathfinding3d.finder.theta_star import ThetaStarFinder


class PathPlanner3D:
    """
    3D path planning system that converts continuous 3D space to voxel grid
    and finds paths using various algorithms.
    """

    # Available algorithms
    ALGORITHMS = {
        "astar": ("A* (A-Star)", AStarFinder),
        "theta_star": ("Theta* (Path Smoothing)", ThetaStarFinder),
        "dijkstra": ("Dijkstra", DijkstraFinder),
        "bfs": ("Breadth-First Search", BreadthFirstFinder),
        "greedy": ("Greedy Best-First", BestFirst),
        "bi_astar": ("Bidirectional A*", BiAStarFinder),
        "msp": ("Minimum Spanning Tree", MinimumSpanningTree),
    }

    def __init__(
        self,
        bounds_min,
        bounds_max,
        voxel_size=2.0,
        algorithm="astar",
        diagonal_movement=True,
    ):
        """
        Initialize 3D path planner.

        Args:
            bounds_min: Minimum bounds [x_min, y_min, z_min]
            bounds_max: Maximum bounds [x_max, y_max, z_max]
            voxel_size: Size of each voxel/grid cell
            algorithm: Path planning algorithm to use
            diagonal_movement: Allow diagonal movement in grid
        """
        self.bounds_min = np.array(bounds_min)
        self.bounds_max = np.array(bounds_max)
        self.voxel_size = voxel_size
        self.algorithm = algorithm

        # Calculate grid dimensions
        grid_size = np.ceil((self.bounds_max - self.bounds_min) / voxel_size).astype(
            int
        )
        self.grid_size = grid_size

        # Diagonal movement configuration
        if diagonal_movement:
            self.diagonal_movement = DiagonalMovement.always
        else:
            self.diagonal_movement = DiagonalMovement.never

        # Initialize empty grid (will be updated with obstacles)
        self.grid_matrix = np.ones(grid_size, dtype=int)
        self.grid = None
        self.obstacles = []

        print("PathPlanner3D initialized:")
        print(f"  Bounds: {bounds_min} to {bounds_max}")
        print(f"  Voxel size: {voxel_size}")
        print(f"  Grid dimensions: {grid_size}")
        print(f"  Algorithm: {self.ALGORITHMS[algorithm][0]}")

    def world_to_grid(self, position):
        """Convert world coordinates to grid indices."""
        grid_pos = ((position - self.bounds_min) / self.voxel_size).astype(int)
        # Clamp to grid bounds
        grid_pos = np.clip(grid_pos, 0, self.grid_size - 1)
        return tuple(grid_pos)

    def grid_to_world(self, grid_pos):
        """Convert grid indices to world coordinates (center of voxel)."""
        world_pos = self.bounds_min + (np.array(grid_pos) + 0.5) * self.voxel_size
        return world_pos

    def update_obstacles(self, obstacles):
        """
        Update grid with obstacles (spheres in 3D space) with type-based costs.

        Args:
            obstacles: List of tuples, either:
                - (x, y, z, radius) - legacy format, treated as blocked
                - (x, y, z, radius, type) - new format with type-based costs
                
        Obstacle types and their grid costs:
            - "physical": 0 (blocked, infinite cost)
            - "high_jam": 1 (very expensive, strongly avoid)
            - "low_jam": 5 (moderate cost, soft preference to avoid)
            - Free space: 10 (normal cost)
        """
        self.obstacles = obstacles

        # Reset grid to all walkable with base cost of 10
        self.grid_matrix = np.full(self.grid_size, 10, dtype=int)

        # Mark obstacle voxels based on type
        for obstacle in obstacles:
            # Parse obstacle data (support both old and new formats)
            if len(obstacle) >= 5:
                x, y, z, radius, obs_type = obstacle[:5]
            else:
                x, y, z, radius = obstacle[:4]
                obs_type = "physical"  # Legacy: treat as blocked
            
            # Determine grid value based on obstacle type
            # Higher value = easier to traverse (lower cost)
            # Lower value = harder to traverse (higher cost)
            # 0 = blocked
            if obs_type == "physical":
                grid_value = 0  # Blocked - must path around
            elif obs_type == "high_jam":
                grid_value = 1  # Very high cost (10x normal)
            elif obs_type == "low_jam":
                grid_value = 5  # Moderate cost (2x normal)
            else:
                grid_value = 0  # Unknown type - block it
            
            center = np.array([x, y, z])

            # Find grid cells within sphere
            grid_center = self.world_to_grid(center)
            radius_voxels = int(np.ceil(radius / self.voxel_size))

            # Check all voxels in bounding box
            for i in range(
                max(0, grid_center[0] - radius_voxels),
                min(self.grid_size[0], grid_center[0] + radius_voxels + 1),
            ):
                for j in range(
                    max(0, grid_center[1] - radius_voxels),
                    min(self.grid_size[1], grid_center[1] + radius_voxels + 1),
                ):
                    for k in range(
                        max(0, grid_center[2] - radius_voxels),
                        min(self.grid_size[2], grid_center[2] + radius_voxels + 1),
                    ):
                        # Check if voxel center is inside sphere
                        voxel_world = self.grid_to_world([i, j, k])
                        if np.linalg.norm(voxel_world - center) <= radius:
                            # Only reduce cost (don't overwrite a more restrictive value)
                            # E.g., if a cell is already blocked (0), don't set it to 5
                            current_val = self.grid_matrix[i, j, k]
                            self.grid_matrix[i, j, k] = min(current_val, grid_value)

        # Create pathfinding3d Grid object
        self.grid = Grid(matrix=self.grid_matrix)

    def find_path(self, start, end):
        """
        Find path from start to end using configured algorithm.

        Args:
            start: Start position in world coordinates [x, y, z]
            end: End position in world coordinates [x, y, z]

        Returns:
            path: List of waypoints in world coordinates, or None if no path found
            algorithm_name: Name of algorithm used
            nodes_expanded: Number of nodes expanded (computational cost)
        """
        if self.grid is None:
            print("Error: Grid not initialized. Call update_obstacles() first.")
            return None, self.algorithm, 0

        # Convert to grid coordinates
        start_grid = self.world_to_grid(start)
        end_grid = self.world_to_grid(end)

        # Get start and end nodes
        start_node = self.grid.node(*start_grid)
        end_node = self.grid.node(*end_grid)

        # Check if start or end is inside obstacle
        if not start_node.walkable:
            # print(f"⚠️  Start position inside obstacle! Finding nearest safe position...")
            # Find nearest walkable node
            start_node = self._find_nearest_walkable(start_grid)
            if start_node is None:
                return None, self.algorithm, 0

        if not end_node.walkable:
            # print(f"⚠️  End position inside obstacle! Finding nearest safe position...")
            # Find nearest walkable node
            end_node = self._find_nearest_walkable(end_grid)
            if end_node is None:
                return None, self.algorithm, 0

        # Get finder class and create instance
        _, finder_class = self.ALGORITHMS[self.algorithm]
        finder = finder_class(diagonal_movement=self.diagonal_movement)

        # Find path
        path_grid, runs = finder.find_path(start_node, end_node, self.grid)

        # Reset grid for next search
        self.grid.cleanup()

        if not path_grid:
            print(f"No path found from {start} to {end}")
            return None, self.algorithm, runs

        # Convert path to world coordinates
        # pathfinding3d returns GridNode objects or tuples depending on algorithm
        path_world = []
        for node in path_grid:
            if hasattr(node, "x"):  # GridNode object
                path_world.append(self.grid_to_world((node.x, node.y, node.z)))
            else:  # Tuple
                path_world.append(self.grid_to_world(node))

        # Smooth path (remove redundant waypoints on straight lines)
        path_world = self._smooth_path(path_world)

        return path_world, self.algorithm, runs

    def _find_nearest_walkable(self, grid_pos, max_distance=10):
        """Find nearest walkable grid cell."""
        for distance in range(1, max_distance + 1):
            for dx in range(-distance, distance + 1):
                for dy in range(-distance, distance + 1):
                    for dz in range(-distance, distance + 1):
                        check_pos = (
                            grid_pos[0] + dx,
                            grid_pos[1] + dy,
                            grid_pos[2] + dz,
                        )
                        if (
                            0 <= check_pos[0] < self.grid_size[0]
                            and 0 <= check_pos[1] < self.grid_size[1]
                            and 0 <= check_pos[2] < self.grid_size[2]
                        ):
                            node = self.grid.node(*check_pos)
                            if node.walkable:
                                return node
        return None

    def _smooth_path(self, path):
        """Remove redundant waypoints that are on straight lines."""
        if len(path) <= 2:
            return path

        smoothed = [path[0]]

        for i in range(1, len(path) - 1):
            # Check if current point is on line between previous and next
            prev = np.array(smoothed[-1])
            curr = np.array(path[i])
            next_pt = np.array(path[i + 1])

            # Calculate if points are collinear (within tolerance)
            v1 = curr - prev
            v2 = next_pt - curr

            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                v1_norm = v1 / np.linalg.norm(v1)
                v2_norm = v2 / np.linalg.norm(v2)

                # If directions are similar, skip current point
                if np.dot(v1_norm, v2_norm) < 0.99:  # Not collinear
                    smoothed.append(path[i])
            else:
                smoothed.append(path[i])

        smoothed.append(path[-1])

        return smoothed

    def set_algorithm(self, algorithm):
        """Change the path planning algorithm."""
        if algorithm not in self.ALGORITHMS:
            available = ", ".join(self.ALGORITHMS.keys())
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {available}")
        self.algorithm = algorithm
        print(f"Path planning algorithm changed to: {self.ALGORITHMS[algorithm][0]}")

    @classmethod
    def list_algorithms(cls):
        """List all available algorithms."""
        print("Available 3D path planning algorithms:")
        for key, (name, _) in cls.ALGORITHMS.items():
            print(f"  '{key}': {name}")
        return list(cls.ALGORITHMS.keys())


def test_path_planner():
    """Test the 3D path planner."""
    print("=" * 60)
    print("Testing 3D Path Planner")
    print("=" * 60)

    # Create planner
    planner = PathPlanner3D(
        bounds_min=[-50, -50, -50],
        bounds_max=[150, 200, 100],
        voxel_size=3.0,
        algorithm="astar",
    )

    # Define obstacles
    obstacles = [
        (35, 75, 15, 20),
        (50, 120, 10, 15),
    ]

    planner.update_obstacles(obstacles)

    # Test path finding
    start = np.array([0, 0, 0])
    end = np.array([35, 150, 30])

    print(f"\nFinding path from {start} to {end}...")
    path, algo, nodes = planner.find_path(start, end)

    if path:
        print(f"Path found using {algo}!")
        print(f"  Waypoints: {len(path)}")
        print(f"  Nodes expanded: {nodes}")
        print("  First few waypoints:")
        for i, wp in enumerate(path[:5]):
            print(f"    {i}: {wp}")
    else:
        print("No path found!")

    # Test different algorithms
    print("\n" + "=" * 60)
    print("Comparing algorithms:")
    print("=" * 60)

    for algo_key in planner.list_algorithms():
        planner.set_algorithm(algo_key)
        path, algo, nodes = planner.find_path(start, end)
        if path:
            print(
                f"{algo_key:12s}: {len(path):3d} waypoints, {nodes:5d} nodes expanded"
            )


if __name__ == "__main__":
    test_path_planner()
