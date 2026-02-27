import matplotlib.pyplot as plt
import numpy as np


def calculate_distance(agent_i, agent_j):
    """
    Calculate the distance between two agents in 3D space

    Parameters:
        agent_i (list/array): The position of agent i [x, y, z]
        agent_j (list/array): The position of agent j [x, y, z]

    Returns:
        float: The distance between agent i and agent j
    """
    return np.sqrt(
        (agent_i[0] - agent_j[0]) ** 2
        + (agent_i[1] - agent_j[1]) ** 2
        + (agent_i[2] - agent_j[2]) ** 2
    )


def calculate_aij(alpha, delta, rij, r0, v):
    """
    Calculate the aij value

    Parameters:
        alpha (float): System parameter about antenna characteristics
        delta (float): The required application data rate
        rij (float): The distance between two agents
        r0 (float): Reference distance value
        v (float): Path loss exponent

    Returns:
        float: The calculated aij (communication quality in antenna far-field) value
    """
    return np.exp(-alpha * (2**delta - 1) * (rij / r0) ** v)


def calculate_gij(rij, r0):
    """
    Calculate the gij value

    Parameters:
        rij (float): The distance between two agents
        r0 (float): Reference distance value

    Returns:
        float: The calculated gij (communication quality in antenna near-field) value
    """
    return rij / np.sqrt(rij**2 + r0**2)


def calculate_rho_ij(beta, v, rij, r0):
    """
    Calculate the rho_ij (the derivative of phi_ij) value

    Parameters:
        beta (float): alpha * (2**delta - 1)
        v (float): Path loss exponent
        rij (float): The distance between two agents
        r0 (float): Reference distance value

    Returns:
        float: The calculated rho_ij value
    """
    return (
        (-beta * v * rij ** (v + 2) - beta * v * (r0**2) * (rij**v) + r0 ** (v + 2))
        * np.exp(-beta * (rij / r0) ** v)
        / np.sqrt((rij**2 + r0**2) ** 3)
    )


def calculate_Jn(communication_qualities_matrix, neighbor_agent_matrix, PT):
    """
    Calculate the Jn (average communication performance indicator) value

    Parameters:
        communication_qualities_matrix (numpy.ndarray): The communication qualities matrix among agents
        neighbor_agent_matrix (numpy.ndarray): The neighbor_agent matrix which is adjacency matrix of aij value
        PT (float): The reception probability threshold

    Returns:
        float: The calculated Jn value
    """
    total_communication_quality = 0
    total_neighbors = 0
    swarm_size = communication_qualities_matrix.shape[0]
    for i in range(swarm_size):
        for j in [x for x in range(swarm_size) if x != i]:
            if neighbor_agent_matrix[i, j] > PT:
                total_communication_quality += communication_qualities_matrix[i, j]
                total_neighbors += 1
    return total_communication_quality / total_neighbors if total_neighbors > 0 else 0


def calculate_rn(distances_matrix, neighbor_agent_matrix, PT):
    """
    Calculate the rn (average neighboring distance performance indicator) value

    Parameters:
        distances_matrix (numpy.ndarray): The distances matrix among agents
        neighbor_agent_matrix (numpy.ndarray): The neighbor_agent matrix which is adjacency matrix of aij value
        PT (float): The reception probability threshold

    Returns:
        float: The calculated rn value
    """
    total_distance = 0
    total_neighbors = 0
    swarm_size = distances_matrix.shape[0]
    for i in range(swarm_size):
        for j in [x for x in range(swarm_size) if x != i]:
            if neighbor_agent_matrix[i, j] > PT:
                total_distance += distances_matrix[i, j]
                total_neighbors += 1
    return total_distance / total_neighbors if total_neighbors > 0 else 0


def find_closest_agent(swarm_position, swarm_centroid):
    """
    Find the index of the agent with the minimum distance to the destination in 3D

    Parameters:
        swarm_position (numpy.ndarray): The positions of the swarm (Nx3)
        swarm_centroid (numpy.ndarray): The centroid of the swarm (3,)

    Returns:
        int: The index of the agent with the minimum distance to the destination
    """
    # Calculate the Euclidean distance from each agent to the destination
    distances_matrix = np.sqrt(np.sum((swarm_position - swarm_centroid) ** 2, axis=1))

    # Find the index of the agent with the minimum distance
    closest_agent_index = np.argmin(distances_matrix)

    return closest_agent_index


def draw_sphere(ax, center, radius, color="red", alpha=0.2):
    """
    Draw a sphere in 3D space

    Parameters:
        ax: The 3D axes object
        center: (x, y, z) center of the sphere
        radius: Radius of the sphere
        color: Color of the sphere
        alpha: Transparency of the sphere
    """
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, color=color, alpha=alpha, shade=True)


def plot_figures_3d(
    axs,
    t_elapsed,
    Jn,
    rn,
    swarm_position,
    PT,
    communication_qualities_matrix,
    swarm_size,
    swarm_paths,
    node_colors,
    line_colors,
    obstacles,
    swarm_destination,
    undiscovered_obstacles=None,
):
    """
    Plot 4 figures in 3D (Formation Scene, Swarm Trajectories, Jn Performance, rn Performance)

    Parameters:
        axs (numpy.ndarray): The axes of the figure
        t_elapsed (list): The elapsed time
        Jn (list): The Jn values
        rn (list): The rn values
        swarm_position (numpy.ndarray): The positions of the swarm (Nx3)
        PT (float): The reception probability threshold
        communication_qualities_matrix (numpy.ndarray): The communication qualities matrix among agents
        swarm_size (int): The number of agents in the swarm
        swarm_paths (list): The paths of the swarm
        node_colors (list): The colors of the nodes
        line_colors (list): The colors of the lines
        obstacles (list): The list of discovered/visible obstacles (x, y, z, radius) - shown in red
        swarm_destination (numpy.ndarray): The destination of the swarm (3,)
        undiscovered_obstacles (list): The list of undiscovered obstacles - shown in gray/transparent

    Returns:
        None
    """
    if undiscovered_obstacles is None:
        undiscovered_obstacles = []
    # Clear axes but preserve 3D projection
    for ax in axs.flatten():
        ax.clear()

    ########################
    # Plot formation scene #
    ########################
    # Check if axes need to be converted to 3D
    if not hasattr(axs[0, 0], "zaxis"):
        axs[0, 0].remove()
        axs[0, 0] = plt.gcf().add_subplot(2, 2, 1, projection="3d")

    axs[0, 0].set_title("Formation Scene (3D)")
    axs[0, 0].set_xlabel("$x$")
    axs[0, 0].set_ylabel("$y$")
    axs[0, 0].set_zlabel("$z$")

    # Plot the nodes
    for i in range(swarm_position.shape[0]):
        axs[0, 0].scatter(
            swarm_position[i, 0],
            swarm_position[i, 1],
            swarm_position[i, 2],
            color=node_colors[i],
            s=100,
            edgecolors="black",
            linewidth=1.5,
        )

    # Plot the edges
    for i in range(swarm_position.shape[0]):
        for j in range(i + 1, swarm_position.shape[0]):
            if communication_qualities_matrix[i, j] > PT:
                axs[0, 0].plot(
                    [swarm_position[i, 0], swarm_position[j, 0]],
                    [swarm_position[i, 1], swarm_position[j, 1]],
                    [swarm_position[i, 2], swarm_position[j, 2]],
                    color=line_colors[i, j],
                    linestyle="--",
                    linewidth=1,
                )

    # Add discovered obstacles to formation scene (red spheres)
    for obstacle in obstacles:
        x, y, z, radius = obstacle
        draw_sphere(axs[0, 0], [x, y, z], radius, color="red", alpha=0.3)

    # Add undiscovered obstacles (gray, very transparent - "fog of war")
    for obstacle in undiscovered_obstacles:
        if obstacle not in obstacles:  # Don't draw if already discovered
            x, y, z, radius = obstacle
            draw_sphere(axs[0, 0], [x, y, z], radius, color="gray", alpha=0.08)

    # Plot destination in formation scene
    axs[0, 0].scatter(
        swarm_destination[0],
        swarm_destination[1],
        swarm_destination[2],
        marker="s",
        s=200,
        color="none",
        edgecolors="black",
        linewidths=2,
    )
    axs[0, 0].text(
        swarm_destination[0],
        swarm_destination[1],
        swarm_destination[2] + 3,
        "Destination",
        ha="center",
        va="bottom",
    )

    # Set equal aspect ratio for 3D
    set_axes_equal(axs[0, 0])

    ###########################
    # Plot swarm trajectories #
    ###########################
    # Check if axes need to be converted to 3D
    if not hasattr(axs[0, 1], "zaxis"):
        axs[0, 1].remove()
        axs[0, 1] = plt.gcf().add_subplot(2, 2, 2, projection="3d")

    axs[0, 1].set_title("Swarm Trajectories (3D)")
    axs[0, 1].set_xlabel("$x$")
    axs[0, 1].set_ylabel("$y$")
    axs[0, 1].set_zlabel("$z$")

    # Store the current swarm positions
    swarm_paths.append(swarm_position.copy())

    # Convert the list of positions to a numpy array
    trajectory_array = np.array(swarm_paths)

    # Plot the trajectories
    for i in range(swarm_position.shape[0]):
        axs[0, 1].plot(
            trajectory_array[:, i, 0],
            trajectory_array[:, i, 1],
            trajectory_array[:, i, 2],
            color=node_colors[i],
            linewidth=2,
        )

    # Plot the initial positions
    axs[0, 1].scatter(
        trajectory_array[0, :, 0],
        trajectory_array[0, :, 1],
        trajectory_array[0, :, 2],
        color=node_colors,
        s=100,
        edgecolors="black",
        linewidth=1.5,
        marker="o",
    )

    # Plot current positions
    axs[0, 1].scatter(
        trajectory_array[-1, :, 0],
        trajectory_array[-1, :, 1],
        trajectory_array[-1, :, 2],
        color=node_colors,
        s=100,
        edgecolors="black",
        linewidth=1.5,
        marker="^",
    )

    # Add discovered obstacles to trajectory plot (red spheres)
    for obstacle in obstacles:
        x, y, z, radius = obstacle
        draw_sphere(axs[0, 1], [x, y, z], radius, color="red", alpha=0.3)

    # Add undiscovered obstacles (gray, very transparent)
    for obstacle in undiscovered_obstacles:
        if obstacle not in obstacles:  # Don't draw if already discovered
            x, y, z, radius = obstacle
            draw_sphere(axs[0, 1], [x, y, z], radius, color="gray", alpha=0.08)

    # Plot destination in trajectory plot
    axs[0, 1].scatter(
        swarm_destination[0],
        swarm_destination[1],
        swarm_destination[2],
        marker="s",
        s=200,
        color="none",
        edgecolors="black",
        linewidths=2,
    )
    axs[0, 1].text(
        swarm_destination[0],
        swarm_destination[1],
        swarm_destination[2] + 3,
        "Destination",
        ha="center",
        va="bottom",
    )

    # Set equal aspect ratio for 3D
    set_axes_equal(axs[0, 1])

    #######################
    # Plot Jn performance #
    #######################
    axs[1, 0].set_title("Average Communication Performance Indicator")
    axs[1, 0].plot(t_elapsed, Jn)
    axs[1, 0].set_xlabel("$t(s)$")
    axs[1, 0].set_ylabel("$J_n$", rotation=0, labelpad=20)
    axs[1, 0].grid(True, alpha=0.3)
    if len(Jn) > 0:  # Only add text if there are values
        axs[1, 0].text(
            t_elapsed[-1], Jn[-1], "Jn={:.4f}".format(Jn[-1]), ha="right", va="top"
        )

    #######################
    # Plot rn performance #
    #######################
    axs[1, 1].set_title("Average Distance Performance Indicator")
    axs[1, 1].plot(t_elapsed, rn)
    axs[1, 1].set_xlabel("$t(s)$")
    axs[1, 1].set_ylabel("$r_n$", rotation=0, labelpad=20)
    axs[1, 1].grid(True, alpha=0.3)
    if len(rn) > 0:  # Only add text if there are values
        axs[1, 1].text(
            t_elapsed[-1], rn[-1], "$r_n$={:.4f}".format(rn[-1]), ha="right", va="top"
        )

    plt.tight_layout()


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Parameters:
        ax: A matplotlib axis, e.g., as output from plt.gca().
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
