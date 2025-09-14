from typing import Any, Callable, Tuple, List
from node import Node
from boundary_conditions import ConstantTemperatureBC, ConvectionBC, HeatFluxBC
from mesh import Mesh
from math import sqrt
import matplotlib.pyplot as plt


def create_and_solve_mesh(k: float, q_dot_gen: float, h: float, tinf: float, const_temp: float, q_prime_prime: float, deltax: float, deltay: float, render: bool = True) -> Tuple[Mesh, Any]:
    """Create, configure, and solve the thermal mesh simulation."""

    def node_factory(x: float, y: float, delta_x: float, delta_y: float) -> Node:
        """Factory function to create nodes with proper material properties."""
        return Node(x, y, delta_x=delta_x, delta_y=delta_y, k=k, q_dot_gen=q_dot_gen)

    def apply_boundary_conditions(mesh, const_temp: float, h: float, T_inf: float, q_prime_prime: float):
        """Apply boundary conditions to the mesh after initialization."""

        # Bottom boundary (i=0) - constant temperature
        for j in range(mesh.width):
            mesh.nodes[0][j].add_boundary_condition(ConstantTemperatureBC(const_temp))

        # Top boundary (i=5) - convection (skip top left i.e j=0, top right is skip in the creation later)
        for j in range(1, mesh.width-1):  # Skip j=0 (left) and j=4 (right)
            mesh.nodes[5][j].add_boundary_condition(ConvectionBC(h, T_inf))

        # Remove top left (i=5, j=0) and top right (i=5, j=4) nodes
        # First, clear neighbor references to these nodes
        if mesh.nodes[5][0] is not None:
            # Clear references from neighbors
            mesh._clear_neighbor_references(5, 0)
        if mesh.nodes[5][4] is not None:
            # Clear references from neighbors
            mesh._clear_neighbor_references(5, 4)

        mesh.nodes[5][0] = None
        mesh.nodes[5][4] = None

        # Right boundary (j=4) - heat flux (skip top right)
        for i in range(mesh.height-1):  # Skip i=5 (top row)
            mesh.nodes[i][4].add_boundary_condition(HeatFluxBC(q_prime_prime))

        mesh.nodes[4][0].add_boundary_condition(ConvectionBC(h, T_inf))
        mesh.nodes[4][4].add_boundary_condition(ConvectionBC(h, T_inf))

    # Create a 5 wide x 6 tall mesh with nodes having proper material properties
    mesh: Mesh = Mesh(5, 6, delta_x=deltax, delta_y=deltay, node_factory=node_factory)

    # Apply boundary conditions by indexing the mesh
    apply_boundary_conditions(mesh, const_temp, h, tinf, q_prime_prime)

    # Set overrides for specific nodes
    mesh[0,4].override = (3/8)*q_dot_gen/k*deltax**2+h*(deltax/sqrt(2))*(tinf-mesh[0,4].symbolic_temp)+(deltay*q_prime_prime/2)+k*(deltay/2)*((mesh[0,4].neighbors["right"].symbolic_temp-mesh[0,4].symbolic_temp)/deltax)+k*(deltax/2)*((mesh[0,4].neighbors["down"].symbolic_temp-mesh[0,4].symbolic_temp)/deltay)
    mesh[4,4].override = (3/8)*q_dot_gen/k*deltax**2+h*(deltax/sqrt(2))*(tinf-mesh[4,4].symbolic_temp)+(deltay*q_prime_prime/2)+k*(deltay/2)*((mesh[4,4].neighbors["left"].symbolic_temp-mesh[4,4].symbolic_temp)/deltax)+k*(deltax/2)*((mesh[4,4].neighbors["down"].symbolic_temp-mesh[4,4].symbolic_temp)/deltay)

    mesh.solve()

    # Render the mesh if requested
    fig: Any = None
    if render:
        fig = mesh.render()

    return mesh, fig


def get_node_temperatures(mesh: Mesh, node_indices: List[Tuple[int, int]]) -> List[float]:
    """Extract temperatures of specified nodes from the mesh.

    Args:
        mesh: Solved mesh object
        node_indices: List of (i, j) indices for nodes

    Returns:
        List of temperatures corresponding to the node indices
    """
    temperatures: List[float] = []
    for i, j in node_indices:
        node = mesh[i, j]
        if node is not None:
            temperatures.append(node.temperature)
    return temperatures


def create_node_plots() -> None:
    """Create three additional plots for different parameter variations."""

    # Default parameters
    k_default = 20.0
    q_dot_gen_default = 1.0e4
    h_default = 10.0
    tinf = 30.0
    const_temp = 100.0
    q_prime_prime = 10.0
    deltax = 0.2
    deltay = 0.2

    # Node indices to plot: [0,1], [1,1], [2,1], [3,1], [4,1]
    node_indices = [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)]

    # Plot 1: Vary h values
    h_values = [1.0, 10.0, 50.0]
    h_temperatures = {}

    for h in h_values:
        mesh, _ = create_and_solve_mesh(k_default, q_dot_gen_default, h, tinf, const_temp, q_prime_prime, deltax, deltay, render=False)
        h_temperatures[h] = get_node_temperatures(mesh, node_indices)

    # Plot 2: Vary k values with default h
    k_values = [10.0, 20.0, 50.0]
    k_temperatures = {}

    for k in k_values:
        mesh, _ = create_and_solve_mesh(k, q_dot_gen_default, h_default, tinf, const_temp, q_prime_prime, deltax, deltay, render=False)
        k_temperatures[k] = get_node_temperatures(mesh, node_indices)

    # Plot 3: Vary q_dot_gen values with default params
    q_dot_gen_values = [0.0, 1e4, 2e4]
    q_temperatures = {}

    for q in q_dot_gen_values:
        mesh, _ = create_and_solve_mesh(k_default, q, h_default, tinf, const_temp, q_prime_prime, deltax, deltay, render=False)
        q_temperatures[q] = get_node_temperatures(mesh, node_indices)

    # Create individual plots
    node_positions = [i for i, _ in node_indices]

    # Plot 1: h variation
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    for h in h_values:
        ax1.plot(node_positions, h_temperatures[h], label=f'h = {h}', marker='o')
    ax1.set_xlabel('Node Position (i)')
    ax1.set_ylabel('Temperature')
    ax1.set_title('Varying h (convection coefficient)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.savefig('node_profiles_h.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("h variation plot saved to node_profiles_h.png")

    # Plot 2: k variation
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for k in k_values:
        ax2.plot(node_positions, k_temperatures[k], label=f'k = {k}', marker='s')
    ax2.set_xlabel('Node Position (i)')
    ax2.set_ylabel('Temperature')
    ax2.set_title('Varying k (thermal conductivity) with h = 10.0')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.savefig('node_profiles_k.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("k variation plot saved to node_profiles_k.png")

    # Plot 3: q_dot_gen variation
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    for q in q_dot_gen_values:
        ax3.plot(node_positions, q_temperatures[q], label=f'q_dot_gen = {q/1000:.1f}k', marker='^')
    ax3.set_xlabel('Node Position (i)')
    ax3.set_ylabel('Temperature')
    ax3.set_title('Varying q_dot_gen (heat generation) with k = 20.0, h = 10.0')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    fig3.savefig('node_profiles_q_dot_gen.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print("q_dot_gen variation plot saved to node_profiles_q_dot_gen.png")


if __name__ == "__main__":
    # Material and boundary condition constants
    k = 20.0  # thermal conductivity
    q_dot_gen = 1.0e4  # heat generation
    h = 10.0  # convection coefficient
    tinf = 30.0  # ambient temperature for convection
    const_temp = 100.0  # constant temperature for bottom boundary
    q_prime_prime = 10.0  # heat flux for right boundary
    deltax = 0.2  # grid spacing in x-direction
    deltay = 0.2  # grid spacing in y-direction

    # Generate original mesh and plot
    mesh, fig = create_and_solve_mesh(k, q_dot_gen, h, tinf, const_temp, q_prime_prime, deltax, deltay)

    print(mesh.get_finite_difference_equations())
    print(mesh.build_matrix_equation())

    if fig:
        fig.savefig('mesh_plot.png', dpi=300, bbox_inches='tight')
        print("Mesh plot saved to mesh_plot.png")
    else:
        print("Error: No figure to save")

    # Generate additional node profile plots
    create_node_plots()
