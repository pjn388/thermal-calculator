from typing import Any, Callable
from node import Node
from boundary_conditions import ConstantTemperatureBC, ConvectionBC, HeatFluxBC
from mesh import Mesh

# Material and boundary condition constants
k = 20.0  # thermal conductivity
q_dot_gen = 1.0e4  # heat generation
h = 10.0  # convection coefficient
const_temp = 100.0  # constant temperature for bottom boundary
tinf = 30.0  # ambient temperature for convection
q_prime_prime = 10.0  # heat flux for right boundary
deltax = 0.2  # grid spacing in x-direction
deltay = 0.2  # grid spacing in y-direction

def node_factory(x: float, y: float, delta_x: float, delta_y: float) -> Node:
    """Factory function to create nodes with proper material properties."""
    return Node(x, y, delta_x=delta_x, delta_y=delta_y, k=k, q_dot_gen=q_dot_gen)

def _clear_neighbor_references(mesh: Mesh, i: int, j: int):
    """Clear all neighbor references pointing to node at position (i, j)."""
    # Define direction mappings: (di, dj, reverse_direction)
    direction_map = [
        (-1, 0, 'up'),       # if we're removing a node, its 'down' neighbor should clear 'up' reference
        (1, 0, 'down'),      # if we're removing a node, its 'up' neighbor should clear 'down' reference
        (0, -1, 'right'),    # if we're removing a node, its 'left' neighbor should clear 'right' reference
        (0, 1, 'left'),      # if we're removing a node, its 'right' neighbor should clear 'left' reference
        (-1, -1, 'right_up'),
        (1, -1, 'right_down'),
        (-1, 1, 'left_up'),
        (1, 1, 'left_down')
    ]

    for di, dj, reverse_direction in direction_map:
        ni, nj = i + di, j + dj
        if 0 <= ni < mesh.height and 0 <= nj < mesh.width:
            neighbor = mesh.nodes[ni][nj]
            if neighbor is not None and isinstance(neighbor, Node):
                neighbor.neighbors[reverse_direction] = None

def apply_boundary_conditions(mesh: Mesh):
    """Apply boundary conditions to the mesh after initialization."""

    # Bottom boundary (i=0) - constant temperature
    for j in range(mesh.width):
        mesh[0, j].add_boundary_condition(ConstantTemperatureBC(const_temp))

    # Top boundary (i=5) - convection (skip top left i.e j=0, top right is skip in the creation later)
    for j in range(1, mesh.width-1):  # Skip j=0 (left) and j=4 (right)
        mesh[5, j].add_boundary_condition(ConvectionBC(h, tinf))

    # Remove top left (i=5, j=0) and top right (i=5, j=4) nodes
    # First, clear neighbor references to these nodes
    if mesh.nodes[5][0] is not None:
        # Clear references from neighbors
        _clear_neighbor_references(mesh, 5, 0)
    if mesh.nodes[5][4] is not None:
        # Clear references from neighbors
        _clear_neighbor_references(mesh, 5, 4)

    mesh.nodes[5][0] = None
    mesh.nodes[5][4] = None

    # Right boundary (j=4) - heat flux (skip top right)
    for i in range(mesh.height-1):  # Skip i=5 (top row)
        mesh[i, 4].add_boundary_condition(HeatFluxBC(q_prime_prime))

    mesh[4, 0].add_boundary_condition(ConvectionBC(h, tinf))
    mesh[4, 4].add_boundary_condition(ConvectionBC(h, tinf))

# Create a 5 wide x 6 tall mesh with nodes having proper material properties
mesh: Mesh = Mesh(5, 6, delta_x=deltax, delta_y=deltay, node_factory=node_factory)

# Apply boundary conditions by indexing the mesh
apply_boundary_conditions(mesh)
mesh.solve()

if __name__ == "__main__":
    print(mesh.get_finite_difference_equations())
    print(mesh.build_matrix_equation())
    # Render the mesh
    fig: Any = mesh.render()
    if fig:
        fig.savefig('mesh_plot.png', dpi=300, bbox_inches='tight')
        print("Plot saved to mesh_plot.png")
    else:
        print("Error: No figure to save")
