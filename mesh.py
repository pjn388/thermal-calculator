from typing import Any, Callable
from typing import Optional
import sympy as sp
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from node import BaseNode, Node
from boundary_conditions import ConstantTemperatureBC, ConvectionBC, HeatFluxBC

class Mesh:
    def __init__(self, width: int, height: int, delta_x: float = 1.0, delta_y: float = 1.0, node_factory: Optional[Callable[[float, float, float, float], BaseNode]] = None):
        self.width: int = width
        self.height: int = height
        self.delta_x: float = delta_x
        self.delta_y: float = delta_y
        self.nodes: list[list[BaseNode]] = []
        if node_factory is None:
            node_factory = lambda x, y, dx, dy: Node(x, y, delta_x=dx, delta_y=dy)
        for i in range(height):
            row: list[BaseNode] = []
            for j in range(width):
                x_pos = j * delta_x
                y_pos = i * delta_y
                row.append(node_factory(x_pos, y_pos, delta_x, delta_y))
            self.nodes.append(row)
        self.connect_neighbors()

    def connect_neighbors(self):
        """Automatically connect adjacent nodes."""
        # Define direction mappings: (di, dj, direction_name)
        directions = [
            (-1, 0, 'down'),    # moving down (decreasing i)
            (1, 0, 'up'),   # moving up (increasing i)
            (0, -1, 'left'),  # moving left (decreasing j)
            (0, 1, 'right'),   # moving right (increasing j)
            (-1, -1, 'left_down'),  # left and down
            (1, -1, 'left_up'),   # left and up
            (-1, 1, 'right_down'),  # right and down
            (1, 1, 'right_up')   # right and up
        ]

        # Clear existing neighbor connections for all nodes
        for row in self.nodes:
            for node in row:
                if node is not None:
                    node.neighbors = {'up': None, 'down': None, 'left': None, 'right': None, 'left_down': None, 'left_up': None, 'right_down': None, 'right_up': None}

        for i in range(self.height):
            for j in range(self.width):
                node = self.nodes[i][j]
                # All nodes get neighbors set (including boundary nodes)
                if node is not None:
                    for di, dj, direction in directions:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.height and 0 <= nj < self.width:
                            neighbor = self.nodes[ni][nj]
                            # Both regular nodes and boundary nodes can be neighbors
                            node.set_neighbor(direction, neighbor)
                        # If out of bounds, neighbor remains None


    def __getitem__(self, indices: tuple[int, int]) -> BaseNode:
        if isinstance(indices, tuple) and len(indices) == 2:
            m, n = indices
            if 0 <= n < self.height and 0 <= m < self.width:
                return self.nodes[n][m]
            else:
                raise IndexError("Mesh index out of range")
        else:
            raise TypeError("Mesh indices must be a tuple (m, n)")

    def get_finite_difference_equations(self) -> list[sp.Expr]:
        equations = []
        for row in self.nodes:
            for node in row:
                if isinstance(node, Node):
                    equations.append(node.compute_finite_difference())
        return equations
    def build_matrix_equation(self):

        symbols = []

        for row in self.nodes:

            for node in row:

                if isinstance(node, Node):

                    symbols.append(node.symbolic_temp)

        equations = self.get_finite_difference_equations()

        A, b = sp.linear_eq_to_matrix(equations, symbols)

        return A, symbols, b

    def solve(self):

        symbols = []

        nodes_list = []

        for row in self.nodes:

            for node in row:

                if isinstance(node, Node):

                    symbols.append(node.symbolic_temp)

                    nodes_list.append(node)

        equations = self.get_finite_difference_equations()

        A, b = sp.linear_eq_to_matrix(equations, symbols)

        x = A.solve(b)

        for idx, node in enumerate(nodes_list):

            node.temperature = float(x[idx])

    def render(self) -> Figure:
        if not self.nodes:
            return None  # type: ignore

        x_coords: list[float] = [node.x for row in self.nodes for node in row if node is not None]
        y_coords: list[float] = [node.y for row in self.nodes for node in row if node is not None]
        temperatures: list[float] = [node.temperature for row in self.nodes for node in row if node is not None]

        fig, ax = plt.subplots(figsize=(16, 8))
        scatter = ax.scatter(x_coords, y_coords, c=temperatures, cmap='coolwarm', marker='o')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Mesh with Temperatures')

        # Add grid lines to show node positions
        ax.grid(True, alpha=0.3, linestyle=':', color='blue')

        # Draw grid lines at all unique x and y positions
        x_positions = sorted(set(node.x for row in self.nodes for node in row if node is not None))
        y_positions = sorted(set(node.y for row in self.nodes for node in row if node is not None))

        for x in x_positions:
            ax.axvline(x, alpha=0.2, color='gray', linestyle='--')
        for y in y_positions:
            ax.axhline(y, alpha=0.2, color='gray', linestyle='--')

        # Draw lines between connected nodes (only for Node objects)
        lines = []
        for row in self.nodes:
            for node in row:
                # Only regular Node objects have neighbors and should render connections
                if isinstance(node, Node):
                    for neighbor in node.neighbors.values():
                        if neighbor is not None and isinstance(neighbor, Node):
                            lines.append([(node.x, node.y), (neighbor.x, neighbor.y)])
        if lines:
            lc = LineCollection(lines, colors='black', linewidths=0.5)
            ax.add_collection(lc)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, location='left')
        cbar.set_label('Temperature')

        # Highlight boundary condition nodes
        bc_nodes = [node for row in self.nodes for node in row if isinstance(node, Node) and len(node.boundary_conditions) > 0]
        if bc_nodes:
            bc_x_coords = [node.x for node in bc_nodes]
            bc_y_coords = [node.y for node in bc_nodes]
            ax.scatter(bc_x_coords, bc_y_coords, marker='s', s=80, facecolors='none', edgecolors='red', linewidth=2, label='Boundary Conditions')

        # Have each node render its own visual representation
        # Only regular Node objects display finite difference equations and render them
        # BoundaryNode objects display their boundary condition information in their render method
        for row in self.nodes:
            for node in row:
                if isinstance(node, BaseNode):
                    node.render(ax)

        # Add equations legend on the right
        equations = self.get_finite_difference_equations()
        _, symbols, _ = self.build_matrix_equation()
        equations_text = "\n".join(f"{str(sym)}: {str(eq)} = 0" for sym, eq in zip(symbols, equations))
        ax.text(1.02, 0.5, equations_text, transform=ax.transAxes, ha='left', va='center', fontsize=8, rotation=0)

        return fig

    def _clear_neighbor_references(self, i: int, j: int):
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
            if 0 <= ni < self.height and 0 <= nj < self.width:
                neighbor = self.nodes[ni][nj]
                if neighbor is not None and isinstance(neighbor, Node):
                    neighbor.neighbors[reverse_direction] = None