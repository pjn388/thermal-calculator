from typing import List, Dict, Optional
from abc import ABC, abstractmethod
import sympy as sp
from boundary_conditions import BoundaryCondition, ConstantTemperatureBC, ConvectionBC, HeatFluxBC


class BaseNode(ABC):
    """Abstract base class for all node types in the mesh."""

    def __init__(self, x: float, y: float, temperature: float = 0.0, delta_x: float = 1.0, delta_y: float = 1.0, k: float = 1.0, q_dot_gen: float = 0.0):
        self.x: float = x
        self.y: float = y
        self.temperature: float = temperature
        self.delta_x: float = delta_x
        self.delta_y: float = delta_y
        
        self.k: float = k # Thermal conductivity (W/mK)
        self.q_dot_gen: float = q_dot_gen # Volumetric heat generation (W/m3)
        self.symbolic_temp = sp.Symbol(f'T_{int(x/delta_x)}_{int(y/delta_y)}') # Create symbolic temperature variable for this node (e.g., T_0_1 for coordinates 0,1)

    @abstractmethod
    def render(self, ax: 'matplotlib.axes.Axes'):
        """Render this node on the plot."""
        pass

class Node(BaseNode):
    def __init__(self, x: float, y: float, temperature: float = 0.0, delta_x: float = 1.0, delta_y: float = 1.0, k: float = 1.0, q_dot_gen: float = 0.0):
        super().__init__(x, y, temperature, delta_x, delta_y, k, q_dot_gen)
        
        self.boundary_conditions: List[BoundaryCondition] = []
        self.override: Optional[sp.Expr] = None # Override for finite difference formula

        self.neighbors: Dict[str, Optional['BaseNode']] = {
            'up': None,
            'down': None,
            'left': None,
            'right': None,
            'left_down': None,
            'left_up': None,
            'right_down': None,
            'right_up': None
        }

    def add_boundary_condition(self, bc: BoundaryCondition):
        """Add a boundary condition to this node."""
        self.boundary_conditions.append(bc)

    def set_neighbor(self, direction: str, neighbor: Optional['BaseNode']):
        """Set a neighbor in a specific direction."""
        if direction in self.neighbors:
            self.neighbors[direction] = neighbor

    def get_verticle_neighbors(self):
        return (1 if self.neighbors['up'] is not None else 0) + (1 if self.neighbors['down'] is not None else 0)

    def get_horizontal_neighbors(self):
        return (1 if self.neighbors['left'] is not None else 0) + (1 if self.neighbors['right'] is not None else 0)

    def compute_finite_difference(self) -> sp.Expr:
        # Check if override is provided
        if self.override is not None:
            return self.override

        # Note: Boundary conditions are now handled by BoundaryNode objects in the mesh
        # Regular nodes compute heat transfer to all neighbors (including boundary nodes)

        # Count actual neighbors (non-None values)
        actual_neighbors = sum(1 for neighbor in self.neighbors.values() if neighbor is not None)


        # check inline existence - only count regular Node objects, not boundary nodes
        left = isinstance(self.neighbors["left"], Node)
        right = isinstance(self.neighbors["right"], Node)
        up = isinstance(self.neighbors["up"], Node)
        down = isinstance(self.neighbors["down"], Node)
        
        # TODO: account for diagonals connections will require understanding of desired mesh shape
        # existance of all 4 possible rectangles that contribute to this node
        left_up = isinstance(self.neighbors["left_up"], Node) and left and up
        right_up = isinstance(self.neighbors["right_up"], Node) and right and up
        left_down = isinstance(self.neighbors["left_down"], Node) and left and down
        right_down = isinstance(self.neighbors["right_down"], Node) and right and down

        heat_transfer = 0
        heat_gen = 0

        # Check each quadrant case and account for its contribution

        # up left
        if left_up:
            heat_transfer += self.k * (self.delta_y / 2) * (self.neighbors['left'].symbolic_temp - self.symbolic_temp) / self.delta_x
            heat_transfer += self.k * (self.delta_x / 2) * (self.neighbors['up'].symbolic_temp - self.symbolic_temp) / self.delta_y

            heat_gen += self.q_dot_gen * 0.25 * self.delta_x * self.delta_y

        # up right
        if right_up:
            heat_transfer += self.k * (self.delta_y / 2) * (self.neighbors['right'].symbolic_temp - self.symbolic_temp) / self.delta_x
            heat_transfer += self.k * (self.delta_x / 2) * (self.neighbors['up'].symbolic_temp - self.symbolic_temp) / self.delta_y

            heat_gen += self.q_dot_gen * 0.25 * self.delta_x * self.delta_y

        # down left
        if left_down:
            heat_transfer += self.k * (self.delta_y / 2) * (self.neighbors['left'].symbolic_temp - self.symbolic_temp) / self.delta_x
            heat_transfer += self.k * (self.delta_x / 2) * (self.neighbors['down'].symbolic_temp - self.symbolic_temp) / self.delta_y

            heat_gen += self.q_dot_gen * 0.25 * self.delta_x * self.delta_y

        # down right
        if right_down:
            heat_transfer += self.k * (self.delta_y / 2) * (self.neighbors['right'].symbolic_temp - self.symbolic_temp) / self.delta_x
            heat_transfer += self.k * (self.delta_x / 2) * (self.neighbors['down'].symbolic_temp - self.symbolic_temp) / self.delta_y

            heat_gen += self.q_dot_gen * 0.25 * self.delta_x * self.delta_y

        base_equation = heat_transfer + heat_gen

        if len(self.boundary_conditions) not in (0, 1, 2):
            raise Exception("Must have 0, 1, or 2 boundary conditions per node. Note when 2 boundary conditions assumed to apply half of avialbe surface to each")

        # number of connected rectangles a node has
        number_rects = left_up + right_up + left_down + right_down

        if number_rects == 1:
            area = (self.delta_x + self.delta_y)/2
        if number_rects == 2:
            if (left_up and left_down) or (right_up and right_down):
                area = self.delta_y
            if (left_up and right_up) or (left_down and right_down):
                area = self.delta_x
            if (left_up and right_down) or (right_up and left_down):
                area = self.delta_x + self.delta_y # I assume thats what makes logical sense
        if number_rects == 3:
            area = (self.delta_x + self.delta_y)/2
        if number_rects == 4 and len(self.boundary_conditions) > 0:
            raise Exception("Cannot have boundary conditions in interior nodes")

        boundary_contribution = 0
        has_const_temp_bc = False

        # Process each boundary condition
        for bc in self.boundary_conditions:
            # Constant temperature overwrites the entire equation
            if isinstance(bc, ConstantTemperatureBC):
                has_const_temp_bc = True
                const_temp_value = bc.temperature
                break

            if len(self.boundary_conditions) == 1:
                boundary_contribution += area * bc.get_contribution(self)
            if len(self.boundary_conditions) == 2:
                boundary_contribution += 0.5*area*bc.get_contribution(self) # for 2 bc give each half the surface area (if ur applying bc on different sides with different dlta x and y this will mess up)

        # Apply constant temperature BC if present
        if has_const_temp_bc:
            total_equation = self.symbolic_temp - const_temp_value
        else:
            total_equation = base_equation + boundary_contribution

        return total_equation

    def render(self, ax: 'matplotlib.axes.Axes'):
        """
        Render this node's temperature and boundary condition information.

        Args:
            ax: Matplotlib axes object
        """
        equation = self.compute_finite_difference()
        eq_display = str(equation)

        # Only display temperature
        ax.text(self.x, self.y + 0.035,
                f'T_{int(self.x/self.delta_x)}_{int(self.y/self.delta_y)} = {self.temperature:.2f}',
                ha='left', va='bottom', fontsize=7,
                color='black', style='normal',
                rotation=45, bbox=None)

        # If there are boundary conditions, also display them above the equation
        if self.boundary_conditions:
            y_offset = 0.07
            for i, bc in enumerate(self.boundary_conditions):
                bc_label = bc.get_display_info()
                color = bc.get_color()

                # Map colors to facecolors
                facecolor_map = {
                    'purple': 'violet',
                    'blue': 'lightblue',
                    'green': 'lightgreen'
                }
                facecolor = facecolor_map.get(color, 'lightblue')

                # Display each BC above the equation with proper spacing
                ax.text(self.x, self.y + y_offset + (i * 0.03),
                        bc_label,
                        ha='left', va='bottom', fontsize=6,
                        color=color, weight='bold',
                        rotation=45,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor=facecolor, alpha=0.8))







