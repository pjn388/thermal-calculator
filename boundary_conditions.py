from abc import ABC, abstractmethod
import sympy as sp
from typing import Any


class BoundaryCondition(ABC):
    """Abstract base class for boundary conditions."""

    @abstractmethod
    def get_contribution(self, node: Any) -> sp.Expr:
        """Get the contribution of this boundary condition."""
        pass

    @abstractmethod
    def get_display_info(self) -> str:
        """Get display information for this boundary condition."""
        pass

    @abstractmethod
    def get_color(self) -> str:
        """Get the color for rendering this boundary condition."""
        pass


class ConstantTemperatureBC(BoundaryCondition):
    """Constant temperature boundary condition."""

    def __init__(self, temperature: float):
        self.temperature = temperature

    def get_contribution(self, node: Any) -> sp.Expr:
        """Constant temperature overrides the entire equation."""
        # This will be handled specially in compute_finite_difference
        return sp.sympify(0)

    def get_display_info(self) -> str:
        return f"T={self.temperature}"

    def get_color(self) -> str:
        return 'purple'


class ConvectionBC(BoundaryCondition):
    """Convection boundary condition."""

    def __init__(self, h: float, T_inf: float):
        self.h = h  # convection coefficient
        self.T_inf = T_inf  # fluid temperature

    def get_contribution(self, node: Any) -> sp.Expr:
        return self.h * (self.T_inf - node.symbolic_temp)

    def get_display_info(self) -> str:
        return f"h={self.h}, T∞={self.T_inf}"

    def get_color(self) -> str:
        return 'blue'


class HeatFluxBC(BoundaryCondition):
    """Heat flux boundary condition."""

    def __init__(self, heat_flux: float):
        self.heat_flux = heat_flux

    def get_contribution(self, node: Any) -> sp.Expr:
        return self.heat_flux

    def get_display_info(self) -> str:
        return f"q''={self.heat_flux}"

    def get_color(self) -> str:
        return 'green'