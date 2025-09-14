# Thermal Calculator

This project provides a Python-based thermal calculator that uses the finite difference method to simulate heat transfer in a 2D mesh. It allows for the application of various boundary conditions and material properties, and visualizes the temperature distribution across the mesh.

## Features

-   **Finite Difference Method**: Solves 2D steady-state heat conduction problems using a numerical approach.
-   **Configurable Mesh**: Create meshes with adjustable dimensions and grid spacing.
-   **Node-based Simulation**: Each node in the mesh represents a point where temperature is calculated, incorporating thermal conductivity and heat generation.
-   **Boundary Conditions**: Supports:
    -   [`ConstantTemperatureBC`](boundary_conditions.py:25)
    -   [`ConvectionBC`](boundary_conditions.py:43)
    -   [`HeatFluxBC`](boundary_conditions.py:60)
-   **Visualization**: Generates matplotlib plots of the mesh with temperature contours and node-specific information.
-   **Parameter Study**: Includes functionality to generate plots showing the effect of varying key parameters (thermal conductivity, convection coefficient, heat generation) on node temperatures.

## Project Structure

-   [`main.py`](main.py): The main script to run the simulation, configure parameters, and generate plots.
-   [`node.py`](node.py): Defines the `Node` and `BaseNode` classes, which represent individual points in the thermal mesh and their thermal properties, neighbors, and boundary conditions.
-   [`mesh.py`](mesh.py): Defines the `Mesh` class, responsible for creating the grid of nodes, connecting them, applying boundary conditions, solving the system of finite difference equations using SymPy, and rendering the results.
-   [`boundary_conditions.py`](boundary_conditions.py): Defines abstract base classes and concrete implementations for various boundary conditions (constant temperature, convection, heat flux).
-   `requirements.txt`: Lists the Python dependencies.
-   `mesh_plot.png`: Output image showing the temperature distribution in the solved mesh.
-   `node_profiles_h.png`: Output image showing temperature profiles with varying convection coefficients.
-   `node_profiles_k.png`: Output image showing temperature profiles with varying thermal conductivities.
-   `node_profiles_q_dot_gen.png`: Output image showing temperature profiles with varying heat generation rates.

## How to Run

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/pjn388/thermal-calculator.git
    cd thermal-calculator
    ```

2.  **Install dependencies**:
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Run the simulation**:
    ```bash
    python main.py
    ```
    This will execute the thermal simulation, generate the `mesh_plot.png`, and the node profile plots (`node_profiles_h.png`, `node_profiles_k.png`, `node_profiles_q_dot_gen.png`) in the project directory.

## Examples

The `main.py` script contains default parameters for a sample simulation. It demonstrates:
-   Creating a 5x6 mesh.
-   Applying a constant temperature boundary condition at the bottom.
-   Applying convection boundary conditions at the top.
-   Applying heat flux boundary conditions at the right.
-   Solving the system and plotting the results.
-   Generating plots to visualize the impact of varying `h`, `k`, and `q_dot_gen` values on node temperatures.

## Contributing

Feel free to fork the repository, open issues, or submit pull requests.

## Donations

* monero:83B495T1N3sje9vXMqNShbSx99g1QjKyL8YKjvU6rt6hAkmwbVUrQ65QGEUsL3QxVPdtiK91GnCP7bG2oCz7h1PDKsoCPB1
* ![monero:83B495T1N3sje9vXMqNShbSx99g1QjKyL8YKjvU6rt6hAkmwbVUrQ65QGEUsL3QxVPdtiK91GnCP7bG2oCz7h1PDKsoCPB1](https://raw.githubusercontent.com/pjn388/FEA/refs/heads/main/images/uni_recieve.png?raw=true)

