# 2D Truss FEM Solver

This is a graphical user interface (GUI) application for solving 2D truss problems using the Finite Element Method (FEM). The application allows users to input the truss structure, boundary conditions, and forces, and then calculates the displacements, reactions, and stresses in each truss member.

## Features

- Input the number of elements and nodes.
- Input node coordinates (x, y).
- Specify local destination array (LDA), applied forces (Fx, Fy), and boundary conditions.
- Input material properties like elastic modulus (E) and cross-sectional area (A) for each element.
- Calculate displacements, reactions, and stresses in the truss structure.
- Visualize the truss structure with forces and the deformed shape.

## Requirements

- Python 3.x
- Required Python libraries:
  - `numpy`
  - `matplotlib`
  - `tkinter`
  - `customtkinter`

## Usage Instructions

1. **Input Truss Parameters:**

   - Enter the number of elements and nodes.
   - Provide the coordinates for each node.
   - Specify the local destination array (LDA) to define element connectivity.
   - Input nodal forces and support conditions.
   - Input material properties: Elastic modulus (E) and cross-sectional area (A)
   - Input fixed points (nb).

2. **Perform FEM Analysis:**

   - Click "Calculate"  button to perform the FEM calculations (displacements, reactions, and stresses).
   - The results will be displayed, including node displacements, reactions, and stresses in each member.

3. **Visualize Truss Structure:**

   - The deformed truss structure will be plotted for visual inspection.

### Contributor

Pongpera Sutthisopha-Arporn - pongpera.s2020@outlook.com
