import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt

class FEMSolverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("2D Truss FEM Solver")
        
        # Variables to store inputs (use Tkinter variables)
        self.e = ctk.IntVar()
        self.n = ctk.IntVar()
        self.x_y = []
        self.lda = []
        self.nf = []
        self.Fx = []
        self.Fy = []
        self.Es = ctk.DoubleVar()
        self.As = ctk.DoubleVar()
        self.E = []
        self.A = []
        self.nb = ctk.StringVar()
        
        # Frame for inputs
        input_frame = ctk.CTkFrame(self.root)
        input_frame.pack(padx=10, pady=10, fill=ctk.BOTH, expand=True)

        # Number of elements and nodes
        ctk.CTkLabel(input_frame, text="Number of Elements (e):").grid(row=0, column=1, padx=5, pady=5, sticky=ctk.E)
        ctk.CTkEntry(input_frame, textvariable=self.e).grid(row=0, column=2, padx=5, pady=5)

        ctk.CTkLabel(input_frame, text="Number of Nodes (n):").grid(row=0, column=3, padx=5, pady=5, sticky=ctk.E)
        ctk.CTkEntry(input_frame, textvariable=self.n).grid(row=0, column=4, padx=5, pady=5)

        # Node coordinates
        ctk.CTkLabel(input_frame, text="Node x, y Coordinates [example: X Y ]:").grid(row=1, column=1, padx=5, pady=5, sticky=ctk.E)
        self.coordinates_entry = ctk.CTkTextbox(input_frame, width=200, height=100)
        self.coordinates_entry.grid(row=1, column=2, padx=5, pady=5)

        # Local destination array (lda)
        ctk.CTkLabel(input_frame, text="Local Destination Array (lda) [example: 1 2 ]:").grid(row=1, column=3, padx=5, pady=5, sticky=ctk.E)
        self.lda_entry = ctk.CTkTextbox(input_frame, width=200, height=100)
        self.lda_entry.grid(row=1, column=4, padx=5, pady=5)

        # Forces applied (nf)
        ctk.CTkLabel(input_frame, text="Forces Applied (nf) [example: 1 2 ]:").grid(row=2, column=1, padx=5, pady=5, sticky=ctk.E)
        self.nf_entry = ctk.CTkTextbox(input_frame, width=200, height=25)
        self.nf_entry.grid(row=2, column=2, padx=5, pady=5)

        # Forces applied (Fx)
        ctk.CTkLabel(input_frame, text="Forces in X-direction (Fx) [example: 1 2 ]:").grid(row=3, column=1, padx=5, pady=5, sticky=ctk.E)
        self.Fx_entry = ctk.CTkTextbox(input_frame, width=200, height=25)
        self.Fx_entry.grid(row=3, column=2, padx=5, pady=5)

        # Forces applied (Fy)
        ctk.CTkLabel(input_frame, text="Forces in Y-direction (Fy) [example: 1 2 ]:").grid(row=3, column=3, padx=5, pady=5, sticky=ctk.E)
        self.Fy_entry = ctk.CTkTextbox(input_frame, width=200, height=25)
        self.Fy_entry.grid(row=3, column=4, padx=5, pady=5)

        # Elastic modulus (E) single
        ctk.CTkLabel(input_frame, text="For Single Elastic Modulus (E):").grid(row=5, column=1, padx=5, pady=5, sticky=ctk.E)
        ctk.CTkEntry(input_frame, textvariable=self.Es).grid(row=5, column=2, padx=5, pady=5)

        # Cross-sectional area (A) single
        ctk.CTkLabel(input_frame, text="For Single Cross-sectional Area (A):").grid(row=5, column=3, padx=5, pady=5, sticky=ctk.E)
        ctk.CTkEntry(input_frame, textvariable=self.As).grid(row=5, column=4, padx=5, pady=5)

        # Elastic modulus (E)
        ctk.CTkLabel(input_frame, text="Multiple Elastic Modulus (E) [in elements order]:").grid(row=4, column=1, padx=5, pady=5, sticky=ctk.E)
        self.E_entry = ctk.CTkTextbox(input_frame, width=200, height=25)
        self.E_entry.grid(row=4, column=2, padx=5, pady=5)

        # Cross-sectional area (A)
        ctk.CTkLabel(input_frame, text="Multiple Cross-sectional Area (A) [in elements order]:").grid(row=4, column=3, padx=5, pady=5, sticky=ctk.E)
        self.A_entry = ctk.CTkTextbox(input_frame, width=200, height=25)
        self.A_entry.grid(row=4, column=4, padx=5, pady=5)

        # Fixed Points (nb)
        ctk.CTkLabel(input_frame, text="Fixed Points (nb) [example: 1 2 ]:").grid(row=2, column=3, padx=5, pady=5, sticky=ctk.E)
        self.nb_entry = ctk.CTkTextbox(input_frame, width=200, height=25)
        self.nb_entry.grid(row=2, column=4, padx=5, pady=5)

        # Additional note
        ctk.CTkLabel(input_frame, text="Note: If there are multiple E or A, leave the single box 0.0,\nBut if there is only one E or A, leave the multiple box blank.").grid(row=6, column=3, padx=5, pady=5, sticky=ctk.E)

        # Button to calculate
        ctk.CTkButton(input_frame, text="Calculate", command=self.calculate).grid(row=6, column=2, padx=5, pady=10)

        # Button to cancel
        ctk.CTkButton(input_frame, text="Cancel", command=self.abort_calculation).grid(row=6, column=4, padx=5, pady=10)

        # Frame for results
        result_frame = ctk.CTkFrame(self.root)
        result_frame.pack(padx=10, pady=10, fill=ctk.BOTH, expand=True)
        result_frame2 = ctk.CTkFrame(self.root)
        result_frame2.pack(padx=10, pady=10, fill=ctk.BOTH, expand=True)

        # Text widget to display results
        self.result_text = ctk.CTkTextbox(result_frame, width=600, height=175)
        self.result_text.pack(padx=5, pady=5)
        self.result_text2 = ctk.CTkTextbox(result_frame2, width=600, height=150)
        self.result_text2.pack(padx=5, pady=5)

    
    def calculate(self):
        try:
            # Get inputs from the GUI
            self.x_y = np.array([[float(coord) for coord in line.split()] for line in self.coordinates_entry.get("1.0", ctk.END).strip().splitlines()])
            self.lda = np.array([[int(node) for node in line.split()] for line in self.lda_entry.get("1.0", ctk.END).strip().splitlines()])
            nf = self.nf_entry.get("1.0", ctk.END).strip().split()
            self.nf = np.array([int(node) for node in nf])
            Fx = self.Fx_entry.get("1.0", ctk.END).strip().split()
            self.Fx = np.array([float(force) for force in Fx])
            Fy = self.Fy_entry.get("1.0", ctk.END).strip().split()
            self.Fy = np.array([float(force) for force in Fy])
            nb = self.nb_entry.get("1.0", ctk.END).strip().split()
            self.nb = np.array([int(node) for node in nb])

            E = self.E_entry.get("1.0", ctk.END).strip().split()
            self.E = np.array([float(Elastic) for Elastic in E])
            A = self.A_entry.get("1.0", ctk.END).strip().split()
            self.A = np.array([float(Area) for Area in A])
            
            # Calculate truss parameters
            e = self.e.get()
            n = self.n.get()
            x = self.x_y[:, 0]
            y = self.x_y[:, 1]
            lda = self.lda
            nf = self.nf
            Fx = self.Fx
            Fy = self.Fy
            E = self.E
            A = self.A
            Es = self.Es.get()
            As = self.As.get()
            nb = self.nb

            # Perform FEM calculation
            U, Ux, Uy, R, Rx, Ry, sigma = self.perform_fem_calculation(e, n, x, y, lda, nf, Fx, Fy, E, A, Es, As, nb)

            # Display results in the GUI
            self.result_text.delete("1.0", ctk.END)  # Clear previous results
            self.result_text2.delete("1.0", ctk.END)  # Clear previous results

            # Displacements (U) in separate x y
            self.result_text.insert(ctk.END, "Displacements (U) in Each Node:\n")
            for i in range(n):
                self.result_text.insert(ctk.END, f"Nodes {i + 1}:\nDisplacement in x-direction = {Ux[i]:.4f}\nDisplacement in y-direction = {Uy[i]:.4f}\n")

            # Reactions (R) in separate x y
            self.result_text.insert(ctk.END, "\nReactions (R) in Each Node:\n")
            for i in range(n):
                self.result_text.insert(ctk.END, f"Nodes {i + 1}:\nReaction in x-direction = {Rx[i]:.4f}\nReaction in y-direction = {Ry[i]:.4f}\n")

            # Stress in Each Member
            self.result_text2.insert(ctk.END, "Axial Stress in Each Member:\n")
            for i in range(e):
                self.result_text2.insert(ctk.END, f"Element {i + 1} (Nodes {lda[i, 0]} to {lda[i, 1]}): Stress = {sigma[i]:.4f}\n")

            # Plot the deformed truss
            self.plot_deformed_truss(x, y, lda, U, sigma)

        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please check your entries and try again.")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")


    def perform_fem_calculation(self, e, n, x, y, lda, nf, Fx, Fy, E, A, Es, As, nb):
        
        print("\nEntered Truss Parameters:")
        print(f"Number of elements (e): {e}")
        print(f"Number of nodes (n): {n}")
        print(f"Node x-coordinates: {x}")
        print(f"Node y-coordinates: {y}")
        print("Local destination array (lda):")
        for i in range(e):
            print(f"Element {i + 1}: {lda[i, 0]} -> {lda[i, 1]}")
        print(f"Forces applied at nodes (nf): {nf}")
        print(f"Fx values: {Fx}")
        print(f"Fy values: {Fy}")

        # Plot the truss structure with forces display
        self.plot_truss(x, y, lda, nf=nf, Fx=Fx, Fy=Fy)

        F = np.zeros(2 * n)
        for i in range(len(nf)):
            F[2 * nf[i] - 2] = Fx[i]
            F[2 * nf[i] - 1] = Fy[i]

        # Length of truss members (in)
        L = np.zeros(e)
        for i in range(e):
            L[i] = np.sqrt((x[lda[i, 1] - 1] - x[lda[i, 0] - 1]) ** 2 + (y[lda[i, 1] - 1] - y[lda[i, 0] - 1]) ** 2)
        
        # Orientation of truss members (degree)
        theta = np.zeros(e)
        for i in range(e):
            theta[i] = np.degrees(np.arctan2((y[lda[i, 1] - 1] - y[lda[i, 0] - 1]), (x[lda[i, 1] - 1] - x[lda[i, 0] - 1])))
            if theta[i] < 0:
                theta[i] += 180

        # Element stiffness matrices
        ke = np.zeros((4, 4, e))
        for i in range(e):
            if Es == 0 and As == 0 and len(E) > 0 and len(A) > 0:
                k = E[i] * A[i] / L[i]
            elif Es != 0 and As != 0 and len(E) == 0 and len(A) == 0:
                k = Es * As / L[i]
            elif Es != 0 and As == 0 and len(E) == 0 and len(A) > 0:
                k = Es * A[i] / L[i] 
            elif Es == 0 and As != 0 and len(E) > 0 and len(A) == 0:
                k = E[i] * As / L[i]                     
            ke[:, :, i] = k * np.array([[np.cos(np.radians(theta[i])) ** 2, np.sin(np.radians(theta[i])) * np.cos(np.radians(theta[i])),
                                      -np.cos(np.radians(theta[i])) ** 2, -np.sin(np.radians(theta[i])) * np.cos(np.radians(theta[i]))],
                                     [np.sin(np.radians(theta[i])) * np.cos(np.radians(theta[i])), np.sin(np.radians(theta[i])) ** 2,
                                      -np.sin(np.radians(theta[i])) * np.cos(np.radians(theta[i])), -np.sin(np.radians(theta[i])) ** 2],
                                     [-np.cos(np.radians(theta[i])) ** 2, -np.sin(np.radians(theta[i])) * np.cos(np.radians(theta[i])),
                                      np.cos(np.radians(theta[i])) ** 2, np.sin(np.radians(theta[i])) * np.cos(np.radians(theta[i]))],
                                     [-np.sin(np.radians(theta[i])) * np.cos(np.radians(theta[i])), -np.sin(np.radians(theta[i])) ** 2,
                                      np.sin(np.radians(theta[i])) * np.cos(np.radians(theta[i])), np.sin(np.radians(theta[i])) ** 2]])
            
        # Assemble the global stiffness matrix
        K = np.zeros((2 * n, 2 * n))
        for i in range(e):
            K[2 * (lda[i, 0] - 1):2 * (lda[i, 0] - 1) + 2, 2 * (lda[i, 0] - 1):2 * (lda[i, 0] - 1) + 2] += ke[0:2, 0:2, i]
            K[2 * (lda[i, 0] - 1):2 * (lda[i, 0] - 1) + 2, 2 * (lda[i, 1] - 1):2 * (lda[i, 1] - 1) + 2] += ke[0:2, 2:4, i]
            K[2 * (lda[i, 1] - 1):2 * (lda[i, 1] - 1) + 2, 2 * (lda[i, 0] - 1):2 * (lda[i, 0] - 1) + 2] += ke[2:4, 0:2, i]
            K[2 * (lda[i, 1] - 1):2 * (lda[i, 1] - 1) + 2, 2 * (lda[i, 1] - 1):2 * (lda[i, 1] - 1) + 2] += ke[2:4, 2:4, i]

        # Apply displacement boundary condition
        KG = K.copy()  # for keeping the original K matrix to be used for calculating the reactions

        for i in range(len(nb)):
            K[2 * nb[i] - 2, :] = 0
            K[2 * nb[i] - 2, 2 * nb[i] - 2] = 1
            K[2 * nb[i] - 1, :] = 0
            K[2 * nb[i] - 1, 2 * nb[i] - 1] = 1

        # Nodal solution
        U = np.linalg.solve(K, F)

        # Determine the middle index U
        middle_index_U = len(U) // 2

        # Split the matrix U into two parts
        Ux = U[:middle_index_U]
        Uy = U[middle_index_U:]

        # Calculate reaction at the supports
        R = np.dot(KG, U) - F

        # Determine the middle index R
        middle_index_R = len(R) // 2

        # Split the matrix R into two parts
        Rx = R[:middle_index_R]
        Ry = R[middle_index_R:]

        # Local displacement of each member
        u = np.zeros((4, 1, e))
        for i in range(e):
            T = np.array([[np.cos(np.radians(theta[i])), -np.sin(np.radians(theta[i])), 0, 0],
                          [np.sin(np.radians(theta[i])), np.cos(np.radians(theta[i])), 0, 0],
                          [0, 0, np.cos(np.radians(theta[i])), -np.sin(np.radians(theta[i]))],
                          [0, 0, np.sin(np.radians(theta[i])), np.cos(np.radians(theta[i]))]])
            u[:, :, i] = np.dot(np.linalg.inv(T), np.array([[U[2 * (lda[i, 0] - 1)], U[2 * (lda[i, 0] - 1) + 1], U[2 * (lda[i, 1] - 1)], U[2 * (lda[i, 1] - 1) + 1]]]).T)

        # Calculate the stress in each element
        sigma = np.zeros(e)
        for i in range(e):
            if Es == 0 and len(E) > 0 :
                sigma[i] = E[i] * (u[2, 0, i] - u[0, 0, i]) / L[i]
            elif Es != 0 and len(E) == 0 :
                sigma[i] = Es * (u[2, 0, i] - u[0, 0, i]) / L[i]
        
        return U, Ux, Uy, R, Rx, Ry, sigma
    
    
    def plot_truss(self, x, y, lda, nf=None, Fx=None, Fy=None):
        plt.figure()

        # Plot truss elements
        for i in range(len(lda)):
            node1 = lda[i, 0] - 1
            node2 = lda[i, 1] - 1
            plt.plot([x[node1], x[node2]], [y[node1], y[node2]], 'bo-')
        
        # Plot node labels
        node_labels = {}  # Dictionary to store labels to avoid duplication
        for i in range(len(lda)):
            node1 = lda[i, 0] - 1
            node2 = lda[i, 1] - 1
            if node1 not in node_labels:
                plt.text(x[node1], y[node1], f'{node1 + 1}', fontsize=12, ha='right')
                node_labels[node1] = True
            if node2 not in node_labels:
                if i < len(lda) - 1:  # Labels to the right for all elements except the last one
                    plt.text(x[node2], y[node2], f'{node2 + 1}', fontsize=12, ha='right')
                else:  # Labels to the left for the last element
                    plt.text(x[node2], y[node2], f'{node2 + 1}', fontsize=12, ha='left')
                node_labels[node2] = True

        # Plot forces as arrows in front of the truss elements, labeled by Fx and Fy (if either is non-zero)
        if nf is not None and Fx is not None and Fy is not None:
            arrow_length = np.mean(y)  # Adjust the arrow length as needed
            for i in range(len(nf)):
                node_index = nf[i] - 1
                if Fx[i] != 0 or Fy[i] != 0:  # Check if either Fx or Fy is non-zero
                    force_mag = np.sqrt(Fx[i]**2 + Fy[i]**2)  # Magnitude of the force vector
                    if force_mag > 0:
                        scale_factor = arrow_length / force_mag
                        dx = Fx[i] * scale_factor
                        dy = Fy[i] * scale_factor
                        plt.arrow(x[node_index], y[node_index], dx, dy, head_width = (arrow_length / 10), head_length = (arrow_length / 10), fc='r', ec='r', zorder=10, alpha=0.5)
                        # Label the arrow with Fx and Fy values
                        label_text = f'{Fx[i]} ' if Fx[i] != 0 else ''
                        label_text += f'{Fy[i]}' if Fy[i] != 0 else ''
                        plt.text(x[node_index] + dx, y[node_index] + dy, label_text, fontsize=10, ha='center', va='center', color='r')
        
        plt.title('Truss Structure')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.grid(True)
        plt.axis('equal')
        plt.show()


    def plot_deformed_truss(self, x, y, lda, U, sigma):
        plt.figure()
        for i in range(len(lda)):
            xu = [x[lda[i, 0] - 1], x[lda[i, 1] - 1]]  # x-coordinates for undeformed shape
            yu = [y[lda[i, 0] - 1], y[lda[i, 1] - 1]]  # y-coordinates for undeformed shape
            xd = [xu[0] + 200 * U[2 * (lda[i, 0] - 1)], xu[1] + 200 * U[2 * (lda[i, 1] - 1)]]  # x-coordinates for deformed shape
            yd = [yu[0] + 200 * U[2 * (lda[i, 0] - 1) + 1], yu[1] + 200 * U[2 * (lda[i, 1] - 1) + 1]]  # y-coordinates for deformed shape
            plt.plot(xu, yu, '--r')
            plt.plot(xu, yu, 'ob')
            plt.plot(xd, yd, '-k')

        # Add labels
        node_labels = {}  # Dictionary to store labels to avoid duplication
        for i in range(len(lda)):
            node1 = lda[i, 0] - 1
            node2 = lda[i, 1] - 1
            if node1 not in node_labels:
                plt.text(x[node1], y[node1], f'{node1 + 1}', fontsize=12, ha='right')
                node_labels[node1] = True
            if node2 not in node_labels:
                if i < len(lda) - 1:  # Labels to the right for all elements except the last one
                    plt.text(x[node2], y[node2], f'{node2 + 1}', fontsize=12, ha='right')
                else:  # Labels to the left for the last element
                    plt.text(x[node2], y[node2], f'{node2 + 1}', fontsize=12, ha='left')
                node_labels[node2] = True

        # Find the highest and lowest stress values
        max_stress = np.max(sigma)
        min_stress = np.min(sigma)
        max_stress_element = np.argmax(sigma)
        min_stress_element = np.argmin(sigma)

        # Annotate the highest and lowest stress values on the plot
        max_stress_coords = [(x[lda[max_stress_element, 0] - 1], y[lda[max_stress_element, 0] - 1]), 
                            (x[lda[max_stress_element, 1] - 1], y[lda[max_stress_element, 1] - 1])]
        min_stress_coords = [(x[lda[min_stress_element, 0] - 1], y[lda[min_stress_element, 0] - 1]), 
                            (x[lda[min_stress_element, 1] - 1], y[lda[min_stress_element, 1] - 1])]

        plt.text(np.mean([coord[0] for coord in max_stress_coords]), 
                np.mean([coord[1] for coord in max_stress_coords]), 
                f'Stress: {max_stress:.2f}', fontsize=12, color='green')

        plt.text(np.mean([coord[0] for coord in min_stress_coords]), 
                np.mean([coord[1] for coord in min_stress_coords]), 
                f'Stress: {min_stress:.2f}', fontsize=12, color='magenta')

        plt.title('Deformed Truss Structure with Stress Annotations')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.grid(True)
        plt.axis('equal')
        plt.show()


    def abort_calculation(self):
        # Placeholder for aborting the calculation
        if messagebox.askyesno("Abort", "Do you want to abort the calculation?"):
            self.root.destroy()


def main():
    root = ctk.CTk()
    app = FEMSolverApp(root)
    root.mainloop()
    

if __name__ == "__main__":
    main()