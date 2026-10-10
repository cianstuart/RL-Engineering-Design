from dataclasses import dataclass
import numpy as np
import parameters as default_params

@dataclass
class FEMResult:
    """
    Dataclass to hold the displacements, stresses, and compliance resulting from the truss analysis.
    """
    displacements: np.ndarray
    stresses: np.ndarray
    compliance: float


class Truss2DConstructive:
    """
    Solver for the 2D constructive truss problem using the finite element method (FEM).
    Design:
      Node 0: Bottom-left fixed support (0,0)
      Node 1: Top-left fixed support (0,H)
      Node 2: Bottom-right tip node (L,0) subject to downward force F
      Node 3: First node (X1,Y1) controlled by the RL agent
      Node 4: Second node (X2,Y2) controlled by the RL agent
    Material properties: 
      E: Young's modulus
      A: Cross-sectional area
    """
    def __init__(self, params=default_params):
        self.L = params.L
        self.H = params.H
        self.E = params.E
        self.A = params.A
        self.F_tip = params.F_TIP

        # Degrees of freedom (DOFs) for each node: [x, y]
        # Nodes 0 and 1 are fixed
        # Nodes 2,3, and 4 are free
        self.fixed_dofs = np.array([0, 1, 2, 3]) 
        self.free_dofs = np.array([4, 5, 6, 7, 8, 9])  

    def get_nodes(self, pos1, pos2):
        """
        Returns the coordinates of the nodes based on the positions of the first and second nodes. 
        """
        nodes = np.array([
            [0.0, 0.0],          
            [0.0, self.H],     
            [self.L, 0.0],       
            [pos1[0], pos1[1]], 
            [pos2[0], pos2[1]]
        ], dtype=float)
        return nodes

    def transformation_matrix(self, node_start, node_end):
        """
        Computes the transformation matrix for a bar element between two nodes and return the length of the bar.
        """
        x1, y1 = node_start
        x2, y2 = node_end
        L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        if L < 1e-12: # Filter out 'zero-length' bars before division
            return None, L

        c = (x2 - x1) / L
        s = (y2 - y1) / L
        T = np.array([ -c,  -s, c, s])

        return T, L
    
    def solve(self, pos1, pos2, connections):
        """
        Solves the truss problem for given node positions and bar connections.
        Returns a dataclass containing the displacements of the nodes and the stresses in the bars.
        """
        # Construct the global stiffness matrix
        nodes = self.get_nodes(pos1, pos2)
        num_dofs = 10  
        K = np.zeros((num_dofs, num_dofs))

        # construct invalid result
        invalid = FEMResult(np.full(num_dofs, np.nan), np.full(len(connections), np.nan), np.nan)

        for i, (start_node, end_node) in enumerate(connections):
            T, L = self.transformation_matrix(nodes[start_node], nodes[end_node])
            if not np.isfinite(L) or L < 1e-6:  # filter out 'zero-length' bars
                return invalid
            
            k_local = (self.E * self.A / L) * np.outer(T, T)
            K[np.ix_([start_node*2, start_node*2+1, end_node*2, end_node*2+1],
                     [start_node*2, start_node*2+1, end_node*2, end_node*2+1])] += k_local

        # Construct the force vector
        F = np.zeros(num_dofs)
        F[5] = -self.F_tip  


        # Apply boundary conditions by removing fixed DOFs
        K_free = K[np.ix_(self.free_dofs, self.free_dofs)]
        F_free = F[self.free_dofs]

        # Check for unconnected nodes
        if np.linalg.matrix_rank(K_free) < len(self.free_dofs):
            return invalid       

        # Solve for displacements at free DOFs
        try:
            displacements_free = np.linalg.solve(K_free, F_free)
        except np.linalg.LinAlgError:
            return invalid

        displacements = np.zeros(num_dofs)
        displacements[self.free_dofs] = displacements_free
        
        # Calculate compliance
        compliance = np.dot(F_free, displacements_free)

        # Calculate stresses in each bar
        stresses = np.zeros(len(connections))
        for i, (start_node, end_node) in enumerate(connections):
            u_local = np.array([displacements[start_node*2], displacements[start_node*2+1],
                                displacements[end_node*2], displacements[end_node*2+1]])
            T, L = self.transformation_matrix(nodes[start_node], nodes[end_node])
            stress = (self.E / L) * (T @ u_local)
            stresses[i] = stress

        return FEMResult(displacements=displacements, stresses=stresses, compliance=compliance)