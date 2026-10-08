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


class Truss2DParametric:
    """
    Solver for the 2D parametric truss problem using the finite element method (FEM).
    Design:
      Node 0: Bottom-left fixed support (0,0)
      Node 1: Top-left fixed support (0,H)
      Node 2: Bottom-right tip node (L,0) subject to downward force F
      Node 3: Middle node (X,Y) controlled by the RL agent
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

        # Connections between nodes: 
        # Bar 0: Node 0 to Node 3 (BL to M)
        # Bar 1: Node 1 to Node 3 (TL to M)
        # Bar 2: Node 0 to Node 2 (BL to BR)
        # Bar 3: Node 2 to Node 3 (BR to M)
        self.connections = np.array([(0, 3), (1, 3), (0, 2), (2, 3)], dtype=int)

        # Degrees of freedom (DOFs) for each node: [x, y]
        # Nodes 0 and 1 are fixed
        # Nodes 2 and 3 are free
        self.fixed_dofs = np.array([0, 1, 2, 3]) 
        self.free_dofs = np.array([4, 5, 6, 7])  

    def get_nodes(self, x_middle, y_middle):
        """
        Returns the coordinates of the nodes based on the middle node position.
        """
        nodes = np.array([
            [0.0, 0.0],          
            [0.0, self.H],     
            [self.L, 0.0],       
            [x_middle, y_middle] 
        ], dtype=float)
        return nodes

    def transformation_matrix(self, node_start, node_end):
        """
        Computes the transformation matrix for a bar element between two nodes and return the length of the bar.
        """
        x1, y1 = node_start
        x2, y2 = node_end
        L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        c = (x2 - x1) / L
        s = (y2 - y1) / L
        T = np.array([ -c,  -s, c, s])

        return T, L
    
    def solve(self, x_middle, y_middle):
        """
        Solves the truss problem for a given middle node position (x_middle, y_middle).
        Returns a dataclass containing the displacements of the nodes and the stresses in the bars.
        """
        # Construct the global stiffness matrix
        nodes = self.get_nodes(x_middle, y_middle)
        num_dofs = 8  
        K = np.zeros((num_dofs, num_dofs))

        for i, (start_node, end_node) in enumerate(self.connections):
            T, L = self.transformation_matrix(nodes[start_node], nodes[end_node])
            k_local = (self.E * self.A / L) * np.outer(T, T)
            K[np.ix_([start_node*2, start_node*2+1, end_node*2, end_node*2+1],
                     [start_node*2, start_node*2+1, end_node*2, end_node*2+1])] += k_local

        # Construct the force vector
        F = np.zeros(num_dofs)
        F[5] = -self.F_tip  

        # Apply boundary conditions by removing fixed DOFs
        K_free = K[np.ix_(self.free_dofs, self.free_dofs)]
        F_free = F[self.free_dofs]

        # Solve for displacements at free DOFs
        try:
            displacements_free = np.linalg.solve(K_free, F_free)
        except np.linalg.LinAlgError:
            return FEMResult(displacements=np.full(num_dofs, np.nan), stresses=np.full(len(self.connections), np.nan), compliance=np.nan)

        displacements = np.zeros(num_dofs)
        displacements[self.free_dofs] = displacements_free
        
        # Calculate compliance
        compliance = np.dot(F_free, displacements_free)

        # Calculate stresses in each bar
        stresses = np.zeros(len(self.connections))
        for i, (start_node, end_node) in enumerate(self.connections):
            u_local = np.array([displacements[start_node*2], displacements[start_node*2+1],
                                displacements[end_node*2], displacements[end_node*2+1]])
            T, L = self.transformation_matrix(nodes[start_node], nodes[end_node])
            stress = (self.E / L) * (T @ u_local)
            stresses[i] = stress

        return FEMResult(displacements=displacements, stresses=stresses, compliance=compliance)