import torch
import torch.nn as nn

"""
Generate all index pairs (i, j) with 0 <= i < j < m for construction of Givens Unitary

Inputs:
    int m: degrees of freedom for rotations (maxed out at N where matrix size is N x N)
Outputs:
    list pairs: List of pairings as can be seen here: [(i1,j1), (i2,j2), ...]
"""
def generate_pairs(m):
    pairs = []

    for i in range(m):
        for j in range(i + 1, m):
            pairs.append((i, j))

    return pairs

class GParameters(nn.Module):

    def __init__(self, m, device=None, dtype=torch.float64):
        super().__init__()
        self.m = m
        self.device = device
        self.dtype = dtype

        self.pairs = generate_pairs(self.m)
        
        num_angles = len(self.pairs)
        self.theta = nn.Parameter(
            torch.zeros(num_angles, device=device, dtype=dtype)
        )

    """
    This function applys a Givens formed Orthogonal Matrix to the input matrix by right multiplication MU

    Inputs:
        Tensor mat: Matrix to multiply the unitary with
    Outputs:
        Tensor out: Matrix after the multiplication, MU
    """
    def forward(self, mat: torch.Tensor):

        out = mat.clone()

        for k, (i, j) in enumerate(self.pairs):
            angle = self.theta[k]
            c = torch.cos(angle)
            s = torch.sin(angle)

            col_i = out[:, i].clone()
            col_j = out[:, j].clone()

            out[:, i] = c * col_i - s * col_j
            out[:, j] = s * col_i + c * col_j

        return out
    
    """
        This function builds and returns the explicit orthogonal matrix Q of shape (m, m).
        
        Inputs:
            void
        Outputs:
            Q: The orthogonal matrix of built from our models final Givens Parameters.
    """
    def build_matrix(self):
        Q = torch.eye(self.m, device=self.theta.device, dtype=self.theta.dtype)

        for k, (i, j) in enumerate(self.pairs):
            angle = self.theta[k]
            c = torch.cos(angle)
            s = torch.sin(angle)

            col_i = Q[:, i].clone()
            col_j = Q[:, j].clone()

            Q[:, i] = c * col_i - s * col_j
            Q[:, j] = s * col_i + c * col_j

        return Q