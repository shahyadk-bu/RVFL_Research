import torch
import torch.nn as nn


class OrthogonalParameters(nn.Module):
    """
    Directly stores a trainable orthogonal matrix Q. This matrix is projected back into its orthogonal state after each training step to ensure 
    the RVFLs hidden layer remains a sample of the Normal Distribution.
    """

    def __init__(self, d, device, dtype, init="identity", generator=None):
        super().__init__()

        self.d = d

        if init == "identity":
            Q = torch.eye(d, device=device, dtype=dtype)

        elif init == "random":
            A = torch.randn(d, d, device=device, dtype=dtype, generator=generator)
            Q, R = torch.linalg.qr(A)

            # Make QR sign convention stable
            signs = torch.sign(torch.diagonal(R))
            signs[signs == 0] = 1
            Q = Q * signs.unsqueeze(0)

        else:
            raise ValueError(f"Invalid inital orthogonal matrix: {init}")

        self.Q = nn.Parameter(Q)

    """
    Returns the rotation matrix.

    Inputs:
        None
    Outputs:
        self.Q: The rotation matrix
    """
    def build_matrix(self):
        return self.Q

    """
    Applies the rotation to the input weight matrix.

    Inputs:
        W: weight matrix
    Outputs:
        WQ = rotated weight matrix
    """
    def forward(self, W: torch.Tensor):
        return W @ self.Q

    """
    Project Q back onto the orthogonal group using QR, then sets self.Q to the orthogonal projection.

    Inputs:
        None
    Outputs:
        None
    """
    @torch.no_grad()
    def project(self):
        Q, R = torch.linalg.qr(self.Q)

        signs = torch.sign(torch.diagonal(R))
        signs[signs == 0] = 1

        Q = Q * signs.unsqueeze(0)

        self.Q.copy_(Q)