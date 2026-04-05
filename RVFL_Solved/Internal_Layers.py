import torch
from typing import Optional

"""
This function will generate one internal hidden layer.

Inputs:
    int input_dim: the input data dimension
    int layer_dim: the number of nodes in our layer
    int seed: the seed used to generate our random numbers
    string weight_dist: the distribution for the weight terms
    float weight_var: the variance of the weight distribution
    float gamma_k: the constant used for the gamma distribution statistics
    bool bias_switch: switch for having or not having bias terms
    str bias_dist: the distribution for the bias terms
    float bias_var: the variance of the bias distribution
    str device: the device we run this function with
    torch.dtype dtype: the datatype for our tensor elements
Outputs:
    tensor W: Returns our tensor of weights for the layer
    tensor b: Returns our tensor of biases for the layer
"""  
def hidden_layer(input_dim: int,
                 layer_dim: int,
                 gen: torch.Generator,
                 *,
                 weight_dist: str = "normal",
                 weight_var: float = 1.0,
                 gamma_k: Optional[float] = None,
                 bias_switch: bool = False,
                 bias_dist: str = "normal",
                 bias_var: float = 1.0,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float64):
    
    W = sample_matrix(
        (input_dim, layer_dim),
        dist=weight_dist,
        var=weight_var,
        gammaK=gamma_k,
        generator=gen,
        device=device,
        dtype=dtype,
    )

    if bias_switch:
        b = sample_matrix(
            (layer_dim,),
            dist=bias_dist,
            var=bias_var,
            gammaK=gamma_k,
            generator=gen,
            device=device,
            dtype=dtype,
        )
    else:
        b = None

    return W, b
    
"""
Sample a random matrix of given shape from a distribution with target variance. Note that mean is set to 0 for each distribution

Inputs:
    tuple shape: shape of matrix, (rows,cols)
    string dist: ;must be 'normal' | 'uniform' | 'gamma'
    float var: variance (needs to be >0).
    float gammaK: Used as the parameter to define the gamma distruibution
    Generator generator: The random number generator
    str device: What device to run it on. cpu vs. gpu
    dtype dtype: just tells us what data type to use for our output
Outputs:
    tensor g: A randomly sampled tensor according to inputs
"""
def sample_matrix(shape: tuple[int, ...], 
                  dist: str, 
                  var: float, 
                  *,
                  gammaK: Optional[float] = None,
                  generator: Optional[torch.Generator] = None,
                  device: str = "cpu",
                  dtype: torch.dtype = torch.float64):
    
    if var <= 0:
        raise ValueError("Variance of random matrix must be positive and nonzero")
    
    if dist == "normal":
        std = var ** 0.5
        return torch.randn(
            shape, generator=generator, device=device, dtype=dtype) * std

    elif dist == 'uniform':
        # For Uniform(-a, a), variance = a^2 / 3  => a = sqrt(3*var)
        a = (3.0 * var) ** 0.5
        return (2.0 * torch.rand(
            shape, generator=generator, device=device, dtype=dtype) - 1.0) * a      

    elif dist == 'gamma':
        # Gamma(k, theta) has mean k*theta, var k*theta^2.
        if gammaK is None or gammaK <= 0:
            raise ValueError("Need a positive k parameter for Gamma distribution")

        theta = (var / gammaK) ** 0.5

        gamma_dist = torch.distributions.Gamma(
            concentration=torch.tensor(gammaK, device=device, dtype=dtype),
            rate=torch.tensor(1.0 / theta, device=device, dtype=dtype),
        )

        g = gamma_dist.sample(shape)
        g = g - (gammaK * theta)  # center to zero-mean
        return g.to(dtype=dtype, device=device)

    else:
        raise ValueError(f"Invalid distribution: {dist}")
