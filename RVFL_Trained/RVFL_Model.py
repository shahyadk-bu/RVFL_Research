import torch
import torch.nn as nn
from Utility_Functions import sample_matrix

class RandomFeatureLayer(nn.Module):
    """
    Frozen linear layer with random weights/bias.
    The weights are sampled once at initalization from a specified distribution and variance.
    """
    def __init__(self, input_size, outputLayer_size,
                 dist='normal', var=1.0, gammaK=None,
                 bias=False, bias_dist='normal', bias_var=1.0, bias_gammaK=None,
                 activation=torch.sigmoid,
                 gamma=0.0):
        super().__init__()
        self.input_size = input_size
        self.outputLayer_size = outputLayer_size
        self.dist = dist
        self.var = float(var)
        self.bias = bias
        self.bias_dist = bias_dist
        self.bias_var = float(bias_var)
        self._act = activation
        self.gamma = float(gamma)

        self.register_buffer('W', sample_matrix((input_size, outputLayer_size),
                                                dist, self.var, gammaK))

        if bias:
            self.register_buffer('b', sample_matrix((outputLayer_size,), bias_dist,
                                                    self.bias_var, bias_gammaK))
        else:
            self.register_buffer('b', torch.zeros((outputLayer_size,), dtype=self.W.dtype))

        self.W.requires_grad_(False)
        self.b.requires_grad_(False)

        self.register_buffer("scale",
                torch.tensor(self.outputLayer_size ** (-self.gamma),dtype=self.W.dtype))

    def forward(self, x):
        # x: (N, input_size) because we do batching
        z = x @ self.W + self.b
        h = self._act(z)
        h = self.scale * h
        return h


class RVFLReadout(nn.Module):
    """
    Trainable linear readout (logits).
    Here we keep a weight matrix 'Beta' and bias 'c' that can be set via 'set_weights'.
    """
    def __init__(self, feature_dim, output_size):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_size = output_size
        self.Beta = nn.Parameter(torch.zeros(self.feature_dim, output_size), requires_grad=True)
        self.c = nn.Parameter(torch.zeros(output_size), requires_grad=True)

    @torch.no_grad()
    def set_weights(self, Beta, c=None):
        self.Beta.copy_(Beta)
        if c is not None:
            self.c.copy_(c)

    def forward(self, H):
        logits = H @ self.Beta + self.c
        return logits

class RVFL3(nn.Module):
    """
    RVFL with three random hidden layers.
    """
    def __init__(self,
                 hidden_units_1, hidden_units_2, hidden_units_3,
                 gamma_1=0.0, gamma_2=0.0, gamma_3=0.0, gamma_link = 0.0, gamma_out = 0.0,
                 dist1='normal', var1=1.0, gammaK1 = None,
                 dist2='normal', var2=1.0, gammaK2 = None,
                 dist3='normal', var3=1.0, gammaK3 = None,
                 bias=True, bias_dist='normal', bias_var=1.0, bias_gammaK = None,
                 activation=torch.sigmoid,
                 output_size=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.in_dim = 784  # 28x28
        self.outerHiddenUnits = hidden_units_3
        self.gamma_link = gamma_link
        self.gamma_out = gamma_out
        self.register_buffer("x_scale", torch.tensor(self.in_dim ** (-self.gamma_link)))
        self.register_buffer("h_scale", torch.tensor(self.outerHiddenUnits ** (-self.gamma_out)))


        self.rf1 = RandomFeatureLayer(self.in_dim, hidden_units_1, dist=dist1, var=var1, gammaK=gammaK1,
                                      bias=bias, bias_dist=bias_dist, bias_var=bias_var, bias_gammaK=bias_gammaK,
                                      activation=activation, gamma=gamma_1)
        self.rf2 = RandomFeatureLayer(hidden_units_1, hidden_units_2, dist=dist2, var=var2, gammaK=gammaK2,
                                      bias=bias, bias_dist=bias_dist, bias_var=bias_var, bias_gammaK=bias_gammaK,
                                      activation=activation, gamma=gamma_2)
        self.rf3 = RandomFeatureLayer(hidden_units_2, hidden_units_3, dist=dist3, var=var3, gammaK=gammaK3,
                                      bias=bias, bias_dist=bias_dist, bias_var=bias_var, bias_gammaK=bias_gammaK,
                                      activation=activation, gamma=gamma_3)

        self.readout = RVFLReadout(self.in_dim + self.outerHiddenUnits, output_size)

    def features(self, x):
        x = self.flatten(x)
        h1 = self.rf1(x)
        h2 = self.rf2(h1)
        h3 = self.rf3(h2)

        x_scaled = self.x_scale * x
        h3_scaled = self.h_scale * h3

        H = torch.cat([x_scaled, h3_scaled], dim=1)

        return H


    def forward(self, x):
        H = self.features(x)
        return self.readout(H)


if __name__ == "__main__":



    print("Ok")
