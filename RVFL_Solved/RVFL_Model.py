import os
import torch
from typing import Optional
from Internal_Layers import hidden_layer

# Personal Note to create some get functins for general Info, layer Info, etc.

class RVFL:
    """
    Inputs:
        List layersInfo: This contains a list of dicts which are used to define the 
            internal layers of the RVFL

            The data format for each dict is as follows
            {
            int layer_dim: the number of nodes in our layer
            string weight_dist: the distribution for the weight terms
            float weight_var: the variance of the weight distribution
            float gamma_k: the constant used for the gamma distribution statistics
            bool bias_switch: switch for having or not having bias terms
            str bias_dist: the distribution for the bias terms
            float bias_var: the variance of the bias distribution
            }

        Dict GeneralInfo: a dict holding info about seed, device, and data type as follows:
            {
            int seed: the seed used to generate our random numbers
            str device: the device we run this function with
            torch.dtype dtype: the datatype for our tensor elements
            }
            
        String activation: a string denoting which activation function to use
        String linkOption: tells us if we have no link, just directlink, or links for input and every hidden layer. 
            The string is in the set: {"none", "direct", "multi"}
        Float lamb: The constant lambda term used in the ridge regression
        List scalings: The scalings for each layer

    Class Vairables:
        List int_Layers: a list of small 2 term dicts containing the weight 
            and bias terms {W: _, b: _}
        Tensor beta: the solved layer(s) 
    """
    def __init__(self, layersInfo, generalInfo, activation, linkOption, lamb, scalings):
        self.layersInfo = layersInfo
        self.activation = activation
        self.linkOption = linkOption
        self.lamb = lamb
        self.scalings = scalings

        if (len(self.scalings) != len(self.layersInfo)):
            raise IndexError("The number of scalings and number of layers are not equal.")

        if self.linkOption not in {"none", "direct", "multi"}:
            raise ValueError(f"Invalid linkOption: {self.linkOption}")

        self.seed = generalInfo["seed"]
        self.device = generalInfo["device"]
        self.dtype = generalInfo["dtype"]

        self.input_dim = 784 # This is temporairly hard coded for MNIST
        self.output_dim = 10 # This is temporairly hard coded for MNIST

        # Initalize our generator for random numbers using our chosen seed
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.seed)


        self.internal_layers = []
        self.beta = None

    """
    This function creates the hidden layers of our RVFL.

    Inputs:
        void
    Outputs:
        void
    """
    def create_hidden_layers(self):

        if self.internal_layers:
            print("The internal layers have already been created.")
            return

        # This is a changing variable which says what the last layers output dim is (aka the new layers input dim)
        input_dim = self.input_dim

        for layer in self.layersInfo:
            W, b = hidden_layer(
                input_dim,
                layer["layer_dim"],
                self.generator,
                weight_dist = layer["weight_dist"],
                weight_var = layer["weight_var"],
                gamma_k = layer["gamma_k"],
                bias_switch = layer["bias_switch"],
                bias_dist = layer["bias_dist"],
                bias_var= layer["bias_var"],
                device = self.device,
                dtype = self.dtype
            )

            input_dim = layer["layer_dim"]

            d = {"W": W, "b": b}

            self.internal_layers.append(d)
    
    """
    This function applies the activation function elementwise to the data.

    Inputs:
        Tensor X: The input data
    Outputs:
        Tensor activation(X): The data after the activation function is applied.
    """
    def actFunc(self, X):
        if self.activation == "relu":
            return torch.relu(X)
        elif self.activation == "sigmoid":
            return torch.sigmoid(X)
        elif self.activation == "tanh":
            return torch.tanh(X)
        else:
            raise ValueError(f"Invalid activation: {self.activation}")
        
    """
    The actual model function. The data is input and runs forward throuh the RVFL and we get our output.
    
    Double check scalings implementation.

    Inputs:
        Tensor X: The input data
    Outputs:
        Torch RVFL(X): The output data
    """
    def forward(self, X):
        if not self.internal_layers:
            raise RuntimeError("Hidden layers have not been made yet.")
        
        H = X

        if self.linkOption == "multi":
            blocks = [X]

            for i, layer in enumerate(self.internal_layers):
                W = layer["W"]
                b = layer["b"]

                H = H @ W
                if b is not None:
                    H = H + b

                layer_dim = W.shape[1]
                H = (1 / layer_dim**self.scalings[i]) * self.actFunc(H)

                blocks.append(H)

            design_matrix = torch.cat(blocks, dim=1)
            
            return design_matrix
            
        else:
            for i, layer in enumerate(self.internal_layers):
                W = layer["W"]
                b = layer["b"]

                H = H @ W
                if b is not None:
                    H = H + b

                layer_dim = W.shape[1]
                H = (1 / layer_dim**self.scalings[i]) * self.actFunc(H)

            if self.linkOption == "direct":
                return torch.cat([X, H], dim=1)
            else:
                return H
        
    """
    This function solves for our "trained" layers directly. This is called beta in our code and 
    is the direct links, and final layer to output (if there are any direct links).

    Inputs:
        void
    Outputs: 
        void
    """
    def solve_beta(self, Phi, Y):
        # N is number of training samples, p is number of features, m is output size
        p = Phi.shape[1] # Phi is (N,p)
        N = Y.shape[0]   # Y is (N,m)

        if p <= N:
            I_p = torch.eye(p, device=self.device, dtype=self.dtype)
            A = Phi.T @ Phi + self.lamb * I_p
            B = Phi.T @ Y

            self.beta = torch.linalg.solve(A, B) #This solves A @ Beta = B
        elif p > N:
            I_N = torch.eye(N, device=self.device, dtype=self.dtype)
            A = Phi @ Phi.T + self.lamb * I_N
            B = Y
            Z = torch.linalg.solve(A, B)
            self.beta = Phi.T @ Z

    """
    This fits our model architecture (solves for Beta) with MNIST data.

    Inputs:
        void
    Outputs: 
        void
    """
    def fit(self, X_train, y_train):
        self.input_dim = X_train.shape[1]

        if not self.internal_layers:
            self.create_hidden_layers()

        Phi = self.forward(X_train)
        self.solve_beta(Phi, y_train)

    """
    This function uses our model to predict the output for a single input. This is currently hard-coded for MNIST data.

    Inputs:
        Tensor X: The input data point
    Outputs:
        Tensor int score: The number which the model predicts 
    """
    def predict(self, X):
        if self.beta is None:
            raise ValueError("Model has not been fit to our data yet, run fit().")
        
        with torch.no_grad():
            Phi = self.forward(X)
            scores = Phi @ self.beta
            return torch.argmax(scores, dim=1)
    
    """
    This function saves the model to a .pt file.

    Inputs:
        void
    Outputs: 
        void
    """
    def saveModel(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(base_dir, "model_saves")
        os.makedirs(save_dir, exist_ok=True)

        layer_dims = "-".join(str(layer["layer_dim"]) for layer in self.layersInfo)
        weight_dists = "-".join(str(layer["weight_dist"]) for layer in self.layersInfo)
        weight_vars = "-".join(str(layer["weight_var"]) for layer in self.layersInfo)
        bias_switches = "-".join(str(layer["bias_switch"]) for layer in self.layersInfo)
        bias_dists = "-".join(str(layer["bias_dist"]) for layer in self.layersInfo)
        bias_vars = "-".join(str(layer["bias_var"]) for layer in self.layersInfo)
        scales = "-".join(str(scale) for scale in self.scalings)

        filename = (
        f"RVFL"
        f"_seed-{self.seed}"
        f"_act-{self.activation}"
        f"_link-{self.linkOption}"
        f"_lam-{self.lamb:.0e}"
        f"_dims-{layer_dims}"
        f"_wdist-{weight_dists}"
        f"_wvar-{weight_vars}"
        f"_bias-{bias_switches}"
        f"_bdist-{bias_dists}"
        f"_bvar-{bias_vars}"
        f"_scale-{scales}"
    )
        
        # Only include gamma_k info if gamma distribution is actually used
        if any(layer["weight_dist"] == "gamma" for layer in self.layersInfo):
            gamma_ks = "-".join(
                str(layer["gamma_k"]) if layer["weight_dist"] == "gamma" else "NA"
                for layer in self.layersInfo
            )
            filename += f"_gk-{gamma_ks}"

        filename += ".pt"

        path = os.path.join(save_dir, filename)
        torch.save({
            "layersInfo": self.layersInfo,
            "activation": self.activation,
            "linkOption": self.linkOption,
            "lamb": self.lamb,
            "scalings": self.scalings,
            "seed": self.seed,
            "device": self.device,
            "dtype": self.dtype,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "internal_layers": self.internal_layers,
            "beta": self.beta
        }, path)

        print("Model saved to path: " + path)
        return filename

    """
    This function is used to load in the saved data easily with the new dict save format.

    Input:
        string path: Path to the model data you wish to load in
    Output:
        RVFL model: The loaded in RVFL object.
    """
    @classmethod
    def loadModel(cls, path):
        loadedModelDict = torch.load(path)

        generalInfo = {
            "seed": loadedModelDict["seed"],
            "device": loadedModelDict["device"],
            "dtype": loadedModelDict["dtype"]
        }

        model = cls(
            layersInfo=loadedModelDict["layersInfo"],
            generalInfo=generalInfo,
            activation=loadedModelDict["activation"],
            linkOption=loadedModelDict["linkOption"],
            lamb=loadedModelDict["lamb"],
            scalings=loadedModelDict["scalings"]
        )

        model.input_dim = loadedModelDict["input_dim"]
        model.output_dim = loadedModelDict["output_dim"]
        model.internal_layers = loadedModelDict["internal_layers"]
        model.beta = loadedModelDict["beta"]

        return model
    
