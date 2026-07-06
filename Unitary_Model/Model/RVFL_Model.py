import os
import torch
from typing import Optional
from RVFL_Research.Unitary_Model.Model.Internal_Layers import hidden_layer
from RVFL_Research.Unitary_Model.Model.Givens_Parameters import GParameters
from RVFL_Research.Unitary_Model.Model.OrthogonalParams import OrthogonalParameters
import torch.nn as nn
import torch.nn.functional as F
import time

# Personal Note to create some get functins for general Info, layer Info, etc.

class RVFL(nn.Module):
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
            unitary_init: the inital unitary state before training
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
    def __init__(
    self,
    layersInfo,
    generalInfo,
    orthoMatMethod,
    activation,
    linkOption,
    lamb,
    scalings,
    input_dim=None,
    output_dim=None,
    task="classification",
    ):
        super().__init__()

        self.layersInfo = layersInfo
        self.activation = activation
        self.linkOption = linkOption
        self.lamb = lamb
        self.scalings = scalings
        self.orthoMatMethod = orthoMatMethod

        if (len(self.scalings) != len(self.layersInfo)):
            raise IndexError("The number of scalings and number of layers are not equal.")

        if self.linkOption not in {"none", "direct", "multi"}:
            raise ValueError(f"Invalid linkOption: {self.linkOption}")

        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'.")

        self.seed = generalInfo["seed"]
        self.device = generalInfo["device"]
        self.dtype = generalInfo["dtype"]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task = task

        # Initalize our generator for random numbers using our chosen seed
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.seed)

        self.internal_layers = []
        self.beta = None

        # Initalize the parameters for training
        self.unitaryParams = nn.ModuleList([])

        if self.orthoMatMethod not in {"givens", "qr"}:
            raise ValueError(f"Invalid orthoMatMethod: {self.orthoMatMethod}")

        for layer in layersInfo:

            if self.orthoMatMethod == "givens":
                self.unitaryParams.append(
                    GParameters(
                        layer["layer_dim"],
                        self.device,
                        self.dtype,
                    )
                )

            elif self.orthoMatMethod == "qr":
                self.unitaryParams.append(
                    OrthogonalParameters(
                        layer["layer_dim"],
                        self.device,
                        self.dtype,
                        init=layer.get("unitary_init", "identity"),
                        generator=self.generator,
                    )
                )

        self.eye_cache = {}

    """
        Converts input data to a 2D tensor of shape (N, d).

        This allows:
            tabular data: (N, d)
            single point: (d,)
            images: (N, H, W) or (N, C, H, W)

        Inputs:
            Tensor X: Data
    """
    def _as_feature_tensor(self, X):
        if torch.is_tensor(X):
            X = X.to(device=self.device, dtype=self.dtype)
        else:
            X = torch.as_tensor(X, device=self.device, dtype=self.dtype)

        if X.ndim == 1:
            X = X.reshape(1, -1)
        elif X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        if X.ndim != 2:
            raise ValueError(f"Expected X to become shape (N, d), but got {tuple(X.shape)}.")

        return X

    def _set_or_check_input_dim(self, X):
        input_dim = X.shape[1]

        if self.input_dim is None:
            self.input_dim = input_dim
        elif self.input_dim != input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, but got data with {input_dim} features."
            )

    def _prepare_targets(self, y, *, fit_output_dim=False):
        """
        For classification:
            accepts integer labels of shape (N,)
            or one-hot labels of shape (N, C)

        For regression:
            accepts targets of shape (N,) or (N, m)

        Returns:
            Y: target matrix used in ridge regression
            labels: integer labels for accuracy, or None for regression
        """
        if torch.is_tensor(y):
            y_tensor = y.to(device=self.device)
        else:
            y_tensor = torch.as_tensor(y, device=self.device)

        if self.task == "classification":
            if y_tensor.ndim == 1:
                labels = y_tensor.to(dtype=torch.long)
                inferred_output_dim = int(labels.max().item()) + 1

                if fit_output_dim or self.output_dim is None:
                    self.output_dim = inferred_output_dim
                elif inferred_output_dim > self.output_dim:
                    raise ValueError(
                        f"Labels require at least {inferred_output_dim} classes, "
                        f"but output_dim={self.output_dim}."
                    )

                Y = F.one_hot(labels, num_classes=self.output_dim).to(
                    device=self.device,
                    dtype=self.dtype,
                )
                return Y, labels

            if y_tensor.ndim == 2:
                Y = y_tensor.to(device=self.device, dtype=self.dtype)
                labels = torch.argmax(Y, dim=1).to(dtype=torch.long)

                if fit_output_dim or self.output_dim is None:
                    self.output_dim = Y.shape[1]
                elif Y.shape[1] != self.output_dim:
                    raise ValueError(
                        f"Expected y to have {self.output_dim} columns, got {Y.shape[1]}."
                    )

                return Y, labels

            raise ValueError("Classification targets must have shape (N,) or (N, C).")

        # Regression case
        Y = y_tensor.to(device=self.device, dtype=self.dtype)

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        if Y.ndim != 2:
            raise ValueError("Regression targets must have shape (N,) or (N, m).")

        if fit_output_dim or self.output_dim is None:
            self.output_dim = Y.shape[1]
        elif Y.shape[1] != self.output_dim:
            raise ValueError(
                f"Expected y to have {self.output_dim} columns, got {Y.shape[1]}."
            )

        return Y, None

    """
    Returns the cached identity matrix.

    Inputs:
        p: the dimension of the identity matrix (i.e. a p by p matrix)
    Outputs:
        self.eye_cache[p]: the identity matrix
    """
    def get_eye(self, p):
        if p not in self.eye_cache:
            self.eye_cache[p] = torch.eye(
                p,
                device=self.device,
                dtype=self.dtype
            )
        return self.eye_cache[p]

    """
    This function creates the hidden layers of our RVFL.

    Inputs:
        void
    Outputs:
        void
    """
    def create_hidden_layers(self, input_dim=None):

        if self.internal_layers:
            print("The internal layers have already been created.")
            return

        if input_dim is not None:
            if self.input_dim is not None and self.input_dim != input_dim:
                raise ValueError(
                    f"Model input_dim is already {self.input_dim}, cannot reset it to {input_dim}."
                )
            self.input_dim = input_dim

        if self.input_dim is None:
            raise ValueError(
                "input_dim is unknown. Pass input_dim to RVFL(...)."
            )

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

    Inputs:
        Tensor X: The input data
    Outputs:
        Torch RVFL(X): The output data
    """
    def forward(self, X, Z=None):
        if not self.internal_layers:
            raise RuntimeError("Hidden layers have not been made yet.")
        
        if (len(self.internal_layers) == 1) and Z is not None:
            return self.forward_from_precomputed_XW(X, Z)
        
        H = X

        if self.linkOption == "multi":
            blocks = [X]

            for i, layer in enumerate(self.internal_layers):
                W = layer["W"]
                b = layer["b"]

                W_rotated = self.unitaryParams[i](W) # Calls the unitaryParams forward method which applied the rotation to W
                H = H @ W_rotated

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

                W_rotated = self.unitaryParams[i](W) # Calls the unitaryParams forward method which applied the rotation to W
                H = H @ W_rotated
                
                if b is not None:
                    H = H + b

                layer_dim = W.shape[1]
                H = (1 / layer_dim**self.scalings[i]) * self.actFunc(H)

            if self.linkOption == "direct":
                return torch.cat([X, H], dim=1)
            else:
                return H

    """
    This function outputs X @ W so that we do not keep recomputing it each epoch in the one-layer case.

    Inputs:
        Tensor X: data
    Outputs: 
        Tensor Z: X @ W
    """      
    def precompute_one_layer_XW(self, X):
        if len(self.internal_layers) != 1:
            raise ValueError("precompute_one_layer_XW is only for one-layer models.")

        W = self.internal_layers[0]["W"]

        with torch.no_grad():
            return X @ W

    """
    This is used in the forward function for the one-layer case using the precomputed XW.

    Inputs:
        Tensor X: data
        Tensor Z: X @ W
    Outputs: 
        Tensor RVFL(X): Output data after forward through RVFL
    """ 
    def forward_from_precomputed_XW(self, X, Z):
        layer = self.internal_layers[0]
        b = layer["b"]

        H = self.unitaryParams[0](Z)

        if b is not None:
            H = H + b

        layer_dim = layer["W"].shape[1]
        H = (1 / layer_dim**self.scalings[0]) * self.actFunc(H)

        if self.linkOption in {"direct", "multi"}:
            return torch.cat([X, H], dim=1)
        elif self.linkOption == "none":
            return H
        else:
            raise ValueError(f"Invalid linkOption: {self.linkOption}")

    """
    This function solves for our "trained" layers directly. This is called beta in our code and 
    is the direct links, and final layer to output (if there are any direct links).

    Inputs:
        Tensor Phi: 
        Tensor Y: The target data
    Outputs: 
        Tensor beta: The solved beta layer(s)
    """
    def solve_beta(self, Phi, Y):
        # N is number of training samples, p is number of features, m is output size
        p = Phi.shape[1] # Phi is (N,p)
        N = Y.shape[0]   # Y is (N,m)

        if p <= N:
            I_p = self.get_eye(p)
            A = Phi.T @ Phi + self.lamb * I_p
            B = Phi.T @ Y

            return torch.linalg.solve(A, B) #This solves A @ Beta = B
        elif p > N:
            I_N = self.get_eye(N)
            A = Phi @ Phi.T + self.lamb * I_N
            B = Y
            Z = torch.linalg.solve(A, B)
            return Phi.T @ Z

    """
    This function just sets self.beta to the solved beta.

    Inputs:
        void
    Outputs:
        void
    """
    def set_beta(self, Phi, Y):
        self.beta = self.solve_beta(Phi, Y)

    """
    This fits our model architecture (solves for Beta). Note we for this model only use MNIST data as of right now.

    Inputs:
        X_train: The training input data
        y_train: The training target data
    Outputs: 
        void
    """
    def fit(self, X_train, y_train):
        X_train = self._as_feature_tensor(X_train)
        self._set_or_check_input_dim(X_train)
        Y_train, _ = self._prepare_targets(y_train, fit_output_dim=True)

        if len(X_train) != len(Y_train):
            raise ValueError(
                f"X_train and y_train have different lengths: {len(X_train)} vs {len(Y_train)}."
            )

        if not self.internal_layers:
            self.create_hidden_layers()

        Phi = self.forward(X_train)
        self.set_beta(Phi, Y_train)

        return self

    def predict_scores(self, X):
        if self.beta is None:
            raise ValueError("Model has not been fit to data yet, run fit().")

        X = self._as_feature_tensor(X)
        self._set_or_check_input_dim(X)

        with torch.no_grad():
            Phi = self.forward(X)
            return Phi @ self.beta

    """
    This function uses our model to predict the output for a single input. This is currently hard-coded for MNIST data.

    Inputs:
        Tensor X: The input data point
    Outputs:
        Tensor int score: The number which the model predicts 
    """
    # def predict(self, X):
    #     if self.beta is None:
    #         raise ValueError("Model has not been fit to our data yet, run fit().")
        
    #     with torch.no_grad():
    #         Phi = self.forward(X)
    #         scores = Phi @ self.beta
    #         return torch.argmax(scores, dim=1)

    def predict(self, X):
        scores = self.predict_scores(X)

        if self.task == "classification":
            return torch.argmax(scores, dim=1)

        return scores
    
    """
    This function finds the loss of our model.

    Inputs:
        Tensor Phi: 
        Tensor Y:
    Outputs:
        Tensor int score: The number which the model predicts 
    """
    def loss(self, Phi, Y, beta=None):

        if beta is None:
            beta = self.solve_beta(Phi, Y)

        preds = Phi @ beta
        return F.mse_loss(preds, Y)

    """
    This function evaluates the accuracy of our model and returns the loss we get from our chosen loss function.

    Inputs:
        Tensor X: Input data
        Tensor y_onehot: The one-hot encoded labels used to find loss
        Tensor y_labels: The normal labels used to find accuracy
    Outputs:
        float loss_val: The loss found from the loss functions
        accuracy: The accuracy of our model against true labels
    """
    # def evaluate(self, X, y_onehot, y_labels):
    #     if self.beta is None:
    #         raise ValueError("Model has not been fit yet, self.beta is None.")

    #     with torch.no_grad():
    #         Phi = self.forward(X)
    #         preds = Phi @ self.beta
    #         loss_val = F.mse_loss(preds, y_onehot).item()
    #         pred_labels = torch.argmax(preds, dim=1)
    #         accuracy = (pred_labels == y_labels).double().mean().item()

    #     return loss_val, accuracy

    def evaluate(self, X, y):
        if self.beta is None:
            raise ValueError("Model has not been fit yet, self.beta is None.")

        X = self._as_feature_tensor(X)
        self._set_or_check_input_dim(X)
        Y, labels = self._prepare_targets(y, fit_output_dim=False)

        with torch.no_grad():
            Phi = self.forward(X)
            preds = Phi @ self.beta
            loss_val = F.mse_loss(preds, Y).item()

            if self.task == "classification":
                pred_labels = torch.argmax(preds, dim=1)
                accuracy = (pred_labels == labels).double().mean().item()
                return loss_val, accuracy

            return loss_val, None

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
            "beta": self.beta,   
            "unitaryParams_state_dict": self.unitaryParams.state_dict(), 
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
        loadedModelDict = torch.load(path, map_location="cpu")

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

        model.unitaryParams.load_state_dict(loadedModelDict["unitaryParams_state_dict"])

        return model

    """
    This function trains the model similar to a normal neural net except it is training the unitary we use to rotate our matrix.
    The parameters trained are the Givens angles which are used to form our Unitary.

    At each epoch:
        1. build Phi(theta)
        2. solve beta(theta) by ridge regression
        3. compute MSE loss
        4. backprop through the whole pipeline into theta
        5. update theta with Adam

        where theta is the vector of givens angles

    Inputs:
        X_train,
        y_train,
        y_train_labels=None,
        X_val=None,
        y_val=None,
        y_val_labels=None,
        epochs=50,
        lr=1e-3,
        printUpdates=True,
    Outputs:
        history: dict containing losses and optional accuracies
    """
    def train_Unitary(
    self,
    X_train,
    y_train,
    y_train_labels=None,
    X_val=None,
    y_val=None,
    y_val_labels=None,
    epochs=50,
    lr=1e-3,
    printUpdates=True,
    profile=False,
    ):
        X_train = self._as_feature_tensor(X_train)
        self._set_or_check_input_dim(X_train)
        y_train, inferred_train_labels = self._prepare_targets(y_train, fit_output_dim=True)

        if y_train_labels is None:
            y_train_labels = inferred_train_labels
        elif y_train_labels is not None:
            y_train_labels = torch.as_tensor(y_train_labels, device=self.device, dtype=torch.long)

        if X_val is not None:
            X_val = self._as_feature_tensor(X_val)
            self._set_or_check_input_dim(X_val)

        if y_val is not None:
            y_val, inferred_val_labels = self._prepare_targets(y_val, fit_output_dim=False)

            if y_val_labels is None:
                y_val_labels = inferred_val_labels
            elif y_val_labels is not None:
                y_val_labels = torch.as_tensor(y_val_labels, device=self.device, dtype=torch.long)

        if not self.internal_layers:
            self.create_hidden_layers()

        optimizer = torch.optim.Adam(self.unitaryParams.parameters(), lr=lr)

        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "epoch_time": [],
            "avg_epoch_time": [],

            # profiling
            "forward_time": [],
            "solve_beta_time": [],
            "pred_loss_time": [],
            "backward_time": [],
            "optimizer_time": [],
        }

        epoch_times = []

        if len(self.internal_layers) == 1:
            Z_train = self.precompute_one_layer_XW(X_train)
            if X_val is not None:
                Z_val = self.precompute_one_layer_XW(X_val)
            else:
                Z_val = None
        else:
            Z_train = None
            Z_val = None

        for epoch in range(epochs):
            epoch_start = time.time()
            self.train()
            optimizer.zero_grad(set_to_none=True)

           # Forward pass
            t0 = time.time()
            Phi = self.forward(X_train, Z_train)
            forward_time = time.time() - t0

            # Solve for Beta
            with torch.no_grad():
                t0 = time.time()
                beta = self.solve_beta(Phi.detach(), y_train)
                solve_beta_time = time.time() - t0

            # Compute predictions and loss
            t0 = time.time()
            train_preds = Phi @ beta
            loss = F.mse_loss(train_preds, y_train)
            pred_loss_time = time.time() - t0

            # Backprop
            t0 = time.time()
            loss.backward()
            backward_time = time.time() - t0

            # Adam update
            t0 = time.time()
            optimizer.step()

            for U in self.unitaryParams:
                if hasattr(U, "project"):
                    U.project()

            optimizer_time = time.time() - t0

           # Record training metrics using beta from this epoch
            with torch.no_grad():
                train_loss = loss.item()
                history["train_loss"].append(train_loss)

                if self.task == "classification" and y_train_labels is not None:
                    train_labels_pred = torch.argmax(train_preds, dim=1)
                    train_acc = (train_labels_pred == y_train_labels).double().mean().item()
                    history["train_acc"].append(train_acc)
                else:
                    train_acc = None

                if X_val is not None and y_val is not None:
                    Phi_val = self.forward(X_val, Z_val)
                    val_preds = Phi_val @ beta
                    val_loss = F.mse_loss(val_preds, y_val).item()
                    history["val_loss"].append(val_loss)

                    if self.task == "classification" and y_val_labels is not None:
                        val_labels_pred = torch.argmax(val_preds, dim=1)
                        val_acc = (val_labels_pred == y_val_labels).double().mean().item()
                        history["val_acc"].append(val_acc)
                    else:
                        val_acc = None
                else:
                    val_loss = None
                    val_acc = None

                
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)

            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            epochs_left = epochs - epoch - 1
            time_remaining = avg_epoch_time * epochs_left

            if printUpdates:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"epoch time: {epoch_time:.2f}s | "
                    f"avg epoch time: {avg_epoch_time:.2f}s | "
                    f"ETA: {time_remaining / 60:.2f} min"
                )
            history["epoch_time"].append(epoch_time)
            history["avg_epoch_time"].append(avg_epoch_time)
            history["forward_time"].append(forward_time)
            history["solve_beta_time"].append(solve_beta_time)
            history["pred_loss_time"].append(pred_loss_time)
            history["backward_time"].append(backward_time)
            history["optimizer_time"].append(optimizer_time)

            if printUpdates:
                msg = f"Epoch {epoch+1}/{epochs} | train_loss={train_loss:.6f}"
                if train_acc is not None:
                    msg += f" | train_acc={train_acc:.4f}"
                if val_loss is not None:
                    msg += f" | val_loss={val_loss:.6f}"
                if val_acc is not None:
                    msg += f" | val_acc={val_acc:.4f}"
                print(msg)

                if profile:
                    print(
                        f"Profile | "
                        f"forward={forward_time:.3f}s | "
                        f"solve_beta={solve_beta_time:.3f}s | "
                        f"pred_loss={pred_loss_time:.3f}s | "
                        f"backward={backward_time:.3f}s | "
                        f"optimizer={optimizer_time:.3f}s"
                    )

        with torch.no_grad():
            Phi_final = self.forward(X_train, Z_train)
            self.set_beta(Phi_final, y_train)

        return history
