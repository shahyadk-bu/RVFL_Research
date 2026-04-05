import torch
from RVFL_Model import RVFL
from utils import load_mnist_tensors, accuracy

link_option = "direct"

layersInfo = [
    {
        "layer_dim": 100,
        "weight_dist": "normal",
        "weight_var": 1.0,
        "gamma_k": None,
        "bias_switch": True,
        "bias_dist": "normal",
        "bias_var": 1.0
    }
]

generalInfo = {
    "seed": 3,
    "device": "cpu",
    "dtype": torch.float64
}

scalings = [1]


model = RVFL(
    layersInfo=layersInfo,
    generalInfo=generalInfo,
    activation="relu",
    linkOption=link_option,
    lamb=1e-12,
    scalings=scalings
)

X_train, y_train, X_test, y_test, y_train_labels, y_test_labels = load_mnist_tensors(
    device=generalInfo["device"],
    dtype=generalInfo["dtype"]
)

model.MNIST_fit(X_train, y_train)

test_preds = model.predict(X_test)
print("Test accuracy:", accuracy(test_preds, y_test_labels))