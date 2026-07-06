import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np

"""
Loads MNIST data for "training" model.
Inputs:
    string device: the device to load in the data with
    torch.dtype dtype: The data type for the MNIST data
outputs:
    Tensor X_train: the training images, normalized and flattened to shape (num_train_samples, 784)
    Tensor y_train: the one-hot encoded training labels of shape (num_train_samples, 10)
    Tensor X_test: the test images, normalized and flattened to shape (num_test_samples, 784)
    Tensor y_test: the one-hot encoded test labels of shape (num_test_samples, 10)
    Tensor y_train_labels: the integer class labels for the training set of shape (num_train_samples,)
    Tensor y_test_labels: the integer class labels for the test set of shape (num_test_samples,)
"""
def load_mnist_tensors(device="cpu", dtype=torch.float64):
        transform = transforms.ToTensor()

        train_data = datasets.MNIST(
            root="data/", train=True, transform=transform, download=True
        )
        test_data = datasets.MNIST(
            root="data/", train=False, transform=transform, download=True
        )

        X_train = train_data.data.to(dtype=dtype, device=device) / 255.0
        X_test = test_data.data.to(dtype=dtype, device=device) / 255.0

        # Normalize using usual MNIST preprocessing
        X_train = (X_train - 0.1307) / 0.3081
        X_test = (X_test - 0.1307) / 0.3081

        # Flatten from (n, 28, 28) to (n, 784)
        X_train = X_train.view(X_train.shape[0], -1)
        X_test = X_test.view(X_test.shape[0], -1)

        y_train_labels = train_data.targets.to(device)
        y_test_labels = test_data.targets.to(device)

        y_train = F.one_hot(y_train_labels, num_classes=10).to(dtype=dtype)
        y_test = F.one_hot(y_test_labels, num_classes=10).to(dtype=dtype)

        return X_train, y_train, X_test, y_test, y_train_labels, y_test_labels

def load_fashion_mnist_tensors(
    device="cpu",
    dtype=torch.float64,
    normalize="standardize",
):
    """
    Loads Fashion-MNIST and prepares it for the generalized RVFL model.

    Returns a dictionary:
        X_train, y_train, X_test, y_test,
        input_dim, output_dim, task
    """
    transform = transforms.ToTensor()

    train_data = datasets.FashionMNIST(
        root="data/",
        train=True,
        transform=transform,
        download=True,
    )

    test_data = datasets.FashionMNIST(
        root="data/",
        train=False,
        transform=transform,
        download=True,
    )

    X_train = train_data.data.to(dtype=dtype, device=device) / 255.0
    X_test = test_data.data.to(dtype=dtype, device=device) / 255.0

    y_train = train_data.targets.to(device=device)
    y_test = test_data.targets.to(device=device)

    data = prepare_supervised_tensors(
        X_train,
        y_train,
        X_test,
        y_test,
        task="classification",
        normalize=normalize,
        device=device,
        dtype=dtype,
        flatten=True,
    )

    return data

"""
This function returns the accuracy of our predictions vs the ground truth

Inputs:
    Tensor preds: The predictions of our model
    Tensor labels: The actual ground truth
"""
def accuracy(preds, labels):
    return (preds == labels).double().mean().item()

def to_feature_tensor(X, device="cpu", dtype=torch.float64, flatten=True):
    """
    Converts input data to a 2D torch tensor of shape (N, d).

    Examples:
        tabular: (N, d) stays (N, d)
        single sample: (d,) becomes (1, d)
        image batch: (N, H, W) becomes (N, H*W)
        image batch: (N, C, H, W) becomes (N, C*H*W)
    """
    if torch.is_tensor(X):
        X = X.to(device=device, dtype=dtype)
    else:
        X = torch.as_tensor(X, device=device, dtype=dtype)

    if X.ndim == 1:
        X = X.reshape(1, -1)
    elif flatten and X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    if X.ndim != 2:
        raise ValueError(f"Expected X to have shape (N, d), got {tuple(X.shape)}.")

    return X

def standardize_train_test(X_train, X_test=None, eps=1e-8):
    """
    Standardizes features using the training-set mean and standard deviation.

    X_train_new = (X_train - mean_train) / std_train
    X_test_new = (X_test - mean_train) / std_train

    This prevents test-set leakage.
    """
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True)

    std = torch.where(std < eps, torch.ones_like(std), std)

    X_train = (X_train - mean) / std

    if X_test is None:
        return X_train, mean, std

    X_test = (X_test - mean) / std
    return X_train, X_test, mean, std

def minmax_train_test(X_train, X_test=None, eps=1e-8):
    """
    Min-max normalizes features using training-set min and max.

    X_new = (X - min_train) / (max_train - min_train)
    """
    min_val = X_train.min(dim=0, keepdim=True).values
    max_val = X_train.max(dim=0, keepdim=True).values
    denom = max_val - min_val

    denom = torch.where(denom < eps, torch.ones_like(denom), denom)

    X_train = (X_train - min_val) / denom

    if X_test is None:
        return X_train, min_val, max_val

    X_test = (X_test - min_val) / denom
    return X_train, X_test, min_val, max_val

def encode_class_labels(y_train, y_test=None, device="cpu"):
    """
    Converts arbitrary class labels into integer labels 0, 1, ..., C-1.

    Handles labels like:
        [2, 4, 4, 9]
        ["cat", "dog", "cat"]
        numpy arrays
        torch tensors

    Returns:
        y_train_encoded
        y_test_encoded
        label_to_index
        index_to_label
    """
    if torch.is_tensor(y_train):
        y_train_list = y_train.detach().cpu().tolist()
    else:
        y_train_list = np.asarray(y_train).tolist()

    unique_labels = sorted(set(y_train_list))
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    index_to_label = {i: label for label, i in label_to_index.items()}

    y_train_encoded = torch.tensor(
        [label_to_index[label] for label in y_train_list],
        device=device,
        dtype=torch.long,
    )

    if y_test is None:
        return y_train_encoded, None, label_to_index, index_to_label

    if torch.is_tensor(y_test):
        y_test_list = y_test.detach().cpu().tolist()
    else:
        y_test_list = np.asarray(y_test).tolist()

    unknown_labels = set(y_test_list) - set(label_to_index.keys())
    if unknown_labels:
        raise ValueError(
            f"Test labels contain classes not present in training labels: {unknown_labels}"
        )

    y_test_encoded = torch.tensor(
        [label_to_index[label] for label in y_test_list],
        device=device,
        dtype=torch.long,
    )

    return y_train_encoded, y_test_encoded, label_to_index, index_to_label

def prepare_supervised_tensors(
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    *,
    task="classification",
    normalize="standardize",
    device="cpu",
    dtype=torch.float64,
    flatten=True,
):
    """
    General data preparation function for RVFL experiments.

    Inputs:
        X_train, y_train: training data and targets
        X_test, y_test: optional test/validation data and targets
        task: "classification" or "regression"
        normalize: "standardize", "minmax", or None
        device: "cpu" or "cuda"
        dtype: torch.float32 or torch.float64
        flatten: whether to flatten image-like inputs to vectors

    Returns:
        A dictionary containing prepared tensors and metadata.
    """
    if task not in {"classification", "regression"}:
        raise ValueError("task must be 'classification' or 'regression'.")

    if normalize not in {"standardize", "minmax", None}:
        raise ValueError("normalize must be 'standardize', 'minmax', or None.")

    X_train = to_feature_tensor(
        X_train,
        device=device,
        dtype=dtype,
        flatten=flatten,
    )

    if X_test is not None:
        X_test = to_feature_tensor(
            X_test,
            device=device,
            dtype=dtype,
            flatten=flatten,
        )

    if normalize == "standardize":
        if X_test is None:
            X_train, mean, std = standardize_train_test(X_train)
            norm_stats = {"mean": mean, "std": std}
        else:
            X_train, X_test, mean, std = standardize_train_test(X_train, X_test)
            norm_stats = {"mean": mean, "std": std}

    elif normalize == "minmax":
        if X_test is None:
            X_train, min_val, max_val = minmax_train_test(X_train)
            norm_stats = {"min": min_val, "max": max_val}
        else:
            X_train, X_test, min_val, max_val = minmax_train_test(X_train, X_test)
            norm_stats = {"min": min_val, "max": max_val}

    else:
        norm_stats = None

    if task == "classification":
        y_train, y_test, label_to_index, index_to_label = encode_class_labels(
            y_train,
            y_test,
            device=device,
        )

        output_dim = len(label_to_index)

    else:
        y_train = torch.as_tensor(y_train, device=device, dtype=dtype)

        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        if y_test is not None:
            y_test = torch.as_tensor(y_test, device=device, dtype=dtype)
            if y_test.ndim == 1:
                y_test = y_test.reshape(-1, 1)

        label_to_index = None
        index_to_label = None
        output_dim = y_train.shape[1]

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "input_dim": X_train.shape[1],
        "output_dim": output_dim,
        "task": task,
        "normalize": normalize,
        "norm_stats": norm_stats,
        "label_to_index": label_to_index,
        "index_to_label": index_to_label,
    }

def decode_class_labels(y_encoded, index_to_label):
    """
    Converts encoded predictions 0, 1, ..., C-1 back to original labels.
    """
    if torch.is_tensor(y_encoded):
        y_encoded = y_encoded.detach().cpu().tolist()

    return [index_to_label[int(i)] for i in y_encoded]