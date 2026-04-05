import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

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

"""
This function returns the accuracy of our predictions vs the ground truth

Inputs:
    Tensor preds: The predictions of our model
    Tensor labels: The actual ground truth
"""
def accuracy(preds, labels):
    return (preds == labels).double().mean().item()
