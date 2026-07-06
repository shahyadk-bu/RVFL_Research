import sys
from pathlib import Path

import torch


THIS_FILE = Path(__file__).resolve()
INNER_REPO = THIS_FILE.parents[1]
OUTER_REPO = INNER_REPO.parent

sys.path.insert(0, str(INNER_REPO))
sys.path.insert(0, str(OUTER_REPO))


from RVFL_Research.Unitary_Model.Model.utils import (
    load_mnist_tensors,
    load_fashion_mnist_tensors,
    load_cifar10_tensors,
)


def main():
    device = "cpu"
    dtype = torch.float32

    print("Downloading/loading MNIST...")
    load_mnist_tensors(device=device, dtype=dtype)

    print("Downloading/loading FashionMNIST...")
    load_fashion_mnist_tensors(device=device, dtype=dtype)

    print("Downloading/loading CIFAR10...")
    load_cifar10_tensors(device=device, dtype=dtype)

    print("Finished downloading/loading datasets.")


if __name__ == "__main__":
    main()