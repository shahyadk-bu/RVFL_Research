import os
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from Utility_Functions import validate_directory

def load_mnist_data(batch_size):
    """Return train and test data loaders for MNIST dataset

    Parameters
    ----------
    batch_size: int
        the number of images per batch
    """

    # Data normalization values from:
    # https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457
    transformation = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST(
        'data/', train=True, transform=transformation, download=True)
    test_data = datasets.MNIST(
        'data/', train=False, transform=transformation, download=True)
    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader



def train(model, optimizer, loss_func, train_loader, device, do_encoding):
    """Train model using specified optimizer, loss_func, and training data

    Parameters
    ----------
    model: class inheriting nn.Module
        neural network model
    optimizer: from torch.optim
        optimization algorithm
    loss_func: from torch.nn
        loss function
    train_loader: torch.utils.data.DataLoader
        training data loader
    device: torch.device("cuda") or torch.device("cpu")
        the device on which to run the model
    do_encoding: bool
        the MSE loss_func requires one-hot encoding to calculate the loss
    """

    model.train()
    printed_debug = False

    # Loop over train data
    for i, (inputs, targets) in enumerate(train_loader):

        optimizer.zero_grad()

        inputs = inputs.to(device)
        targets = targets.to(device)

        if not do_encoding and targets.dtype != torch.long:
            targets = targets.long()

        output = model(inputs)

        # Finds Loss
        if do_encoding:
            # Set up for Mean Squared Error (MSE) Loss
            targets_encoded = F.one_hot(targets, num_classes=10).float()
            loss = loss_func(output, targets_encoded)
        else:
            # Set up for Cross-Entropy Loss
            loss = loss_func(output, targets)

        
        loss.backward()
        if hasattr(model, 'scale_learning_rates'):
            model.scale_learning_rates()

      # Print so we can check for issues
        if not printed_debug:
            with torch.no_grad():
                probs = output.softmax(dim=1)
                pred = probs.argmax(dim=1)
                acc_b = (pred == targets).float().mean().item()
                print(f"[train dbg] loss={loss.item():.4f} acc={acc_b:.4f} "
                      f"logits Î¼={output.mean().item():.3e} Ïƒ={output.std().item():.3e}")
            printed_debug = True

        optimizer.step()

    return

def calculate_accuracy(model, loader, device):
    """Check model accuracy for train or test data

    Parameters
    ----------
    model: class inheriting nn.Module
        neural network model
    loader: torch.utils.data.DataLoader
        train or test data loader
    device: torch.device("cuda") or torch.device("cpu")
        the device we use to run the model
    """

    num_correct = 0
    num_attempt = 0

    # Loop over data
    for batch in loader:

        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)


        output = model(inputs)

        # Find the accuracy for the batch
        predictions = output.max(dim=1, keepdim=True)[1]
        is_correct = predictions.eq(targets.view_as(predictions))
        num_correct += is_correct.sum().item()
        num_attempt += len(inputs)

    # Calculate test accuracy
    accuracy = num_correct / num_attempt
    return accuracy



def run_model(model, optimizer, loss_func, train_loader, test_loader, device,
              do_encoding, epochs):
    """Train model

    Parameters
    ----------
    model: class inheriting nn.Module
        neural network model
    optimizer: from torch.optim
        optimization algorithm
    loss_func: from torch.nn
        loss function
    train_loader: torch.utils.data.DataLoader
        training data loader
    test_loader: torch.utils.data.DataLoader
        test data loader
    device: torch.device("cuda") or torch.device("cpu")
        the device on which to run the model
    do_encoding: bool
        the MSE loss_func requires one-hot encoding to calculate the loss
    epochs: int
        number of times to iterate through the data set for training the model
        and calculating accuracy
    """

    # Store the accuracy results
    results = pd.DataFrame(
        None,
        index=range(epochs),
        columns=['Train', 'Test'],
        dtype=float)

    for epoch in range(epochs):
        
        train(
            model=model,
            optimizer=optimizer,
            loss_func=loss_func,
            train_loader=train_loader,
            device=device,
            do_encoding=do_encoding)


        model.eval()
        with torch.no_grad():
            train_accuracy = calculate_accuracy(
                model=model,
                loader=train_loader,
                device=device)
            results.loc[epoch, 'Train'] = train_accuracy

            test_accuracy = calculate_accuracy(
                model=model,
                loader=test_loader,
                device=device)
            results.loc[epoch, 'Test'] = test_accuracy

            msg = 'Epoch: {}, Train Accuracy = {:.2f}, Test Accuracy = {:.2f}'.format(
                epoch, train_accuracy, test_accuracy)
            print(msg)

    return results

def save_results(results, directory, file_name):
    """Saves a csv file of the train and test accuracy vs. epoch

    Parameters
    ----------
    results: DataFrame
        index = epochs, columns = 'Train' and 'Test', data = accuracy
    directory: str
        the location to where the file will be saved
    file_name: str
        the name of the file
    """

    validate_directory(directory)
    extension = '.csv'
    file_path = os.path.join(directory, file_name) + extension
    results.to_csv(file_path)
    msg = 'Saved results to {file_path}.'.format(
        file_path=file_path)
    print(msg)
    return


def save_state(model, directory, file_name):
    """Saves a file of the model parameters

    Parameters
    ----------
    model: class inheriting nn.Module
        neural network model
    directory: str
        the location to where the file will be saved
    file_name: str
        the name of the file
    """

    validate_directory(directory)
    file_path = os.path.join(directory, file_name)
    state = model.state_dict()
    torch.save(obj=state, f=file_path)
    msg = 'Saved model to {file_path}.'.format(
        file_path=file_path)
    print(msg)
