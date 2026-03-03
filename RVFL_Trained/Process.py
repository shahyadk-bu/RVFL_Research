import os
import torch.nn as nn
import torch.optim as optim

from Functions import load_mnist_data,run_model, save_state, save_results
from Utility_Functions import determine_device, generate_file_name
from RVFL_Model import RVFL3



def process(dataset_name, model_name, loss_func_name, gamma_1, gamma_2, gamma_3, gamma_link, gamma_out,
            dist1, var1, dist2, var2, dist3, var3, bias_dist,
            hidden_units_1, hidden_units_2, hidden_units_3,
            epochs, batch_size, directory):
    """Trains a NN model on a dataset and saves the model accuracy and model parameters

    Parameters
    ----------
    dataset_name: str
        'mnist'
    model_name: str
        'rvfl2' (two-layer perceptron) or 'rvfl3' (three-layer perceptron)
    loss_func_name: str
        'ce' (Cross Entropy loss) or 'mse' (Mean Squared Error loss)
    gamma_N: float
        the mean-field scaling parameter for the Nth layer
    hidden_units_K: int
        the number of nodes in the Kth layer
    epochs: int
        number of times to iterate through the data set for training the model
        and calculating accuracy
    batch_size: int
        the number of images per batch
    directory: str
        the local where accuracy results and model parameters are saved
        (requires folders 'results' and 'models')
        :param gamma_out:
        :param gamma_link:
    """

    # Prints your input info
    print("Dataset:    {}".format(dataset_name.upper()))
    print("Model:      {}".format(model_name.upper()))
    print("loss_func:  {}".format(loss_func_name.upper()))
    print("Parameters: g_1={g_1}, g_2={g_2}, g_3={g_3}, h_1={h_1}, h_2={h_2}, h_3={h_3}, e={e}, b={b}".format(
        g_1=gamma_1, g_2=gamma_2, g_3=gamma_3, h_1=hidden_units_1, h_2=hidden_units_2, h_3=hidden_units_3, e=epochs,
        b=batch_size))

    device = determine_device(do_print=True)

    if dataset_name.upper() == 'MNIST':
        train_loader, test_loader = load_mnist_data(batch_size=batch_size)
    else:
        raise ValueError("Dataset '{0}' unknown".format(dataset_name))

    if model_name.upper() == 'RVFL3':
        lr_beta = 1 / ((hidden_units_1 ** (1 - 2 * 0.75)) * (hidden_units_2 ** (3 - 2 * 0.75)))
        lr_c = 1 / ((hidden_units_1 ** (1 - 2 *0.75)) * (hidden_units_2 ** (3 - 2 * 0.75)))
        lambda_beta = 0.0
        lambda_c = 0.0
        print(f"Trainable: readout only (Beta, c). LRs -> Beta={lr_beta}, c={lr_c}; "
              f"weight_decay -> Beta={lambda_beta}, c={lambda_c}")

        model = RVFL3(hidden_units_1=hidden_units_1, hidden_units_2=hidden_units_2, hidden_units_3=hidden_units_3,
                     gamma_1=gamma_1, gamma_2=gamma_2, gamma_3=gamma_3, gamma_link=gamma_link, gamma_out=gamma_out,
                     dist1=dist1, var1=var1, dist2=dist2, var2=var2, dist3=dist3, var3=var3, bias_dist=bias_dist)
    else:
        raise ValueError("Model '{0}' unknown".format(model_name))
    model.to(device)

    # loss function
    if loss_func_name.upper() == 'CE':
        loss_func = nn.CrossEntropyLoss()
        do_encoding = False
    elif loss_func_name.upper() == 'MSE':
        loss_func = nn.MSELoss()
        do_encoding = True
    else:
        raise ValueError("loss_func '{0}' unknown".format(loss_func_name))

    # Optimizer (Just set to use SGD)
    optimizer = optim.SGD(
        [
            {"params": [model.readout.Beta], "lr": lr_beta, "weight_decay": lambda_beta},
            {"params": [model.readout.c], "lr": lr_c, "weight_decay": lambda_c},
        ],
    )

    # Run model
    results = run_model(
        model=model,
        optimizer=optimizer,
        loss_func=loss_func,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        do_encoding=do_encoding,
        epochs=epochs)

    # Get the file name
    file_name = generate_file_name(
        dataset_name=dataset_name,
        model_name=model_name,
        loss_func_name=loss_func_name,
        gamma_1=gamma_1,
        gamma_2=gamma_2,
        gamma_3=gamma_3,
        gamma_link=gamma_link,
        gamma_out=gamma_out,
        hidden_units_1=hidden_units_1,
        hidden_units_2=hidden_units_2,
        hidden_units_3=hidden_units_3,
        epochs=epochs,
        batch_size=batch_size)

    # Save accuracy results
    results_directory = os.path.join(directory, 'results/')
    save_results(
        results=results,
        directory=results_directory,
        file_name=file_name)

    # Save model state
    models_directory = os.path.join(directory, 'models/')
    save_state(
        model=model,
        directory=models_directory,
        file_name=file_name)

    return
