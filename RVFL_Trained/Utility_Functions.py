import os
import torch

def determine_device(do_print):
    if torch.cuda.is_available():
        if do_print:
            print("Device:     GPU")
        device = torch.device("cuda")
    else:
        if do_print:
            print("Device:     CPU")
        device = torch.device("cpu")

    return device


def sample_matrix(shape, dist: str, var, gammaK=None):
    """Sample a random matrix of given shape from a distribution with target variance.
    Note that mean is set to 0 for each distribution

        shape: shape of matrix
        dist:  'normal' | 'uniform' | 'gamma'
        var:   variance (needs to be >0).
        theta: For gamma distribution parameter; preset to None; dtype = dict{"k": k, "theta": theta}
    """

    if var <= 0:
        raise ValueError("Variance of random matrix must be positive and nonzero")

    if dist == 'normal':
        std = (var ** 0.5)
        return torch.randn(shape) * std

    elif dist == 'uniform':
        # For Uniform(-a, a), variance = a^2 / 3  => a = sqrt(3*var)
        a = (3.0 * var) ** 0.5
        return (torch.rand(shape) * 2.0 - 1.0) * a

    elif dist == 'gamma':
        # Gamma(k, theta) has mean k*theta, var k*theta^2.
        if gammaK is None or gammaK <= 0:
            raise ValueError("Need a positive k parameter for Gamma distribution")

        theta = (var / gammaK) ** 0.5

        g = torch.distributions.Gamma(concentration=gammaK, rate=1.0 / theta).sample(sample_shape=shape)
        g = g - (gammaK * theta)  # center to zero-mean
        return g

    else:
        raise ValueError(f"Invalid distribution")



def generate_file_name(dataset_name, model_name, loss_func_name, gamma_1, gamma_2, gamma_3, gamma_link, gamma_out,
                       hidden_units_1, hidden_units_2, hidden_units_3, epochs, batch_size):
    """Returns a file name specifying parameters, e.g.,
        'mnist_mlp2_ce_gI06_gII05_hI1000_hII1000_e500_b20'

    Parameters
    ----------
    dataset_name: str
        'mnist'
    model_name: str
        'mlp' or 'cnn'
    loss_func_name: str
        'ce' (for Cross Entropy loss) or 'mse' (for Mean Squared Error loss)
    gamma_1: float
        the mean-field scaling parameter for the first layer
    gamma_2: float
        the mean-field scaling parameter for the second layer
    gamma_3: float
        the mean-field scaling parameter for the third layer
    hidden_units_1: int
        the number of nodes in the first layer
    hidden_units_2: int
        the number of nodes in the second layer
    hidden_units_3: int
        the number of nodes in the third layer
    epochs: int
        number of times to iterate through the data set for training the model
        and calculating accuracy
    batch_size: int
        the number of images per batch
    """

    parts = [dataset_name.lower(), model_name.lower(), loss_func_name.lower()]
    g_1 = 'gI{0}'.format(str(gamma_1).replace('.', ''))
    g_2 = 'gII{0}'.format(str(gamma_2).replace('.', ''))
    g_3 = 'gIII{0}'.format(str(gamma_3).replace('.', ''))
    g_L = 'gL{0}'.format(str(gamma_link).replace('.', ''))
    g_O = 'gO{0}'.format(str(gamma_out).replace('.', ''))
    parts += [g_1, g_2, g_3, g_L, g_O]

    h_1 = 'hI{0}'.format(hidden_units_1)
    h_2 = 'hII{0}'.format(hidden_units_2)
    h_3 = 'hIII{0}'.format(hidden_units_3)
    parts += [h_1, h_2, h_3]

    e = 'e{0}'.format(epochs)
    b = 'b{0}'.format(batch_size)
    parts += [e, b]

    file_name = '_'.join(parts)
    return file_name

def validate_directory(directory):
    """Checks if a directory exists, and creates it if it doesn't exist

    Parameters
    ----------
    directory: str
        the location to where the file will be saved
    """

    if not os.path.isdir(directory):
        os.makedirs(directory)
        msg = 'Created directory:  {directory}.'.format(
            directory=directory)
        print(msg)
    return

if __name__ == "__main__":

    print("Ok")
