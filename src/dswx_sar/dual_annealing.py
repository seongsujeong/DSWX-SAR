import numpy as np
from scipy.optimize import dual_annealing

bimodal_param_names = ['mu1', 'sigma1', 'amplitude1',
                  'mu2', 'sigma2', 'amplitude2']

trimodal_params_names = ['mu1', 'sigma1', 'amplitude1',
                   'mu2', 'sigma2', 'amplitude2',
                   'mu3', 'sigma3', 'amplitude3']

DEFAULT_BOUNDS_BIMODAL = ((-30, 0.01, 0.01,
                           -30, 0, 0.01),
                           (5, 5, 0.95,
                            5, 5, 0.95))

DEFAULT_BOUNDS_TRIMODAL = ((-30, 0.01, 0.01,
                           -30, 0.01, 0.01,
                           -30, 0.01, 0.01),
                           (5, 5, 0.95,
                            5, 5, 0.95,
                            5, 5, 0.95))

# Define trimodal function
def gauss(array, mu, sigma, amplitude):
    return amplitude * np.exp(-(array - mu)**2 / (2 * sigma**2))


def trimodal(array, mu1, sigma1, amplitude1,
             mu2, sigma2, amplitude2,
              mu3, sigma3, amplitude3):
    return gauss(array, mu1, sigma1, amplitude1) + \
           gauss(array, mu2, sigma2, amplitude2) + \
           gauss(array, mu3, sigma3, amplitude3)


def bimodal(array, mu1, sigma1, amplitude1,
             mu2, sigma2, amplitude2):
    return gauss(array, mu1, sigma1, amplitude1) + \
           gauss(array, mu2, sigma2, amplitude2)


def map_param(param_name_list, param_array):
    return dict(zip(param_name_list, param_array))


def loss_trimodal(params, param_names, x, y_true):
    y_pred = trimodal(x, **map_param(param_names, params))
    return np.sqrt(np.mean((y_pred - y_true)**2))


def loss_bimodal(params, param_names, x, y_true):
    y_pred = bimodal(x, **map_param(param_names, params))
    return np.sqrt(np.mean((y_pred - y_true)**2))


def bimodal_fit(x, y, initvel, bounds=DEFAULT_BOUNDS_BIMODAL):
    result_da = dual_annealing(loss_bimodal, x0=initvel, bounds=bounds, args=(x, y))
    return result_da.x


def trimodal_fit(x, y, initvel, bounds=DEFAULT_BOUNDS_TRIMODAL):
    result_da = dual_annealing(loss_trimodal, x0=initvel, bounds=bounds, args=(x, y))
    return result_da.x
