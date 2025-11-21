import mrafit.wavelet_bases as wavelet_bases
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def test_errors(func, N, resolution):
    
    X = np.linspace(-5, 5, N)
    resolution = 1
    
    basis = wavelet_bases.GaussletBasis(resolution = resolution)
    coeffs, Yet, err = basis.get_mra_approx(func, X)
    error_bound = 10e-2 * resolution
    assert err <= error_bound, f"in {basis.wavelet_type}, error bound is {error_bound} while error is {err}"

    basis = wavelet_bases.HatBasis(resolution = resolution)
    coeffs, Yet, err = basis.get_mra_approx(func, X)
    error_bound = 10e-2 * resolution
    assert err <= error_bound, f"in {basis.wavelet_type}, error bound is {error_bound} while error is {err}"

    basis = wavelet_bases.GaussianBasis(resolution = resolution)
    coeffs, Yet, err = basis.get_mra_approx(func, X)
    error_bound = 1.7 * 10e-2 * resolution
    assert err <= error_bound, f"in {basis.wavelet_type}, error bound is {error_bound} while error is {err}"

    basis = wavelet_bases.GaussPleteauBasis(resolution = resolution)
    coeffs, Yet, err = basis.get_mra_approx(func, X)
    error_bound = 10e-2 * resolution
    assert err <= error_bound, f"in {basis.wavelet_type}, error bound is {error_bound} while error is {err}"


resolution = 1
N = 200* resolution
test_errors(sigmoid, N, resolution)

resolution = 0.5
N = int(200* resolution)
test_errors(sigmoid, N, resolution)

resolution = 0.1
N = int(200* resolution)
test_errors(sigmoid, N, resolution)


