import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def plot(func, start, n, unit, name = "fig"):
    plt.plot([unit*(start + i)/n for i in range(0, n)], [func(unit*(start + i)/n) for i in range(0, n)])
    plt.savefig(name + ".png")
    plt.show()

def get_wavelet_values(wavelet, i, j, dim, start, end):
    if(i > wavelet.dim or j > wavelet.dim):
        execption = ValueError("Invalid wavelent dimensions " , i , " and ", j, " for wavelet dimension =  ", wavelet.dim )
        return execption
    X = np.linspace(start, end, dim)
    Y = np.linspace(start, end, dim)    
    return wavelet.get_wavefunc_data(i, j, X, Y)

def get_delta_values(wavelet, i, j, dim):
    if(i > wavelet.dim or j > wavelet.dim):
        execption = ValueError("Invalid wavelent dimensions " , i , " and ", j, " for wavelet dimension =  ", wavelet.dim )
        return execption
    X = np.linspace(0, 1, dim)
    Y = np.linspace(0, 1, dim)    
    return wavelet.get_delta_data(i, j, X, Y)
    
def get_kinetic_term(wavelet, coord1, coord2, resolution_dim):
    psi1 = get_wavelet_values(wavelet, coord1[0], coord1[1], resolution_dim)
    del_psi2 = get_delta_values(wavelet, coord2[0], coord2[1], resolution_dim)
    integral = 0.5*(1/(resolution_dim*resolution_dim))*sum(sum(psi1*del_psi2))
    return integral

def plot_wavelet(wavelet, i, j, dim, start = 0, end = 1, title = "figure1"):
    data = get_wavelet_values(wavelet, i, j, dim, start, end)
    sns.heatmap(data, cbar=True, annot=False)
    plt.title(title)
    plt.show()

def plot_wavelet_comb(wavelet, index_list, coefficient_list, resolution_dim, start, end, title = "figure1"):
    X = np.linspace(start, end, resolution_dim)
    Y = np.linspace(start, end, resolution_dim)   
    data = np.zeros((resolution_dim, resolution_dim))
    for k in range(0, len(index_list)):
        (i, j) = index_list[k]
        coefficient = coefficient_list[k]
        if(i > wavelet.dim or j > wavelet.dim):
            execption = ValueError("Invalid wavelent dimensions " , i , " and ", j, " for wavelet dimension =  ", wavelet.dim )
            return execption    
        data += coefficient*wavelet.get_wavefunc_data(i, j, X, Y)
    sns.heatmap(data, cbar=True, annot=False)
    # plt.savefig(title + ".png")
    plt.title(title)
    plt.show()


def fit_interpol(y0, y1, a):
    return (1 - a)*y0 + a*y1

def fit_select(y0, y1, a):
    return ((1 - a) > 0.5)*y0 + (a > 0.5)*y1

def scaled_sampling(func_val, X, translate, scaling = 1):
    N = len(X)
    X = X - (X[-1] + X[0])/2 #normalization
    if (N!= len(func_val)):
        print("func_val and X not of same length! ")
        return
    fitting_func = fit_select
    if scaling < 1:
        fitting_func = fit_interpol
    dx = (X[-1] - X[0])/N
    y = np.zeros(len(X))
    begin = max(0, int((N - 1)*(translate)/(X[-1] - X[0])))
    for i in range(begin, len(X)):
        x_new = scaling*(X[i] - translate)
        shifted_i = int((N - 1)*(x_new - X[0])/(X[-1] - X[0]))
        shifted_i = min(max(shifted_i, 0), N - 2)
        a = (x_new - X[shifted_i])/dx
        y[i] =  fitting_func( func_val[shifted_i], func_val[shifted_i + 1], a)
    return y


def four_plot(data, main_title, sub_plot_titles):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(main_title, fontsize=16)

    for i, (data, title) in enumerate(zip(data, sub_plot_titles)):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        norm = colors.LogNorm(vmin=10**-12, vmax=np.max(data))
        im = ax.matshow(data, cmap='viridis', norm=norm)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, format='%.0e')
        plt.tight_layout()

def smoothen_function(y):
    n = len(y)
    y_new = np.zeros(n)
    for i in range(1, n-1):
        y_new[i] = 0.5* (y[i] + y[i+1])
    return y_new

def index_converter(i, n):
    return (i + int(n/2) )%n - int(n/2)

import inspect

def fast_scaling_func(X, f_init, coeffs, level = 0, dilation = 2, width = 1, smoothing = False):
    y = f_init
    if inspect.isfunction(f_init):
        y = np.array([f_init(x) for x in X])
    if level == 0:
        return y
    for _ in range(0, level):
        y_new = np.zeros(len(X))
        for i in range(0, len(coeffs)):
            y_temp = np.sqrt(dilation)*coeffs[i]*scaled_sampling(y, X, 1.01001*width*(i - int(len(coeffs)/2))/dilation, dilation)
            y_new += y_temp
            # plt.plot(X, y_temp)
        y = y_new
        if smoothing:
            y = smoothen_function(y)
    return y

def vector_transform(func, n_func = 3, width = 3, vector_dim = 3):
    n_func = 2*int(0.5*n_func) + 1
    vector_dim = 2*int(0.5*vector_dim) + 1
    n_matr = n_func * vector_dim
    e_vector = np.zeros(n_func)
    matr = np.zeros((n_matr, n_func))
    func_val = np.zeros(vector_dim)
    for i in range(0, vector_dim):
        x = (width/(vector_dim - 1))*(i - int(vector_dim/2))
        func_val[i] = func(x)

    for i in range(0, n_func):
        e_vector[i] = 1
        matr[:,i] = np.kron(e_vector, func_val)
        e_vector[i] = 0
    return matr

def get_condition_number(matr):
    eigs = abs(np.linalg.eigvals(matr))
    return np.max(eigs)/np.min(eigs)

def matrix_shift(init_matr, shift, n):
    n_mat = (n-1)*shift + init_matr.shape[0]
    matr = np.zeros((n_mat, n*init_matr.shape[0]))
    for i in range(0, n):
        matr[i*shift:i*shift + init_matr.shape[0], i*init_matr.shape[0]: (i+1)*init_matr.shape[0]] = init_matr
    return matr

def vector_function_fit(X, tofit_func, basis_func, width = 3, vector_dim = 9, dilation = 1):
    distance = X[-1] - X[0]
    n_func = int(distance/dilation + 1)
    vectorized_basis = vector_transform(basis_func, n_func, width, vector_dim)
    unit = width/(vector_dim - 1)
    shift = int(dilation/unit) 
    vectorized_basis = matrix_shift(np.eye(vector_dim), shift, int(vectorized_basis.shape[0]/vector_dim)) @ vectorized_basis
    trim = int((vectorized_basis.shape[0] - (X[-1] - X[0])/unit)/2)
    vectorized_basis = vectorized_basis[trim:-trim, :]
    X_new = np.linspace(X[0], X[-1], vectorized_basis.shape[0])
    data = np.array([tofit_func(x) for x in X_new])
    coeffs =  np.linalg.inv(np.transpose(vectorized_basis) @ vectorized_basis) @ np.transpose(vectorized_basis) @ data
    error = np.sqrt(np.sum((vectorized_basis @ coeffs - data)**2))
    def new_func(x):
        rval = 0
        for i in range(0, len(coeffs)):
            rval += coeffs[i]* basis_func(x - dilation*(i - int(len(coeffs)/2)))
        return rval
    loosefit = np.zeros(vectorized_basis.shape[0])
    for i in range(0, len(coeffs)):
        loosefit += coeffs[i]* vectorized_basis[:,i]
    basisfuncdata = np.zeros((vectorized_basis.shape[0], len(coeffs)))
    for i in range(0, len(coeffs)):
        for j in range(0, vectorized_basis.shape[0]):
            x = X_new[j]
            basisfuncdata[j, i] = basis_func(x - dilation*(i - int(len(coeffs)/2)))
    return coeffs, new_func, vectorized_basis

def raise_error(error_type=Exception, error_message=""):
    raise error_type(error_message)


def save_image(pltref, name):
    pltref.savefig('./../images/' + name + '.pdf', format='pdf', bbox_inches='tight')