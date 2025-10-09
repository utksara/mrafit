import numpy as np
import wavelet_bases
import utils
import itertools
from scipy.optimize import minimize
import unitary_circ_coefficient

def truncate_array(array, length):
    '''
    Truncate an array to a given input length by chipping of equal number of
    elements from both ends.

    Args:
        array(numpy.ndarray) : input array to be truncated
        length (int) : truncation length (should less than the array length)

    Returns:
        truncated array if input length is less than input array lenght
            otherwise return input array unchanged 

    Example truncate_array([1,2,3,4,5], 3) will return [2,3,4]
    '''
    if length >= len(array):
        return array
    new_array = np.zeros(length)
    for i in range(0, length):
        new_array[i] = array[int(len(array)/2) - int(length/2) + i]
    return new_array

def recursive_coeffs (coeff_parent, coeff_child, dilation):
    ''' 
    Given two arrays of coefficients : child and parent,
    returns a new array of coefficients by 
    1) using child coefficients as first basis
    2) muliplying parent coefficients to the fist basis and summing them up 
    after interval of a given dilation factor
    
    Args: 
        coeff_parent (numpy.ndarray) : parent coefficients
        coeff_child (numpy.ndarray) : child coefficients
        dilation (int): dilation factor

    Returns:
        array of new coefficients
    Example: coeff_child [1,2,3], coeff_parent [4,5,6] with dilation 2 will return
    [4*[1,2,3]
            +
         5*[1,2,3]
                +
             6*[1,2,3]]
    = [4*[1,2], 4*[3] + 5*[1], 5*[2], 5*[3] + 6*[1], 6*[2,3]]
    = [4, 8, 17, 10, 21, 12, 18] 
    '''

    new_coeff = np.zeros(dilation*(len(coeff_parent) - 1) + len(coeff_child))
    for i in range(0, len(coeff_parent)):
        new_coeff[i*dilation: i*dilation + len(coeff_child)] += coeff_parent[i]*coeff_child
    return new_coeff

def gausslet_recursion(init_basis, iterations, u4basis, truncate = 63):
    '''
    Generate a new gausslet basis by using an input gausslet basis to fit the
    Unitary circuit haar basis (u4basis) function using the . The step is repeated a given number
    of times

    Args: 
        init_basis (wavelet_bases.Gausslet_Basis) : initial gausslet basis
        iterations (int) : total number of iterations for each step
        truncate (int) : truncate lenght of coefficients using truncate_array function

    Returns :
        gausslet basis with new coefficients (wavelet_bases.Gausslet_Basis), 
        approximate fit of u4basis with the new gausslet basis (numpy.ndarray)
    '''
    
    width = 4
    n_func = 3
    vector_dim = 5
    L1 = - (width*n_func + (n_func - 1)*width/vector_dim)/2
    L2 = - L1
    X = np.linspace(L1, L2, 1000)

    if iterations == 0:
        return init_basis, None
    for i in range(0, iterations):
        coeff_child = init_basis.wavelet_coefficients
        '''
        Wavelet.get_mra_approx

        fits a given function using mra based on given grid

        Args:
            func (function) : function to fit
            X (numpy.ndarray) : uniform 1-d grid created using np.linspace function 

        Returns:
            coeffs (numpy.ndarray): coefficients of the basis function in the wavelet 
            approx_func (numpy.ndarray) : approximate funciton in the wavelet basis calculated on X
            error (numpy.ndarray): error w.r.t actual function values calculated on X
        '''
        coeffs, approx_func, error = init_basis.get_mra_approx(u4basis.wavelet_func, X)
        coeff_parent = coeffs
        new_basis_coeffs = recursive_coeffs (coeff_parent, coeff_child, 3)
        new_basis_coeffs = truncate_array(new_basis_coeffs, truncate)
        init_basis = wavelet_bases.Gausslet_Basis(resolution=3, wavelet_coefficients = list(new_basis_coeffs[int(len(new_basis_coeffs)/2):]))
        # print(error)
    return init_basis, approx_func


# width = 3
# unit = 1/2
# vector_dim = int(width/unit + 1)
# dilation = 4

# L1 = -5
# L2 = 5
# X = np.linspace(L1, L2, 1000)

# ''' 
# vector_function_fit 

# fit a given function against vectorized form of a basis function function by applying
# vector transformation based on given width, vector dimension. The overlap in the
# basis function is calculated using a given dilation factor

# Args:
    
#     numpy.ndarray) : uniform 1-d grid created using np.linspace function  
#     tofit_func (function): function to fit
#     basis_func  (function): basis function 
#     width (float): width for vector_transform function
#     vector_dim (odd int): vector_dim for vector_transform function
#     dilation (int): dilation factor

# Returns
#     coeffs (numpy.ndarray) : coefficients of the basis functions
#     new_func (function): new basis function created using coeffs and basis_func
#     vectorized_basis (basis): matrix containing columns as vectorized form of translated basis_funcs
# '''
# coeffs, new_func, vectorized_basis = utils.vector_function_fit(X, u4basis.wavelet_func, lambda x: np.exp(-0.5*x**2) \
#                                              , width, vector_dim, dilation)
# gausslet_basis2 = wavelet_bases.Gausslet_Basis(resolution=3, wavelet_coefficients = list(coeffs[int(len(coeffs)/2):]))

# iterations = 0
# new_gausslet_basis, approx_func = gausslet_recursion(gausslet_basis2, iterations)

saved_wavelet_coeffs = None
def get_gausslet_coefficients(depth = 4):
    global saved_wavelet_coeffs
    
    if saved_wavelet_coeffs is not None:
        return saved_wavelet_coeffs
    
    u4basis  = wavelet_bases.Ugeneric_Basis(basis_coeffs=unitary_circ_coefficient.get_ugeneric_coefficients(depth), \
                                        resolution = 3)    
    # gausslet_basis = wavelet_bases.Gausslet_Basis(resolution = 3)

    L1 = -10
    L2 = 10
    X = np.linspace(L1, L2, 200)

    width = 11
    unit = 1/2
    vector_dim = int(width/unit + 1)
    dilation = 1       

    coeffs, new_func, vectorized_basis = utils.vector_function_fit(X, u4basis.wavelet_func, lambda x: np.exp(-0.5*x**2) \
                                             , width, vector_dim, dilation)
    # new_gausslet_basis = wavelet_bases.Gausslet_Basis(resolution=3, wavelet_coefficients = list(coeffs[int(len(coeffs)/2):]))

    def objective_ortho(coeffs):
        gausslet_basis = wavelet_bases.Gausslet_Basis(resolution=3, wavelet_coefficients = list(coeffs))
        score = gausslet_basis.get_orthogonality()
        return abs(1 - score)
    
    def objective_comple(coeffs):
        gausslet_basis = wavelet_bases.Gausslet_Basis(resolution=3, wavelet_coefficients = list(coeffs))
        score = gausslet_basis.get_completeness()
        return abs(1 - score)
    
    def objective(coeffs):
        return objective_ortho(coeffs) + objective_comple(coeffs)

    coeffs = coeffs[int(len(coeffs)/2):]
    coeffs = coeffs[0:7]
    new_gausslet_basis = wavelet_bases.Gausslet_Basis(resolution=3, wavelet_coefficients = list(coeffs))
    max_orthogonality = new_gausslet_basis.get_orthogonality()
    max_completeness = new_gausslet_basis.get_completeness()
    list_of_elements = [[k - 0.2,k + 0.2] for k in coeffs]
    coeffs_list = list(itertools.product(*list_of_elements))
    bounds = [[-0.5, 0.5] for _ in range(len(coeffs))]
    bounds[0] = [0.5, 1.5]

    final_coeffs = coeffs
    for coeffs in coeffs_list:
        results = minimize(objective, list(coeffs), bounds=bounds, method = 'Nelder-Mead', options={'maxiter':1000}, tol=10e-5)
        new_gausslet_basis = wavelet_bases.Gausslet_Basis(resolution=3, wavelet_coefficients = list(results.x))
        orthogonality = new_gausslet_basis.get_orthogonality()
        completeness = new_gausslet_basis.get_completeness()
        if orthogonality > max_orthogonality and completeness > max_completeness:
            # results = minimize(objective_comple, results.x, bounds=bounds, method = 'Nelder-Mead', options={'maxiter':1000}, tol=10e-5)
            # new_gausslet_basis = wavelet_bases.Gausslet_Basis(resolution=3, wavelet_coefficients = list(results.x))
            # orthogonality = new_gausslet_basis.get_orthogonality()
            
            overalap_matr = new_gausslet_basis.get_overlap_matrix(np.linspace(-10, 10, 100))
            max_orthogonality = orthogonality
            max_completeness = completeness
            final_coeffs = results.x
            print("Gausslets : Orthogonality values after optimization  (closer to 1, the better)", orthogonality)
        if orthogonality >= 0.999999 and completeness >= 0.999999:
            break

    new_gausslet_basis = wavelet_bases.Gausslet_Basis(resolution=3, wavelet_coefficients = list(final_coeffs))
    overalap_matr = new_gausslet_basis.get_overlap_matrix(np.linspace(-10, 10, 200))
    
    import matplotlib.pyplot as plt

    plt.imshow(overalap_matr)
    plt.colorbar()
    plt.show()

    norm_factor = np.sqrt(np.trace(overalap_matr)/overalap_matr.shape[0])
    saved_wavelet_coeffs = new_gausslet_basis.wavelet_coefficients/norm_factor
    print(saved_wavelet_coeffs)
    return saved_wavelet_coeffs