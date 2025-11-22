import numpy as np
import mrafit.utils as utils
from typing import Callable, Union, Literal
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------------
# Constant values:
#     Certain Wavelet bases require pre-calculated G4_coefficient values which are listed here
# --------------------------------------------------------------------------------------------

G4_coefficients = [ 0.3503355455763915, 0.2653497528194342, -0.009493682948873735, 
    -0.08104201303620272, -0.023264880483848605, 0.005274123077909805, 0.023792334520446615, 
    -0.015191089433463032, 0.006951114919505941, -0.0023742278283798764, 0.00018221012749184177, 
    0.0005141714444618312, -0.00044080480876810234, 9.553999697818239e-05, -6.137393609482799e-05, 
    8.601393086578256e-05, -5.343801368492391e-05, 3.4016183940411115e-05, -1.985965403024716e-05, 
    8.052464633857624e-06, -3.2275842260623264e-06, 1.2698733862216701e-06, -4.48805902138133e-07, 
    1.051587241281154e-07, -6.45842667073618e-09, 3.323982326399466e-09, 1.012959330904841e-09, 
    -4.52644996207932e-09, 1.6865217510327753e-09, -6.443316774147104e-10, 3.837461530283764e-10, 
    -6.240302185603385e-11,]

U4coeffs = [0.70063302 , 0.45201996, 0.19821879, -0.08288878, -0.05343915, -0.02585453 , 
            0.02258198 , 0.00609471, 0.00072552, -0.00102534, -0.00072453]

# --------------------------------------------------------------------------------------------
# Pre-defined basis functions:
# --------------------------------------------------------------------------------------------

def f_haar(x):
    if ( x < -0.5 ) : return 0
    if (x >= -0.5 and x <= 0.5): return 1
    if (x > 0.5): return 0

def f_hat(x, a = 1):
    if (x < -a): return 0
    if (x >=-a and x <= 0): return x/a + 1
    if (x > 0 and x < a): return 1 - x/a
    if (x >= a ): return 0

def f_sharpgauss(x):
    y = 0
    coeffs = [np.float64(-0.002839361613601305), np.float64(0.008793934142845222), np.float64(-0.017185949687694345), np.float64(0.025714038738946694), np.float64(-0.034419963712823756), np.float64(0.04188225893883571), np.float64(-0.04971829451544142), np.float64(0.057766981683443054), np.float64(-0.06859819077340289), np.float64(0.0832077615544251), np.float64(-0.10610150058244837), np.float64(0.1406640262572276), np.float64(-0.18693565419301592), np.float64(0.1943169599056), np.float64(0.06457093847852348), np.float64(0.19553866500764192), np.float64(0.36738050141078615), np.float64(0.19551521420229423), np.float64(0.0646099852438759), np.float64(0.1942758357155953), np.float64(-0.18690820873339042), np.float64(0.14066693867052293), np.float64(-0.1061510777897899), np.float64(0.08332080410873652), np.float64(-0.06879243570223072), np.float64(0.05806311934190555), np.float64(-0.050140664917501764), np.float64(0.042458618447651975), np.float64(-0.03517230290160782), np.float64(0.026630772769315865), np.float64(-0.018167560989438124), np.float64(0.009645399666036938), np.float64(-0.0034723272911514054)]
    for i in range(-int(len(coeffs)/2), int(len(coeffs)/2) + 1):
        y += coeffs[i + int(len(coeffs)/2)]*np.exp(-4*(x - i/4)**2)
    return y

def approxintegral(y1 : np.array, y2 : np.array, dt : float):
    y_larger = y1
    y_smallr = y2
    if len(y1) != len(y2):
        if len(y1) < len(y2) : 
            y_larger = y2
            y_smallr = y1 
        X = np.linspace(0, 1, len(y_larger))
        X_smallr = np.linspace(0, 1, len(y_smallr))
        y_smallr = np.interp(X, X_smallr, y_smallr)
    return np.inner(y_smallr, y_larger)*dt

# --------------------------------------------------------------------------------------------
# Utlity function:
# --------------------------------------------------------------------------------------------

def _console_print(statement):
    print("_/\\__/\\_" + statement)
# --------------------------------------------------------------------------------------------
# Wavelet classes:
# --------------------------------------------------------------------------------------------
class Wavelet():
    """
    This is general Wavelet class which contains functions common to all wavelet basis. ALl wavelet
    bases inherits this class
    """
    def __init__(self, wavefunction : Callable = f_haar, wavelet_type : str = "Haar", resolution : int = 1, dilation = 1, dim : int = 1):
        self.dilation = dilation
        self.resolution = resolution
        self.scaling = dilation*1/resolution
        self.wavefunction = wavefunction
        self.wavelet_type = wavelet_type
        self.grids = None
        self.base_scale = 1
        self.dim = dim
        self._state = {
            "X" : None,
            "resolution" : resolution
        }
        self._obser = {
            "orthogonality" : None,
            "overlap_matrix" : None
        }
        _console_print("_/\\__/\\__/\\__/\\__/\\__/\\__/\\__/\\_")
        _console_print(f"MRA started with {self.wavelet_type} wavelets")
        _console_print(f"resolution {self.resolution}; dilation {dilation}")
        
    def _update_state(self, update_dict):
        for key in update_dict.keys():
            self._state[key] = update_dict[key]
            
        # state_update of even single variable triggers nullification of observables
        self._obser["orthogonality"] = None
        

    def _gridify(self, wavelets_per_dim : int, X : np.array):
        wavelet_grids = []
        scaling = self.dilation/(self.resolution*self.base_scale)
        Y = np.sqrt(scaling)*np.array([self.wavefunction(scaling*(x - 0.5*(X[0] + X[-1]))) for x in X])
        dx = X[1] - X[0]
        norm_offset = np.sqrt(approxintegral(Y, Y, dx))
        Y = Y/norm_offset
        
        if (self.dim == 2): 
            for i in range(0, wavelets_per_dim):
                wavelet_grids.append([])
                for j in range(0, wavelets_per_dim):
                    def temp_function(temp_i = i, temp_j = j) :
                        Yi = self._eval_wavefunction(Y, utils.index_converter(temp_i, wavelets_per_dim), X, 1)
                        Yj = self._eval_wavefunction(Y, utils.index_converter(temp_j, wavelets_per_dim), X, 1)
                        return np.outer(Yi, Yj)
                    wavelet_grids[i].append(temp_function)
        
        if (self.dim == 1): 
            for i in range(0, wavelets_per_dim):
                def temp_function(temp_i = i) : 
                    return self._eval_wavefunction(Y, utils.index_converter(temp_i, wavelets_per_dim), X, 1)
                wavelet_grids.append(temp_function)
        
        return wavelet_grids

    def _eval_wavefunction(self, Y : np.array, i : int, X : np.array, scaling : float):
        return np.sqrt(scaling)*utils.scaled_sampling(function_val=Y, X = X, translate= i*self.resolution, scaling = scaling)
        
    def get_num_wavelets(self, X : np.array) -> int:
        """
        Get total number of basis functions (wavelets) inside give domain 
        Args:
            X : uniform 1-d space (np.array)
        Returns :
            n_wavelets : number of wavelets (int)
        """
        n_wavelets = int((X[-1] - X[0])/self.resolution)
        n_wavelets = 2*int(n_wavelets/2) + 1 # converting to odd number of wavelets
        return n_wavelets
    
    def get_wavefunction_values(self, indices : int, X : np.array):
        n_wavelets = self.get_num_wavelets(X)
        if self.grids is None:
            self.grids = self._gridify(wavelets_per_dim=n_wavelets, X = X)
        
        if (isinstance(indices, tuple)):
            if (len(indices) != self.dim):
                utils.raise_error(error_type=Exception, error_message="index dimension mismatch!")
            return self.grids[indices[0]][indices[1]]()
        
        if indices < -int(0.5*n_wavelets) or indices > int(0.5*n_wavelets): return None
        return self.grids[indices]()
        
    def get_overlap_matrix(self, X : np.array):
        dx = X[1] - X[0]
        n_wavelets = self.get_num_wavelets(X)
        wavelets = self._gridify(n_wavelets, X)
        overlap_matrix = []
        overlap_cache = {}
        n_half = int(0.5*n_wavelets)
        if self.dim == 1:
            overlap_matrix = np.zeros((n_wavelets, n_wavelets))
            for i in range(0, n_wavelets):
                for j in range(0, n_wavelets):
                    i_mod = utils.index_converter(i, n_wavelets)
                    j_mod = utils.index_converter(j, n_wavelets)
                    if (abs(i_mod - j_mod) not in overlap_cache):
                        overlap_cache[abs(i_mod - j_mod)] = approxintegral(wavelets[i_mod]() \
                                                                 , wavelets[j_mod](), dx)
                    overlap_matrix[(i + n_half)%n_wavelets, (j + n_half)%n_wavelets] =  overlap_cache[abs(i_mod - j_mod)] 
        return overlap_matrix
    
    def get_orthogonality(self):
        """
        Prints orthogonality scrore of a given basis-
        
        If score >= 0.99, the overlap matrix is highly diagonal and hence the basis is orthogonal
        If score > 0.90, the overlap matrix is mildly diagonal and the basis is almost diagonal
        In other cases, the basis is non diagonal
        """
        if self._obser["orthogonality"] is None:
            X = self._state["X"]
            A = self.get_overlap_matrix(X)
            diag_norm_sq = np.sum(np.diag(A)**2)
            total_norm_sq = np.sum(A**2)
            self._obser["orthogonality"] =  diag_norm_sq / total_norm_sq
        return self._obser["orthogonality"]
    
    def get_reconstructed_func(self, coeffs : Union[list, np.array], X : np.array):
        n_wavelets = self.get_num_wavelets(X)
        resized_coeffs = 1*np.ones(n_wavelets)
        resized_coeffs[0:len(coeffs)] = coeffs
        reconstructed_value = np.zeros(len(X))
        for i in range(0, n_wavelets):
            reconstructed_value +=  resized_coeffs[i] *  self.get_wavefunction_values(i - int(n_wavelets/2), X)
        return reconstructed_value

    def get_completeness(self):
        """
        measure of completeness
        Returns:
            Score between 0(least complete) to 1 (most complete) 
        """
        X = np.linspace(-3*self.resolution, 3*self.resolution, 100)
        y = np.zeros(100)
        st = 3
        for i in range(-st , st + 1):
            y += self.get_wavefunction_values(i, X)
        y = y[30:70]
        score =  1 - np.var(y)
        return score
    
    def get_mra_approx(self, function : Callable, X : np.array, plotting_enabled = False) -> tuple[np.array, np.array, np.array]:
        """
            Get multi-resolution approximation of any given function
            Args:
                function : a callable function of function values (function or numpy array)
                X : Uniform 1-d grid (numpy array)
            
            Returns:
                coeffs : values of coefficients of given basis function (numpy array)
                approx_function : values of approximated function using given basis (numpy array)
                error : error of the approximate values with respect to original value of function
            
        """
        function_val = None
        if isinstance(function, Callable):        
            function_val = np.array([function(x) for x in X])
        elif isinstance(function, np.array):
            function_val = function
            assert len(function_val) == len(X), "sizes of function and X do not match!"
        else: raise TypeError("Invalid data type for function")
        
        self._update_state({"X" : X})
        approx_function = np.zeros(len(X))
        dx = X[1] - X[0]
        n_wavelets = self.get_num_wavelets(X)
        coeffs = np.zeros(n_wavelets) 
        
        if self.get_orthogonality() >= 0.99 :
            _console_print(f"Orthogonal basis (score = {self.get_orthogonality()}): Calculating coefficients of input function")
            Y0 = self.get_wavefunction_values(0, X)
            _console_print("normalization offset value : " + str(approxintegral(Y0, Y0, X[1] - X[0])))
            _console_print("total basis function : " + str(n_wavelets))
            for i in range(0, n_wavelets):
                Yi = self.get_wavefunction_values(i - int(n_wavelets/2), X)
                coeffs[i] = approxintegral(function_val, Yi, dx)
                # _console_print("calculating approx function : " + str(i))
                approx_function += coeffs[i] * self.get_wavefunction_values(i - int(n_wavelets/2), X)
        else:
            _console_print(f"Non orthogonal basis (score = {self.get_orthogonality()}): Calculating overlap matrix")
            _console_print("total basis function : " + str(n_wavelets))
            A = self.get_overlap_matrix(X)
            b = np.zeros(n_wavelets)
            for i in range(0, n_wavelets):
                y_new = self.get_wavefunction_values(i - int(n_wavelets/2), X)
                b[i] = approxintegral(function_val, y_new, dx)

            _console_print("Calculating inverse of overlap matrix")
            coeffs = np.linalg.inv(A) @ b
            
            _console_print("Calculating coefficients of input function")
            for i in range(0, n_wavelets):
                approx_function +=  coeffs[i] *  self.get_wavefunction_values(i - int(n_wavelets/2), X)
        error = np.mean((approx_function - function_val)**2)
        _console_print(f"MRA Complete! : mean error = {error}")
        _console_print("_/\\__/\\__/\\__/\\__/\\__/\\__/\\__/\\_")
        
        if plotting_enabled:
            for i in range(0, n_wavelets):
                plt.plot(X, coeffs[i] * self.get_wavefunction_values(i - int(n_wavelets/2), X))
        return coeffs, approx_function, error

class GaussletBasis(Wavelet):
    """
    Gausslet Basis : Basis functions composed of summation gaussian of form c_i x exp(-(x - a_i)^2/2)
    where i denotes indices ranging from 0 to 2*n + 1 or -n to n (based on convention).
    """
    
    def _gausslet_function(self, x : float, centre : float, scaling : float = 1):
        c = self.wavelet_coefficients
        rval = 0
        beta = 3
        sigma = 1/scaling**2
        for i in range(0, len(self.wavelet_coefficients)):
            r = (x - beta*centre/scaling - (i - 0.5*(len(self.wavelet_coefficients) - 1))/scaling)
            rval += c[abs(i)]*np.exp(-r*r/(2*sigma))
        return  np.sqrt(scaling)*rval
    
    def _haar_function(self, x : float, centre : float):
        c = self.wavelet_coefficients
        rval = 0
        for i in range(0, len(self.wavelet_coefficients)):
            r = (x - 3*centre - (i - 0.5*(len(self.wavelet_coefficients) - 1)))
            rval += c[abs(i)]*f_haar(r)
        return rval

    def _spline_function(self, x : float, centre : float):
        c = self.wavelet_coefficients
        rval = 0
        for i in range(0, len(self.wavelet_coefficients)):
            r = (x- 3*centre - (i - 0.5*(len(self.wavelet_coefficients) - 1)))
            rval += c[abs(i)]*f_hat(r)
        return rval
    
    def get_laplacian_values(self, i : int, j : int, X : np.array, Y : np.array):
        data = np.zeros((len(X), len(Y))) 
        for x_i in range(0, len(X)):
            for y_i in range(0, len(Y)):
                x = X[x_i]
                y = Y[y_i]
                data[x_i][y_i] = self.wavelet_grids[i][j](x, y)*(x*x + y*y - 2)
        return data
    
    def get_overlap_matrix(self, X : np.array):
        """
        A seperate function to obtain overlap matrix is created for Gausslet in order to have
        more efficient and accurate calculation
        """
        n_wavelets = self.get_num_wavelets(X)
        L = len(self.wavelet_coefficients)
        M = np.outer(self.wavelet_coefficients, self.wavelet_coefficients)
        gausslet_integral = []
        M_orth = np.zeros((L,L))
        overlap_matrix = np.zeros((n_wavelets, n_wavelets))
        for d in range(0, n_wavelets):
            for i in range(0, L):
                for j in range(0, L):
                    d_gausslet = self.dilation*d
                    d_gaussian =  (i - j)  +  d_gausslet
                    M_orth[i, j] = np.exp( -(0.5*d_gaussian)**2) * M[i, j]
            gausslet_integral.append(np.sum(M_orth))

        for i in range(0, n_wavelets):
            for j in range(0, n_wavelets):
                overlap_matrix[i,j] = self.scaling*gausslet_integral[abs(i-j)]
        return overlap_matrix
    
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # deprecated
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    
    # def get_wavefunction_values(self, centre : float, X : np.array, scaling : float = 1, type : Literal["gaussian", "spline", "haar"] = "gaussian"):
    #     data = np.zeros(len(X))
    #     for i in range(0, len(X)):
    #         x = X[i]
    #         if (type == "spline"):
    #             data[i] = self._spline_function(x, centre, scaling)
    #         if (type == "haar"):
    #             data[i] = self._haar_function(x, centre, scaling)
    #         else:data[i] = self._gausslet_function(x, centre, scaling)
    #     return data
    
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def __init__(self, resolution = 1, dim = 1, dilation = 3, wavelet_coefficients = None):
        super().__init__(wavefunction = lambda x : self._gausslet_function(x, 0, 1) \
                         , resolution = resolution, dilation = dilation, dim=dim)
        if wavelet_coefficients is not None:
            if wavelet_coefficients[0] == wavelet_coefficients[-1]:
                raise("Wrong format of wavelet_coefficients: input the semi-coeffiecients") 
            self.wavelet_coefficients = np.array(wavelet_coefficients[::-1][0:-1] + wavelet_coefficients)  
        else:
            self.wavelet_coefficients = np.array(G4_coefficients[::-1][0:-1] + G4_coefficients) 
class HaarBasis(Wavelet):
    def __init__(self, resolution = 1, dim = 1):
        super().__init__(wavefunction = f_haar, resolution = resolution \
                         , dilation = 3, dim=dim)

rootpi = np.sqrt(1/np.pi)
class GaussianBasis(Wavelet):
    def __init__(self, resolution = 1, dim = 1, dilation = 1):
        super().__init__(wavefunction = lambda r: np.exp(-0.5*r*r)/rootpi, wavelet_type="GaussianBasis" \
                         , resolution = resolution, dilation = dilation, dim=dim)
class HatBasis(Wavelet):
    def __init__(self, resolution : float = 1, dim : int = 1):
        super().__init__(wavefunction=f_hat, resolution = resolution, dilation = 0.5, dim=dim, wavelet_type="HatBasis")

class GaussPleteauBasis(Wavelet):
    def __init__(self, resolution = 1):
        super().__init__(wavefunction=f_sharpgauss, resolution = resolution, dilation=1, dim=1, wavelet_type="GaussPleteauBasis")
class U4Basis(Wavelet):
    def u4basis_function(self, x):
        rval = 0
        for i in range(0, len(self.wavelet_coefficients)):
            rval += self.wavelet_coefficients[i]*f_haar(x - (i - int(len(self.wavelet_coefficients)/2)))
        return rval

    def __init__(self, resolution = 1, dim = 1):
        self.wavelet_coefficients = np.array(U4coeffs[::-1][0:-1] + U4coeffs)
        super().__init__(self.u4basis_function, resolution, dilation = 3, dim = dim, wavelet_type="U4Basis")
        
class Ugeneric_Basis(Wavelet):
    def u4basis_function(self, x):
        rval = 0
        for i in range(0, len(self.wavelet_coefficients)):
            rval += self.wavelet_coefficients[i]*f_haar(x - (i - int(len(self.wavelet_coefficients)/2)))
        return rval

    def __init__(self, basis_coeffs, resolution = 1, dim = 1, dilation = 3):
        self.wavelet_coefficients = basis_coeffs
        super().__init__(self.u4basis_function, resolution, dilation, dim, wavelet_type="Ugeneric_Basis")
        