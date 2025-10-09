import numpy as np
import mrafit.utils as utils
import yaml

def f_haar(x):
    if ( x < -0.5 ) : return 0
    if (x >= -0.5 and x <= 0.5): return 1
    if (x > 0.5): return 0


def f_hat(x, a = 0.5):
    if (x < -a): return 0
    if (x >=-a and x <= 0): return x/a + 1
    if (x > 0 and x < a): return 1 - x/a
    if (x >= a ): return 0


def approxintegral(y1, y2, dt):
    return np.inner(y1, y2)*dt

class Wavelet():
    def __init__(self, wavelet_func = f_haar, resolution = 1, dilation = 1, dim = 1, support = 1):
        self.support = support
        self.dilation = dilation
        self.scaling = dilation*1/resolution
        self.resolution = resolution
        self.wavelet_func = wavelet_func
        self.grids = None
        self.base_scale = 1
        self.dim = dim

    def _gridify(self, wavelets_per_dim, X):
        wavelet_grids = []
        # offset = 0.5*((X[-1] - X[0]) - n_wavelets*self.resolution)
        base_scale = self.support/(X[-1] - X[0])
        y_val = np.sqrt(base_scale)*np.array([self.wavelet_func(base_scale*(x - 0.5*(X[0] + X[-1]))) for x in X])
        scaling = self.dilation/(self.resolution*base_scale)
        
        if (self.dim == 2): 
            for i in range(0, wavelets_per_dim):
                wavelet_grids.append([])
                for j in range(0, wavelets_per_dim):
                    def temp_func(temp_i = i, temp_j = j) : return np.outer(self.get_wavefunc(y_val, utils.index_converter(temp_i, wavelets_per_dim), X, scaling), self.get_wavefunc(y_val, utils.index_converter(temp_j, wavelets_per_dim), X, scaling))
                    wavelet_grids[i].append(temp_func)
        
        if (self.dim == 1): 
            for i in range(0, wavelets_per_dim):
                def temp_func(temp_i = i) : return self.get_wavefunc(y_val, utils.index_converter(temp_i, wavelets_per_dim), X, scaling)
                wavelet_grids.append(temp_func)
        return wavelet_grids

    def get_wavefunc(self, y_val, i, X, scaling):
        return np.sqrt(scaling)*utils.scaled_sampling(func_val=y_val, X =X, translate= i*self.resolution, scaling = scaling)
        
    def get_num_wavelets(self, X):
        n_wavelets = int((X[-1] - X[0])/self.resolution)
        n_wavelets = 2*int(n_wavelets/2) + 1 # converting to odd number of wavelets
        return n_wavelets
    
    def get_wavefunc_data(self, indices, X):
        n_wavelets = self.get_num_wavelets(X)
        if self.grids is None:
            self.grids = self._gridify(wavelets_per_dim=n_wavelets, X = X)
        
        if (isinstance(indices, tuple)):
            if (len(indices) != self.dim):
                utils.raise_error(error_type=Exception, error_message="index dimension mismatch!")
            return self.grids[indices[0]][indices[1]]()
        
        if indices < -int(0.5*n_wavelets) or indices > int(0.5*n_wavelets): return None
        return self.grids[indices]()
        

    def get_overlap_matrix (self, X):
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
        X = np.linspace(-10*self.resolution, 10*self.resolution, 100)
        A = self.get_overlap_matrix(X)
        diag_norm_sq = np.sum(np.diag(A)**2)
        total_norm_sq = np.sum(A**2)
        score =  diag_norm_sq / total_norm_sq

        # if score >0.99 :
        #     print("very diagonal")
        # elif score > 0.90:
        #     print("almost diagonal")
        # else : print("non diagonal")
        return score
    

    def get_completeness(self):
        X = np.linspace(-3*self.resolution, 3*self.resolution, 100)
        y = np.zeros(100)
        st = 3
        for i in range(-st , st + 1):
            y += self.get_wavefunc_data(i, X)
        y = y[30:70]
        score =  1 - np.var(y)

        # if score >0.99 :
        #     print("very diagonal")
        # elif score > 0.90:
        #     print("almost diagonal")
        # else : print("non diagonal")
        return score
    
    def get_mra_approx(self, func, X):
        
        # X_augmented = np.linspace(X[0] - (X[-1] - X[0])/2, X[-1] + (X[-1] - X[0])/2, 2*len(X))
        func_val = np.array([func(x) for x in X])
        approx_func = np.zeros(len(X))
        dx = X[1] - X[0]
        norm_offset = approxintegral( self.get_wavefunc_data(0, X), self.get_wavefunc_data(0, X), dx)
        n_wavelets = self.get_num_wavelets(X)
        coeffs = np.zeros(n_wavelets) 
        
        if self.get_orthogonality() >= 0.99 :
            for i in range(0, n_wavelets):
                y_new =  self.get_wavefunc_data(i - int(n_wavelets/2), X)
                coeffs[i] = approxintegral(func_val, y_new, dx)/norm_offset
                approx_func +=  coeffs[i] * self.get_wavefunc_data(i - int(n_wavelets/2), X)
                # plt.plot(X, y_new)
        else:
            A = self.get_overlap_matrix(X)
            b = np.zeros(n_wavelets)
            for i in range(0, n_wavelets):
                y_new = self.get_wavefunc_data(i - int(n_wavelets/2), X)
                b[i] = approxintegral(func_val, y_new, dx)/norm_offset

            coeffs = np.linalg.inv(A) @ b
            for i in range(0, n_wavelets):
                approx_func +=  coeffs[i] *  self.get_wavefunc_data(i - int(n_wavelets/2), X)
        error = np.mean((approx_func - func_val)**2)
        return coeffs, approx_func, error


# data_loaded = {}
# with open("coeffs.yaml", 'r') as stream:
#     data_loaded = yaml.safe_load(stream)
    
G4_coefficients = [ 0.3503355455763915, 0.2653497528194342, -0.009493682948873735, 
    -0.08104201303620272, -0.023264880483848605, 0.005274123077909805, 0.023792334520446615, 
    -0.015191089433463032, 0.006951114919505941, -0.0023742278283798764, 0.00018221012749184177, 
    0.0005141714444618312, -0.00044080480876810234, 9.553999697818239e-05, -6.137393609482799e-05, 
    8.601393086578256e-05, -5.343801368492391e-05, 3.4016183940411115e-05, -1.985965403024716e-05, 
    8.052464633857624e-06, -3.2275842260623264e-06, 1.2698733862216701e-06, -4.48805902138133e-07, 
    1.051587241281154e-07, -6.45842667073618e-09, 3.323982326399466e-09, 1.012959330904841e-09, 
    -4.52644996207932e-09, 1.6865217510327753e-09, -6.443316774147104e-10, 3.837461530283764e-10, 
    -6.240302185603385e-11,]

U4coeffs =  [0.70063302 , 0.45201996 \
    , 0.19821879, -0.08288878, -0.05343915, -0.02585453 , 0.02258198 , 0.00609471 \
    , 0.00072552, -0.00102534, -0.00072453]

class Gausslet_Basis(Wavelet):

    def get_laplace_data(self, i, j, X, Y):
        data = np.zeros((len(X), len(Y))) 
        for x_i in range(0, len(X)):
            for y_i in range(0, len(Y)):
                x = X[x_i]
                y = Y[y_i]
                data[x_i][y_i] = self.wavelet_grids[i][j](x, y)*(x*x + y*y - 2)
        return data
    
    def get_overlap_matrix(self, X):
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
    
    def _gausslet_func(self, x, centre, scaling = 1):
        c = self.wavelet_coefficients
        rval = 0
        beta = 3
        sigma = 1/scaling**2
        for i in range(0, len(self.wavelet_coefficients)):
            r = (x - beta*centre/scaling - (i - 0.5*(len(self.wavelet_coefficients) - 1))/scaling)
            rval += c[abs(i)]*np.exp(-r*r/(2*sigma))
        return  np.sqrt(scaling)*rval
    
    def _haar_func(self, x, centre):
        c = self.wavelet_coefficients
        rval = 0
        for i in range(0, len(self.wavelet_coefficients)):
            r = (x - 3*centre - (i - 0.5*(len(self.wavelet_coefficients) - 1)))
            rval += c[abs(i)]*f_haar(r)
        return rval

    def _spline_func(self, x, centre):
        c = self.wavelet_coefficients
        rval = 0
        for i in range(0, len(self.wavelet_coefficients)):
            r = (x- 3*centre - (i - 0.5*(len(self.wavelet_coefficients) - 1)))
            rval += c[abs(i)]*f_hat(r)
        return rval
    
    def get_1d_wavefunc_data(self, centre, X, scaling = 1, type = "gaussian"):
        data = np.zeros(len(X))
        for i in range(0, len(X)):
            x = X[i]
            if (type == "spline"):
                data[i] = self._spline_func(x, centre, scaling)
            if (type == "haar"):
                data[i] = self._haar_func(x, centre, scaling)
            else:data[i] = self._gausslet_func(x, centre, scaling)
        return data

    def __init__(self, resolution = 1, dim = 1, dilation = 3, wavelet_coefficients = None):
        super().__init__(wavelet_func = lambda x : self._gausslet_func(x, 0, 1) \
                         , resolution = resolution, dilation = dilation, dim=dim, support=20)
        if wavelet_coefficients is not None:
            if wavelet_coefficients[0] == wavelet_coefficients[-1]:
                raise("Wrong format of wavelet_coefficients: input the semi-coeffiecients") 
            self.wavelet_coefficients = np.array(wavelet_coefficients[::-1][0:-1] + wavelet_coefficients)  
        else:
            self.wavelet_coefficients = np.array(G4_coefficients[::-1][0:-1] + G4_coefficients) 
class HaarBasis(Wavelet):
    def __init__(self, resolution = 1, dim = 1, dilation = 1):
        super().__init__(wavelet_func = f_haar, resolution = resolution \
                         , dilation = dilation, dim=dim)

rootpi = np.sqrt(1/np.pi)
class GaussianBasis(Wavelet):
    def __init__(self, resolution = 1, dim = 1, dilation = 1):
        super().__init__(wavelet_func = lambda r: np.exp(-0.5*r*r)*rootpi \
                         , resolution = resolution, dilation = dilation, dim=dim, support = 20)

class HatBasis(Wavelet):
    def __init__(self, wavelet_func=f_hat, resolution=1, dilation=0.5, dim=1):
        super().__init__(wavelet_func, resolution, dilation, dim, support= 1.6)


class U4Basis(Wavelet):
    
    def u4basis_func(self, x):
        rval = 0
        for i in range(0, len(self.wavelet_coefficients)):
            rval += self.wavelet_coefficients[i]*f_haar(x - (i - int(len(self.wavelet_coefficients)/2)))
        return rval

    def __init__(self, resolution = 1, dim = 1, dilation = 3):
        self.wavelet_coefficients = np.array(U4coeffs[::-1][0:-1] + U4coeffs)
        super().__init__(self.u4basis_func, resolution, dilation, dim, support= len(self.wavelet_coefficients))
        

class Ugeneric_Basis(Wavelet):
    
    def u4basis_func(self, x):
        rval = 0
        for i in range(0, len(self.wavelet_coefficients)):
            rval += self.wavelet_coefficients[i]*f_haar(x - (i - int(len(self.wavelet_coefficients)/2)))
        return rval

    def __init__(self, basis_coeffs, resolution = 1, dim = 1, dilation = 3):
        self.wavelet_coefficients = basis_coeffs
        super().__init__(self.u4basis_func, resolution, dilation, dim, support= len(self.wavelet_coefficients))
        