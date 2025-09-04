import numpy as np
from scipy.optimize import minimize
import itertools

swap = np.array([
    [0, 1],
    [1, 0]])

'''
gate/unitary to rotate by angle pi/4
'''
pi4 = (1/np.sqrt(2))*np.array([
    [1,  1],
    [1, -1]])

def get_unitary(theta):
    '''
    Creates a 3x3 unitary Matrix of form-

    [cost + 1,    root2*sint,   cost - 1],
    [-root2*sint, 2*cost,    -root2*sint],
    [cost - 1,    root2*sint,   cost + 1]

    '''

    root2 = np.sqrt(2)
    cost = np.cos(theta)
    sint = np.sin(theta)
    return 0.5*np.array([
        [cost + 1,    root2*sint,   cost - 1],
        [-root2*sint, 2*cost,    -root2*sint],
        [cost - 1,    root2*sint,   cost + 1]
    ])

class OrthogonalCoefficientsTernary():
    '''
    Creates an object for scaling-wavelet pair of coefficients  
    '''
    def __init__(self, n, factor):
        self.sitesym = np.zeros((factor*(2*n - 1), 1))
        self.edgesym_p = np.zeros((factor*(2*n), 1))
        self.edgesym_m = np.zeros((factor*(2*n), 1))
        self.length = len(self.sitesym)

    def sum_function(self, func, x):
        v = 0
        X = np.linspace(-len(self.sitesym)/2, len(self.sitesym)/2, len(self.sitesym))
        for i in range(0, len(X)):
            h = self.sitesym[i]
            v += h*func(2*x - X[i])
        return v*np.sqrt(3)
   
    def verify_orthogonality(self, scaling_factor):
        i = 0
        inner_product = []
        while(i < self.length):
            inner_product.append(np.inner(self.sitesym[i:], self.sitesym[:self.length - i]))
            i += scaling_factor
        return inner_product
    
    def multiscale_func(self, func, order, x):
        if order == 0 :
            return func(x)
        if order == 1 :
            return self.sum_function(func, x)
        return self.sum_function(lambda t: self.multiscale_func(func, order - 1, t), x) 
    
    def multiwave_func(self, func, order, x):
        v = 0
        for i in range(0, len(self.wavelet)):
            g = self.wavelet[i]
            v += g*self.multiscale_func(func, order, 2*x - i)
        return v*np.sqrt(2)

class CircuitTernary():
    '''
    Class to create a Unitary circuit of arbitrary depth
    to calcualte scaling and wavelet coefficients for ternary
    circuits
    '''
    def __init__(self, theta):
        self.depth = len(theta)
        self.theta = np.array(theta)
        self.coefficients = OrthogonalCoefficientsTernary(self.depth, 3)
        self.calculated_coefficients = None
        if self.theta is not None:
            self.theta = theta

    def moment_calcuator(self):
        n = np.size(self.theta, 0)
        L = self.coefficients.length + 3
        R = np.zeros((n, L))
        r = np.linspace(1, L, L)
        for i in range(0, n):
            R[i,:] = np.power(r, i)
        moments_m = np.matmul(R, self.coefficients.edgesym_m)
        moments_p = np.matmul(R, self.coefficients.edgesym_p)
        return np.concatenate((moments_m, moments_p), axis= 0)

    def calculate_coefficients(self):
        '''
        In case of ternary wavelets there are some changes in terms of
        scaling and wavelet coefficients. There are cite cetric (s+) 
        coefficients and a pair of edge centric (b+, b-) coefficients
        for a given depth

        example : for depth N = 2
        we circuits beginning with 
        s+ = [0 0 0 0 1 0 0 0 0]
        b+ = [0 0 0 0 0 0 1 0 0 0 0 0]
        b- = [0 0 0 0 0 1 0 0 0 0 0 0]

        and each of them (s, b) have slightly different circuits
        '''
        
        '''
        First calculating s+
        '''
        L = np.size(self.coefficients.sitesym, 0)
        L_half = int(L/2) 
        self.coefficients.sitesym[L_half] = 1 # initializing scaling coefficients
        u_start = L_half - 1
        s_start = u_start - 1 
        for i in range(0, 2*self.depth - 1):
            if (i%2 == 0):
                # print("\n")
                j = u_start
                for _ in range(0, i + 1):
                    # print(j, j+1, j+2, end=", ")
                    self.coefficients.sitesym[j:j + 3] = np.matmul(get_unitary(self.theta[int(i/2)]), self.coefficients.sitesym[j:j + 3])
                    j = j + 3
                u_start = u_start - 3
            else:
                # print("\n")
                j = s_start
                for _ in range(0, i+1):
                    # print(j, j+1, end=", ")
                    self.coefficients.sitesym[j:j + 2] = np.matmul(swap, self.coefficients.sitesym[j:j + 2])
                    j = j + 3
                s_start = u_start - 1

        self.coefficients.sitesym = self.coefficients.sitesym.flatten()
        '''
        Calculating b+ b-
        '''
        L = np.size(self.coefficients.edgesym_p, 0)
        L_half = int(L/2)  
        self.coefficients.edgesym_p[L_half] = 1
        self.coefficients.edgesym_m[L_half - 1] = 1
        
        self.coefficients.edgesym_m[L_half - 1: L_half + 1] = np.matmul(pi4, self.coefficients.edgesym_m[L_half - 1: L_half + 1])
        self.coefficients.edgesym_p[L_half - 1: L_half + 1] = np.matmul(pi4, self.coefficients.edgesym_p[L_half - 1: L_half + 1])
        
        u_start = L_half - 3
        s_start = u_start - 1

        def _edge_transfor(coeffs, depth, theta, u_start, s_start):
            for i in range(0, 2*depth - 1):
                if (i%2 == 0):
                    j = u_start
                    # print("\n")
                    # print(int(i/2))
                    for _ in range(0, i + 2):
                        # print(j, j+1, j+2, end=", ")
                        coeffs[j:j + 3] = np.matmul(get_unitary(theta[int(i/2)]), coeffs[j:j + 3])
                        j = j + 3
                    u_start = u_start - 3
                else:
                    j = s_start
                    # print("\n")
                    for _ in range(0, i + 2):
                        # print(j, j+1, end=", ")
                        coeffs[j:j + 2] = np.matmul(swap, coeffs[j:j + 2])
                        j = j + 3
                    s_start = u_start - 1
                # print(np.transpose(coeffs))
            return coeffs
        
        self.coefficients.edgesym_p = _edge_transfor(self.coefficients.edgesym_p, self.depth, self.theta, u_start, s_start)
        self.coefficients.edgesym_m = _edge_transfor(self.coefficients.edgesym_m, self.depth, self.theta, u_start, s_start)
          
'''
Defining objective function to minimize second upto first moment (n = 2)
'''
def objective(theta):
    circ = CircuitTernary(theta)
    circ.calculate_coefficients()
    moments = circ.moment_calcuator()
    # print(np.power(moments, 2))
    return np.sum(np.power(moments, 2)) # squaring values of moment to ensure optima is achieved for postive value of moment

saved_unitary_coeffs = None
def get_ugeneric_coefficients(depth = 4):
    global saved_unitary_coeffs
    if saved_unitary_coeffs is not None:
        return saved_unitary_coeffs
    bounds = []
    min_moment = 10
    # theta = None # expected theta = [0.595157579, -0.840085482, -0.314805259, 1.515049781]
    value_set = {- 1, -0.5, 0, 0.5, 1}
    thetas = itertools.product(value_set, repeat=depth)
    for theta in thetas:
        theta = [0.595157579, -0.840085482, -0.314805259, 1.515049781]
        bounds = [(-np.pi/2, np.pi/2) for _ in range(depth)]
        results = minimize(objective, list(theta), bounds=bounds, method = 'Nelder-Mead', options={'maxiter':1000}, tol=10e-20)
        circ = CircuitTernary(results.x)
        circ.calculate_coefficients()
        current_moment = circ.moment_calcuator()
        if np.max(current_moment) < min_moment:
            min_moment = np.max(circ.moment_calcuator())
            theta = results.x
            print("Unitary circuit: Moment values after optimization (closer to 0, the better)", np.max(current_moment))
        if np.max(current_moment) < 1e-12:
            break
    circ = CircuitTernary(theta)
    circ.calculate_coefficients()
    saved_unitary_coeffs = circ.coefficients.sitesym
    return saved_unitary_coeffs