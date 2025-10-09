# MRA FIT

Fit curves with high precision using [Multi resolution analysis](https://www.sciencedirect.com/topics/mathematics/multiresolution-analysis) and [Wavelet transform](https://en.wikipedia.org/wiki/Wavelet_transform). The module direclty implements Orthongonal Gausslets as described in the [paper](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.99.081110), however in addition the module also have hat basis function, and also allow to construct custom Gausslets.

# Installation

simply run

```pip install mrafit```

# Use cases

### Approximating curve using basis function
    
mrafit allows to reconstruct a curve by expressing them in terms of localized basis function. In other words a curve represented by list of points $x_i$, $y_i$ can be represented as a much smaller list of coefficients $c_i$ which reduces dimensionality of representation. Same can be done for any 3-d curve represneted by $x_i$, $y_i$, $z_i$.
 
### Quantum chemistry and molecular structure
Quantum chemisty invovlves calcuation elecotronic staets in Atoms and molecules with large number of electron using Schrodinger's equation, which makes it particularly challenging due complexity involved in electron-electron interactions. Orthogonal wavelet transformation can drastically simplify quantum chemistry calculation by reducing complexity from O(4) to O(2) while calculating electron interaction Potential 

### ML and AI
Since any 2-d, 3-d curve can be reduced to 1-d representation of of coeffiecients, this can provide massive computational advantage in ML problems where parametric learning of curves is required. Instead of learnign points, one can simply apply mrafit and learn mra coefficients instead.

# Examples

To use existing wavelet bases

```
import mrafit.wavelet_bases as wavelet_bases
```

From wavelet_bases, an instance of any available basis can be created, for example to use orthogonal gausslet basis, we can define

```
gb = wavelet_bases.Gausslet_Basis()
```

To approximate any given function defined over domain \(-1, 1\) with respect to a basis

```
X = np.linspace(-1, 1, 100)
coeffs, approx_func, error = gb.get_mra_approx(func, X)
```

The example below includes list of all steps to fit a synthetic function using mrafit
```
""" This paramter controls how preicely you want to approximate a function, smaller the value better the approximation"""
resolution = 0.5

wid = 10
N = 800
error_bound = 10e-2 * resolution

""" Use this section if you want to test gausslet with Stephen White's coefficients"""
gb = wavelet_bases.Gausslet_Basis(resolution=resolution, wavelet_coefficients=coeffs[int(len(coeffs)/2):])

""" Sample function to be approximated, you can change it as per your need"""
func = lambda x : np.exp(-x**2/3) * (x**2 - x + 1)

""" Finally applying the mra approximation"""
X = np.linspace(-wid, wid, N)
coeffs, approx_func, error = gb.get_mra_approx(func, X)

```

The image below shows the approximate function vs the actual function
![alt text](https://github.com/utksara/mrafit/blob/main/images/output.png)


