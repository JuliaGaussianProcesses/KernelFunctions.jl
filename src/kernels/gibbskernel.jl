"""

    GibbsKernel(x, y)

# Definition

The Gibbs kernel is non-stationary generalisation of the Squared-Exponential
kernel. The lengthscale parameter ``\\ell`` becomes a function of
position ``\\ell(x)``.

``\\ell(x) = 1.`` then you recover the standard Squared-Exponential kernel
with constant lengthscale.

```math
k(x, y) = \\sqrt{ \\left( \\frac{2 \\ell(x) \\ell(y)}{\\ell(x)^2 + \\ell(y)^2} \\right) } \\quad \\rm{exp} \\left( - \\frac{(x - y)^2}{\\ell(x)^2 + \\ell(y)^2} \\right)
```

[1] - Mark N. Gibbs. "Bayesian Gaussian Processes for Regression and Classication."
    PhD thesis, 1997

[2] - Christopher J. Paciorek and Mark J. Schervish. "Nonstationary Covariance
    Functions for Gaussian Process Regression". NEURIPS, 2003

[3] - Sami Remes, Markus Heinonen, Samuel Kaski.
    "Non-Stationary Spectral Kernels". arXiV:1705.08736, 2017

[4] - Sami Remes, Markus Heinonen, Samuel Kaski.
    "Neural Non-Stationary Spectral Kernel". arXiv:1811.10978, 2018
"""
function GibbsKernel(x::Any, y::Any, ell::Function)
    fac = ell(x)^2 + ell(y)^2
    term1 = 2.0 * ell(x) * ell(y) / fac
    term2 = - (x-y)^2 / fac
    return term1 * exp(term2)
end
