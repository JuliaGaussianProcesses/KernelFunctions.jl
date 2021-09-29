"""
    GibbsKernel(x, y)

# Definition

The Gibbs kernel is non-stationary generalisation of the Squared-Exponential
kernel. The lengthscale parameter ``l`` becomes a function of
position ``l(x)``.

``l(x) = 1.`` then you recover the standard Squared-Exponential kernel
with constant lengthscale.

```math
k(x, y) = \\sqrt{ \\left(\\frac{2 l(x) l(y)}{l(x)^2 + l(y)^2} \\right) }
\\quad \\rm{exp} \\left( - \\frac{(x - y)^2}{l(x)^2 + l(y)^2} \\right)
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

struct GibbsKernel{T} <: Kernel
    lengthscale::T
end

GibbsKernel(; lengthscale) = GibbsKernel(lengthscale)

function (k::GibbsKernel)(x, y)
    lengthscale = k.lengthscale
    lx = lengthscale(x)
    ly = lengthscale(y)
    l = hypot(lx, ly)
    kernel = sqrt(2 * lx * ly) / l * with_lengthscale(SqExponentialKernel(), l)
    return kernel(x, y)
end