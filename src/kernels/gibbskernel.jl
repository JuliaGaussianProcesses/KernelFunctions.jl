"""
    GibbsKernel(; lengthscale)

# Definition

The Gibbs kernel is non-stationary generalisation of the squared exponential
kernel. The lengthscale parameter ``l`` becomes a function of
position ``l(x)``.

For a constant function``l(x) = c``, one recovers the standard squared exponential kernel
with lengthscale `c`.

```math
k(x, y; l) = \\sqrt{ \\left(\\frac{2 l(x) l(y)}{l(x)^2 + l(y)^2} \\right) }
\\quad \\rm{exp} \\left( - \\frac{(x - y)^2}{l(x)^2 + l(y)^2} \\right)
```

# References

Mark N. Gibbs. "Bayesian Gaussian Processes for Regression and Classication." PhD thesis, 1997

Christopher J. Paciorek and Mark J. Schervish. "Nonstationary Covariance Functions
for Gaussian Process Regression". NeurIPS, 2003

Sami Remes, Markus Heinonen, Samuel Kaski. "Non-Stationary Spectral Kernels". arXiV:1705.08736, 2017

Sami Remes, Markus Heinonen, Samuel Kaski. "Neural Non-Stationary Spectral Kernel". arXiv:1811.10978, 2018
"""
struct GibbsKernel{T} <: Kernel
    lengthscale::T
end

GibbsKernel(; lengthscale) = GibbsKernel(lengthscale)

function (k::GibbsKernel)(x, y)
    lengthscale = k.lengthscale
    lx = lengthscale(x)
    ly = lengthscale(y)
    l = invsqrt2 * hypot(lx, ly)
    kernel = (sqrt(lx * ly) / l) * with_lengthscale(SqExponentialKernel(), l)
    return kernel(x, y)
end
