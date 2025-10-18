@doc raw"""
    GibbsKernel(; lengthscale)

Gibbs Kernel with lengthscale function `lengthscale`.

The Gibbs kernel is a non-stationary generalisation of the squared exponential
kernel. The lengthscale parameter ``l`` becomes a function of
position ``l(x)``.

# Definition

For inputs ``x, x'``, the Gibbs kernel with lengthscale function ``l(\cdot)``
is defined as
```math
k(x, x'; l) = \sqrt{\left(\frac{2 l(x) l(x')}{l(x)^2 + l(x')^2}\right)}
\quad \exp{\left(-\frac{(x - x')^2}{l(x)^2 + l(x')^2}\right)}.
```

For a constant function ``l \equiv c``, one recovers the [`SqExponentialKernel`](@ref)
with lengthscale `c`.

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

function (k::GibbsKernel)(x, y, base_kernel::Kernel=SqExponentialKernel)
    lengthscale = k.lengthscale
    lx = lengthscale(x)
    ly = lengthscale(y)
    l = invsqrt2 * hypot(lx, ly)
    kernel = (sqrt(lx * ly) / l) * with_lengthscale(base_kernel(), l)
    return kernel(x, y)
end
