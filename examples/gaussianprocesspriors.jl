# -*- coding: utf-8 -*-
# # Gaussian process prior samples
#
# The kernels defined in this package can also be used to specify the
# covariance of a Gaussian process prior.
# A Gaussian process (GP) is defined by its mean function $m(\cdot)$ and its covariance function or kernel $k(\cdot, \cdot')$:
# ```math
#   f \sim \mathcal{GP}\big(m(\cdot), k(\cdot, \cdot')\big)
# ```
# The function values of the GP at a finite number of points $X = \{x_n\}_{n=1}^N$ follow a multivariate normal distribution with mean vector $\mathbf{m}$ and covariance matrix $\mathrm{K}$, where
# ```math
# \begin{aligned}
#   \mathbf{m}_i &= m(x_i) \\
#   \mathrm{K}_{i,j} &= k(x_i, x_j)
# \end{aligned}
# ```
# where $1 \le i, j \le N$.
#
# In this notebook we show samples from zero-mean GPs with different kernels.

## Load required packages
using KernelFunctions
using LinearAlgebra
using Distributions
using Plots
default(; lw=1.0, legendfontsize=15.0)
using Random: seed!
seed!(42);

# We now define a function that visualizes a kernel
# for us. We use the same randomness to obtain
# comparable samples.

num_inputs = 101
xlim = (-5, 5)
X = reshape(collect(range(xlim...; length=num_inputs)), :, 1)
num_samples = 11
v = randn(num_inputs, num_samples);

function visualize(k::Kernel; xref=0.0)
    K = kernelmatrix(k, X; obsdim=1)
    L = cholesky(K + 1e-6 * I)
    f = L.L * v

    p_kernel_2d = heatmap(
        K;
        yflip=true,
        colorbar=false,
        ylabel=string(k),
        framestyle=:none,
        #color=:blues,
        vlim=(0, 1),
        title=raw"$k(x, x')$",
    )

    p_kernel_cut = plot(
        X,
        k.(X, xref);
        title=string(raw"$k(x, ", xref, raw")$"),
        xlim=xlim,
        xticks=(xlim, xlim),
        label=nothing,
    )

    p_samples = plot(
        X,
        f;
        c="blue",
        title=raw"$f(x)$",
        ylim=(-3, 3),
        xlim=xlim,
        xticks=(xlim, xlim),
        label=nothing,
    )

    return plot(p_kernel_2d, p_kernel_cut, p_samples; layout=(1, 3), xlabel=raw"$x$")
end

# We can now visualize a kernel and show samples from
# a Gaussian process with this kernel:

visualize(SqExponentialKernel())

# This also allows us to compare different kernels:

kernel_classes = [Matern12Kernel, Matern32Kernel, Matern52Kernel, SqExponentialKernel]
plot(
    [visualize(k()) for k in kernel_classes]...,
    #layout=(length(kernel_classes), 1)
)
