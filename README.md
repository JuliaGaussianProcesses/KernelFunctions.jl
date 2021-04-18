# KernelFunctions.jl

![CI](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl/workflows/CI/badge.svg?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/JuliaGaussianProcesses/KernelFunctions.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaGaussianProcesses/KernelFunctions.jl?branch=master)
[![Documentation (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliagaussianprocesses.github.io/KernelFunctions.jl/stable)
[![Documentation (latest)](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliagaussianprocesses.github.io/KernelFunctions.jl/dev)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)


## Kernel functions for machine learning

KernelFunctions.jl provide a flexible and complete framework for kernel functions, pretransforming the input data.

The aim is to make the API as model-agnostic as possible while still being user-friendly.

## Examples

```julia
x = range(-3.0, 3.0; length=100)

# A simple standardised squared-exponential / exponentiated-quadratic kernel.
k₁ = SqExponentialKernel()
K₁ = kernelmatrix(k₁, x)

# Set a function transformation on the data
k₂ = Matern32Kernel() ∘ FunctionTransform(x -> sin.(x))
K₂ = kernelmatrix(k₂, x)

# Set a matrix premultiplication on the data
k₃ = PolynomialKernel(; c=2.0, degree=2) ∘ LinearTransform(randn(4, 1))
K₃ = kernelmatrix(k₃, x)

# Add and sum kernels
k₄ = 0.5 * SqExponentialKernel() * LinearKernel(; c=0.5) + 0.4 * k₂
K₄ = kernelmatrix(k₄, x)

plot(
    heatmap.([K₁, K₂, K₃, K₄]; yflip=true, colorbar=false)...;
    layout=(2, 2), title=["K₁" "K₂" "K₃" "K₄"],
)
```
<p align=center>
  <img src="docs/src/assets/heatmap_combination.png" width=400px>
</p>

## Packages goals (by priority)
- Ensure AD Compatibility (already the case for Zygote, ForwardDiff)
- Toeplitz Matrices compatibility

Directly inspired by the [MLKernels](https://github.com/trthatcher/MLKernels.jl) package.

## Issues/Contributing

If you notice a problem or would like to contribute by adding more kernel functions or features please [submit an issue](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl/issues), or open a PR.
