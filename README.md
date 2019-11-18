[![Build Status](https://travis-ci.org/theogf/KernelFunctions.jl.svg?branch=master)](https://travis-ci.org/theogf/AugmentedGaussianProcesses.jl)
[![Coverage Status](https://coveralls.io/repos/github/theogf/KernelFunctions.jl/badge.svg?branch=master)](https://coveralls.io/github/theogf/KernelFunctions.jl?branch=master)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://theogf.github.io/KernelFunctions.jl/dev/)
# KernelFunctions.jl
## Kernel functions for machine learning

KernelFunctions.jl provide a flexible and complete framework for kernel functions, pretransforming the input data.

The aim is to make the API as model-agnostic as possible while still being user-friendly.

## Examples

```julia
  X = reshape(collect(range(-3.0,3.0,length=100)),:,1)
  # Set simple scaling of the data
  k₁ = SqExponentialKernel(1.0)
  K₁ = kernelmatrix(k,X,obsdim=1)

  # Set a function transformation on the data
  k₂ = MaternKernel(FunctionTransform(x->sin.(x)))
  K₂ = kernelmatrix(k,X,obsdim=1)

  # Set a matrix premultiplication on the data
  k₃ = PolynomialKernel(LowRankTransform(randn(4,1)),0.0,2.0)
  K₃ = kernelmatrix(k,X,obsdim=1)

  # Add and sum kernels
  k₄ = 0.5*SqExponentialKernel()*LinearKernel(0.5) + 0.4*k₂
  K₄ = kernelmatrix(k,X,obsdim=1)

  plot(heatmap.([K₁,K₂,K₃,K₄],yflip=true,colorbar=false)...,layout=(2,2))
```
<p align=center>
  <img src="docs/src/assets/heatmap_combination.png" width=400px>
</p>

## Packages goals (by priority)
- Ensure AD Compatibility (already the case for Zygote, ForwardDiff)
- Toeplitz Matrices compatibility
- BLAS backend

Directly inspired by the [MLKernels](https://github.com/trthatcher/MLKernels.jl) package.

## Issues/Contributing

If you notice a problem or would like to contribute by adding more kernel functions or features please [submit an issue](https://github.com/theogf/KernelFunctions.jl/issues).
