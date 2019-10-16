[![Build Status](https://travis-ci.org/theogf/KernelFunctions.jl.svg?branch=master)](https://travis-ci.org/theogf/AugmentedGaussianProcesses.jl)
[![Coverage Status](https://coveralls.io/repos/github/theogf/KernelFunctions.jl/badge.svg?branch=master)](https://coveralls.io/github/theogf/KernelFunctions.jl?branch=master)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://theogf.github.io/KernelFunctions.jl/dev/)
# KernelFunctions.jl
## Kernel functions for machine learning

KernelFunctions.jl provide a flexible and complete framework for kernel functions, pretransforming the input data.

The aim is to make the API as model-agnostic as possible while still being user-friendly.

## Objectives (by priority)
- ARD Kernels
- AD Compatible (Zygote, ForwardDiff, ReverseDiff)
- Kernel sum and product
- Toeplitz Matrices
- BLAS backend


Directly inspired by the [MLKernels](https://github.com/trthatcher/MLKernels.jl) package
