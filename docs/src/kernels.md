```@meta
  CurrentModule = KernelFunctions
```

# Kernel Functions

## [Base Kernels](@id base_kernels)

These are the basic kernels without any transformation of the data. They are the building blocks of KernelFunctions.

### Constant Kernels

```@docs
ZeroKernel
ConstantKernel
WhiteKernel
EyeKernel
```

### Cosine Kernel

```@docs
CosineKernel
```

### Exponential Kernels

```@docs
ExponentialKernel
GibbsKernel
LaplacianKernel
SqExponentialKernel
SEKernel
GaussianKernel
RBFKernel
GammaExponentialKernel
```

### Exponentiated Kernel

```@docs
ExponentiatedKernel
```

### Fractional Brownian Motion Kernel

```@docs
FBMKernel
```

### Gabor Kernel

```@docs
gaborkernel
```

### Matérn Kernels

```@docs
MaternKernel
Matern12Kernel
Matern32Kernel
Matern52Kernel
Matern72Kernel
```

### Neural Network Kernel

```@docs
NeuralNetworkKernel
```

### Periodic Kernel

```@docs
PeriodicKernel
PeriodicKernel(::DataType, ::Int)
```

### Piecewise Polynomial Kernel

```@docs
PiecewisePolynomialKernel
```

### Polynomial Kernels

```@docs
LinearKernel
PolynomialKernel
```

### Rational Kernels

```@docs
RationalKernel
RationalQuadraticKernel
GammaRationalKernel
```

### Spectral Mixture Kernels

```@docs
spectral_mixture_kernel
spectral_mixture_product_kernel
```

### Wiener Kernel

```@docs
WienerKernel
```

## Composite Kernels

The modular design of KernelFunctions uses [base kernels](@ref base_kernels) as building
blocks for more complex kernels. There are a variety of composite kernels implemented,
including those which [transform the inputs](@ref input_transforms) to a wrapped kernel
to implement length scales, scale the variance of a kernel, and sum or multiply collections
of kernels together.

```@docs
TransformedKernel
∘(::Kernel, ::Transform)
ScaledKernel
KernelSum
KernelProduct
KernelTensorSum
KernelTensorProduct
NormalizedKernel
```

## Multi-output Kernels
Kernelfunctions implements multi-output kernels as scalar kernels on an extended output domain. For more details on this read [the section on inputs for multi-output GPs](@ref Inputs-for-Multiple-Outputs).

For a function ``f(x) \rightarrow y`` denote the inputs as ``x, x'``, such that we compute the covariance between output components ``y_{p}`` and ``y_{p'}``. The total number of outputs is ``m``.

```@docs
MOKernel
IndependentMOKernel
LatentFactorMOKernel
IntrinsicCoregionMOKernel
LinearMixingModelKernel
```
