# API Library

---
```@contents
Pages = ["api.md"]
```

```@meta
CurrentModule = KernelFunctions
```

## Module
```@docs
KernelFunctions
```

## Base Kernels API

```@docs
ConstantKernel
WhiteKernel
EyeKernel
ZeroKernel
CosineKernel
SqExponentialKernel
GaussianKernel
RBFKernel
SEKernel
ExponentialKernel
LaplacianKernel
GammaExponentialKernel
ExponentiatedKernel
FBMKernel
GaborKernel
MaternKernel
Matern12Kernel
Matern32Kernel
Matern52Kernel
NeuralNetworkKernel
LinearKernel
PolynomialKernel
PiecewisePolynomialKernel
RationalQuadraticKernel
GammaRationalQuadraticKernel
spectral_mixture_kernel
spectral_mixture_product_kernel
PeriodicKernel
WienerKernel
MOKernel
IndependentMOKernel
LatentFactorMOKernel
```

## Composite Kernels

```@docs
TransformedKernel
ScaledKernel
KernelSum
KernelProduct
KernelTensorProduct
```

## Transforms

```@docs
Transform
IdentityTransform
ScaleTransform
ARDTransform
LinearTransform
FunctionTransform
SelectTransform
ChainTransform
PeriodicTransform
```

## Functions

```@docs
kernelmatrix
kernelmatrix!
kerneldiagmatrix
kerneldiagmatrix!
kernelpdmat
nystrom
transform
```

## Utilities

```@docs
ColVecs
RowVecs
MOInput
NystromFact
```

## Index

```@index
Pages = ["api.md"]
Module = ["KernelFunctions"]
Order = [:type, :function]
```
