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
ExponentialKernel
GammaExponentialKernel
ExponentiatedKernel
FBMKernel
GaborKernel
MaternKernel
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
```

## Composite Kernels

```@docs
TransformedKernel
ScaledKernel
KernelSum
KernelProduct
TensorProduct
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
```

## Functions

```@docs
kernelmatrix
kernelmatrix!
kerneldiagmatrix
kerneldiagmatrix!
kernelpdmat
kernelkronmat
nystrom
transform
```

## Utilities

```@docs
ColVecs
RowVecs
NystromFact
```

## Index

```@index
Pages = ["api.md"]
Module = ["KernelFunctions"]
Order = [:type, :function]
```
