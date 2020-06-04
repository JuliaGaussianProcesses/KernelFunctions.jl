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

## Base Kernels

```@docs
SqExponentialKernel
ExponentialKernel
GammaExponentialKernel
ExponentiatedKernel
MaternKernel
Matern32Kernel
Matern52Kernel
NeuralNetworkKernel
GaborKernel
EyeKernel
FBMKernel
CosineKernel
LinearKernel
PolynomialKernel
PiecewisePolynomialKernel
MahalanobisKernel
RationalQuadraticKernel
GammaRationalQuadraticKernel
PeriodicKernel
ZeroKernel
ConstantKernel
WienerKernel
WhiteKernel
```

## Composite Kernels

```@docs
TransformedKernel
ScaledKernel
KernelSum
KernelProduct
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
```

## Index

```@index
Pages = ["api.md"]
Module = ["KernelFunctions"]
Order = [:type, :function]
```
