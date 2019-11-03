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

## Kernel Functions

```@docs
SqExponentialKernel
ExponentialKernel
GammaExponentialKernel
ExponentiatedKernel
MaternKernel
Matern32Kernel
Matern52Kernel
LinearKernel
PolynomialKernel
RationalQuadraticKernel
GammaRationalQuadraticKernel
```

## Kernel Combinations

```@docs
KernelSum
KernelProduct
```

## Transforms

```@docs
Transform
IdentityTransform
ScaleTransform
LowRankTransform
FunctionTransform
ChainTransform
```

## Functions

```@docs
kernelmatrix
kernelmatrix!
kerneldiagmatrix
kerneldiagmatrix!
kernelpdmat
transform
```


## Index

```@index
Pages = ["api.md"]
Module = ["KernelFunctions"]
Order = [:type, :function]
```
