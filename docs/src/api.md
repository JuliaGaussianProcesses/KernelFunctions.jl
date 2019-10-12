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
Exponential
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
```

## Functions

```@docs
kernelmatrix
kernelmatrix!
kerneldiagmatrix
kerneldiagmatrix!
transform
```


## Index

```@index
Pages = ["api.md"]
Module = ["KernelFunctions"]
Order = [:type, :function]
```
