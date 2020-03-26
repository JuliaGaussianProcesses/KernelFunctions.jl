```@meta
  CurrentModule = KernelFunctions
```

# Base Kernels

These are the basic kernels without any transformation of the data. They are the building blocks of KernelFunctions

## Exponential Kernels

### Exponential Kernel

The [Exponential Kernel](@ref ExponentialKernel) is defined as
```math
  k(x,x') = \exp\left(-|x-x'|\right)
```

### Square Exponential Kernel

The [Square Exponential Kernel](@ref KernelFunctions.SqExponentialKernel) is defined as
```math
  k(x,x') = \exp\left(-\|x-x'\|^2\right)
```

### Gamma Exponential Kernel

The [Gamma Exponential Kernel](@ref KernelFunctions.GammaExponentialKernel) is defined as
```math
  k(x,x';\gamma) = \exp\left(-\|x-x'\|^{2\gamma}\right)
```

## Matern Kernels

### Matern Kernel

The [Matern Kernel](@ref KernelFunctions.MaternKernel) is defined as

```math
  k(x,x';\nu) = \frac{2^{1-\nu}}{\Gamma(\nu)}\left(\sqrt{2\nu}|x-x'|\right)K_\nu\left(\sqrt{2\nu}|x-x'|\right)
```

### Matern 3/2 Kernel

The [Matern 3/2 Kernel](@ref KernelFunctions.Matern32Kernel) is defined as

```math
  k(x,x') = \left(1+\sqrt{3}|x-x'|\right)\exp\left(\sqrt{3}|x-x'|\right)
```

### Matern 5/2 Kernel

The [Matern 5/2 Kernel](@ref KernelFunctions.Matern52Kernel) is defined as

```math
  k(x,x') = \left(1+\sqrt{5}|x-x'|+\frac{5}{2}\|x-x'\|^2\right)\exp\left(\sqrt{5}|x-x'|\right)
```

## Rational Quadratic

### Rational Quadratic Kernel

The [Rational Quadratic Kernel](@ref KernelFunctions.RationalQuadraticKernel) is defined as

```math
  k(x,x';\alpha) = \left(1+\frac{\|x-x'\|^2}{\alpha}\right)^{-\alpha}
```

### Gamma Rational Quadratic Kernel

The [Gamma Rational Quadratic Kernel](@ref KernelFunctions.GammaRationalQuadraticKernel) is defined as

```math
  k(x,x';\alpha,\gamma) = \left(1+\frac{\|x-x'\|^{2\gamma}}{\alpha}\right)^{-\alpha}
```

## Polynomial Kernels

### LinearKernel

The [Linear Kernel](@ref KernelFunctions.LinearKernel) is defined as

```math
  k(x,x';c) = \langle x,x'\rangle + c
```

### PolynomialKernel

The [Polynomial Kernel](@ref KernelFunctions.PolynomialKernel) is defined as

```math
  k(x,x';c,d) = \left(\langle x,x'\rangle + c\right)^d
```

## Constant Kernels

### ConstantKernel

The [Constant Kernel](@ref KernelFunctions.ConstantKernel) is defined as

```math
  k(x,x';c) = c
```

### WhiteKernel

The [White Kernel](@ref KernelFunctions.WhiteKernel) is defined as

```math
  k(x,x') = \delta(x-x')
```

### ZeroKernel

The [Zero Kernel](@ref KernelFunctions.ZeroKernel) is defined as

```math
  k(x,x') = 0
```

# Composite Kernels

### TransformedKernel

The [Transformed Kernel](@ref KernelFunctions.TransformedKernel) is a kernel where input are transformed via a function `f`

```math
  k(x,x';f,\widetile{k}) = \widetilde{k}(f(x),f(x'))
```

Where `k̃` is another kernel

### ScaledKernel

The [Scalar Kernel](@ref KernelFunctions.ScaledKernel) is defined as

```math
  k(x,x';\sigma^2,\widetilde{k}) = \sigma^2\widetilde{k}(x,x')
```

### KernelSum

The [Kernel Sum](@ref KernelFunctions.KernelSum) is defined as a sum of kernel

```math
  k(x,x';\{w_i\},\{k_i\}) = \sum_i w_i k_i(x,x')
```

### KernelProduct

The [Kernel Product](@ref KernelFunctions.KernelProduct) is defined as a product of kernel

```math
  k(x,x';\{k_i\}) = \prod_i k_i(x,x')
```
