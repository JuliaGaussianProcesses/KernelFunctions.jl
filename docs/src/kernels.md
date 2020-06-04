```@meta
  CurrentModule = KernelFunctions
```

# Base Kernels

These are the basic kernels without any transformation of the data. They are the building blocks of KernelFunctions

## Exponential Kernels

### Exponential Kernel

The [`ExponentialKernel`](@ref) is defined as
```math
  k(x,x') = \exp\left(-|x-x'|\right).
```

### Square Exponential Kernel

The [`SqExponentialKernel`](@ref) is defined as
```math
  k(x,x') = \exp\left(-\|x-x'\|^2\right).
```

### Gamma Exponential Kernel

The [`GammaExponentialKernel`](@ref) is defined as
```math
  k(x,x';\gamma) = \exp\left(-\|x-x'\|^{2\gamma}\right),
```
where $\gamma > 0$.

## Matern Kernels

### Matern Kernel

The [`MaternKernel`](@ref) is defined as

```math
  k(x,x';\nu) = \frac{2^{1-\nu}}{\Gamma(\nu)}\left(\sqrt{2\nu}|x-x'|\right)K_\nu\left(\sqrt{2\nu}|x-x'|\right),
```

where $\nu > 0$.

### Matern 3/2 Kernel

The [`Matern32Kernel`](@ref) is defined as

```math
  k(x,x') = \left(1+\sqrt{3}|x-x'|\right)\exp\left(\sqrt{3}|x-x'|\right).
```

### Matern 5/2 Kernel

The [`Matern52Kernel`](@ref) is defined as

```math
  k(x,x') = \left(1+\sqrt{5}|x-x'|+\frac{5}{2}\|x-x'\|^2\right)\exp\left(\sqrt{5}|x-x'|\right).
```

## Rational Quadratic

### Rational Quadratic Kernel

The [`RationalQuadraticKernel`](@ref) is defined as

```math
  k(x,x';\alpha) = \left(1+\frac{\|x-x'\|^2}{\alpha}\right)^{-\alpha},
```

where $\alpha > 0$.

### Gamma Rational Quadratic Kernel

The [`GammaRationalQuadraticKernel`](@ref) is defined as

```math
  k(x,x';\alpha,\gamma) = \left(1+\frac{\|x-x'\|^{2\gamma}}{\alpha}\right)^{-\alpha},
```

where $\alpha > 0$ and $\gamma > 0$.

## Polynomial Kernels

### Linear Kernel

The [`LinearKernel`](@ref) is defined as

```math
  k(x,x';c) = \langle x,x'\rangle + c,
```

where $c \in \mathbb{R}$

### Polynomial Kernel

The [`PolynomialKernel`](@ref) is defined as

```math
  k(x,x';c,d) = \left(\langle x,x'\rangle + c\right)^d,
```

where $c \in \mathbb{R}$ and $d>0$

## Periodic Kernels

### Periodic Kernel

The [`PeriodicKernel`](@ref) is defined as

```math
  k(x,x';r) = \exp\left(-0.5 \sum_i (sin (π(x_i - x'_i))/r_i)^2\right),
```

where $r$ has the same dimension as $x$ and $r_i >0$.

## Constant Kernels

### Constant Kernel

The [`ConstantKernel`](@ref) is defined as

```math
  k(x,x';c) = c,
```

where $c \in \mathbb{R}$.

### White Kernel

The [`WhiteKernel`](@ref) is defined as

```math
  k(x,x') = \delta(x-x').
```

### Zero Kernel

The [`ZeroKernel`](@ref) is defined as

```math
  k(x,x') = 0.
```

# Composite Kernels

### Transformed Kernel

The [`TransformedKernel`](@ref) is a kernel where input are transformed via a function `f`

```math
  k(x,x';f,\widetile{k}) = \widetilde{k}(f(x),f(x')),
```

Where $\widetilde{k}$ is another kernel and $f$ is an arbitrary mapping.

### Scaled Kernel

The [`ScaledKernel`](@ref) is defined as

```math
  k(x,x';\sigma^2,\widetilde{k}) = \sigma^2\widetilde{k}(x,x')
```

Where $\widetilde{k}$ is another kernel and $\sigma^2 > 0$.

### Kernel Sum

The [`KernelSum`](@ref) is defined as a sum of kernels

```math
  k(x,x';\{w_i\},\{k_i\}) = \sum_i w_i k_i(x,x'),
```
Where $w_i > 0$.
### KernelProduct

The [`KernelProduct`](@ref) is defined as a product of kernels

```math
  k(x,x';\{k_i\}) = \prod_i k_i(x,x').
```
