```@meta
  CurrentModule = KernelFunctions
```

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

```math
  k(x,x';\gamma) = \exp\left(-\|x-x'\|^{2\gamma}\right)
```

## Matern Kernels

### Matern Kernel

```math
  k(x,x';\nu) = \frac{2^{1-\nu}}{\Gamma(\nu)}\left(\sqrt{2\nu}|x-x'|\right)K_\nu\left(\sqrt{2\nu}|x-x'|\right)
```

### Matern 3/2 Kernel

```math
  k(x,x') = \left(1+\sqrt{3}|x-x'|\right)\exp\left(\sqrt{3}|x-x'|\right)
```

### Matern 5/2 Kernel

```math
  k(x,x') = \left(1+\sqrt{5}|x-x'|+\frac{5}{2}\|x-x'\|^2\right)\exp\left(\sqrt{5}|x-x'|\right)
```

## Rational Quadratic

### Rational Quadratic Kernel

```math
  k(x,x';\alpha) = \left(1+\frac{\|x-x'\|^2}{\alpha}\right)^{-\alpha}
```

### Gamma Rational Quadratic Kernel

```math
  k(x,x';\alpha,\gamma) = \left(1+\frac{\|x-x'\|^{2\gamma}}{\alpha}\right)^{-\alpha}
```

## Polynomial Kernels

### LinearKernel

```math
  k(x,x';c) = \langle x,x'\rangle + c
```

### PolynomialKernel

```math
  k(x,x';c,d) = \left(\langle x,x'\rangle + c\right)^d
```

## Constant Kernels

### ConstantKernel

```math
  k(x,x';c) = c
```

### WhiteKernel

```math
  k(x,x') = \delta(x-x')
```

### ZeroKernel

```math
  k(x,x') = 0
```
