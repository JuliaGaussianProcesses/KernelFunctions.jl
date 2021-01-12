```@meta
  CurrentModule = KernelFunctions
```

# Base Kernels

These are the basic kernels without any transformation of the data. They are the building blocks of KernelFunctions.


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

## Cosine Kernel

The [`CosineKernel`](@ref) is defined as

```math
  k(x, x') = \cos(\pi |x-x'|),
```

where $x\in\mathbb{R}$.

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

## Exponentiated Kernel

The [`ExponentiatedKernel`](@ref) is defined as

```math
  k(x,x') = \exp\left(\langle x,x'\rangle).
```

## Fractional Brownian Motion

The [`FBMKernel`](@ref) is defined as

```math
  k(x,x';h) =  \frac{|x|^{2h} + |x'|^{2h} - |x-x'|^{2h}}{2},
```

where $h$ is the [Hurst index](https://en.wikipedia.org/wiki/Hurst_exponent#Generalized_exponent) and $0 < h < 1$.

## Gabor Kernel

The [`GaborKernel`](@ref) is defined as

```math
  k(x,x'; l,p) = \exp\left(-\cos\left(\pi \sum_i \frac{x_i - x'_i}{p_i}\right)\sum_i \frac{(x_i - x'_i)^2}{l_i^2}\right),
```
where $l_i > 0$ is the lengthscale and $p_i > 0$ is the period.

## Matérn Kernels

### General Matérn Kernel

The [`MaternKernel`](@ref) is defined as

```math
  k(x,x';\nu) = \frac{2^{1-\nu}}{\Gamma(\nu)}\left(\sqrt{2\nu}|x-x'|\right)K_\nu\left(\sqrt{2\nu}|x-x'|\right),
```

where $\nu > 0$.

### Matérn 1/2 Kernel

The Matérn 1/2 kernel is defined as
```math
  k(x,x') = \exp\left(-|x-x'|\right),
```
equivalent to the Exponential kernel. `Matern12Kernel` is an alias for [`ExponentialKernel`](@ref).

### Matérn 3/2 Kernel

The [`Matern32Kernel`](@ref) is defined as

```math
  k(x,x') = \left(1+\sqrt{3}|x-x'|\right)\exp\left(\sqrt{3}|x-x'|\right).
```

### Matérn 5/2 Kernel

The [`Matern52Kernel`](@ref) is defined as

```math
  k(x,x') = \left(1+\sqrt{5}|x-x'|+\frac{5}{2}\|x-x'\|^2\right)\exp\left(\sqrt{5}|x-x'|\right).
```

## Neural Network Kernel

The [`NeuralNetworkKernel`](@ref) (as in the kernel for an infinitely wide neural network interpreted as a Gaussian process) is defined as

```math
  k(x, x') = \arcsin\left(\frac{\langle x, x'\rangle}{\sqrt{(1+\langle x, x\rangle)(1+\langle x',x'\rangle)}}\right).
```

## Periodic Kernel

The [`PeriodicKernel`](@ref) is defined as

```math
  k(x,x';r) = \exp\left(-0.5 \sum_i (\sin (\pi(x_i - x'_i))/r_i)^2\right),
```

where $r$ has the same dimension as $x$ and $r_i > 0$.

## Piecewise Polynomial Kernel

The [`PiecewisePolynomialKernel`](@ref) of degree $v \in \{0,1,2,3\}$ is defined for
inputs $x, x' \in \mathbb{R}^d$ of dimension $d$ as
```math
k(x, x'; v) = \max(1 - \|x - x'\|, 0)^{\alpha} f_{v,d}(\|x - x'\|),
```
where $\alpha = \lfloor \frac{d}{2}\rfloor + 2v + 1$, and $f_{v,d}$ are polynomials of
degree $v$ given by
```math
\begin{aligned}
f_{0,d}(r) &= 1, \\
f_{1,d}(r) &= 1 + (j + 1) r, \\
f_{2,d}(r) &= 1 + (j + 2) r + \big((j^2 + 4j + 3) / 3\big) r^2, \\
f_{3,d}(r) &= 1 + (j + 3) r + \big((6 j^2 + 36j + 45) / 15\big) r^2 + \big((j^3 + 9 j^2 + 23j + 15) / 15\big) r^3,
\end{aligned}
```
where $j = \lfloor \frac{d}{2}\rfloor + v + 1$.

## Polynomial Kernels

### Linear Kernel

The [`LinearKernel`](@ref) is defined as

```math
  k(x,x';c) = \langle x,x'\rangle + c,
```

where $c \in \mathbb{R}$.

### Polynomial Kernel

The [`PolynomialKernel`](@ref) is defined as

```math
  k(x,x';c,d) = \left(\langle x,x'\rangle + c\right)^d,
```

where $c \in \mathbb{R}$ and $d>0$.


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

## Spectral Mixture Kernel

The spectral mixture kernel is called by [`spectral_mixture_kernel`](@ref).


## Wiener Kernel

The [`WienerKernel`](@ref) is defined as

```math
k(x,x';i) = \left\{\begin{array}{cc}
  \delta(x, x') & i = -1\\
  \min(x,x') & i = 0\\
  \frac{\min(x,x')^{2i+1}}{a_i} + b_i \min(x,x')^{i+1}|x-x'|r_i(x,x') & i\geq 1
\end{array}\right.,
```
where $i\in\{-1,0,1,2,3\}$ and coefficients $a_i$, $b_i$ are fixed and residuals $r_i$ are defined in the code.

# Composite Kernels

### Transformed Kernel

The [`TransformedKernel`](@ref) is a kernel where inputs are transformed via a function `f`:

```math
  k(x,x';f,\widetilde{k}) = \widetilde{k}(f(x),f(x')),
```
where $\widetilde{k}$ is another kernel and $f$ is an arbitrary mapping.

### Scaled Kernel

The [`ScaledKernel`](@ref) is defined as

```math
  k(x,x';\sigma^2,\widetilde{k}) = \sigma^2\widetilde{k}(x,x') ,
```
where $\widetilde{k}$ is another kernel and $\sigma^2 > 0$.

### Kernel Sum

The [`KernelSum`](@ref) is defined as a sum of kernels:

```math
  k(x, x'; \{k_i\}) = \sum_i k_i(x, x').
```

### Kernel Product

The [`KernelProduct`](@ref) is defined as a product of kernels:

```math
  k(x,x';\{k_i\}) = \prod_i k_i(x,x').
```

### Tensor Product

The [`TensorProduct`](@ref) is defined as:

```math
  k(x,x';\{k_i\}) = \prod_i k_i(x_i,x'_i)
```
