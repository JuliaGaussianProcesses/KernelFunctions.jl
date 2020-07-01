```@meta
  CurrentModule = KernelFunctions
```

# Base Kernels

These are the basic kernels without any transformation of the data. They are the building blocks of `KernelFunctions.jl`
We show for all examples some samples of a GP from [AbstractGPs.jl](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl).
To run any of these examples, start with

```@example plots
using AbstractGPs, Plots, KernelFunctions
x = range(-5, 5, length = 100)
default(legend = false,
        linewidth = 3.0,
        background_color = :transparent,
        foreground_color = :black,
)
```



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

```@example plots
k = CosineKernel()
gp = GP(k)(x, 1e-5)
plot(x, rand(gp, 4))
savefig("cosine.png"); nothing #hide
```

![](cosine.png)

## Exponential Kernels

### Exponential Kernel

The [`ExponentialKernel`](@ref) is defined as
```math
  k(x,x') = \exp\left(-|x-x'|\right).
```

```@example plots
k = ExponentialKernel()
gp = GP(k)(x, 1e-5)
plot(x, rand(gp, 4))
savefig("exponential.png"); nothing #hide
```

![](exponential.png)

### Square Exponential Kernel

The [`SqExponentialKernel`](@ref) is defined as
```math
  k(x,x') = \exp\left(-\|x-x'\|^2\right).
```

```@example plots
k = SqExponentialKernel()
gp = GP(k)(x, 1e-5)
plot(x, rand(gp, 4))
savefig("sqexponential.png"); nothing #hide
```

![](sqexponential.png)

### Gamma Exponential Kernel

The [`GammaExponentialKernel`](@ref) is defined as

```math
  k(x,x';\gamma) = \exp\left(-\|x-x'\|^{2\gamma}\right),
```
where $\gamma > 0$.

```@example plots
k1 = GammaExponentialKernel(γ=0.1)
k2 = GammaExponentialKernel(γ=0.8)
gp1 = GP(k1)(x, 1e-5)
gp2 = GP(k2)(x, 1e-5)
plot(x, [rand(gp1, 1) rand(gp2, 1)], label = ["γ=0.1" "γ=0.8"], legend=true)
savefig("gammaexponential.png"); nothing #hide
```

![](gammaexponential.png)

## Exponentiated Kernel

The [`ExponentiatedKernel`](@ref) is defined as

```math
  k(x,x') = \exp\left(\langle x,x'\rangle)\right).
```

```@example plots
k = ExponentiatedKernel()
gp = GP(k)(x, 1e-5)
plot(x, rand(gp, 4))
savefig("exponentiated.png"); nothing #hide
```

![](exponentiated.png)


## Fractional Brownian Motion

The [`FBMKernel`](@ref) is defined as

```math
  k(x,x';h) =  \frac{|x|^{2h} + |x'|^{2h} - |x-x'|^{2h}}{2},
```

where $h$ is the [Hurst index](https://en.wikipedia.org/wiki/Hurst_exponent#Generalized_exponent) and $0<h<1$.

```@example plots
k1 = FBMKernel(h=0.3)
k2 = FBMKernel(h=0.6)
gp1 = GP(k1)(x, 1e-3)
gp2 = GP(k2)(x, 1e-3)
plot(x, [rand(gp1, 1) rand(gp2, 1)], label = ["h=0.3" "h=0.6"], legend = true)
savefig("fbm.png"); nothing #hide
```

![](fbm.png)


## Gabor Kernel

The [`GaborKernel`](@ref) is defined as

```math
  k(x,x'; l,p) = h(x-x';l,p)\\
  h(u;l,p) = \exp\left(-\cos\left(\pi \sum_i \frac{u_i}{p_i}\right)\sum_i \frac{u_i^2}{l_i^2}\right),
```
where $l_i >0 $ is the lengthscale and $p_i>0$ is the period.

```@example plots
k1 = GaborKernel()
k2 = GaborKernel(p=4.0)
gp1 = GP(k1)(x, 1e-5)
gp2 = GP(k2)(x, 1e-5)
plot(x, [rand(gp1, 1) rand(gp2, 1)], label = ["p=1" "p=4"], legend = true)
savefig("gabor.png"); nothing #hide
```

![](gabor.png)

## Matern Kernels

### Matern Kernel

The [`MaternKernel`](@ref) is defined as

```math
  k(x,x';\nu) = \frac{2^{1-\nu}}{\Gamma(\nu)}\left(\sqrt{2\nu}|x-x'|\right)K_\nu\left(\sqrt{2\nu}|x-x'|\right),
```

where $\nu > 0$.

```@example plots
k1 = MaternKernel(ν=6.0)
k2 = MaternKernel(nu=2.0)
gp1 = GP(k1)(x, 1e-5)
gp2 = GP(k2)(x, 1e-5)
plot(x, [rand(gp1, 1) rand(gp2, 1)], label = ["ν=6" "ν=2"], legend = true)
savefig("matern.png"); nothing #hide
```

![](matern.png)

### Matern 3/2 Kernel

The [`Matern32Kernel`](@ref) is defined as

```math
  k(x,x') = \left(1+\sqrt{3}|x-x'|\right)\exp\left(\sqrt{3}|x-x'|\right).
```


```@example plots
k = Matern32Kernel()
gp = GP(k)(x, 1e-5)
plot(x, rand(gp, 4))
savefig("matern32.png"); nothing #hide
```

![](matern32.png)

### Matern 5/2 Kernel

The [`Matern52Kernel`](@ref) is defined as

```math
  k(x,x') = \left(1+\sqrt{5}|x-x'|+\frac{5}{2}\|x-x'\|^2\right)\exp\left(\sqrt{5}|x-x'|\right).
```


```@example plots
k =  Matern52Kernel()
gp = GP(k)(x, 1e-5)
plot(x, rand(gp, 4))
savefig("matern52.png"); nothing #hide
```

![](matern52.png)

## Neural Network Kernel

The [`NeuralNetworkKernel`](@ref) (as in the kernel for an infinitely wide neural network interpretated as a Gaussian process) is defined as

```math
  k(x, x') = \arcsin\left(\frac{\langle x, x'\rangle}{\sqrt{(1+\langle x, x\rangle)(1+\langle x',x'\rangle)}}\right).
```


```@example plots
k = NeuralNetworkKernel()
gp = GP(k)(x, 1e-5)
plot(x, rand(gp, 4))
savefig("nn.png"); nothing #hide
```

![](nn.png)

## Periodic Kernel

The [`PeriodicKernel`](@ref) is defined as

```math
  k(x,x';r) = \exp\left(-0.5 \sum_i (sin (π(x_i - x'_i))/r_i)^2\right),
```

where $r$ has the same dimension as $x$ and $r_i >0$.

```@example plots
k1 = PeriodicKernel(r=[2.0])
k2 = PeriodicKernel(r=[8.0])
gp1 = GP(k1)(x, 1e-5)
gp2 = GP(k2)(x, 1e-5)
plot(x, [rand(gp1, 1) rand(gp2, 1)], label = ["r=2" "r=8"], legend = true)
savefig("periodic.png"); nothing #hide
```

![](periodic.png)

## Piecewise Polynomial Kernel

The [`PiecewisePolynomialKernel`](@ref) is defined as

```math
  k(x,x'; P, V) = \max(1 - r, 0)^{j + V} f(r, j),\\
  r = x^\top P x',\\
  j = \lfloor \frac{D}{2}\rfloor + V + 1,
```
where $x\in \mathbb{R}^D$, $V \in \{0,1,2,3\} and $P$ is a positive definite matrix.
$f$ is a piecewise polynomial (see source code).

```@example plots
a = rand(1,1)
k1 = PiecewisePolynomialKernel{1}(a)
k2 = PiecewisePolynomialKernel{3}(a)
gp1 = GP(k1)(x, 1e-5)
gp2 = GP(k2)(x, 1e-5)
plot(x, [rand(gp1, 1) rand(gp2, 1)], label = ["P=1" "P=3"], legend = true)
savefig("piecewise.png"); nothing #hide
```

![](piecewise.png)

## Polynomial Kernels

### Linear Kernel

The [`LinearKernel`](@ref) is defined as

```math
  k(x,x';c) = \langle x,x'\rangle + c,
```

where $c \in \mathbb{R}$


```@example plots
k1 = LinearKernel(c=0.0)
k2 = LinearKernel(c=2.0)
gp1 = GP(k1)(x, 1e-5)
gp2 = GP(k2)(x, 1e-5)
plot(x, [rand(gp1, 1) rand(gp2, 1)], label = ["c=0" "c=2"], legend = true)
savefig("linear.png"); nothing #hide
```

![](linear.png)

### Polynomial Kernel

The [`PolynomialKernel`](@ref) is defined as

```math
  k(x,x';c,d) = \left(\langle x,x'\rangle + c\right)^d,
```

where $c \in \mathbb{R}$ and $d>0$

```@example plots
k1 = PolynomialKernel(d=2.0)
k2 = PolynomialKernel(d=3.0)
gp1 = GP(k1)(x, 1e-5)
gp2 = GP(k2)(x, 1e-5)
plot(x, [rand(gp1, 1) rand(gp2, 1)], label = ["d=2" "d=3"], legend = true)
savefig("poly.png"); nothing #hide
```

![](poly.png)


## Rational Quadratic

### Rational Quadratic Kernel

The [`RationalQuadraticKernel`](@ref) is defined as

```math
  k(x,x';\alpha) = \left(1+\frac{\|x-x'\|^2}{\alpha}\right)^{-\alpha},
```

where $\alpha > 0$.

```@example plots
k1 = RationalQuadraticKernel(α=0.5)
k2 = RationalQuadraticKernel(alpha=2.0)
gp1 = GP(k1)(x, 1e-5)
gp2 = GP(k2)(x, 1e-5)
plot(x, [rand(gp1, 1) rand(gp2, 1)], label = ["α=0.5" "α=2.0"], legend = true)
savefig("ratquad.png"); nothing #hide
```

![](ratquad.png)


### Gamma Rational Quadratic Kernel

The [`GammaRationalQuadraticKernel`](@ref) is defined as

```math
  k(x,x';\alpha,\gamma) = \left(1+\frac{\|x-x'\|^{2\gamma}}{\alpha}\right)^{-\alpha},
```

where $\alpha > 0$ and $\gamma > 0$.

```@example plots
k1 = GammaRationalQuadraticKernel(γ=1.3)
k2 = GammaRationalQuadraticKernel(gamma=1.5)
gp1 = GP(k1)(x[1:10:end], 1e-5)
gp2 = GP(k2)(x[1:10:end], 1e-5)
plot(x[1:10:end], [rand(gp1, 1) rand(gp2, 1)], label = ["γ=1.3" "γ=1.5"], legend = true)
savefig("gammaratquad.png"); nothing #hide
```

![](gammaratquad.png)

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

```@example plots
ks = [WienerKernel{i}() for i in -1:2]
gps = AbstractGPs.FiniteGP.(GP.(ks), Ref(x[x.>0]), 1e-5)
plot(x[x.>0], rand.(gps, 1), label = reshape(["i=$i" for i in -1:2], 1, :), legend = true)
savefig("wiener.png"); nothing #hide
```

![](wiener.png)

# Composite Kernels

### Transformed Kernel

The [`TransformedKernel`](@ref) is a kernel where input are transformed via a function `f`

```math
  k(x,x';f,\widetilde{k}) = \widetilde{k}(f(x),f(x')),
```

Where $\widetilde{k}$ is another kernel and $f$ is an arbitrary mapping.

### Scaled Kernel

The [`ScaledKernel`](@ref) is defined as

```math
  k(x,x';\sigma^2,\widetilde{k}) = \sigma^2\widetilde{k}(x,x')
```

Where $\widetilde{k}$ is another kernel and $\sigma^2 > 0$.

```@example plots
k1 = 1.0 * SqExponentialKernel()
k2 = ScaledKernel(SqExponentialKernel(), 10.0)
gp1 = GP(k1)(x, 1e-5)
gp2 = GP(k2)(x, 1e-5)
plot(x, [rand(gp1, 1) rand(gp2, 1)], label = ["σ²=1" "σ²=10"], legend = true)
savefig("scaled.png"); nothing #hide
```

![](scaled.png)

### Kernel Sum

The [`KernelSum`](@ref) is defined as a sum of kernels

```math
  k(x,x';\{w_i\},\{k_i\}) = \sum_i w_i k_i(x,x'),
```
Where $w_i > 0$.

```@example plots
k1 = SqExponentialKernel() + LinearKernel()
k2 = KernelSum([Matern32Kernel(), PeriodicKernel(r=[1.0])])
gp1 = GP(k1)(x, 1e-5)
gp2 = GP(k2)(x, 1e-5)
plot(x, [rand(gp1, 1) rand(gp2, 1)], label = ["SqExp + Lin" "Mat32 + Per"], legend = true)
savefig("ksum.png"); nothing #hide
```

![](ksum.png)

### KernelProduct

The [`KernelProduct`](@ref) is defined as a product of kernels

```math
  k(x,x';\{k_i\}) = \prod_i k_i(x,x').
```

```@example plots
k1 = SqExponentialKernel() * LinearKernel()
k2 = KernelProduct([Matern32Kernel(), PeriodicKernel(r=[1.0])])
gp1 = GP(k1)(x, 1e-5)
gp2 = GP(k2)(x, 1e-5)
plot(x, [rand(gp1, 1) rand(gp2, 1)], label = ["SqExp * Lin" "Mat32 * Per"], legend = true)
savefig("kprod.png"); nothing #hide
```

![](kprod.png)


### Tensor Product

The [`TensorProduct`](@ref) is defined as :

```math
  k(x,x';\{k_i\}) = \prod_i k_i(x_i,x'_i)
```
