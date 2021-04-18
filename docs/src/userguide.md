# User guide

## Kernel creation

To create a kernel object, choose one of the pre-implemented kernels, see [Kernel Functions](@ref), or create your own, see [Creating your own kernel](@ref).
For example, a squared exponential kernel is created by
```julia
  k = SqExponentialKernel()
```

!!! tip "How do I set the lengthscale?"
    Instead of having lengthscale(s) for each kernel we use [`Transform`](@ref) objects which act on the inputs before passing them to the kernel. Note that the transforms such as [`ScaleTransform`](@ref) and [`ARDTransform`](@ref) _multiply_ the input by a scale factor, which corresponds to the _inverse_ of the lengthscale.
    For example, a lengthscale of 0.5 is equivalent to premultiplying the input by 2.0, and you can create the corresponding kernel as follows:
    ```julia
      k = transform(SqExponentialKernel(), ScaleTransform(2.0))
      k = transform(SqExponentialKernel(), 2.0)  # implicitly constructs a ScaleTransform(2.0)
    ```
    Check the [Input Transforms](@ref input_transforms) page for more details.

!!! tip "How do I set the kernel variance?"
    To premultiply the kernel by a variance, you can use `*` with a scalar number:
    ```julia
      k = 3.0 * SqExponentialKernel()
    ```

!!! tip "How do I use a Mahalanobis kernel?"
    The `MahalanobisKernel(; P=P)`, defined by
    ```math
    k(x, x'; P) = \exp{\big(- (x - x')^\top P (x - x')\big)}
    ```
    for a positive definite matrix $P = Q^\top Q$, was removed in 0.9. Instead you can
    use a squared exponential kernel together with a [`LinearTransform`](@ref) of
    the inputs:
    ```julia
    k = transform(SqExponentialKernel(), LinearTransform(sqrt(2) .* Q))
    ```
    Analogously, you can combine other kernels such as the
    [`PiecewisePolynomialKernel`](@ref) with a [`LinearTransform`](@ref) of the
    inputs to obtain a kernel that is a function of the Mahalanobis distance
    between inputs.

## Using a kernel function

To evaluate the kernel function on two vectors you simply call the kernel object:
```julia
  k = SqExponentialKernel()
  x1 = rand(3)
  x2 = rand(3)
  k(x1, x2)
```

## Creating a kernel matrix

Kernel matrices can be created via the `kernelmatrix` function or `kernelmatrix_diag` for only the diagonal.
For example, for a collection of `10` `Real`-valued inputs:
```julia
  k = SqExponentialKernel()
  x = rand(10)
  kernelmatrix(k, x) # 10x10 matrix
```
If your inputs are multi-dimensional, it is common to represent them as a matrix.
For example
```julia
  X = rand(10, 5)
```
However, it is ambiguous whether this represents a collection of 10 5-dimensional row-vectors, or 5 10-dimensional column-vectors.
Therefore, we require users to provide some more information.

You can write `RowVecs(X)` to declare that `X` contains 10 5-dimensional row-vectors, or `ColVecs(X)` to declare that `X` contains 5 10-dimensional column-vectors, then
```julia
  kernelmatrix(k, RowVecs(X)) # returns a 10x10 matrix -- each row of X treated as input
  kernelmatrix(k, ColVecs(X)) # returns a 5x5 matrix -- each column of X treated as input
```
This is the mechanism used throughout KernelFunctions.jl to handle multi-dimensional inputs.

You can also utilise the `obsdim` keyword argument if you prefer:
```julia
  kernelmatrix(k, X; obsdim=1) # same as RowVecs(X)
  kernelmatrix(k, X; obsdim=2) # same as ColVecs(X)
```
This is similar to the convention used in [Distances.jl](https://github.com/JuliaStats/Distances.jl).

See [Input Types](@ref) for a more thorough discussion of these two approaches.



We also support specific kernel matrix outputs:
- For a positive-definite matrix object`PDMat` from [`PDMats.jl`](https://github.com/JuliaStats/PDMats.jl), you can call the following:
```julia
  using PDMats
  k = SqExponentialKernel()
  K = kernelpdmat(k, RowVecs(X)) # PDMat
  K = kernelpdmat(k, X; obsdim=1) # PDMat
```
It will create a matrix and in case of bad conditioning will add some diagonal noise until the matrix is considered positive-definite; it will then return a `PDMat` object. For this method to work in your code you need to include `using PDMats` first.
- For a Kronecker matrix, we rely on [`Kronecker.jl`](https://github.com/MichielStock/Kronecker.jl). Here are two examples:
```julia
using Kronecker
x = range(0, 1; length=10)
y = range(0, 1; length=50)
K = kernelkronmat(k, [x, y]) # Kronecker matrix
K = kernelkronmat(k, x, 5) # Kronecker matrix
```
Make sure that `k` is a kernel compatible with such constructions (with `iskroncompatible(k)`). Both methods will return a Kronecker matrix. For those methods to work in your code you need to include `using Kronecker` first.
- For a Nystrom approximation: `kernelmatrix(nystrom(k, X, ρ, obsdim=1))` where `ρ` is the fraction of data samples used in the approximation.

## Composite kernels

Sums and products of kernels are also valid kernels. They can be created via `KernelSum` and `KernelProduct` or using simple operators `+` and `*`.
For example:
```julia
  k1 = SqExponentialKernel()
  k2 = Matern32Kernel()
  k = 0.5 * k1 + 0.2 * k2 # KernelSum
  k = k1 * k2 # KernelProduct
```

## Kernel parameters

What if you want to differentiate through the kernel parameters? This is easy even in a highly nested structure such as:
```julia
  k = transform(
        0.5 * SqExponentialKernel() * Matern12Kernel()
      + 0.2 * (transform(LinearKernel(), 2.0) + PolynomialKernel()),
      [0.1, 0.5])
```
One can access the named tuple of trainable parameters via `Functors.functor` from `Functors.jl`.
This means that in practice you can implicitly optimize the kernel parameters by calling:
```julia
using Flux
kernelparams = Flux.params(k)
Flux.gradient(kernelparams) do
    # ... some loss function on the kernel ....
end
```
