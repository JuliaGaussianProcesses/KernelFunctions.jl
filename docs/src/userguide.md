# User guide

## Kernel creation

To create a kernel chose one of the kernels proposed, see [Base Kernels](@ref), or create your own, see [Creating your own kernel](@ref)
For example to create a square exponential kernel
```julia
  k = SqExponentialKernel()
```
!!! tip "How do I set the lengthscale?" Instead of having lengthscale(s) for each kernel we use `Transform` objects (see [Transform](@ref)) which are directly going to act on the inputs before passing them to the kernel. 
For example, if you want to to premultiply the input by 2.0, you can create your kernel with the following options:
```julia
  k = transform(SqExponentialKernel(), 2.0)) # returns a TransformedKernel
  k = TransformedKernel(SqExponentialKernel(), ScaleTransform(2.0))
```
In the example of the [SqExponentialKernel](@ref), you can reproduce the usual definition, $$\exp\left(-\frac{\|x-x'\|^2}{\rho^2}$$, by using the following `Transform` : `transform(SqExponentialKernel(), 1 / ρ`. 
Check the [`Transform`](@ref) page to see the other options.

To premultiply the kernel by a variance, you can use `*` or create a `ScaledKernel`
```julia
  k = 3.0*SqExponentialKernel()
  k = ScaledKernel(SqExponentialKernel(),3.0)
  @kernel 3.0*SqExponentialKernel()
```

## Using a kernel function

To compute the kernel function on two vectors you can call
```julia
  k = SqExponentialKernel()
  x1 = rand(3)
  x2 = rand(3)
  k(x1,x2)
```

## Creating a kernel matrix

Kernel matrices can be created via the `kernelmatrix` function or `kerneldiagmatrix` for only the diagonal.
An important argument to give is the dimensionality of the input `obsdim`. It tells if the matrix is of the type `# samples X # features` (`obsdim`=1) or `# features X # samples`(`obsdim`=2) (similarly to [Distances.jl](https://github.com/JuliaStats/Distances.jl))
For example:
```julia
  k = SqExponentialKernel()
  A = rand(10,5)
  kernelmatrix(k,A,obsdim=1) # Return a 10x10 matrix
  kernelmatrix(k,A,obsdim=2) # Return a 5x5 matrix
  k(A,obsdim=1) # Syntactic sugar
```

We also support specific kernel matrices outputs:
- For a positive-definite matrix object`PDMat` from [`PDMats.jl`](https://github.com/JuliaStats/PDMats.jl), you can call the following:
```julia
  using PDMats
  k = SqExponentialKernel()
  K = kernelpdmat(k,A,obsdim=1) # PDMat
```
It will create a matrix and in case of bad conditionning will add some diagonal noise until the matrix is considered PSD, it will then return a `PDMat` object. For this method to work in your code you need to include `using PDMats` first
- For a Kronecker matrix, we rely on [`Kronecker.jl`](https://github.com/MichielStock/Kronecker.jl). Here are two examples:
```julia
using Kronecker
x = range(0,1,length=10)
y = range(0,1,length=50)
K = kernelkronmat(k,[x,y]) # Kronecker matrix
K = kernelkronmat(k,x,5) # Kronecker matrix
```
Make sure that `k` is a vector compatible with such constructions (with `iskroncompatible`). Both method will return a . For those methods to work in your code you need to include `using Kronecker` first
- For a Nystrom approximation : `kernelmatrix(nystrom(k, X, ρ, obsdim = 1))` where `ρ` is the proportion of sampled used.

## Composite kernels

One can create combinations of kernels via `KernelSum` and `KernelProduct` or using simple operators `+` and `*`.
For example :
```julia
  k1 = SqExponentialKernel()
  k2 = Matern32Kernel()
  k = 0.5 * k1 + 0.2 * k2 # KernelSum
  k = k1 * k2 # KernelProduct
```

## Kernel Parameters

What if you want to differentiate through the kernel parameters? Even in a highly nested structure such as :
```julia
  k = transform(
        0.5 * SqExponentialKernel() * MaternKernel()
      + 0.2 * (transform(LinearKernel(), 2.0) + PolynomialKernel()),
      [0.1, 0.5])
```
One can access the array of trainable parameters via `params` from `Flux.jl`

```julia
  using Flux
  params(k)
```
