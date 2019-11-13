# User guide

## Kernel creation

To create a kernel chose one of the kernels proposed, see [Kernels](@ref), or create your own, see [Creating Kernels](@ref)
For example to create a square exponential kernel
```julia
  k = SqExponentialKernel()
```
All kernels can take as argument a `Transform` object (see [Transform](@ref)) which is directly going to act on the inputs before it's processes.
But it's also possible to simply give a scalar or a vector if all you are interested in is to modify the lengthscale, respectively for all dimensions or independently for each dimension.

## Kernel matrix creation

Matrix are created via the `kernelmatrix` function or `kerneldiagmatrix`.
An important argument to give is the dimensionality of the input `obsdim`. It tells if the matrix is of the type `# samples X # features` (`obsdim`=1) or `# features X # samples`(`obsdim`=2) (similarly to [Distances.jl](https://github.com/JuliaStats/Distances.jl))
For example:
```julia
  k = SqExponentialKernel()
  A = rand(10,5)
  kernelmatrix(k,A,obsdim=1) # Return a 10x10 matrix
  kernelmatrix(k,A,obsdim=2) # Return a 5x5 matrix
```

## Kernel manipulation

One can create combinations of kernels via `KernelSum` and `KernelProduct` or using simple operators `+` and `*`.
For
