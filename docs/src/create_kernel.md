## Creating your own kernel

KernelFunctions.jl contains the most popular kernels already but you might want to make your own!

Here is for example how one can define the Squared Exponential Kernel again :

```julia
struct MyKernel <: Kernel end

KernelFunctions.kappa(::MyKernel, d2::Real) = exp(-d2)
KernelFunctions.metric(::MyKernel) = SqEuclidean()
```

For a "Base" kernel, where the kernel function is simply a function applied on some metric between two vectors of real, you only need to:
 - Define your struct inheriting from `Kernel`.
 - Define a `kappa` function.
 - Define the metric used `SqEuclidean`, `DotProduct` etc. Note that the term "metric" is here overabused.
 - Optional : Define any parameter of your kernel as `trainable` by Flux.jl if you want to perform optimization on the parameters. We recommend wrapping all parameters in arrays to allow them to be mutable.

Once these functions are defined, you can use all the wrapping functions of KernelFuntions.jl
