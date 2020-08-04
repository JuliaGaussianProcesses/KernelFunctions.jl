## Creating your own kernel

KernelFunctions.jl contains the most popular kernels already but you might want to make your own!

Here are a few ways depending on how complicated your kernel is :

### SimpleKernel for kernels function depending on a metric

If your kernel function is of the form `k(x, y) = f(d(x, y))` where `d(x, y)` is a `PreMetric`,
you can construct your custom kernel by defining `kappa` and `metric` for your kernel.
Here is for example how one can define the `SqExponentialKernel` again :

```julia
struct MyKernel <: KernelFunctions.SimpleKernel end

KernelFunctions.kappa(::MyKernel, d2::Real) = exp(-d2)
KernelFunctions.metric(::MyKernel) = SqEuclidean()
```

### Kernel for more complex kernels

If your kernel does not satisfy such a representation, all you need to do is define `(k::MyKernel)(x, y)` and inherit from `Kernel`.
For example we recreate here the `NeuralNetworkKernel`

```julia
struct MyKernel <: KernelFunctions.Kernel end

(::MyKernel)(x, y) = asin(dot(x, y) / sqrt((1 + sum(abs2, x)) * (1 + sum(abs2, y))))
```

Note that `BaseKernel` do not use `Distances.jl` and can therefore be a bit slower.

### Additional Options

Finally there are additional functions you can define to bring in more features:
 - `KernelFunctions.trainable(k::MyKernel)`: it defines the trainable parameters of your kernel, it should return a `Tuple` of your parameters.
These parameters will be passed to the `Flux.params` function. For some examples see the `trainable.jl` file in `src/`
 - `KernelFunctions.iskroncompatible(k::MyKernel)`: if your kernel factorizes in dimensions, you can declare your kernel as `iskroncompatible(k) = true` to use Kronecker methods.
 - `KernelFunctions.dim(x::MyDataType)`: by default the dimension of the inputs will only be checked for vectors of type `AbstractVector{<:Real}`. If you want to check the dimensionality of your inputs, dispatch the `dim` function on your datatype. Note that `0` is the default.
 - You can also directly overload `KernelFunctions.validate_inputs(x::MyDataType, y::MyDataType)` if you want to run special checks for your input types.
 - `kernelmatrix(k::MyKernel, ...)`: you can redefine the diverse `kernelmatrix` functions to eventually optimize the computations.
 - `Base.print(io::IO, k::MyKernel)`: if you want to specialize the printing of your kernel
