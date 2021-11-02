"""
    ParameterKernel(params, kernel)

Kernel with parameters `params` that can be instantiated by calling `kernel(params)`.

This kernel is particularly useful if you want to optimize a vector of,
usually unconstrained, kernel parameters `params` with e.g.
[Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) or
[Flux.jl](https://github.com/FluxML/Flux.jl).

# Examples

There are two different approaches for obtaining the parameters `params` and the function
`kernel` from which a `ParameterKernel` can be constructed.

## Extracting parameters from an existing kernel

You can extract the parameters `params` and the function `kernel` from an existing kernel
`k` with `ParameterHandling.flatten`:
```jldoctest parameterkernel1
julia> k = 2.0 * (RationalQuadraticKernel(; α=1.0) + ConstantKernel(; c=2.5));

julia> params, kernel = ParameterHandling.flatten(k);
```

Here `params` is a vector of the three parameters of kernel `k`. In this example, all these
parameters must be positive (otherwise `k` would not be a positive-definite kernel). To
simplify unconstrained optimization with e.g. Optim.jl or Flux.jl,
`ParameterHandling.flatten` automatically transforms the parameters to unconstrained values:
```jldoctest parameterkernel1
julia> params ≈ map(log, [1.0, 2.5, 2.0])
true
```

Kernel `k` can be reconstructed with the `kernel` function:
```jldoctest parameterkernel1
julia> kernel(params)
Sum of 2 kernels:
        Rational Quadratic Kernel (α = 1.0, metric = Distances.Euclidean(0.0))
        Constant Kernel (c = 2.5)
        - σ² = 2.0
```

As expected, different parameter values yield a kernel of the same structure with different
parameters:
```jldoctest parameterkernel1
julia> kernel([log(0.25), log(0.5), log(2.0)])
Sum of 2 kernels:
        Rational Quadratic Kernel (α = 0.25, metric = Distances.Euclidean(0.0))
        Constant Kernel (c = 0.5)
        - σ² = 2.0
```

## Defining a function that constructs the kernel

Instead of extracting parameters and a reconstruction function from an existing kernel you
can explicitly define a function that constructs the kernel of interest and a set of
parameters.

```jldoctest parameterkernel2
julia> using LogExpFunctions

julia> function kernel(params)
           length(params) == 1 || throw(ArgumentError("incorrect number of parameters"))
           p = first(params)
           return 2 * (RationalQuadraticKernel(; α=log1pexp(p)) + ConstantKernel(; c=exp(p)))
       end;
```

With the function `kernel` kernels of the same structure as in the example above can be
constructed:
```jldoctest parameterkernel2
julia> kernel([log(0.5)])
Sum of 2 kernels:
        Rational Quadratic Kernel (α = 0.4054651081081644, metric = Distances.Euclidean(0.0))
        Constant Kernel (c = 0.5)
        - σ² = 2
```

This example shows that defining `kernel` manually has some advantages over using
`ParameterHandling.flatten`:
- Kernel parameters can be fixed (scale parameter is always set to `2` in this example)
- Kernel parameters can be transformed from unconstrained to constrained space with
  non-default mappings (shape parameter `α` is transformed with `log1pexp`)
- Kernel parameters can be linked (parameters `α` and `c` are computed from a single
  parameter `p`)

See also: [ParameterHandling.jl](https://github.com/invenia/ParameterHandling.jl)
"""
struct ParameterKernel{P,K} <: Kernel
    params::P
    kernel::K
end

Functors.@functor ParameterKernel (params,)

function ParameterHandling.flatten(::Type{T}, kernel::ParameterKernel) where {T<:Real}
    params_vec, unflatten_to_params = flatten(T, kernel.params)
    k = kernel.kernel
    function unflatten_to_parameterkernel(v::Vector{T})
        return ParameterKernel(unflatten_to_params(v), k)
    end
    return params_vec, unflatten_to_parameterkernel
end

(k::ParameterKernel)(x, y) = k.kernel(k.params)(x, y)

function kernelmatrix(k::ParameterKernel, x::AbstractVector)
    return kernelmatrix(k.kernel(k.params), x)
end

function kernelmatrix(k::ParameterKernel, x::AbstractVector, y::AbstractVector)
    return kernelmatrix(k.kernel(k.params), x, y)
end

function kernelmatrix!(K::AbstractMatrix, k::ParameterKernel, x::AbstractVector)
    return kernelmatrix!(K, k.kernel(k.params), x)
end

function kernelmatrix!(
    K::AbstractMatrix, k::ParameterKernel, x::AbstractVector, y::AbstractVector
)
    return kernelmatrix!(K, k.kernel(k.params), x, y)
end

function kernelmatrix_diag(k::ParameterKernel, x::AbstractVector)
    return kernelmatrix_diag(k.kernel(k.params), x)
end

function kernelmatrix_diag(k::ParameterKernel, x::AbstractVector, y::AbstractVector)
    return kernelmatrix_diag(k.kernel(k.params), x, y)
end

function kernelmatrix_diag!(K::AbstractVector, k::ParameterKernel, x::AbstractVector)
    return kernelmatrix_diag!(K, k.kernel(k.params), x)
end

function kernelmatrix_diag!(
    K::AbstractVector, k::ParameterKernel, x::AbstractVector, y::AbstractVector
)
    return kernelmatrix_diag!(K, k.kernel(k.params), x, y)
end
