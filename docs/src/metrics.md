# Metrics

`SimpleKernel` implementations rely on [Distances.jl](https://github.com/JuliaStats/Distances.jl) for efficiently computing the pairwise matrix.
This requires a distance measure or metric, such as the commonly used `SqEuclidean` and `Euclidean`.

The metric used by a given kernel type is specified as
```julia
KernelFunctions.metric(::CustomKernel) = SqEuclidean()
```

However, there are kernels that can be implemented efficiently using "metrics" that do not respect all the definitions expected by Distances.jl. For this reason, KernelFunctions.jl provides additional "metrics" such as `DotProduct` ($\langle x, y \rangle$) and `Delta` ($\delta(x,y)$).


## Adding a new metric

If you want to create a new "metric" just implement the following:

```julia
struct Delta <: Distances.PreMetric
end

@inline function Distances._evaluate(::Delta,a::AbstractVector{T},b::AbstractVector{T}) where {T}
    @boundscheck if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end
    return a==b
end

@inline (dist::Delta)(a::AbstractArray,b::AbstractArray) = Distances._evaluate(dist,a,b)
@inline (dist::Delta)(a::Number,b::Number) = a==b
```
