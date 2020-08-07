# Metrics

KernelFunctions.jl relies on [Distances.jl](https://github.com/JuliaStats/Distances.jl) for computing the pairwise matrix.
To do so a distance measure is needed for each kernel. Two very common ones can already be used : `SqEuclidean` and `Euclidean`.
However all kernels do not rely on distances metrics respecting all the definitions. That's why  additional metrics come with the package such as `DotProduct` (`<x,y>`) and `Delta` (`δ(x,y)`).
Note that every `SimpleKernel` must have a defined metric defined as :
```julia
    KernelFunctions.metric(::CustomKernel) = SqEuclidean()
```

## Adding a new metric

If you want to create a new distance just implement the following :

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
