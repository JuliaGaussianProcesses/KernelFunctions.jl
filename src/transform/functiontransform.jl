"""
    FunctionTransform(f)

Transformation that applies function `f` to the input.

Make sure that `f` can act on an input. For instance, if the inputs are vectors, use
`f(x) = sin.(x)` instead of `f = sin`.

# Examples

```jldoctest
julia> f(x) = sum(x); t = FunctionTransform(f); X = randn(100, 10);

julia> map(t, ColVecs(X)) == ColVecs(sum(X; dims=1))
true
```
"""
struct FunctionTransform{F} <: Transform
    f::F
end

@functor FunctionTransform

(t::FunctionTransform)(x) = t.f(x)

_map(t::FunctionTransform, x::AbstractVector{<:Real}) = map(t.f, x)

function _map(t::FunctionTransform, x::ColVecs)
    vals = map(axes(x.X, 2)) do i
        t.f(view(x.X, :, i))
    end
    return ColVecs(reduce(hcat, vals))
end

function _map(t::FunctionTransform, x::RowVecs)
    vals = map(axes(x.X, 1)) do i
        t.f(view(x.X, i, :))
    end
    return RowVecs(reduce(hcat, vals)')
end

duplicate(t::FunctionTransform, f) = FunctionTransform(f)

Base.show(io::IO, t::FunctionTransform) = print(io, "Function Transform (f = ", t.f, ")")
