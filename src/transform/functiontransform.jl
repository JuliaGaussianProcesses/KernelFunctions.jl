"""
    FunctionTransform(f)

Take a function or object `f` as an argument which is going to act on each vector individually.
Make sure that `f` is supposed to act on a vector.
For example replace `f(x)=sin(x)` by `f(x)=sin.(x)`
```
    f(x) = abs.(x)
    tr = FunctionTransform(f)
```
"""
struct FunctionTransform{F} <: Transform
    f::F
end

(t::FunctionTransform)(x) = t.f(x)

Base.map(t::FunctionTransform, x::AbstractVector{<:Real}) = map(t.f, x)
_map(t::FunctionTransform, x::ColVecs) = ColVecs(mapslices(t.f, x.X; dims=1))
_map(t::FunctionTransform, x::RowVecs) = RowVecs(mapslices(t.f, x.X; dims=2))

duplicate(t::FunctionTransform,f) = FunctionTransform(f)

Base.show(io::IO, t::FunctionTransform) = print(io, "Function Transform: ", t.f)
