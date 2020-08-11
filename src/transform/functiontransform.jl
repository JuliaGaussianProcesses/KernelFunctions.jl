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

_map(t::FunctionTransform, x::AbstractVector{<:Real}) = map(t.f, x)
     

function _map(t::FunctionTransform, x::ColVecs)
    vals = map(axes(x.X, 2)) do i
        t.f(view(x.X, :, i))
    end
    return ColVecs(hcat(vals...))
end

function _map(t::FunctionTransform, x::RowVecs)
    vals = map(axes(x.X, 1)) do i
        t.f(view(x.X, i, :))
    end
    return RowVecs(hcat(vals...)')
end

duplicate(t::FunctionTransform,f) = FunctionTransform(f)

Base.show(io::IO, t::FunctionTransform) = print(io, "Function Transform: ", t.f)
