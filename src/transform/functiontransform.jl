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

apply(t::FunctionTransform, X::T; obsdim::Int = defaultobs) where {T} = mapslices(t.f, X, dims = feature_dim(obsdim))

duplicate(t::FunctionTransform,f) = FunctionTransform(f)

Base.show(io::IO, t::FunctionTransform) = print(io, "Function Transform: ", t.f)
