"""
FunctionTransform
```
    f(x) = abs.(x)
    tr = FunctionTransform(f)
```
Take a function `f` as an argument which is going to act on each vector individually.
Make sure that `f` is supposed to act on a vector by eventually using broadcasting
For example `f(x)=sin(x)` -> `f(x)=sin.(x)`
"""
struct FunctionTransform{F} <: Transform
    f::F
end

apply(t::FunctionTransform, X::T; obsdim::Int = defaultobs) where {T} = mapslices(t.f, X, dims = feature_dim(obsdim))

duplicate(t::FunctionTransform,f) = FunctionTransform(f)
params(t::FunctionTransform) = t.f
