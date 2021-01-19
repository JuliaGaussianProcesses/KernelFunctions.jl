# Input Transforms

[`Transform`](@ref) is the object that takes care of transforming the input data before distances are being computed.

It can be as standard as [`IdentityTransform`](@ref) returning the same input, or
multiplying the data by a scalar with [`ScaleTransform`](@ref) or by a vector with
[`ARDTransform`](@ref).
There is a more general [`FunctionTransform`](@ref) that uses a function and applies it to
each input.

You can also create a pipeline of [`Transform`](@ref)s via [`ChainTransform`](@ref), e.g.,
```julia
LowRankTransform(rand(10, 5)) ∘ ScaleTransform(2.0)
```

A transformation `t` can be applied to a vector `v` with `map`.

## List of Input Transforms

```@docs
Transform
IdentityTransform
ScaleTransform
ARDTransform
ARDTransform(::Real, ::Integer)
LinearTransform
FunctionTransform
SelectTransform
ChainTransform
PeriodicTransform
```