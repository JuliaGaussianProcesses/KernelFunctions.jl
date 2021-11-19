# [Input Transforms](@id input_transforms)

## Overview

[`Transform`](@ref)s are designed to change input data before passing it on to a kernel object.

It can be as standard as [`IdentityTransform`](@ref) returning the same input, or
multiplying the data by a scalar with [`ScaleTransform`](@ref) or by a vector with
[`ARDTransform`](@ref).
There is a more general [`FunctionTransform`](@ref) that uses a function and applies it to
each input.

You can also create a pipeline of [`Transform`](@ref)s via [`ChainTransform`](@ref), e.g.,
```julia
LowRankTransform(rand(10, 5)) ∘ ScaleTransform(2.0)
```

A transformation `t` can be applied to a single input `x` with `t(x)` and to multiple inputs
`xs` with `map(t, xs)`.

Kernels can be coupled with input transformations with [`∘`](@ref) or its alias `compose`. It falls
back to creating a [`TransformedKernel`](@ref) but allows more
optimized implementations for specific kernels and transformations.

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

## Convenience functions

```@docs
with_lengthscale
median_heuristic_transform
```
