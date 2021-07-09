"""
    with_lengthscale(kernel::Kernel, lengthscale::Real)
    with_lengthscale(kernel::Kernel, lengthscales::AbstractVector{<:Real})

Constructs a transformed kernel with `lengthscale`.
If a vector `lengthscales` is passed instead, constructs an "ARD" kernel with different lengthscales for each dimension.

The following two ways of constructing a squared-exponential kernel with
a given lengthscale are equivalent:

```jldoctest
julia> ℓ = 2.5;

julia> isequal(SqExponentialKernel() ∘ ScaleTransform(inv(ℓ)), with_lengthscale(SqExponentialKernel(), ℓ))
true
```

and for the ARD case:

```jldoctest
julia> ℓ = [0.5, 2.5];

julia> isequal(SqExponentialKernel() ∘ ARDTransform(inv.(ℓ)), with_lengthscale(SqExponentialKernel(), ℓ))
true
```
"""
function with_lengthscale(kernel::Kernel, lengthscale::Real)
    return compose(kernel, ScaleTransform(inv(lengthscale)))
end
function with_lengthscale(kernel::Kernel, lengthscales::AbstractVector{<:Real})
    return compose(kernel, ARDTransform(map(inv, lengthscales)))
end
