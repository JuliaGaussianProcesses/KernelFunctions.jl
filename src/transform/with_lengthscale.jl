"""
    with_lengthscale(kernel::Kernel, lengthscale::Real)

Construct a transformed kernel with `lengthscale`.

# Examples

```jldoctest
julia> kernel = with_lengthscale(SqExponentialKernel(), 2.5);

julia> x = rand(2);

julia> y = rand(2);

julia> kernel(x, y) ≈ (SqExponentialKernel() ∘ ScaleTransform(0.4))(x, y)
true
```
"""
function with_lengthscale(kernel::Kernel, lengthscale::Real)
    return kernel ∘ ScaleTransform(inv(lengthscale))
end

"""
    with_lengthscale(kernel::Kernel, lengthscales::AbstractVector{<:Real})

Construct a transformed "ARD" kernel with different `lengthscales` for each dimension.

# Examples

```jldoctest
julia> kernel = with_lengthscale(SqExponentialKernel(), [0.5, 2.5]);

julia> x = rand(2);

julia> y = rand(2);

julia> kernel(x, y) ≈ (SqExponentialKernel() ∘ ARDTransform([2, 0.4]))(x, y)
true
```
"""
function with_lengthscale(kernel::Kernel, lengthscales::AbstractVector{<:Real})
    return kernel ∘ ARDTransform(map(inv, lengthscales))
end
