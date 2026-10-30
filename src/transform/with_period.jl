"""
    with_period(kernel::Kernel, period::Real)

Construct a transformed kernel with `period`.

# Examples

```jldoctest
julia> kernel = with_period(SqExponentialKernel(), π/2);

julia> x = rand();

julia> y = rand();

julia> kernel(x, y) ≈ (SqExponentialKernel() ∘ PeriodicTransform(2/π))(x, y)
true
```
"""
function with_period(kernel::Kernel, period::Real)
    return kernel ∘ PeriodicTransform(inv(period))
end
