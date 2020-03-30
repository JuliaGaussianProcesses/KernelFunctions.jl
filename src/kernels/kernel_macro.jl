"""
    @kernel [variance *]kernel::Kernel [l=Real/Vector / t=transform::Transform / transform::Transform]

The `@kernel` macro is an helping alias to the [`transform`](@ref) function.
The first argument should be a kernel multiplied (or not) by a scalar (variance of the kernel).
The second argument (optional) can be a keyword :
 - `l=ρ` where `ρ` is a positive scalar or a vector of scalar
 - `t=transform` where `transform` is a [`Transform`](@ref) object
One can also directly use a `Transform` object without a keyword.
Here are some examples :
```julia
    k = @kernel SqExponentialKernel() l=3.0
    k == transform(SqExponentialKernel(), ScaleTransform(3.0))

    k = @kernel (MaternKernel(ν=3.0) + LinearKernel()) t=LowRankTransform(rand(4,3))
    k == transform(KernelSum(MaternKernel(ν=3.0), LinearKernel()), LowRankTransform(rand(4,3)))

    k = @kernel 4.0*ExponentiatedKernel() ScaleTransform(3.0)
    k == ScaleTransform(transform(ExponentiatedKernel(), ScaleTransform(3.0)), 4.0)
"""
macro kernel(expr::Expr, arg = nothing)
    @capture(expr, (scale_ * k_ | k_)) || throw(error("@kernel first arguments should be of the form `σ*kernel` or `kernel`"))
    t = if @capture(arg, kw_ = val_)
        if kw == :l
            val
        elseif kw == :t
            val
        else
            throw(error("The additional argument could not be intepreted. Please see documentation of `@kernel`"))
        end
    else
        arg
    end
    if isnothing(scale)
        return esc(:(transform($k, $t)))
    else
        return esc(:($scale*transform($k, $t)))
    end
end
