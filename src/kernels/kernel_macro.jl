using MacroTools: @capture

"""

"""
macro kernel(expr::Expr,arg=nothing)
    @capture(expr,(scale_*k_ | k_)) || throw(error("@kernel first arguments should be of the form `σ*Kernel()` or `Kernel()`"))
    @show kw
    t = if @capture(arg,kw_=val_)
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
        return esc(:(transform($k,$t)))
    else
        return esc(:($scale*transform($k,$t)))
    end
end
