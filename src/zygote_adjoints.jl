@adjoint function evaluate(s::DotProduct, x::AbstractVector, y::AbstractVector)
  dot(x, y), Δ -> begin
    (nothing, Δ .* y, Δ .* x)
  end
end

@adjoint function loggamma(x)
    first(logabsgamma(x)) , Δ -> (Δ .* polygamma(0, x), )
end

@adjoint function kappa(κ::MaternKernel, d::Real)
    ν = first(κ.ν)
    val, grad = pullback(_matern, ν, d)
    return ((iszero(d) ? one(d) : val),
    Δ -> begin
        ∇ = grad(Δ)
        return ((ν = [∇[1]],), iszero(d) ? zero(d) : ∇[2])
    end)
end

@adjoint function ColVecs(X::AbstractMatrix)
    back(Δ::NamedTuple) = (Δ.X,)
    back(Δ::AbstractMatrix) = (Δ,)
    function back(Δ::AbstractVector{<:AbstractVector{<:Real}})
        throw(error("In slow method"))
    end
    return ColVecs(X), back
end

@adjoint function RowVecs(X::AbstractMatrix)
    back(Δ::NamedTuple) = (Δ.X,)
    back(Δ::AbstractMatrix) = (Δ,)
    function back(Δ::AbstractVector{<:AbstractVector{<:Real}})
        throw(error("In slow method"))
    end
    return RowVecs(X), back
end

# @adjoint function evaluate(s::Sinus, x::AbstractVector, y::AbstractVector)
#   d = evaluate(s, x, y)
#   s = sum(sin.(π*(x-y)))
#   d, Δ -> begin
#     (Sinus(Δ ./ s.r), 2Δ .* cos.(x - y) * d, -2Δ .* cos.(x - y) * d)
#   end
# end
