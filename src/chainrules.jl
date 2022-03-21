## Forward Rules

# Note that this is type piracy as the derivative should be NaN for x == y.
function ChainRulesCore.frule(
    (_, Δx, Δy), d::Distances.Euclidean, x::AbstractVector, y::AbstractVector
)
    Δ = x - y
    D = sqrt(sum(abs2, Δ))
    if !iszero(D)
        Δ ./= D
    end
    return D, dot(Δ, Δx) - dot(Δ, Δy)
end

## Reverse Rules Delta

function ChainRulesCore.rrule(dist::Delta, x::AbstractVector, y::AbstractVector)
    d = dist(x, y)
    function evaluate_pullback(::Any)
        return NoTangent(), ZeroTangent(), ZeroTangent()
    end
    return d, evaluate_pullback
end

function ChainRulesCore.rrule(
    ::typeof(Distances.pairwise), d::Delta, X::AbstractMatrix, Y::AbstractMatrix; dims=2
)
    P = Distances.pairwise(d, X, Y; dims=dims)
    function pairwise_pullback(::Any)
        return NoTangent(), NoTangent(), ZeroTangent(), ZeroTangent()
    end
    return P, pairwise_pullback
end

function ChainRulesCore.rrule(
    ::typeof(Distances.pairwise), d::Delta, X::AbstractMatrix; dims=2
)
    P = Distances.pairwise(d, X; dims=dims)
    function pairwise_pullback(::Any)
        return NoTangent(), NoTangent(), ZeroTangent()
    end
    return P, pairwise_pullback
end

function ChainRulesCore.rrule(
    ::typeof(Distances.colwise), d::Delta, X::AbstractMatrix, Y::AbstractMatrix
)
    C = Distances.colwise(d, X, Y)
    function colwise_pullback(::Any)
        return NoTangent(), NoTangent(), ZeroTangent(), ZeroTangent()
    end
    return C, colwise_pullback
end

## Reverse Rules DotProduct

function ChainRulesCore.rrule(dist::DotProduct, x::AbstractVector, y::AbstractVector)
    d = dist(x, y)
    function evaluate_pullback(Δ::Any)
        return NoTangent(), Δ .* y, Δ .* x
    end
    return d, evaluate_pullback
end

function ChainRulesCore.rrule(
    ::typeof(Distances.pairwise),
    d::DotProduct,
    X::AbstractMatrix,
    Y::AbstractMatrix;
    dims=2,
)
    P = Distances.pairwise(d, X, Y; dims=dims)
    function pairwise_pullback_cols(Δ::Any)
        if dims == 1
            return NoTangent(), NoTangent(), Δ * Y, Δ' * X
        else
            return NoTangent(), NoTangent(), Y * Δ', X * Δ
        end
    end
    return P, pairwise_pullback_cols
end

function ChainRulesCore.rrule(
    ::typeof(Distances.pairwise), d::DotProduct, X::AbstractMatrix; dims=2
)
    P = Distances.pairwise(d, X; dims=dims)
    function pairwise_pullback_cols(Δ::Any)
        if dims == 1
            return NoTangent(), NoTangent(), 2 * Δ * X
        else
            return NoTangent(), NoTangent(), 2 * X * Δ
        end
    end
    return P, pairwise_pullback_cols
end

function ChainRulesCore.rrule(
    ::typeof(Distances.colwise), d::DotProduct, X::AbstractMatrix, Y::AbstractMatrix
)
    C = Distances.colwise(d, X, Y)
    function colwise_pullback(Δ::Any)
        return NoTangent(), NoTangent(), Δ' .* Y, Δ' .* X
    end
    return C, colwise_pullback
end

## Reverse Rules Sinus

function ChainRulesCore.rrule(s::Sinus, x::AbstractVector, y::AbstractVector)
    d = x - y
    sind = sinpi.(d)
    abs2_sind_r = abs2.(sind) ./ s.r
    val = sum(abs2_sind_r)
    gradx = twoπ .* cospi.(d) .* sind ./ (s.r .^ 2)
    function evaluate_pullback(Δ::Any)
        return (r=-2Δ .* abs2_sind_r,), Δ * gradx, -Δ * gradx
    end
    return val, evaluate_pullback
end

## Reverse Rules SqMahalanobis

function ChainRulesCore.rrule(
    dist::Distances.SqMahalanobis, a::AbstractVector, b::AbstractVector
)
    d = dist(a, b)
    function SqMahalanobis_pullback(Δ::Real)
        a_b = a - b
        ∂qmat = InplaceableThunk(
            X̄ -> mul!(X̄, a_b, a_b', true, Δ), @thunk((a_b * a_b') * Δ)
        )
        ∂a = InplaceableThunk(
            X̄ -> mul!(X̄, dist.qmat, a_b, true, 2 * Δ), @thunk((2 * Δ) * dist.qmat * a_b)
        )
        ∂b = InplaceableThunk(
            X̄ -> mul!(X̄, dist.qmat, a_b, true, -2 * Δ), @thunk((-2 * Δ) * dist.qmat * a_b)
        )
        return Tangent{typeof(dist)}(; qmat=∂qmat), ∂a, ∂b
    end
    return d, SqMahalanobis_pullback
end

## Reverse Rules for matrix wrappers

function ChainRulesCore.rrule(::Type{<:ColVecs}, X::AbstractMatrix)
    ColVecs_pullback(Δ::Tangent) = (NoTangent(), Δ.X)
    function ColVecs_pullback(::AbstractVector{<:AbstractVector{<:Real}})
        return error(
            "Pullback on AbstractVector{<:AbstractVector}.\n" *
            "This might happen if you try to use gradients on the generic `kernelmatrix` or `kernelmatrix_diag`.\n" *
            "To solve this issue overload `kernelmatrix(_diag)` for your kernel for `ColVecs`",
        )
    end
    return ColVecs(X), ColVecs_pullback
end

function ChainRulesCore.rrule(::Type{<:RowVecs}, X::AbstractMatrix)
    RowVecs_pullback(Δ::Tangent) = (NoTangent(), Δ.X)
    function RowVecs_pullback(::AbstractVector{<:AbstractVector{<:Real}})
        return error(
            "Pullback on AbstractVector{<:AbstractVector}.\n" *
            "This might happen if you try to use gradients on the generic `kernelmatrix` or `kernelmatrix_diag`.\n" *
            "To solve this issue overload `kernelmatrix(_diag)` for your kernel for `RowVecs`",
        )
    end
    return RowVecs(X), RowVecs_pullback
end
