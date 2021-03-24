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
        return NO_FIELDS, Zero(), Zero()
    end
    return d, evaluate_pullback
end

function ChainRulesCore.rrule(
    ::typeof(Distances.pairwise), d::Delta, X::AbstractMatrix, Y::AbstractMatrix; dims=2
)
    P = Distances.pairwise(d, X, Y; dims=dims)
    function pairwise_pullback(::AbstractMatrix)
        return NO_FIELDS, NO_FIELDS, Zero(), Zero()
    end
    return P, pairwise_pullback
end

function ChainRulesCore.rrule(
    ::typeof(Distances.pairwise), d::Delta, X::AbstractMatrix; dims=2
)
    P = Distances.pairwise(d, X; dims=dims)
    function pairwise_pullback(::AbstractMatrix)
        return NO_FIELDS, NO_FIELDS, Zero()
    end
    return P, pairwise_pullback
end

function ChainRulesCore.rrule(
    ::typeof(Distances.colwise), d::Delta, X::AbstractMatrix, Y::AbstractMatrix
)
    C = Distances.colwise(d, X, Y)
    function colwise_pullback(::AbstractVector)
        return NO_FIELDS, NO_FIELDS, Zero(), Zero()
    end
    return C, colwise_pullback
end

## Reverse Rules DotProduct

function ChainRulesCore.rrule(dist::DotProduct, x::AbstractVector, y::AbstractVector)
    d = dist(x, y)
    function evaluate_pullback(Δ::Any)
        return NO_FIELDS, Δ .* y, Δ .* x
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
    function pairwise_pullback_cols(Δ::AbstractMatrix)
        if dims == 1
            return NO_FIELDS, NO_FIELDS, Δ * Y, Δ' * X
        else
            return NO_FIELDS, NO_FIELDS, Y * Δ', X * Δ
        end
    end
    return P, pairwise_pullback_cols
end

function ChainRulesCore.rrule(
    ::typeof(Distances.pairwise), d::DotProduct, X::AbstractMatrix; dims=2
)
    P = Distances.pairwise(d, X; dims=dims)
    function pairwise_pullback_cols(Δ::AbstractMatrix)
        if dims == 1
            return NO_FIELDS, NO_FIELDS, 2 * Δ * X
        else
            return NO_FIELDS, NO_FIELDS, 2 * X * Δ
        end
    end
    return P, pairwise_pullback_cols
end

function ChainRulesCore.rrule(
    ::typeof(Distances.colwise), d::DotProduct, X::AbstractMatrix, Y::AbstractMatrix
)
    C = Distances.colwise(d, X, Y)
    function colwise_pullback(Δ::AbstractVector)
        return NO_FIELDS, NO_FIELDS, Δ' .* Y, Δ' .* X
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

## Reverse Rulse SqMahalanobis

function ChainRulesCore.rrule(
    dist::Distances.SqMahalanobis, a::AbstractVector, b::AbstractVector
)
    d = dist(a, b)
    function SqMahalanobis_pullback(Δ::Real)
        B_Bᵀ = dist.qmat + transpose(dist.qmat)
        a_b = a - b
        δa = @thunk((B_Bᵀ * a_b) * Δ)
        return (qmat=(a_b * a_b') * Δ,), δa, @thunk(-δa)
    end
    return d, SqMahalanobis_pullback
end

## Reverse Rules for matrix wrappers

function ChainRulesCore.rrule(::Type{<:ColVecs}, X::AbstractMatrix)
    ColVecs_pullback(Δ::Composite) = (NO_FIELDS, Δ.X)
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
    RowVecs_pullback(Δ::Composite) = (NO_FIELDS, Δ.X)
    function RowVecs_pullback(::AbstractVector{<:AbstractVector{<:Real}})
        return error(
            "Pullback on AbstractVector{<:AbstractVector}.\n" *
            "This might happen if you try to use gradients on the generic `kernelmatrix` or `kernelmatrix_diag`.\n" *
            "To solve this issue overload `kernelmatrix(_diag)` for your kernel for `RowVecs`",
        )
    end
    return RowVecs(X), RowVecs_pullback
end
