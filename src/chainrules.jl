## Forward Rules

function ChainRulesCore.frule((_, Δx, Δy), d::Distances.Euclidean, x, y)
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
    function pairwise_pullback(::Any)
        return NO_FIELDS, NO_FIELDS, Zero(), Zero()
    end
    return P, pairwise_pullback
end

function ChainRulesCore.rrule(
    ::typeof(Distances.pairwise), d::Delta, X::AbstractMatrix; dims=2
)
    P = Distances.pairwise(d, X; dims=dims)
    function pairwise_pullback(::Any)
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
    function evaluate_pullback(Δ)
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
    if dims == 1
        function pairwise_pullback_cols(Δ)
            return NO_FIELDS, NO_FIELDS, Δ * Y, Δ' * X
        end
        return P, pairwise_pullback_cols
    else
        function pairwise_pullback_rows(Δ)
            return NO_FIELDS, NO_FIELDS, Y * Δ', X * Δ
        end
        return P, pairwise_pullback_rows
    end
end

function ChainRulesCore.rrule(
    ::typeof(Distances.pairwise), d::DotProduct, X::AbstractMatrix; dims=2
)
    P = Distances.pairwise(d, X; dims=dims)
    if dims == 1
        function pairwise_pullback_cols(Δ)
            return NO_FIELDS, NO_FIELDS, 2 * Δ * X
        end
        return P, pairwise_pullback_cols
    else
        function pairwise_pullback_rows(Δ)
            return NO_FIELDS, NO_FIELDS, 2 * X * Δ
        end
        return P, pairwise_pullback_rows
    end
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

function ChainRulesCore.rrule(
    ::typeof(Distances.evaluate), s::Sinus, x::AbstractVector, y::AbstractVector
)
    d = x - y
    sind = sinpi.(d)
    val = sum(abs2, sind ./ s.r)
    gradx = twoπ .* cospi.(d) .* sind ./ (s.r .^ 2)
    function evaluate_pullback(Δ)
        return NO_FIELDS, (r=-2Δ .* abs2.(sind) ./ s.r,), Δ * gradx, -Δ * gradx
    end
    return val, evaluate_pullback
end

## Reverse Rules for matrix wrappers

function ChainRulesCore.rrule(::Type{<:ColVecs}, X::AbstractMatrix)
    ColVecs_pullback(Δ::Composite) = (NO_FIELDS, Δ.X)
    ColVecs_pullback(Δ::NamedTuple) = (Δ.X,)
    ColVecs_pullback(Δ::AbstractMatrix) = (Δ,)
    function ColVecs_pullback(::AbstractVector{<:AbstractVector{<:Real}})
        return throw(error("In slow method"))
    end
    return ColVecs(X), ColVecs_pullback
end

function ChainRulesCore.rrule(::Type{<:RowVecs}, X::AbstractMatrix)
    RowVecs_pullback(Δ::Composite) = (NO_FIELDS, Δ.X)
    RowVecs_pullback(Δ::NamedTuple) = (Δ.X,)
    RowVecs_pullback(Δ::AbstractMatrix) = (Δ,)
    function RowVecs_pullback(::AbstractVector{<:AbstractVector{<:Real}})
        return throw(error("In slow method"))
    end
    return RowVecs(X), RowVecs_pullback
end

ZygoteRules.@adjoint function Base.map(t::Transform, X::ColVecs)
    return ZygoteRules.pullback(_map, t, X)
end

ZygoteRules.@adjoint function Base.map(t::Transform, X::RowVecs)
    return ZygoteRules.pullback(_map, t, X)
end

function ChainRulesCore.rrule(dist::Distances.SqMahalanobis, a, b)
    d = dist(a, b)
    function SqMahalanobis_pullback(Δ::Real)
        B_Bᵀ = dist.qmat + transpose(dist.qmat)
        a_b = a - b
        δa = (B_Bᵀ * a_b) * Δ
        return (qmat=(a_b * a_b') * Δ,), δa, -δa
    end
    return d, SqMahalanobis_pullback
end
