## Reverse Rules Delta

function rrule(::typeof(Distances.evaluate), s::Delta, x::AbstractVector, y::AbstractVector)
    d = evaluate(s, x, y)
    function evaluate_pullback(::Any)
        return NO_FIELDS, Zero(), Zero()
    end
    return d, evaluate_pullback
end

function rrule(
    ::typeof(Distances.pairwise), d::Delta, X::AbstractMatrix, Y::AbstractMatrix; dims=2
)
    P = Distances.pairwise(d, X, Y; dims=dims)
    function pairwise_pullback(::Any)
        return NO_FIELDS, Zero(), Zero()
    end
    return P, pairwise_pullback
end

function rrule(::typeof(Distances.pairwise), d::Delta, X::AbstractMatrix; dims=2)
    P = Distances.pairwise(d, X; dims=dims)
    function pairwise_pullback(::Any)
        return NO_FIELDS, Zero()
    end
    return P, pairwise_pullback
end

function rrule(::typeof(Distances.colwise), d::Delta, X::AbstractMatrix, Y::AbstractMatrix)
    C = Distances.colwise(d, X, Y)
    function colwise_pullback(::AbstractVector)
        return NO_FIELDS, Zero(), Zero()
    end
    return C, colwise_pullback
end

## Reverse Rules DotProduct
function rrule(
    ::typeof(Distances.evaluate), s::DotProduct, x::AbstractVector, y::AbstractVector
)
    d = dot(x, y)
    function evaluate_pullback(Δ)
        return NO_FIELDS, Δ .* y, Δ .* x
    end
    return d, evaluate_pullback
end

function rrule(
    ::typeof(Distances.pairwise),
    d::DotProduct,
    X::AbstractMatrix,
    Y::AbstractMatrix;
    dims=2,
)
    P = Distances.pairwise(d, X, Y; dims=dims)
    if dims == 1
        function pairwise_pullback_cols(Δ)
            return NO_FIELDS, Δ * Y, Δ' * X
        end
        return P, pairwise_pullback_cols
    else
        function pairwise_pullback_rows(Δ)
            return NO_FIELDS, Y * Δ', X * Δ
        end
        return P, pairwise_pullback_rows
    end
end

function rrule(::typeof(Distances.pairwise), d::DotProduct, X::AbstractMatrix; dims=2)
    P = Distances.pairwise(d, X; dims=dims)
    if dims == 1
        function pairwise_pullback_cols(Δ)
            return NO_FIELDS, 2 * Δ * X
        end
        return P, pairwise_pullback_cols
    else
        function pairwise_pullback_rows(Δ)
            return NO_FIELDS, 2 * X * Δ
        end
        return P, pairwise_pullback_rows
    end
end

function rrule(
    ::typeof(Distances.colwise), d::DotProduct, X::AbstractMatrix, Y::AbstractMatrix
)
    C = Distances.colwise(d, X, Y)
    function colwise_pullback(Δ::AbstractVector)
        return (nothing, Δ' .* Y, Δ' .* X)
    end
    return C, colwise_pullback
end

## Reverse Rules Sinus
function rrule(::typeof(Distances.evaluate), s::Sinus, x::AbstractVector, y::AbstractVector)
    d = (x - y)
    sind = sinpi.(d)
    val = sum(abs2, sind ./ s.r)
    gradx = 2π .* cospi.(d) .* sind ./ (s.r .^ 2)
    function evaluate_pullback(Δ)
        return (r=-2Δ .* abs2.(sind) ./ s.r,), Δ * gradx, -Δ * gradx
    end
    return val, evaluate_pullback
end

## Reverse Rules for matrix wrappers

function rrule(::ColVecs, X::AbstractMatrix)
    ColVecs_pullback(Δ::NamedTuple) = (Δ.X,)
    ColVecs_pullback(Δ::AbstractMatrix) = (Δ,)
    function ColVecs_pullback(Δ::AbstractVector{<:AbstractVector{<:Real}})
        return throw(error("In slow method"))
    end
    return ColVecs(X), ColVecs_pullback
end

function rrule(::RowVecs, X::AbstractMatrix)
    RowVecs_pullback(Δ::NamedTuple) = (Δ.X,)
    RowVecs_pullback(Δ::AbstractMatrix) = (Δ,)
    function RowVecs_pullback(Δ::AbstractVector{<:AbstractVector{<:Real}})
        return throw(error("In slow method"))
    end
    return RowVecs(X), RowVecs_pullback
end

# function rrule(::typeof(Base.map), t::Transform, X::ColVecs)
#     return pullback(_map, t, X)
# end

# function rrule(::typeof(Base.map), t::Transform, X::RowVecs)
#     return pullback(_map, t, X)
# end

# @adjoint function (dist::Distances.SqMahalanobis)(a, b)
#     function SqMahalanobis_pullback(Δ::Real)
#         B_Bᵀ = dist.qmat + transpose(dist.qmat)
#         a_b = a - b
#         δa = (B_Bᵀ * a_b) * Δ
#         return (qmat=(a_b * a_b') * Δ,), δa, -δa
#     end
#     return evaluate(dist, a, b), SqMahalanobis_pullback
# end
