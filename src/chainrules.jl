## rules for Delta
function ChainRulesCore.rrule(::typeof(evaluate), s::Delta, x::AbstractVector, y::AbstractVector)
    function back(Δ)
        return (NO_FIELDS, DoesNotExist(), DoesNotExist(), DoesNotExist())
    end
    evaluate(s, x, y), back
end

function ChainRulesCore.rrule(::typeof(Distances.pairwise), d::Delta, X::AbstractMatrix, Y::AbstractMatrix; dims=2)
    D = Distances.pairwise(d, X, Y; dims = dims)
    function back(Δ)
        return (NO_FIELDS, DoesNotExist(), DoesNotExist(), DoesNotExist())
    end
    return D, back
end

function ChainRulesCore.rrule(::typeof(Distances.pairwise), d::Delta, X::AbstractMatrix; dims=2)
    D = Distances.pairwise(d, X; dims = dims)
    back(Δ) = (NO_FIELDS, DoesNotExist(), DoesNotExist())
    return D, back
end

## rules for DotProduct
function ChainRulesCore.rrule(::typeof(evaluate), s::DotProduct, x::AbstractVector, y::AbstractVector)
    back(Δ) = (NO_FIELDS, nothing, @thunk(Δ .* y), @thunk(Δ .* x))
    return dot(x, y), back
end

function ChainRulesCore.rrule(::typeof(Distances.pairwise), d::DotProduct, X::AbstractMatrix, Y::AbstractMatrix; dims=2)
    D = Distances.pairwise(d, X, Y; dims = dims)
    
    function back(Δ)
        if dims == 1
            return (NO_FIELDS, nothing, @thunk(Δ * Y), @thunk((X' * Δ)'))
        else
            return (NO_FIELDS, nothing, @thunk((Δ * Y')'), @thunk(X * Δ))  
        end
    end
    return D, back
end

function ChainRulesCore.rrule(::typeof(Distances.pairwise), d::DotProduct, X::AbstractMatrix; dims=2)
    D = Distances.pairwise(d, X; dims = dims)
    
    function back(Δ)
        if dims == 1
            return (NO_FIELDS, nothing, @thunk(2 * Δ * X))
        else
            return (NO_FIELDS, nothing, @thunk(2 * X * Δ))
        end
    end
    return D, back
end

## rules for Sinus
function ChainRulesCore.rrule(::typeof(evaluate), s::Sinus, x::AbstractVector, y::AbstractVector)
    d = @thunk((x - y))
    sind = @thunk(sinpi.(d))
    val = @thunk(sum(abs2, sind ./ s.r))
    gradx = @thunk(2π .* cospi.(d) .* sind ./ (s.r .^ 2))
    function back(Δ)
        return (NO_FIELDS, (r = @thunk(-2Δ .* abs2.(sind) ./ s.r),), @thunk(Δ * gradx), @thunk(- Δ * gradx))
    end
    val, back
end


# rules for ColVecs and RowVecs
vecs_pullback(Δ::NamedTuple) = (NO_FIELDS, Δ.X,)
vecs_pullback(Δ::AbstractMatrix) = (NO_FIELDS, Δ,)
function vecs_pullback(Δ::AbstractVector{<:AbstractVector{<:Real}})
    throw(error("In slow method"))
end

function ChainRulesCore.rrule(::Type{ColVecs}, X::AbstractMatrix)
    return ColVecs(X), vecs_pullback
end

function ChainRulesCore.rrule(::Type{RowVecs}, X::AbstractMatrix)
    return RowVecs(X), vecs_pullback
end


# rules for transforms
@adjoint function Base.map(t::Transform, X::ColVecs)
    return pullback(_map, t, X)
end

@adjoint function Base.map(t::Transform, X::RowVecs)
    return pullback(_map, t, X)
end
