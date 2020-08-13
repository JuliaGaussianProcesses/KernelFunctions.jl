## rules for Delta
function ChainRulesCore.rrule(::typeof(evaluate), s::Delta, x::AbstractVector, y::AbstractVector)
  evaluate(s, x, y), Δ -> begin
    (NO_FIELDS, DoesNotExist(), DoesNotExist(), DoesNotExist())
  end
end

function ChainRulesCore.rrule(::typeof(Distances.pairwise), d::Delta, X::AbstractMatrix, Y::AbstractMatrix; dims=2)
  D = Distances.pairwise(d, X, Y; dims = dims)
  if dims == 1
      return D, Δ -> (NO_FIELDS, DoesNotExist(), DoesNotExist(), DoesNotExist())
  else
      return D, Δ -> (NO_FIELDS, DoesNotExist(), DoesNotExist(), DoesNotExist())
  end
end

function ChainRulesCore.rrule(::typeof(Distances.pairwise), d::Delta, X::AbstractMatrix; dims=2)
  D = Distances.pairwise(d, X; dims = dims)
  if dims == 1
      return D, Δ -> (NO_FIELDS, DoesNotExist(), DoesNotExist())
  else
      return D, Δ -> (NO_FIELDS, DoesNotExist(), DoesNotExist())
  end
end

## rules for DotProduct
function ChainRulesCore.rrule(::typeof(evaluate), s::DotProduct, x::AbstractVector, y::AbstractVector)
  dot(x, y), Δ -> begin
    (NO_FIELDS, nothing, Δ .* y, Δ .* x)
  end
end

function ChainRulesCore.rrule(::typeof(Distances.pairwise), d::DotProduct, X::AbstractMatrix, Y::AbstractMatrix; dims=2)
  D = Distances.pairwise(d, X, Y; dims = dims)
  if dims == 1
      return D, Δ -> (NO_FIELDS, nothing, Δ * Y, (X' * Δ)')
  else
      return D, Δ -> (NO_FIELDS, nothing, (Δ * Y')', X * Δ)
  end
end

function ChainRulesCore.rrule(::typeof(Distances.pairwise), d::DotProduct, X::AbstractMatrix; dims=2)
  D = Distances.pairwise(d, X; dims = dims)
  if dims == 1
      return D, Δ -> (NO_FIELDS, nothing, 2 * Δ * X)
  else
      return D, Δ -> (NO_FIELDS, nothing, 2 * X * Δ)
  end
end

## rules for Sinus
function ChainRulesCore.rrule(::typeof(evaluate), s::Sinus, x::AbstractVector, y::AbstractVector)
  d = (x - y)
  sind = sinpi.(d)
  val = sum(abs2, sind ./ s.r)
  gradx = 2π .* cospi.(d) .* sind ./ (s.r .^ 2)
  val, Δ -> begin
    (NO_FIELDS, (r = -2Δ .* abs2.(sind) ./ s.r,), Δ * gradx, - Δ * gradx)
  end
end


# rules for ColVecs and RowVecs
function ChainRulesCore.rrule(::typeof(ColVecs), X::AbstractMatrix)
    back(Δ::NamedTuple) = (NO_FIELDS, Δ.X,)
    back(Δ::AbstractMatrix) = (NO_FIELDS, Δ,)
    function back(Δ::AbstractVector{<:AbstractVector{<:Real}})
        throw(error("In slow method"))
    end
    return ColVecs(X), back
end

function ChainRulesCore.rrule(::typeof(RowVecs), X::AbstractMatrix)
    back(Δ::NamedTuple) = (NO_FIELDS, Δ.X,)
    back(Δ::AbstractMatrix) = (NO_FIELDS, Δ,)
    function back(Δ::AbstractVector{<:AbstractVector{<:Real}})
        throw(error("In slow method"))
    end
    return RowVecs(X), back
end


# rules for transforms
function ChainRulesCore.rrule(::typeof(Base.map), t::Transform, X::ColVecs)
    ChainRulesCore.rrule(_map, t, X)
end

function ChainRulesCore.rrule(::typeof(Base.map), t::Transform, X::RowVecs)
    ChainRulesCore.rrule(_map, t, X)
end
