## Adjoints Delta
@adjoint function evaluate(s::Delta, x::AbstractVector, y::AbstractVector)
  evaluate(s, x, y), Δ -> begin
    (nothing, nothing, nothing)
  end
end

@adjoint function Distances.pairwise(d::Delta, X::AbstractMatrix, Y::AbstractMatrix; dims=2)
  D = Distances.pairwise(d, X, Y; dims = dims)
  if dims == 1
      return D, Δ -> (nothing, nothing, nothing)
  else
      return D, Δ -> (nothing, nothing, nothing)
  end
end

@adjoint function Distances.pairwise(d::Delta, X::AbstractMatrix; dims=2)
  D = Distances.pairwise(d, X; dims = dims)
  if dims == 1
      return D, Δ -> (nothing, nothing)
  else
      return D, Δ -> (nothing, nothing)
  end
end

## Adjoints DotProduct
@adjoint function evaluate(s::DotProduct, x::AbstractVector, y::AbstractVector)
  dot(x, y), Δ -> begin
    (nothing, Δ .* y, Δ .* x)
  end
end

@adjoint function Distances.pairwise(d::DotProduct, X::AbstractMatrix, Y::AbstractMatrix; dims=2)
  D = Distances.pairwise(d, X, Y; dims = dims)
  if dims == 1
      return D, Δ -> (nothing, Δ * Y, (X' * Δ)')
  else
      return D, Δ -> (nothing, (Δ * Y')', X * Δ)
  end
end

@adjoint function Distances.pairwise(d::DotProduct, X::AbstractMatrix; dims=2)
  D = Distances.pairwise(d, X; dims = dims)
  if dims == 1
      return D, Δ -> (nothing, 2 * Δ * X)
  else
      return D, Δ -> (nothing, 2 * X * Δ)
  end
end

## Adjoints Sinus
@adjoint function evaluate(s::Sinus, x::AbstractVector, y::AbstractVector)
  d = (x - y)
  sind = sinpi.(d)
  val = sum(abs2, sind ./ s.r)
  gradx = 2π .* cospi.(d) .* sind ./ (s.r .^ 2)
  val, Δ -> begin
    ((r = -2Δ .* abs2.(sind) ./ s.r,), Δ * gradx, - Δ * gradx)
  end
end

@adjoint function ColVecs(X::AbstractMatrix)
    ColVecs_pullback(Δ::NamedTuple) = (Δ.X,)
    ColVecs_pullback(Δ::AbstractMatrix) = (Δ,)
    function ColVecs_pullback(Δ::AbstractVector{<:AbstractVector{<:Real}})
        throw(error("In slow method"))
    end
    return ColVecs(X), ColVecs_pullback
end

@adjoint function RowVecs(X::AbstractMatrix)
    RowVecs_pullback(Δ::NamedTuple) = (Δ.X,)
    RowVecs_pullback(Δ::AbstractMatrix) = (Δ,)
    function RowVecs_pullback(Δ::AbstractVector{<:AbstractVector{<:Real}})
        throw(error("In slow method"))
    end
    return RowVecs(X), RowVecs_pullback
end

@adjoint function Base.map(t::Transform, X::ColVecs)
    pullback(_map, t, X)
end

@adjoint function Base.map(t::Transform, X::RowVecs)
    pullback(_map, t, X)
end

@adjoint function (dist::Distances.SqMahalanobis)(a, b)
    function SqMahalanobis_pullback(Δ::Real)
        B_Bᵀ = dist.qmat + transpose(dist.qmat)
        a_b = a - b
        δa = (B_Bᵀ * a_b) * Δ
        return (qmat = (a_b * a_b') * Δ,), δa, -δa 
    end
    return evaluate(dist, a, b), SqMahalanobis_pullback
end
