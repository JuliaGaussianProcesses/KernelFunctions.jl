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

@adjoint function Base.map(t::Transform, X::ColVecs)
    pullback(_map, t, X)
end

@adjoint function Base.map(t::Transform, X::RowVecs)
    pullback(_map, t, X)
end

@adjoint function (dist::Distances.SqMahalanobis)(a, b)
    function back(Δ::Real)
        B_Bᵀ = dist.qmat + transpose(dist.qmat)
        a_b = a - b
        δa = B_Bᵀ * a_b
        return (qmat = a_b * a_b',), δa, -δa 
    end
  return evaluate(dist, a, b), back
end


# FIXME
@adjoint function Distances.pairwise(
    dist::SqMahalanobis, 
    a::AbstractMatrix, 
    b::AbstractMatrix;
    dims::Union{Nothing,Integer}=nothing
    )
    function back(Δ::AbstractMatrix)
        B_Bᵀ = dist.qmat + transpose(dist.qmat)
        a_b = map(
            x -> (first(last(x)) - last(last(x)))*first(x), 
            zip(
                Δ,
                Iterators.product(eachslice(a, dims=dims), eachslice(b, dims=dims))
            )
        )
        δa = reduce(hcat, sum(map(x -> B_Bᵀ*x, a_b), dims=1))
        δB = sum(map(x -> x*transpose(x), a_b))
        return (qmat=δB,), δa, -δa
    end
    return Distances.pairwise(dist, a, b, dims=dims), back
end
