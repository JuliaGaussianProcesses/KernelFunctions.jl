@adjoint function evaluate(s::SqEuclidean, x::AbstractVector, y::AbstractVector)
  δ = x .- y
  sum(abs2, δ), Δ -> begin
    x̄ = (2 * Δ) .* δ
    (nothing, x̄, -x̄)
  end
end

@adjoint function evaluate(s::Euclidean, x::AbstractVector, y::AbstractVector)
  D = x.-y
  δ = sqrt(sum(abs2,D))
  δ, Δ -> begin
    x̄ = Δ .* D / (δ + eps(δ))
    (nothing, x̄, -x̄)
  end
end

@adjoint function evaluate(s::DotProduct, x::AbstractVector, y::AbstractVector)
  dot(x,y), Δ -> begin
    (nothing, Δ.*y, Δ.*x)
  end
end
