@adjoint function evaluate(s::DotProduct, x::AbstractVector, y::AbstractVector)
  dot(x,y), Δ -> begin
    (nothing, Δ.*y, Δ.*x)
  end
end
