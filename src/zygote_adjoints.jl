@adjoint function evaluate(s::DotProduct, x::AbstractVector, y::AbstractVector)
  dot(x, y), Δ -> begin
    (nothing, Δ .* y, Δ .* x)
  end
end

# @adjoint function evaluate(s::Sinus, x::AbstractVector, y::AbstractVector)
#   d = evaluate(s, x, y)
#   s = sum(sin.(π*(x-y)))
#   d, Δ -> begin
#     (Sinus(Δ ./ s.r), 2Δ .* cos.(x - y) * d, -2Δ .* cos.(x - y) * d)
#   end
# end
