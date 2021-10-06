# Delta is not following the PreMetric rules since d(x, x) == 1
struct Delta <: Distances.UnionPreMetric end

@inline Distances.eval_op(::Delta, a::Real, b::Real) = a == b
@inline Distances.eval_reduce(::Delta, a, b) = a && b
@inline Distances.eval_start(::Delta, a, b) = true
@inline (dist::Delta)(a::AbstractArray, b::AbstractArray) = Distances._evaluate(dist, a, b)
@inline (dist::Delta)(a::Number, b::Number) = a == b

Distances.result_type(::Delta, Ta::Type, Tb::Type) = Bool
