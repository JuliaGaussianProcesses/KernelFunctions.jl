function FiniteDifferences.to_vec(x::Tuple{T,Int}) where {T}
    x_vec, first_x_from_vec = to_vec(first(x))
    function Tuple_from_vec(x_vec::AbstractVector{<:Real})
        return (first_x_from_vec(x_vec), last(x))
    end
    return x_vec, Tuple_from_vec
end
