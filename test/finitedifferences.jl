function FiniteDifferences.to_vec(x::Tuple{T, Int}) where {T}
    function MOinput_from_vec(x_vec)
        return first(x_vec)
    end
    return [x], MOinput_from_vec
end

FiniteDifferences.to_vec(x::Vector{Tuple{T, Int}}) where {T} = (x, identity)
