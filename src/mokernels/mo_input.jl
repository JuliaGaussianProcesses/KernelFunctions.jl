"""
    MOInput

A data type to accomodate modelling multi-dimensional input data.
"""
struct MOInput{T,X} <: AbstractVector{Tuple{T,Int}}
    x::X
    out_dim::Int
end

"""
    mo_input(x::AbstractVector, out_dim::Int)

Return `MOInput` to accomodate modelling multi-dimensional output data.
"""
function mo_input(x::X, out_dim::Int) where {T,X<:AbstractVector{T}}
    return MOInput{T,X}(x, out_dim)
end

Base.size(inp::MOInput) = (inp.out_dim * length(inp.x),)

@inline function Base.getindex(inp::MOInput, ind::Integer)
    @boundscheck checkbounds(inp, ind)
    out_dim, ind = fldmod1(ind, length(inp.x))
    return inp.x[ind], out_dim
end
