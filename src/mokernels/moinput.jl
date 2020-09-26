"""
    MOInput(x::AbstractVector, out_dim::Integer)

A data type to accomodate modelling multi-dimensional output data.
"""
struct MOInput{T<:AbstractVector} <: AbstractVector{Tuple{Any,Int}}
    x::T
    out_dim::Integer
end

Base.size(inp::MOInput) = (inp.out_dim * length(inp.x),)

function Base.getindex(inp::MOInput, ind::Integer)
    @boundscheck checkbounds(inp, ind)
    out_dim, ind = fldmod1(ind, length(inp.x))
    return inp.x[ind], out_dim
end

Base.iterate(inp::MOInput) = (inp[1], 1)
Base.iterate(inp::MOInput, state) = (state<length(inp)) ? (inp[state + 1], state + 1) : nothing
