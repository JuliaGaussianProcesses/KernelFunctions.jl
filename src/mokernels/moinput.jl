"""
    MOInput(x::AbstractVector, out_dim::Integer)

A data type to accomodate modelling multi-dimensional output data.
"""
struct MOInput{T<:AbstractVector} <: AbstractVector{Tuple{Any,Int}}
    x::T
    out_dim::Integer
end

Base.size(inp::MOInput) = (inp.out_dim * size(inp.x, 1),)

function Base.getindex(inp::MOInput, ind::Integer)
    @boundscheck if ind <= 0 || ind > length(inp)
        throw(BoundsError(string("Trying to access at ", ind)))
    end 
    
    out_dim = ind ÷ length(inp.x) + 1
    ind = ind % length(inp.x)
    if ind==0 ind = length(inp.x); out_dim-=1 end
    return (inp.x[ind], out_dim::Int)
end
