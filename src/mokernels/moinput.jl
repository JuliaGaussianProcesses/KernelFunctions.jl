"""
    MOInput(x::AbstractVector, out_dim::Integer)

A data type to accomodate modelling multi-dimensional output data.
"""
struct MOInput{T<:AbstractVector} <: AbstractVector{Tuple{Any,Int}}
    x::T
    out_dim::Integer
end

Base.length(inp::MOInput) = inp.out_dim * length(inp.x)

Base.size(inp::MOInput, d) = d::Integer == 1 ? inp.out_dim * size(inp.x, 1) : 1
Base.size(inp::MOInput) = (inp.out_dim * size(inp.x, 1),)

Base.lastindex(inp::MOInput) = length(inp)
Base.firstindex(inp::MOInput) = 1

function Base.getindex(inp::MOInput, ind::Integer)
    if ind > 0
        out_dim = ind ÷ length(inp.x) + 1
        ind = ind % length(inp.x)
        if ind == 0
            ind = length(inp.x)
            out_dim -= 1
        end
        return (inp.x[ind], out_dim::Int)
    else
        throw(BoundsError(string("Trying to access at ", ind)))
    end
end

Base.iterate(inp::MOInput) = (inp[1], 1)
function Base.iterate(inp::MOInput, state)
    return (state < length(inp)) ? (inp[state + 1], state + 1) : nothing
end
