"""
    MOInput(x::AbstractVector, out_dim::Integer)

A data type to accomodate modelling multi-dimensional output data.

`MOInput(x, out_dim)` has length `length(x) * out_dim`.

```jldoctest
julia> x = [1, 2, 3];

julia> MOInput(x, 2)
6-element MOInput{Vector{Int64}}:
 (1, 1)
 (2, 1)
 (3, 1)
 (1, 2)
 (2, 2)
 (3, 2)
```

As shown above, an `MOInput` represents a vector of tuples.
The first `length(x)` elements represent the inputs for the first output, the second
`length(x)` elements represent the inputs for the second output, etc.

See the docs for a more extensive discussion of this design decision.
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
