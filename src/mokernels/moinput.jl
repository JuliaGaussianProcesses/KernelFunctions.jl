"""
    MOInput

An abstract type to accomodate modelling multi-dimensional output data. There are two
subtypes that each specify a unique ordering of the dimensions.

See [Inputs for Multiple Outputs](@ref) in the docs for more info.
"""
abstract type MOInput <: AbstractVector{Tuple{Any,Int}} end

"""
    IsotopicByFeatures(x::AbstractVector, out_dim::Integer)

`IsotopicByFeatures(x, out_dim)` has length `length(x) * out_dim`.

```jldoctest
julia> x = [1, 2, 3];

julia> IsotopicByFeatures(x, 2)
6-element IsotopicByFeatures{Vector{Int64}}:
 (1, 1)
 (2, 1)
 (3, 1)
 (1, 2)
 (2, 2)
 (3, 2)
```

As shown above, an `IsotopicByFeatures` represents a vector of tuples.
The first `length(x)` elements represent the inputs for the first output, the second
`length(x)` elements represent the inputs for the second output, etc.
"""

struct IsotopicByFeatures{T<:AbstractVector} <: MOInput
    x::T
    out_dim::Integer
end

"""
    IsotopicByOutputs(x::AbstractVector, out_dim::Integer)

`IsotopicByOutputs(x, out_dim)` has length `out_dim * length(x)`.

```jldoctest
julia> x = [1, 2, 3];

julia> IsotopicByOutputs(x, 2)
6-element IsotopicByOutputs{Vector{Int64}}:
 (1, 1)
 (1, 2)
 (2, 1)
 (2, 2)
 (3, 1)
 (3, 2)
```

As shown above, an `IsotopicByOutputs` represents a vector of tuples.
The first `out_dim` elements represent all outputs for the first input, the second
`out_dim` elements represent the outputs for the second input, etc.
"""

struct IsotopicByOutputs{T<:AbstractVector} <: MOInput
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
        if inp isa IsotopicByOutputs
            out_dim = ind ÷ length(inp.x) + 1
            ind = ind % length(inp.x)
            if ind == 0
                ind = length(inp.x)
                out_dim -= 1
            end
        elseif inp isa IsotopicByFeatures
            out_dim = (ind - 1) % inp.out_dim + 1
            ind = Int(round((ind - 1) ÷ inp.out_dim) + 1)
        else
            throw(TypeError(string("Trying to access at ", ind)))
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
