"""
    MOInput

This is a data type to accomodate modelling multi-dimensional inputs of a multi-output GP. 

"""
struct MOInput{T,X} <: AbstractVector{Tuple{T,Int}}
    x::X
    out_dim::Int
end

"""
    mo_input(x::AbstractVector, out_dim::Int)

Return `MOInput` object to accomodate modelling multi-dimensional inputs of a multi-output GP. 
This datatype interprets the inputs of the model in such a way that individual GPs for 
each output dimension is simulated by a single Multi Output GP.

For example,
```jldoctest
julia> x1 = [collect(i * 4:i * 4 + 3) for i in 0:4]
5-element Array{Array{Int64,1},1}:
 [0, 1, 2, 3]
 [4, 5, 6, 7]
 [8, 9, 10, 11]
 [12, 13, 14, 15]
 [16, 17, 18, 19]

julia> mo_input(x1, 2)
10-element KernelFunctions.MOInput{Array{Int64,1},Array{Array{Int64,1},1}}:
 ([0, 1, 2, 3], 1)
 ([4, 5, 6, 7], 1)
 ([8, 9, 10, 11], 1)
 ([12, 13, 14, 15], 1)
 ([16, 17, 18, 19], 1)
 ([0, 1, 2, 3], 2)
 ([4, 5, 6, 7], 2)
 ([8, 9, 10, 11], 2)
 ([12, 13, 14, 15], 2)
 ([16, 17, 18, 19], 2)

 julia> x2 = reshape(0:19, 4, 5)
 4×5 reshape(::UnitRange{Int64}, 4, 5) with eltype Int64:
  0  4   8  12  16
  1  5   9  13  17
  2  6  10  14  18
  3  7  11  15  19 

julia> x1 == ColVecs(x2)
true

julia> mo_input(x1, 2) == mo_input(ColVecs(x2), 2)
true

```
We can see that the same input is repeated for every output dimension. `MOInput` enables us to 
simulate repeated inputs without allocating additional memory.
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
