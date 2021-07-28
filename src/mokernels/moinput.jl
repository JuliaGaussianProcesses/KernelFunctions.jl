"""
    MOInputIsotopicByFeatures(x::AbstractVector, out_dim::Integer)

`MOInputIsotopicByFeatures(x, out_dim)` has length `out_dim * length(x)`.

```jldoctest
julia> x = [1, 2, 3];

julia> KernelFunctions.MOInputIsotopicByFeatures(x, 2)
6-element KernelFunctions.MOInputIsotopicByFeatures{Int64, Vector{Int64}}:
 (1, 1)
 (1, 2)
 (2, 1)
 (2, 2)
 (3, 1)
 (3, 2)
```

Accommodates modelling multi-dimensional output data where all outputs are always observed.

As shown above, an `MOInputIsotopicByFeatures` represents a vector of tuples.
The first `out_dim` elements represent all outputs for the first input, the second
`out_dim` elements represent the outputs for the second input, etc.

See [Inputs for Multiple Outputs](@ref) in the docs for more info.
"""
struct MOInputIsotopicByFeatures{S,T<:AbstractVector{S}} <: AbstractVector{Tuple{S,Int}}
    x::T
    out_dim::Integer
end

"""
    MOInputIsotopicByOutputs(x::AbstractVector, out_dim::Integer)

`MOInputIsotopicByOutputs(x, out_dim)` has length `length(x) * out_dim`.

```jldoctest
julia> x = [1, 2, 3];

julia> KernelFunctions.MOInputIsotopicByOutputs(x, 2)
6-element KernelFunctions.MOInputIsotopicByOutputs{Int64, Vector{Int64}}:
 (1, 1)
 (2, 1)
 (3, 1)
 (1, 2)
 (2, 2)
 (3, 2)
```

Accommodates modelling multi-dimensional output data where all outputs are always observed.

As shown above, an `MOInputIsotopicByOutputs` represents a vector of tuples.
The first `length(x)` elements represent the inputs for the first output, the second
`length(x)` elements represent the inputs for the second output, etc.
"""
struct MOInputIsotopicByOutputs{S,T<:AbstractVector{S}} <: AbstractVector{Tuple{S,Int}}
    x::T
    out_dim::Integer
end

const IsotopicMOInputs = Union{MOInputIsotopicByFeatures,MOInputIsotopicByOutputs}

function Base.getindex(inp::MOInputIsotopicByOutputs, ind::Integer)
    @boundscheck checkbounds(inp, ind)
    output_index, feature_index = fldmod1(ind, length(inp.x))
    feature = @inbounds inp.x[feature_index]
    return feature, output_index
end

function Base.getindex(inp::MOInputIsotopicByFeatures, ind::Integer)
    @boundscheck checkbounds(inp, ind)
    feature_index, output_index = fldmod1(ind, inp.out_dim)
    feature = @inbounds inp.x[feature_index]
    return feature, output_index
end

Base.size(inp::IsotopicMOInputs) = (inp.out_dim * length(inp.x),)

"""
    MOInput(x::AbstractVector, out_dim::Integer)

A data type to accommodate modelling multi-dimensional output data.
`MOInput(x, out_dim)` has length `length(x) * out_dim`.

```jldoctest
julia> x = [1, 2, 3];

julia> MOInput(x, 2)
6-element KernelFunctions.MOInputIsotopicByOutputs{Int64, Vector{Int64}}:
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
See [Inputs for Multiple Outputs](@ref) in the docs for more info.

`MOInput` will be deprecated in version 0.11 in favour of `MOInputIsotopicByOutputs`,
and removed in version 0.12.
"""
const MOInput = MOInputIsotopicByOutputs

"""
    prepare_isotopic_multi_output_data(x::AbstractVector, y::ColVecs)

Utility functionality to convert a collection of `N = length(x)` inputs `x`, and a
vector-of-vectors `y` (efficiently represented by a `ColVecs`) into a format suitable for
use with multi-output kernels.

`y[n]` is the vector-valued output corresponding to the input `x[n]`.
Consequently, it is necessary that `length(x) == length(y)`.

For example:
```julia
julia> N, P = 10, 5;

julia> x = randn(N);

julia> y = ColVecs(randn(P, N));

julia> x_canon, y_canon = prepare_isotopic_multi_output_data(x, y);

julia> x_canon isa KernelFunctions.MOInputIsotopicByFeatures
true

julia> length(x_canon) == N * P
true

julia> y_canon isa AbstractVector{<:Real}
true

julia> length(y_canon) == length(x_canon)
true
```
"""
function prepare_isotopic_multi_output_data(x::AbstractVector, y::ColVecs)
    length(x) == length(y) || throw(ArgumentError("length(x) not equal to length(y)."))
    return MOInputIsotopicByFeatures(x, size(y.X, 1)), vec(y.X)
end

"""
    prepare_isotopic_multi_output_data(x::AbstractVector, y::RowVecs)

Utility functionality to convert a collection of `N = length(x)` inputs `x` and output
vectors `y` (efficiently represented by a `RowVecs`) into a format suitable for
use with multi-output kernels.

`y[n]` is the vector-valued output corresponding to the input `x[n]`.
Consequently, it is necessary that `length(x) == length(y)`.

For example:
```jldoctest
julia> N, P = 10, 5;

julia> x = randn(N);

julia> y = RowVecs(randn(N, P));

julia> x_canon, y_canon = prepare_isotopic_multi_output_data(x, y);

julia> x_canon isa KernelFunctions.MOInputIsotopicByOutputs
true

julia> length(x_canon) == N * P
true

julia> y_canon isa AbstractVector{<:Real}
true

julia> length(y_canon) == length(x_canon)
true
```
"""
function prepare_isotopic_multi_output_data(x::AbstractVector, y::RowVecs)
    length(x) == length(y) || throw(ArgumentError("length(x) not equal to length(y)."))
    return MOInputIsotopicByOutputs(x, size(y.X, 2)), vec(y.X)
end
