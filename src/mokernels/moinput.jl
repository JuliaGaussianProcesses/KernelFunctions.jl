"""
    MOInputIsotopicByFeatures(x::AbstractVector, out_dim::Integer)

`MOInputIsotopicByFeatures(x, out_dim)` has length `out_dim * length(x)`.

```jldoctest
julia> x = [1, 2, 3];

julia> KernelFunctions.MOInputIsotopicByFeatures(x, 2)
6-element KernelFunctions.MOInputIsotopicByFeatures{Int64, Vector{Int64}, Int64}:
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
struct MOInputIsotopicByFeatures{S,I,T<:AbstractVector{S},Tout_axis<:AbstractVector{I}} <:
       AbstractVector{Tuple{S,I}}
    x::T
    out_axis::Tout_axis
end

function MOInputIsotopicByFeatures(x::AbstractVector, out_dim::Integer)
    return MOInputIsotopicByFeatures(x, Base.OneTo(out_dim))
end

"""
    MOInputIsotopicByOutputs(x::AbstractVector, out_dim::Integer)

`MOInputIsotopicByOutputs(x, out_dim)` has length `length(x) * out_dim`.

```jldoctest
julia> x = [1, 2, 3];

julia> KernelFunctions.MOInputIsotopicByOutputs(x, 2)
6-element KernelFunctions.MOInputIsotopicByOutputs{Int64, Vector{Int64}, Int64}:
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
struct MOInputIsotopicByOutputs{S,I,T<:AbstractVector{S},Tout_axis<:AbstractVector{I}} <:
       AbstractVector{Tuple{S,I}}
    x::T
    out_axis::Tout_axis
end

function MOInputIsotopicByOutputs(x::AbstractVector, out_dim::Integer)
    return MOInputIsotopicByOutputs(x, Base.OneTo(out_dim))
end

const IsotopicMOInputsUnion = Union{MOInputIsotopicByFeatures,MOInputIsotopicByOutputs}

function Base.getindex(inp::MOInputIsotopicByOutputs, ind::Integer)
    @boundscheck checkbounds(inp, ind)
    output_index, feature_index = fldmod1(ind, length(inp.x))
    feature = @inbounds inp.x[feature_index]
    out_idx = axes(inp.out_axis, 1)[output_index]
    return feature, @inbounds inp.out_axis[out_idx]
end

function Base.getindex(inp::MOInputIsotopicByFeatures, ind::Integer)
    @boundscheck checkbounds(inp, ind)
    feature_index, output_index = fldmod1(ind, length(inp.out_axis))
    feature = @inbounds inp.x[feature_index]
    out_idx = axes(inp.out_axis, 1)[output_index]
    return feature, @inbounds inp.out_axis[out_idx]
end

Base.size(inp::IsotopicMOInputsUnion) = (length(inp.out_axis) * length(inp.x),)

function Base.vcat(x::MOInputIsotopicByFeatures, y::MOInputIsotopicByFeatures)
    x.out_axis == y.out_axis || throw(DimensionMismatch("out_axis mismatch"))
    return MOInputIsotopicByFeatures(vcat(x.x, y.x), x.out_axis)
end

function Base.vcat(x::MOInputIsotopicByOutputs, y::MOInputIsotopicByOutputs)
    x.out_axis == y.out_axis || throw(DimensionMismatch("out_axis mismatch"))
    return MOInputIsotopicByOutputs(vcat(x.x, y.x), x.out_axis)
end

"""
    MOInput(x::AbstractVector, out_dim::Integer)

A data type to accommodate modelling multi-dimensional output data.
`MOInput(x, out_dim)` has length `length(x) * out_dim`.

```jldoctest
julia> x = [1, 2, 3];

julia> MOInput(x, 2)
6-element KernelFunctions.MOInputIsotopicByOutputs{Int64, Vector{Int64}, Int64}:
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

For example, if outputs are initially stored in a `num_outputs × N` matrix:
```julia
julia> x = [1.0, 2.0, 3.0];

julia> Y = [1.1 2.1 3.1; 1.2 2.2 3.2]
2×3 Matrix{Float64}:
 1.1  2.1  3.1
 1.2  2.2  3.2

julia> inputs, outputs = prepare_isotopic_multi_output_data(x, ColVecs(Y));

julia> inputs
6-element KernelFunctions.MOInputIsotopicByFeatures{Float64, Vector{Float64}, Int64}:
 (1.0, 1)
 (1.0, 2)
 (2.0, 1)
 (2.0, 2)
 (3.0, 1)
 (3.0, 2)

julia> outputs
6-element Vector{Float64}:
 1.1
 1.2
 2.1
 2.2
 3.1
 3.2
```

See also [`prepare_heterotopic_multi_output_data`](@ref).
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

For example, if outputs are initial stored in an `N × num_outputs` matrix:
```jldoctest
julia> x = [1.0, 2.0, 3.0];

julia> Y = [1.1 1.2; 2.1 2.2; 3.1 3.2]
3×2 Matrix{Float64}:
 1.1  1.2
 2.1  2.2
 3.1  3.2

julia> inputs, outputs = prepare_isotopic_multi_output_data(x, RowVecs(Y));

julia> inputs
6-element KernelFunctions.MOInputIsotopicByOutputs{Float64, Vector{Float64}, Int64}:
 (1.0, 1)
 (2.0, 1)
 (3.0, 1)
 (1.0, 2)
 (2.0, 2)
 (3.0, 2)

julia> outputs
6-element Vector{Float64}:
 1.1
 2.1
 3.1
 1.2
 2.2
 3.2
```

See also [`prepare_heterotopic_multi_output_data`](@ref).
"""
function prepare_isotopic_multi_output_data(x::AbstractVector, y::RowVecs)
    length(x) == length(y) || throw(ArgumentError("length(x) not equal to length(y)."))
    return MOInputIsotopicByOutputs(x, size(y.X, 2)), vec(y.X)
end

"""
    prepare_heterotopic_multi_output_data(
        x::AbstractVector, y::AbstractVector{<:Real}, output_indices::AbstractVector{Int},
    )

Utility functionality to convert a collection of inputs `x`, observations `y`, and
`output_indices` into a format suitable for use with multi-output kernels.
Handles the situation in which only one (or a subset) of outputs are observed at each
feature.
Ensures that all arguments are compatible with one another, and returns a vector of inputs
and a vector of outputs.

`y[n]` should be the observed value associated with output `output_indices[n]` at feature
`x[n]`.

```jldoctest
julia> x = [1.0, 2.0, 3.0];

julia> y = [-1.0, 0.0, 1.0];

julia> output_indices = [3, 2, 1];

julia> inputs, outputs = prepare_heterotopic_multi_output_data(x, y, output_indices);

julia> inputs
3-element Vector{Tuple{Float64, Int64}}:
 (1.0, 3)
 (2.0, 2)
 (3.0, 1)

julia> outputs
3-element Vector{Float64}:
 -1.0
  0.0
  1.0
```

See also [`prepare_isotopic_multi_output_data`](@ref).
"""
function prepare_heterotopic_multi_output_data(
    x::AbstractVector, y::AbstractVector{<:Real}, output_indices::AbstractVector{Int}
)
    # Ensure validity of arguments.
    if length(x) != length(y)
        throw(ArgumentError("length(x) != length(y)"))
    end
    if length(x) != length(output_indices)
        throw(ArgumentError("length(x) != length(output_indices)"))
    end

    # Construct inputs and outputs for multi-output kernel.
    x_mogp = map(tuple, x, output_indices)
    return x_mogp, y
end
