# Macro for checking arguments
macro check_args(K, param, cond, desc=string(cond))
    quote
        if !($(esc(cond)))
            throw(
                ArgumentError(
                    string(
                        $(string(K)),
                        ": ",
                        $(string(param)),
                        " = ",
                        $(esc(param)),
                        " does not ",
                        "satisfy the constraint ",
                        $(string(desc)),
                        ".",
                    ),
                ),
            )
        end
    end
end

function deprecated_obsdim(obsdim::Union{Int,Nothing})
    _obsdim = if obsdim === nothing
        Base.depwarn(
            "implicit `obsdim=2` argument is deprecated and now has to be passed " *
            "explicitly to specify that each column corresponds to one observation",
            :vec_of_vecs,
        )
        2
    else
        obsdim
    end
    return _obsdim
end

function vec_of_vecs(X::AbstractMatrix; obsdim::Union{Int,Nothing}=nothing)
    _obsdim = deprecated_obsdim(obsdim)
    if _obsdim == 1
        return RowVecs(X)
    elseif _obsdim == 2
        return ColVecs(X)
    else
        throw(ArgumentError("`obsdim` keyword argument should be 1 or 2"))
    end
end

"""
    ColVecs(X::AbstractMatrix)

A lightweight wrapper for an `AbstractMatrix` which interprets it as a vector-of-vectors, in
which each _column_ of `X` represents a single vector.

That is, by writing `x = ColVecs(X)`, you are saying "`x` is a vector-of-vectors, each of
which has length `size(X, 1)`. The total number of vectors is `size(X, 2)`."

Phrased differently, `ColVecs(X)` says that `X` should be interpreted as a vector
of horizontally-concatenated column-vectors, hence the name `ColVecs`.

```jldoctest
julia> X = randn(2, 5);

julia> x = ColVecs(X);

julia> length(x) == 5
true

julia> X[:, 3] == x[3]
true
```

`ColVecs` is related to [`RowVecs`](@ref) via transposition:
```jldoctest
julia> X = randn(2, 5);

julia> ColVecs(X) == RowVecs(X')
true
```
"""
struct ColVecs{T,TX<:AbstractMatrix{T},S} <: AbstractVector{S}
    X::TX
    function ColVecs(X::TX) where {T,TX<:AbstractMatrix{T}}
        S = typeof(view(X, :, 1))
        return new{T,TX,S}(X)
    end
end

Base.size(D::ColVecs) = (size(D.X, 2),)
Base.getindex(D::ColVecs, i::Int) = view(D.X, :, i)
Base.getindex(D::ColVecs, i::CartesianIndex{1}) = view(D.X, :, i)
Base.getindex(D::ColVecs, i) = ColVecs(view(D.X, :, i))
Base.setindex!(D::ColVecs, v::AbstractVector, i) = setindex!(D.X, v, :, i)

Base.vcat(a::ColVecs, b::ColVecs) = ColVecs(hcat(a.X, b.X))
Base.zero(x::ColVecs) = ColVecs(zero(x.X))
Base.reduce(::typeof(hcat), a::ColVecs) = copy(a.X)
Base.reduce(::typeof(vcat), a::ColVecs) = copy(vec(a.X))

dim(x::ColVecs) = size(x.X, 1)

_to_colvecs(x::AbstractVector{<:Real}) = ColVecs(reshape(x, 1, :))

pairwise(d::PreMetric, x::ColVecs) = Distances_pairwise(d, x.X; dims=2)
pairwise(d::PreMetric, x::ColVecs, y::ColVecs) = Distances_pairwise(d, x.X, y.X; dims=2)
function pairwise(d::PreMetric, x::AbstractVector{<:AbstractVector{<:Real}}, y::ColVecs)
    return Distances_pairwise(d, reduce(hcat, x), y.X; dims=2)
end
function pairwise(d::PreMetric, x::ColVecs, y::AbstractVector{<:AbstractVector{<:Real}})
    return Distances_pairwise(d, x.X, reduce(hcat, y); dims=2)
end
function pairwise!(out::AbstractMatrix, d::PreMetric, x::ColVecs)
    return Distances.pairwise!(out, d, x.X; dims=2)
end
function pairwise!(out::AbstractMatrix, d::PreMetric, x::ColVecs, y::ColVecs)
    return Distances.pairwise!(out, d, x.X, y.X; dims=2)
end

"""
    RowVecs(X::AbstractMatrix)

A lightweight wrapper for an `AbstractMatrix` which interprets it as a vector-of-vectors, in
which each _row_ of `X` represents a single vector.

That is, by writing `x = RowVecs(X)`, you are saying "`x` is a vector-of-vectors, each of
which has length `size(X, 2)`. The total number of vectors is `size(X, 1)`."

Phrased differently, `RowVecs(X)` says that `X` should be interpreted as a vector
of vertically-concatenated row-vectors, hence the name `RowVecs`.

Internally, the data continues to be represented as an `AbstractMatrix`, so using this type
does not introduce any kind of performance penalty.

```jldoctest
julia> X = randn(5, 2);

julia> x = RowVecs(X);

julia> length(x) == 5
true

julia> X[3, :] == x[3]
true
```

`RowVecs` is related to [`ColVecs`](@ref) via transposition:
```jldoctest
julia> X = randn(5, 2);

julia> RowVecs(X) == ColVecs(X')
true
```
"""
struct RowVecs{T,TX<:AbstractMatrix{T},S} <: AbstractVector{S}
    X::TX
    function RowVecs(X::TX) where {T,TX<:AbstractMatrix{T}}
        S = typeof(view(X, 1, :))
        return new{T,TX,S}(X)
    end
end

RowVecs(x::AbstractVector) = RowVecs(reshape(x, :, 1))

Base.size(D::RowVecs) = (size(D.X, 1),)
Base.getindex(D::RowVecs, i::Int) = view(D.X, i, :)
Base.getindex(D::RowVecs, i::CartesianIndex{1}) = view(D.X, i, :)
Base.getindex(D::RowVecs, i) = RowVecs(view(D.X, i, :))
Base.setindex!(D::RowVecs, v::AbstractVector, i) = setindex!(D.X, v, i, :)

Base.vcat(a::RowVecs, b::RowVecs) = RowVecs(vcat(a.X, b.X))
Base.zero(x::RowVecs) = RowVecs(zero(x.X))
Base.reduce(::typeof(hcat), a::RowVecs) = permutedims(a.X)
Base.reduce(::typeof(vcat), a::RowVecs) = vec(permutedims(a.X))

dim(x::RowVecs) = size(x.X, 2)

pairwise(d::PreMetric, x::RowVecs) = Distances_pairwise(d, x.X; dims=1)
pairwise(d::PreMetric, x::RowVecs, y::RowVecs) = Distances_pairwise(d, x.X, y.X; dims=1)
function pairwise(d::PreMetric, x::AbstractVector{<:AbstractVector{<:Real}}, y::RowVecs)
    return Distances_pairwise(d, permutedims(reduce(hcat, x)), y.X; dims=1)
end
function pairwise(d::PreMetric, x::RowVecs, y::AbstractVector{<:AbstractVector{<:Real}})
    return Distances_pairwise(d, x.X, permutedims(reduce(hcat, y)); dims=1)
end
function pairwise!(out::AbstractMatrix, d::PreMetric, x::RowVecs)
    return Distances.pairwise!(out, d, x.X; dims=1)
end
function pairwise!(out::AbstractMatrix, d::PreMetric, x::RowVecs, y::RowVecs)
    return Distances.pairwise!(out, d, x.X, y.X; dims=1)
end

# Resolve ambiguity error for ColVecs vs RowVecs. #346
pairwise(d::PreMetric, x::ColVecs, y::RowVecs) = pairwise(d, x, ColVecs(permutedims(y.X)))
pairwise(d::PreMetric, x::RowVecs, y::ColVecs) = pairwise(d, ColVecs(permutedims(x.X)), y)

dim(x) = 0 # This is the passes-by-default choice. For a proper check, implement `KernelFunctions.dim` for your datatype.
dim(x::AbstractVector) = dim(first(x))
dim(x::AbstractVector{<:AbstractVector{<:Real}}) = length(first(x))
dim(x::AbstractVector{<:Real}) = 1

function validate_inputs(x, y)
    if dim(x) != dim(y) # Passes by default if `dim` is not defined
        throw(
            DimensionMismatch(
                "dimensionality of x ($(dim(x))) is not equal to that of y ($(dim(y)))"
            ),
        )
    end
    return nothing
end

function validate_inplace_dims(K::AbstractMatrix, x::AbstractVector, y::AbstractVector)
    validate_inputs(x, y)
    if size(K) != (length(x), length(y))
        throw(
            DimensionMismatch(
                "Size of the target matrix K ($(size(K))) not consistent with lengths of " *
                "inputs x ($(length(x))) and y ($(length(y)))",
            ),
        )
    end
end

function validate_inplace_dims(K::AbstractVector, x::AbstractVector, y::AbstractVector)
    validate_inputs(x, y)
    n = length(x)
    if length(y) != n
        throw(
            DimensionMismatch(
                "Length of input x ($n) not consistent with length of input y " *
                "($(length(y))",
            ),
        )
    end
    if length(K) != n
        throw(
            DimensionMismatch(
                "Length of target vector K ($(length(K))) not consistent with length of " *
                "inputs ($n)",
            ),
        )
    end
end

function validate_inplace_dims(K::AbstractVecOrMat, x::AbstractVector)
    return validate_inplace_dims(K, x, x)
end
