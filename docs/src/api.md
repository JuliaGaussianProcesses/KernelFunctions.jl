# API Library

```@meta
CurrentModule = KernelFunctions
```

## Functions

The KernelFunctions API comprises the following four functions.
```@docs
kernelmatrix
kernelmatrix!
kernelmatrix_diag
kernelmatrix_diag!
```

## Input Types

The above API operates on collections of inputs.
All collections of inputs in KernelFunctions.jl are represented as `AbstractVector`s.
To understand this choice, please see the [design notes on collections of inputs](@ref why_abstract_vectors).
The length of any such `AbstractVector` is equal to the number of inputs in the collection.
For example, this means that
```julia
size(kernelmatrix(k, x)) == (length(x), length(x))
```
is always true, for some `Kernel` `k`, and `AbstractVector` `x`.

### Univariate Inputs

If each input to your kernel is `Real`-valued, then any `AbstractVector{<:Real}` is a valid
representation for a collection of inputs.
More generally, it's completely fine to represent a collection of inputs of type `T` as, for
example, a `Vector{T}`.
However, this may not be the most efficient way to represent collection of inputs.
See [Vector-Valued Inputs](@ref) for an example.


### Vector-Valued Inputs

We recommend that collections of vector-valued inputs are stored in an
`AbstractMatrix{<:Real}` when possible, and wrapped inside a `ColVecs` or `RowVecs` to make
their interpretation clear:
```@docs
ColVecs
RowVecs
```
These types are specialised upon to ensure good performance e.g. when computing Euclidean distances between pairs of elements.
The benefit of using this representation, rather than using a `Vector{Vector{<:Real}}`, is that
optimised matrix-matrix multiplication functionality can be utilised when computing
pairwise distances between inputs, which are needed for `kernelmatrix` computation.

### Inputs for Multiple Outputs

KernelFunctions.jl views multi-output GPs as GPs on an extended input domain.
For an explanation of this design choice, see [the design notes on multi-output GPs](@ref inputs_for_multiple_outputs).

An input to a multi-output `Kernel` should be a `Tuple{T, Int}`, whose first element specifies a location in the domain of the multi-output GP, and whose second element specifies which output the inputs corresponds to.
The type of collections of inputs for multi-output GPs is therefore `AbstractVector{<:Tuple{T, Int}}`.

KernelFunctions.jl provides the following helper function for situations in which all outputs are observed all of the time:
```@docs
prepare_isotopic_multi_output_data(x::AbstractVector, y::ColVecs)
prepare_isotopic_multi_output_data(x::AbstractVector, y::RowVecs)
```

The input types that it constructs can also be constructed manually:
```@docs
MOInput
```
As with [`ColVecs`](@ref) and [`RowVecs`](@ref) for vector-valued input spaces, this
type enables specialised implementations of e.g. [`kernelmatrix`](@ref) for
[`MOInput`](@ref)s in some situations.

To find out more about the background, read this [review of kernels for vector-valued functions](https://arxiv.org/pdf/1106.6251.pdf).

## Generic Utilities

KernelFunctions also provides miscellaneous utility functions.
```@docs
kernelpdmat
nystrom
NystromFact
```
