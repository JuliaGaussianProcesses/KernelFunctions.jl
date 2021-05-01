# API Library

---
```@contents
Pages = ["api.md"]
```

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

All collections of inputs in KernelFunctions.jl are represented as `AbstractVector`s.
To understand this choice, please see the [design notes](@ref why_abstract_vectors).
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

There are two equally-valid perspectives on multi-output kernels: they can either be treated
as matrix-valued kernels, or standard kernels on an extended input domain.
Each of these perspectives are convenient in different circumstances, but the latter
greatly simplifies the incorporation of multi-output kernels in KernelFunctions.

More concretely, let `k_mat` be a matrix-valued kernel, mapping pairs of inputs of type `T` to matrices of size `P x P` to describe the covariance between `P` outputs.
Given inputs `x` and `y` of type `T`, and integers `p` and `q`, we can always find an
equivalent standard kernel `k` mapping from pairs of inputs of type `Tuple{T, Int}` to the
`Real`s as follows:
```julia
k((x, p), (y, q)) = k_mat(x, y)[p, q]
```
This ability to treat multi-output kernels as single-output kernels is very helpful, as it
means that there is no need to introduce additional concepts into the API of
KernelFunctions.jl, just additional kernels!
This in turn simplifies downstream code as they don't need to "know" about the existence of
multi-output kernels in addition to standard kernels. For example, GP libraries built on
top of KernelFunctions.jl just need to know about `Kernel`s, and they get multi-output
kernels, and hence multi-output GPs, for free.

Where there is the need to specialise _implementations_ for multi-output kernels, this is
done in an encapsulated manner -- parts of KernelFunctions that have nothing to do with
multi-output kernels know _nothing_ about the existence of multi-output kernels.

Multi-output kernels in KernelFunctions.jl do support collection of inputs of
type `AbstractVector{Tuple{T, Int}}`, we provide the `MOInput` type to simplify constructing inputs
for situations in which all outputs are observed all of the time:
```@docs
MOInput
```
As with [`ColVecs`](@ref) and [`RowVecs`](@ref) for vector-valued input spaces, this
type enables specialised implementations of e.g. [`kernelmatrix`](@ref) for
[`MOInput`](@ref)s.

To find out more about the background, read this [review of kernels for vector-valued functions](https://arxiv.org/pdf/1106.6251.pdf).

## Utilities

KernelFunctions also provides miscellaneous utility functions.
```@docs
kernelpdmat
nystrom
NystromFact
```
