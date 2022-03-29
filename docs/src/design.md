# Design

## [Why AbstractVectors Everywhere?](@id why_abstract_vectors)

To understand the advantages of using `AbstractVector`s everywhere to represent collections of inputs, first consider the following properties that it is desirable for a collection of inputs to satisfy.

#### Unique Ordering

There must be a clearly-defined first, second, etc element of an input collection.
If this were not the case, it would not be possible to determine a unique mapping between a collection of inputs and the output of `kernelmatrix`, as it would not be clear what order the rows and columns of the output should appear in.

Moreover, ordering guarantees that if you permute the collection of inputs, the ordering of the rows and columns of the `kernelmatrix` are correspondingly permuted.

#### Generality

There must be no restriction on the domain of the input.
Collections of `Real`s, vectors, graphs, finite-dimensional domains, or really anything else that you fancy should be straightforwardly representable.
Moreover, whichever input class is chosen should not prevent optimal performance from being obtained.

#### Unambiguously-Defined Length

Knowing the length of a collection of inputs is important.
For example, a well-defined length guarantees that the size of the output of `kernelmatrix`,
and related functions, are predictable.
It also makes it possible to perform internal error-checking that ensures that e.g. there
are the same number of inputs in two collections of inputs.



### AbstractMatrices Do Not Cut It

Notably, while `AbstractMatrix` objects are often used to represent collections of vector-valued
inputs, they do _not_ immediately satisfy these properties as it is unclear whether a matrix
of size `P x Q` represents a collection of `P` `Q`-dimensional inputs (each row is an
input), or `Q` `P`-dimensional inputs (each column is an input).

Moreover, they occassionally add some aesthetic inconvenience.
For example, a collection of `Real`-valued inputs, which might be straightforwardly
represented as an `AbstractVector{<:Real}`, must be reshaped into a matrix.

There are two commonly used ways to partly resolve these shortcomings:

#### Resolution 1: Specify a Convention

One way that these shortcomings can be partly resolved is by specifying a convention that
everyone adheres to regarding the interpretation of rows vs columns.
However, opinions about the choice of convention are often surprisingly strongly held, and
users regularly have to remind themselves _which_ convention has been chosen.
While this resolves the ordering problem, and in principle defines the "length" of a
collection of inputs, `AbstractMatrix`s already have a `length` defined in Julia, which
would generally disagree with our internal notion of `length`.
This isn't a show-stopper, but it isn't an especially clean situation.

There is also the opportunity for some kinds of silent bugs.
For example, if an input matrix happens to be square because the number of input dimensions
is the same as the number of inputs, it would be hard to know whether the correct
`kernelmatrix` has been computed.
This kind of bug seems unlikely, but it exists regardless.

Finally, suppose that your inputs are some type `T` that is not simply a vector of real
numbers, say a graph.
In this situation, how should a collection of inputs be represented?
A `N x 1` or `1 x N` matrix is the only obvious candidate, but the additional singular
dimension seems somewhat redundant.

#### Resolution 2: Always Specify An `obsdim` Argument

Another way to partly resolve these problems is to not commit to a convention, and instead
to propagate some additional information through the codebase that specifies how the input
data is to be interpretted.
For example, a kernel `k` that represents the sum of two other kernels might implement
`kernelmatrix` as follows:
```julia
function kernelmatrix(k::KernelSum, x::AbstractMatrix; obsdim=1)
    return kernelmatrix(k.kernels[1], x; obsdim=obsdim) +
        kernelmatrix(k.kernels[2], x; obsdim=obsdim)
end
```
While this prevents this package from having to pre-specify a convention, it doesn't resolve
the `length` issue, or the issue of representing collections of inputs which aren't
immediately represented as vectors.
Moreover, it complicates the internals; in contrast, consider what this function looks like
with an `AbstractVector`:
```julia
function kernelmatrix(k::KernelSum, x::AbstractVector)
    return kernelmatrix(k.kernels[1], x) + kernelmatrix(k.kernels[2], x)
end
```
This code is clearer (less visual noise), and has removed a possible bug -- if the
implementer of `kernelmatrix` forgets to pass the `obsdim` kwarg into each subsequent
`kernelmatrix` call, it's possible to get the wrong answer.

This being said, we do support matrix-valued inputs -- see
[Why We Have Support for Both](@ref).


### AbstractVectors 

Requiring all collections of inputs to be `AbstractVector`s resolves all of these problems,
and ensures that the data is self-describing to the extent that KernelFunctions.jl requires.

Firstly, the question of how to interpret the columns and rows of a matrix of inputs is
resolved.
Users _must_ wrap matrices which represent collections of inputs in either a `ColVecs` or
`RowVecs`, both of which have clearly defined semantics which are hard to confuse.

By design, there is also no discrepancy between the number of inputs in the collection, and
the `length` function -- the `length` of a `ColVecs`, `RowVecs`, or `Vector{<:Real}` is
equal to the number of inputs.

There is no loss of performance.

A collection of `N` `Real`-valued inputs can be represented by an
`AbstractVector{<:Real}` of `length` `N`, rather than needing to use an
`AbstractMatrix{<:Real}` of size either `N x 1` or `1 x N`.
The same can be said for any other input type `T`, and new subtypes of `AbstractVector` can
be added if particularly efficient ways exist to store collections of inputs of type `T`.
A good example of this in practice is using `Tuple{S, Int}`, for some input type `S`, as the
[Inputs for Multiple Outputs](@ref).

This approach can also lead to clearer user code.
A user need only wrap their inputs in a `ColVecs` or `RowVecs` once in their code, and this
specification is automatically re-used _everywhere_ in their code.
In this sense, it is straightforward to write code in such a way that there is one unique
source of "truth" about the way in which a particular data set should be interpreted.
Conversely, the `obsdim` resolution requires that the `obsdim` keyword argument is passed
around with the data _every_ _single_ _time_ that you use it.

The benefits of the `AbstractVector` approach are likely most strongly felt when writing a substantial amount of code on top of KernelFunctions.jl -- in the same way that using
`AbstractVector`s inside KernelFunctions.jl removes the need for large amounts of keyword
argument propagation, the same will be true of other code.




### Why We Have Support for Both

In short: many people like matrices, and are familiar with `obsdim`-style keyword
arguments.

All internals are implemented using `AbstractVector`s though, and the `obsdim` interface
is just a thin layer of utility functionality which sits on top of this. To avoid
confusion and silent errors, we do not favour a specific convention (rows or columns)
but instead it is necessary to specify the `obsdim` keyword argument explicitly.

## [Kernels for Multiple-Outputs](@id inputs_for_multiple_outputs)

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
