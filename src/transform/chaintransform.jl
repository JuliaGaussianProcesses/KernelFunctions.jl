"""
    ChainTransform(ts::AbstractVector{<:Transform})

Transformation that applies a chain of transformations `ts` to the input.

The transformation `first(ts)` is applied first.

# Examples

```jldoctest
julia> l = rand(); A = rand(3, 4); t1 = ScaleTransform(l); t2 = LinearTransform(A);

julia> X = rand(4, 10);

julia> map(ChainTransform([t1, t2]), ColVecs(X)) == ColVecs(A * (l .* X))
true

julia> map(t2 ∘ t1, ColVecs(X)) == ColVecs(A * (l .* X))
true
```
"""
struct ChainTransform{V<:AbstractVector{<:Transform}} <: Transform
    transforms::V
end

@functor ChainTransform

Base.length(t::ChainTransform) = length(t.transforms)

# Constructor to create a chain transform with an array of parameters
function ChainTransform(v::AbstractVector{<:Type{<:Transform}}, θ::AbstractVector)
    @assert length(v) == length(θ)
    return ChainTransform(v.(θ))
end

Base.:∘(t₁::Transform, t₂::Transform) = ChainTransform([t₂, t₁])
Base.:∘(t::Transform, tc::ChainTransform) = ChainTransform(vcat(tc.transforms, t))
Base.:∘(tc::ChainTransform, t::Transform) = ChainTransform(vcat(t, tc.transforms))

(t::ChainTransform)(x) = foldl((x, t) -> t(x), t.transforms; init=x)

function _map(t::ChainTransform, x::AbstractVector)
    return foldl((x, t) -> map(t, x), t.transforms; init=x)
end

set!(t::ChainTransform, θ) = set!.(t.transforms, θ)
duplicate(t::ChainTransform, θ) = ChainTransform(duplicate.(t.transforms, θ))

Base.isequal(t::ChainTransform, t2::ChainTransform) = isequal(t.transforms, t2.transforms)

Base.show(io::IO, t::ChainTransform) = printshifted(io, t, 0)

function printshifted(io::IO, t::ChainTransform, shift::Int)
    println(io, "Chain of ", length(t), " transforms:")
    for _ in 1:(shift + 1)
        print(io, "\t")
    end
    print(io, " - ")
    printshifted(io, t.transforms[1], shift + 2)
    for i in 2:length(t)
        print(io, " |> ")
        printshifted(io, t.transforms[i], shift + 2)
    end
end
