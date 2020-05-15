"""
    ChainTransform(ts::AbstractVector{<:Transform})

Chain a series of transform, here `t1` will be called first
```
    t1 = ScaleTransform()
    t2 = LinearTransform(rand(3,4))
    ct = ChainTransform([t1,t2]) #t1 will be called first
    ct == t2 ∘ t1
```
"""
struct ChainTransform{V<:AbstractVector{<:Transform}} <: Transform
    transforms::V
end

Base.length(t::ChainTransform) = length(t.transforms)

# Constructor to create a chain transform with an array of parameters
function ChainTransform(v::AbstractVector{<:Type{<:Transform}},θ::AbstractVector)
    @assert length(v) == length(θ)
    ChainTransform(v.(θ))
end

Base.:∘(t₁::Transform, t₂::Transform) = ChainTransform([t₂, t₁])
Base.:∘(t::Transform, tc::ChainTransform) = ChainTransform(vcat(tc.transforms, t))
Base.:∘(tc::ChainTransform, t::Transform) = ChainTransform(vcat(t, tc.transforms))

(t::ChainTransform)(x) = foldl((x, t) -> t(x), t.transforms; init=x)

function _map(t::ChainTransform, x::AbstractVector)
    return foldl((x, t) -> map(t, x), t.transforms; init=x)
end

set!(t::ChainTransform,θ) = set!.(t.transforms,θ)
duplicate(t::ChainTransform,θ) = ChainTransform(duplicate.(t.transforms,θ))

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
