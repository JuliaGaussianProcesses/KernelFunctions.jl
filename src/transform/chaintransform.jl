"""
Chain a series of transform, here `t1` will be called first
```
    t1 = ScaleTransform()
    t2 = LowRankTransform(rand(3,4))
    ct = ChainTransform([t1,t2]) #t1 will be called first
    ct == t2∘t1
```
"""
struct ChainTransform <: Transform
    transforms::Vector{Transform}
end

Base.length(t::ChainTransform) = length(t.transforms) #TODO Add test

function ChainTransform(v::AbstractVector{<:Transform})
    ChainTransform(v)
end

## Constructor to create a chain transform with an array of parameters
function ChainTransform(v::AbstractVector{<:Type{<:Transform}},θ::AbstractVector)
    @assert length(v) == length(θ)
    ChainTransform(v.(θ))
end

function apply(t::ChainTransform,X::T;obsdim::Int=defaultobs) where {T}
    Xtr = copy(X)
    for tr in t.transforms
        Xtr = apply(tr, Xtr, obsdim = obsdim)
    end
    return Xtr
end

set!(t::ChainTransform,θ) = set!.(t.transforms,θ)
duplicate(t::ChainTransform,θ) = ChainTransform(duplicate.(t.transforms,θ))

Base.:∘(t₁::Transform,t₂::Transform) = ChainTransform([t₂,t₁])
Base.:∘(t::Transform,tc::ChainTransform) = ChainTransform(vcat(tc.transforms,t)) #TODO add test
Base.:∘(tc::ChainTransform,t::Transform) = ChainTransform(vcat(t,tc.transforms))
