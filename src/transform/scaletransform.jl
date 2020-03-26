"""
Scale Transform
```
    l = 2.0
    tr = ScaleTransform(l)
```
Multiply every element of the input by `l`
"""
struct ScaleTransform{T<:Real} <: Transform
    s::Vector{T}
end

function ScaleTransform(s::T=1.0) where {T<:Real}
    @check_args(ScaleTransform, s, s > zero(T), "s > 0")
    ScaleTransform{T}([s])
end

set!(t::ScaleTransform,ρ::Real) = t.s .= [ρ]
dim(str::ScaleTransform) = 1

apply(t::ScaleTransform,x::AbstractVecOrMat;obsdim::Int=defaultobs) = first(t.s) * x

Base.isequal(t::ScaleTransform,t2::ScaleTransform) = isequal(first(t.s),first(t2.s))

Base.show(io::IO,t::ScaleTransform) = print(io,"Scale Transform, s = $(first(t.s))")
