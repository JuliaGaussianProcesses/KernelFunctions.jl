struct IntrinsicCoregionalizationMOKernel{Tu<:Kernel, TA<:AbstractMatrix, Tv<:LinearAlgebra.Diagonal, TB<:AbstractMatrix, Trank<:Integer, Tnoutputs<:Integer} <: MOKernel
    u::Tu
    A::TA
    v::Tv
    B::TB
    rank::Trank
    noutputs::Tnoutputs

    function IntrinsicCoregionalizationMOKernel(u, noutputs, rank)
        noutputs >= rank || error("`noutputs` should be greater or equal than `rank`")

        A = rand(noutputs, rank)
        v = Diagonal(rand(noutputs))
        B = A * transpose(A) + v

        return new{typeof(u), typeof(A),  typeof(v), typeof(B), typeof(rank), typeof(noutputs)}(u,A,v,B,rank,noutputs)
    end

    function IntrinsicCoregionalizationMOKernel(u, A, v::AbstractVector)
        rank = size(A,2)
        noutputs = size(A,1)
        v = Diagonal(v)
        B = A * transpose(A) + v
        return new{typeof(u), typeof(A),  typeof(v), typeof(B), typeof(rank), typeof(noutputs)}(u,A,v,B,rank,noutputs)
    end
end



function (κ::IntrinsicCoregionalizationMOKernel)((x, px)::Tuple{Any,Int}, (y, py)::Tuple{Any,Int})
    return κ.B[px,py] * κ.u(x,y)
end


# function kernelmatrix(κ::IntrinsicCoregionalizationMOKernel,
#                       x::AbstractVector{Tuple{Any,Int}},
#                       y::AbstractVector{Tuple{Any,Int}})

#     return κ.(x, permutedims(y))
# end


function Base.show(io::IO, κ::IntrinsicCoregionalizationMOKernel)
    return print(io, "Intrinsic Coregionalization Multi-Output Kernel")
end
