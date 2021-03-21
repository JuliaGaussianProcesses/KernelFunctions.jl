struct CoregionMOKernel{K<:Kernel,T<:AbstractMatrix} <: MOKernel
    kernel::K
    B::T

    function CoregionMOKernel{K,T}(kernel::K, B::T) where {K,T}
        @check_args(CoregionMOKernel, B, (eigmin(B) >= 0), "B is Positive semi-definite")
        return new{K,T}(kernel, B)
    end
end

function CoregionMOKernel(kernel::Kernel, B::AbstractMatrix)
    return CoregionMOKernel{typeof(kernel),typeof(B)}(kernel, B)
end

function (k::CoregionMOKernel)((x, px)::Tuple{Any,Int}, (y, py)::Tuple{Any,Int})
    return k.B[px,py] * k.kernel(x,y)
end

function Base.show(io::IO, k::CoregionMOKernel)
    return print(io, "Coregion Multi-Output Kernel")
end
