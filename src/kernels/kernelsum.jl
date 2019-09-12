struct KernelSum{T,Tr} <: Kernel{T,Tr}
    kernels::Vector{Kernel}
    weights::Vector{Real}
end

function Base.:+(k1::Kernel,k2::Kernel)
    KernelSum([k1,k2],[1.0,1.0])
end
