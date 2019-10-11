struct KernelProduct{T,Tr} <: Kernel{T,Tr}
    kernels::Vector{Kernel}
end

function KernelProduct(kernels::AbstractVector{<:Kernel})
    KernelProduct{eltype(kernels),Transform}(kernels)
end

Base.:*(k1::Kernel,k2::Kernel) = KernelProduct([k1,k2])
Base.:*(k::Kernel,kp::KernelProduct) = KernelProduct(vcat(k,kp.kernels))
Base.:*(kp::KernelProduct,k::Kernel) = KernelProduct(vcat(kp.kernels,k))

Base.length(k::KernelProduct) = length(k.kernels)
metric(k::KernelProduct) = getmetric.(k.kernels)
transform(k::KernelProduct) = transform.(k.kernels)
transform(k::KernelProduct,x::AbstractVecOrMat) = transform.(k.kernels,[x])
transform(k::KernelProduct,x::AbstractVecOrMat,obsdim::Int) = transform.(k.kernels,[x],obsdim)

hadamard(x,y) = x.*y

function kernelmatrix(
    κ::KernelProduct,
    X::AbstractMatrix;
    obsdim::Int=defaultobs)
    reduce(hadamard,kernelmatrix(κ.kernels[i],X,obsdim=obsdim) for i in 1:length(κ))
end

function kernelmatrix(
    κ::KernelProduct,
    X::AbstractMatrix,
    Y::AbstractMatrix;
    obsdim::Int=defaultobs)
    reduce(hadamard,_kernelmatrix(κ.kernels[i],X,Y,obsdim) for i in 1:length(κ))
end

function kerneldiagmatrix(
    κ::KernelProduct,
    X::AbstractMatrix;
    obsdim::Int=defaultobs)
    reduce(hadamard,kerneldiagmatrix(κ.kernels[i],X,obsdim=obsdim) for i in 1:length(κ))
end
