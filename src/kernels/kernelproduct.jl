"""
    KernelProduct(kernels::Array{Kernel})
Create a multiplication of kernels.
One can also use the operator `*`
```
    kernelmatrix(SqExponentialKernel()*LinearKernel(),X) == kernelmatrix(SqExponentialKernel(),X).*kernelmatrix(LinearKernel(),X)
```
"""
struct KernelProduct{T,Tr} <: Kernel{T,Tr}
    kernels::Vector{Kernel}
end

function KernelProduct(kernels::AbstractVector{<:Kernel})
    KernelProduct{eltype(kernels),Transform}(kernels)
end

Base.:*(k1::Kernel,k2::Kernel) = KernelProduct([k1,k2])
Base.:*(k1::KernelProduct,k2::KernelProduct) = KernelProduct(vcat(k1.kernels,k2.kernels)) #TODO Add test
Base.:*(k::Kernel,kp::KernelProduct) = KernelProduct(vcat(k,kp.kernels))
Base.:*(kp::KernelProduct,k::Kernel) = KernelProduct(vcat(kp.kernels,k))

Base.length(k::KernelProduct) = length(k.kernels)
metric(k::KernelProduct) = getmetric.(k.kernels) #TODO Add test
transform(k::KernelProduct) = transform.(k.kernels) #TODO Add test
transform(k::KernelProduct,x::AbstractVecOrMat) = transform.(k.kernels,[x]) #TODO Add test
transform(k::KernelProduct,x::AbstractVecOrMat,obsdim::Int) = transform.(k.kernels,[x],obsdim) #TODO Add test

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
    obsdim::Int=defaultobs) #TODO Add test
    reduce(hadamard,kerneldiagmatrix(κ.kernels[i],X,obsdim=obsdim) for i in 1:length(κ))
end
