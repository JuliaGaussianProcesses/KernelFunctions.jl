"""
`KernelSum(kernels::Array{Kernel};weights::Array{Real}=ones(length(kernels)))`
Create a positive weighted sum of kernels.
One can also use the operator `+`
```
k1 = SqExponentialKernel()
k2 = LinearKernel()
k = KernelSum([k1,k2])
kernelmatrix(k,X) == kernelmatrix(k1,X).+kernelmatrix(k2,X)
kernelmatrix(k,X) == kernelmatrix(k1+k2,X)
kweighted = 0.5*k1 + 2.0*k2
```
"""
struct KernelSum{T,Tr} <: Kernel{T,Tr}
    kernels::Vector{Kernel}
    weights::Vector{Real}
    function KernelSum{T,Tr}(kernels::AbstractVector{<:Kernel},weights::AbstractVector{<:Real}) where {T,Tr}
        new{T,Tr}(kernels,weights)
    end
end

function KernelSum(kernels::AbstractVector{<:Kernel}; weights::AbstractVector{<:Real}=ones(Float64,length(kernels)))
    @assert length(kernels)==length(weights) "Weights and kernel vector should be of the same length"
    @assert all(weights.>=0) "All weights should be positive"
    KernelSum{eltype(kernels),Transform}(kernels,weights)
end

params(k::KernelSum) = (k.weights,params.(k.kernels))
opt_params(k::KernelSum) = (k.weights,opt_params.(k.kernels))

Base.:+(k1::Kernel,k2::Kernel) = KernelSum([k1,k2],weights=[1.0,1.0])
Base.:+(k1::KernelSum,k2::KernelSum) = KernelSum(vcat(k1.kernels,k2.kernels),weights=vcat(k1.weights,k2.weights))
Base.:+(k::Kernel,ks::KernelSum) = KernelSum(vcat(k,ks.kernels),weights=vcat(1.0,ks.weights))
Base.:+(ks::KernelSum,k::Kernel) = KernelSum(vcat(ks.kernels,k),weights=vcat(ks.weights,1.0))
Base.:*(w::Real,k::Kernel) = KernelSum([k],weights=[w]) #TODO add tests
Base.:*(w::Real,k::KernelSum) = KernelSum(k.kernels,weights=w*k.weights) #TODO add tests


Base.length(k::KernelSum) = length(k.kernels)
metric(k::KernelSum) = metric.(k.kernels)
transform(k::KernelSum) = transform.(k.kernels)
transform(k::KernelSum,x::AbstractVecOrMat) = transform.(k.kernels,[x])
transform(k::KernelSum,x::AbstractVecOrMat,obsdim::Int) = transform.(k.kernels,[x],obsdim)

function kernelmatrix(
    κ::KernelSum,
    X::AbstractMatrix;
    obsdim::Int=defaultobs)
    sum(κ.weights[i]*kernelmatrix(κ.kernels[i],X,obsdim=obsdim) for i in 1:length(κ))
end

function kernelmatrix(
    κ::KernelSum,
    X::AbstractMatrix,
    Y::AbstractMatrix;
    obsdim::Int=defaultobs)
    sum(κ.weights[i]*_kernelmatrix(κ.kernels[i],X,Y,obsdim) for i in 1:length(κ))
end

function kerneldiagmatrix(
    κ::KernelSum,
    X::AbstractMatrix;
    obsdim::Int=defaultobs)
    sum(κ.weights[i]*kerneldiagmatrix(κ.kernels[i],X,obsdim=obsdim) for i in 1:length(κ))
end
