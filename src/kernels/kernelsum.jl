"""
    KernelSum(kernels::Array{Kernel}; weights::Array{Real}=ones(length(kernels)))

Create a positive weighted sum of kernels. All weights should be positive.
One can also use the operator `+`
```
    k1 = SqExponentialKernel()
    k2 = LinearKernel()
    k = KernelSum([k1, k2]) == k1 + k2
    kernelmatrix(k, X) == kernelmatrix(k1, X) .+ kernelmatrix(k2, X)
    kernelmatrix(k, X) == kernelmatrix(k1 + k2, X)
    kweighted = 0.5* k1 + 2.0*k2 == KernelSum([k1, k2], weights = [0.5, 2.0])
```
"""
struct KernelSum <: Kernel
    kernels::Vector{Kernel}
    weights::Vector{Real}
end

function KernelSum(
    kernels::AbstractVector{<:Kernel};
    weights::AbstractVector{<:Real} = ones(Float64, length(kernels)),
)
    @assert length(kernels) == length(weights) "Weights and kernel vector should be of the same length"
    @assert all(weights .>= 0) "All weights should be positive"
    return KernelSum(kernels, weights)
end

Base.:+(k1::Kernel, k2::Kernel) = KernelSum([k1, k2], weights = [1.0, 1.0])
Base.:+(k1::ScaledKernel, k2::ScaledKernel) = KernelSum([kernel(k1), kernel(k2)], weights = [first(k1.σ²), first(k2.σ²)])
Base.:+(k1::KernelSum, k2::KernelSum) =
    KernelSum(vcat(k1.kernels, k2.kernels), weights = vcat(k1.weights, k2.weights))
Base.:+(k::Kernel, ks::KernelSum) =
    KernelSum(vcat(k, ks.kernels), weights = vcat(1.0, ks.weights))
Base.:+(k::ScaledKernel, ks::KernelSum) =
        KernelSum(vcat(kernel(k), ks.kernels), weights = vcat(first(k.σ²), ks.weights))
Base.:+(k::ScaledKernel, ks::Kernel) =
        KernelSum(vcat(kernel(k), ks), weights = vcat(first(k.σ²), 1.0))
Base.:+(ks::KernelSum, k::Kernel) =
    KernelSum(vcat(ks.kernels, k), weights = vcat(ks.weights, 1.0))
Base.:+(ks::KernelSum, k::ScaledKernel) =
        KernelSum(vcat(ks.kernels, kernel(k)), weights = vcat(ks.weights, first(k.σ²)))
Base.:+(ks::Kernel, k::ScaledKernel) =
        KernelSum(vcat(ks, kernel(k)), weights = vcat(1.0, first(k.σ²)))
Base.:*(w::Real, k::KernelSum) = KernelSum(k.kernels, weights = w * k.weights) #TODO add tests

Base.length(k::KernelSum) = length(k.kernels)

kappa(κ::KernelSum, x, y) = sum(κ.weights[i] * kappa(κ.kernels[i], x, y) for i in 1:length(κ))

function kernelmatrix(κ::KernelSum, X::AbstractMatrix; obsdim::Int = defaultobs)
    sum(κ.weights[i] * kernelmatrix(κ.kernels[i], X, obsdim = obsdim) for i in 1:length(κ))
end

function kernelmatrix(
    κ::KernelSum,
    X::AbstractMatrix,
    Y::AbstractMatrix;
    obsdim::Int = defaultobs,
)
    sum(κ.weights[i] * kernelmatrix(κ.kernels[i], X, Y, obsdim = obsdim) for i in 1:length(κ))
end

function kerneldiagmatrix(
    κ::KernelSum,
    X::AbstractMatrix;
    obsdim::Int = defaultobs,
)
    sum(κ.weights[i] * kerneldiagmatrix(κ.kernels[i], X, obsdim = obsdim) for i in 1:length(κ))
end

function Base.show(io::IO, κ::KernelSum)
    printshifted(io, κ, 0)
end

function printshifted(io::IO,κ::KernelSum, shift::Int)
    print(io,"Sum of $(length(κ)) kernels:")
    for i in 1:length(κ)
        print(io, "\n" * ("\t" ^ (shift + 1)) * "- (w = $(κ.weights[i])) ")
        printshifted(io, κ.kernels[i], shift + 2)
    end
end
