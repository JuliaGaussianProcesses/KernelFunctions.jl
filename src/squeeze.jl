base_kernel(k::Kernel) = eval(nameof(typeof(k)))

base_transform(k::Kernel) = base_transform(k.transform)
base_transform(t::Transform) = eval(nameof(typeof(t)))
tail(v::AbstractVector) = view(v,2:length(v))
duplicate(k::Kernel,θ::AbstractVector) = base_kernel(k)(duplicate(transform(k),first(θ)),tail(θ)...)
duplicate(k::Kernel,θ::Tuple) = base_kernel(k)(duplicate(transform(k),first(θ)),Base.tail(θ)...)
Base.one(x::V) where {V<:AbstractArray{T}} where T = V(fill(one(T),size(x)))
duplicate(t::Transform,θ) = base_transform(t)(θ)
duplicate(t::ChainTransform,θ) = ChainTransform(duplicate.(t.transforms,θ))
duplicate(t::FunctionTransform,θ) = t
duplicate(t::IdentityTransform,θ) = t
duplicate(t::SelectTransform,θ) = t


function duplicate(k::KernelSum,θ)
    KernelSum(duplicate.(k.kernels,θ[2]),weights=first(θ))
end

function duplicate(k::KernelProduct,θ)
    KernelProduct(duplicate.(k.kernels,θ))
end

dim(k::Kernel) = length(params(k))
