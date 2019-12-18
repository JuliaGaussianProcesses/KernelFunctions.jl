# Macro for checking arguments
macro check_args(K, param, cond, desc=string(cond))
    quote
        if !($(esc(cond)))
            throw(ArgumentError(string(
                $(string(K)), ": ", $(string(param)), " = ", $(esc(param)), " does not ",
                "satisfy the constraint ", $(string(desc)), ".")))
        end
    end
end


# Take highest Float among possibilities
# function promote_float(Tₖ::DataType...)
#     if length(Tₖ) == 0
#         return Float64
#     end
#     T = promote_type(Tₖ...)
#     return T <: Real ? T : Float64
# end

check_dims(K,X,Y,featdim,obsdim) = check_dims(X,Y,featdim,obsdim) && (size(K) == (size(X,obsdim),size(Y,obsdim)))

check_dims(X,Y,featdim,obsdim) = size(X,featdim) == size(Y,featdim)


feature_dim(obsdim::Int) = obsdim == 1 ? 2 : 1

base_kernel(k::Kernel) = eval(nameof(typeof(k)))

base_transform(k::Kernel) = base_transform(k.transform)
base_transform(t::Transform) = eval(nameof(typeof(t)))
_tail(v::AbstractVector) = view(v,2:length(v))

"""
```julia
    duplicate(k::Kernel,θ)
    duplicate(t::Transform,θ)
```
Recreate a kernel (transform) with the same structure as `k` (`t`) with the appropriate new parameters `θ`.
`theta` should have the same structure then the one given by `params(k)` (`params(t)`).
"""
duplicate

duplicate(k::Kernel,θ::AbstractVector) = base_kernel(k)(duplicate(transform(k),first(θ)),_tail(θ)...)
duplicate(k::Kernel,θ::Tuple) = base_kernel(k)(duplicate(transform(k),first(θ)),Base.tail(θ)...)
duplicate(t::Transform,θ) = base_transform(t)(θ)

dim(k::Kernel) = length(params(k))

"""
```julia
    params(k::Kernel)
    params(t::Transform)
```
For a kernel return a tuple with parameters of the transform followed by the specific parameters of the kernel
For a transform return its parameters, for a `ChainTransform` return a vector of `params(t)`.
"""
params(k::Kernel) = (params(k.transform),)
