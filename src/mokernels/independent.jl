"""
    IndependentMOKernel(k::Kernel)

Kernel for multiple independent outputs with kernel `k` each.

# Definition

For inputs ``x, x'`` and output dimensions ``p, p'``, the kernel ``\\widetilde{k}``
for independent outputs with kernel ``k`` each is defined as
```math
\\widetilde{k}\\big((x, p), (x', p')\\big) = \\begin{cases}
    k(x, x') & \\text{if } p = p', \\\\
    0 & \\text{otherwise}.
\\end{cases}
```
Mathematically, it is equivalent to a matrix-valued kernel defined as
```math
\\widetilde{K}(x, x') = \\mathrm{diag}\\big(k(x, x'), \\ldots, k(x, x')\\big) \\in \\mathbb{R}^{m \\times m},
```
where ``m`` is the number of outputs.
"""
struct IndependentMOKernel{Tkernel<:Kernel} <: MOKernel
    kernel::Tkernel
end

function (κ::IndependentMOKernel)((x, px)::Tuple{Any,Int}, (y, py)::Tuple{Any,Int})
    return κ.kernel(x, y) * (px == py)
end

_mo_output_covariance(k::IndependentMOKernel, out_dim) = Eye{Bool}(out_dim)

function kernelmatrix(
    k::IndependentMOKernel, x::MOI, y::MOI
) where {MOI<:IsotopicMOInputsUnion}
    x.out_dim == y.out_dim ||
        throw(DimensionMismatch("`x` and `y` must have the same `out_dim`"))
    Kfeatures = kernelmatrix(k.kernel, x.x, y.x)
    Koutputs = _mo_output_covariance(k, x.out_dim)
    return _kernelmatrix_kron_helper(MOI, Kfeatures, Koutputs)
end

if VERSION >= v"1.6"
    function kernelmatrix!(
        K::AbstractMatrix, k::IndependentMOKernel, x::MOI, y::MOI
    ) where {MOI<:IsotopicMOInputsUnion}
        x.out_dim == y.out_dim ||
            throw(DimensionMismatch("`x` and `y` must have the same `out_dim`"))
        Kfeatures = kernelmatrix(k.kernel, x.x, y.x)
        Koutputs = _mo_output_covariance(k, x.out_dim)
        return _kernelmatrix_kron_helper!(K, MOI, Kfeatures, Koutputs)
    end
end

function Base.show(io::IO, k::IndependentMOKernel)
    return print(io, string("Independent Multi-Output Kernel\n\t", string(k.kernel)))
end
