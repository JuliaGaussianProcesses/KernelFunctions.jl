"""
    IndependentMOKernel(k::Kernel)

Kernel for multiple independent outputs with kernel `k` each.

# Definition

For inputs ``x, x'`` and output dimensions ``p_x, p_{x'}'``, the kernel ``\\widetilde{k}``
for independent outputs with kernel ``k`` each is defined as
```math
\\widetilde{k}\\big((x, p_x), (x', p_{x'})\\big) = \\begin{cases}
    k(x, x') & \\text{if } p_x = p_{x'}, \\\\
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

# for specialized kernelmatrix implementation see intrinsiccoregion.jl

if VERSION >= v"1.6"
    function _kernelmatrix_kron_helper!(K, ::MOInputIsotopicByFeatures, Kfeatures, B)
        return kron!(K, Kfeatures, B)
    end

    function _kernelmatrix_kron_helper!(K, ::MOInputIsotopicByOutputs, Kfeatures, B)
        return kron!(K, B, Kfeatures)
    end

    function kernelmatrix!(
        K::AbstractMatrix, k::IndependentMOKernel, x::MOI, y::MOI
    ) where {MOI<:IsotopicMOInputsUnion}
        @assert x.out_dim == y.out_dim
        Ktmp = kernelmatrix(k.kernel, x.x, y.x)
        mtype = eltype(Ktmp)
        return _kernelmatrix_kron_helper!(
            K, x, Ktmp, Matrix{mtype}(I, x.out_dim, x.out_dim)
        )
    end
end

function Base.show(io::IO, k::IndependentMOKernel)
    return print(io, string("Independent Multi-Output Kernel\n\t", string(k.kernel)))
end
