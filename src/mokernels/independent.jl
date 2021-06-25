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
    if px == py
        return κ.kernel(x, y)
    else
        return 0.0
    end
end

function kernelmatrix(k::IndependentMOKernel, x::IsotopicByOutputs, y::IsotopicByOutputs)
    @assert x.out_dim == y.out_dim
    temp = k.kernel.(x.x, permutedims(y.x))
    return cat((temp for _ in 1:(y.out_dim))...; dims=(1, 2))
end

function Base.show(io::IO, k::IndependentMOKernel)
    return print(io, string("Independent Multi-Output Kernel\n\t", string(k.kernel)))
end
