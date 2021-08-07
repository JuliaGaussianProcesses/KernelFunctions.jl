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

# kernel function should be symmetric
# would really like (κ::IndependentMOKernel)((x, px)::Tuple{T,Int}, (y, py)::Tuple{T,Int}) where T, but seems to cause autodiff problems
function (κ::IndependentMOKernel)((x, px)::Tuple{Any,Int}, (y, py)::Tuple{Any,Int})
    if px == py
        return κ.kernel(x, y)
    else
        retType = Base.return_types(κ.kernel, (typeof(x), typeof(y)))[1]
        return zero(retType)
    end
end

# this function never gets called it seems
function kernelmatrix(k::IndependentMOKernel, x::MOInput, y::MOInput)
    @assert x.out_dim == y.out_dim
    temp = k.kernel.(x.x, permutedims(y.x))
    return cat((temp for _ in 1:(y.out_dim))...; dims=(1, 2))
end

export kernelmatrix2
function kernelmatrix2(k::IndependentMOKernel, x::MOInputIsotopicByFeatures, y::MOInputIsotopicByFeatures)
    @assert x.out_dim == y.out_dim
    Ktmp = kernelmatrix(k.kernel, x.x, y.x)
    mtype = eltype(Ktmp)
    kron(Ktmp, Matrix{mtype}(I, x.out_dim, x.out_dim))
end

function kernelmatrix2(k::IndependentMOKernel, x::MOInputIsotopicByOutputs, y::MOInputIsotopicByOutputs)
    @assert x.out_dim == y.out_dim
    Ktmp = kernelmatrix(k.kernel, x.x, y.x)
    mtype = eltype(Ktmp)
    kron(Matrix{mtype}(I, x.out_dim, x.out_dim), Ktmp)
end

function Base.show(io::IO, k::IndependentMOKernel)
    return print(io, string("Independent Multi-Output Kernel\n\t", string(k.kernel)))
end
