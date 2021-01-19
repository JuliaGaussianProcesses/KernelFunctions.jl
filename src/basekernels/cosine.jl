"""
    CosineKernel()

Cosine kernel.

# Definition

For inputs ``x, x' \\in \\mathbb{R}^d``, the cosine kernel is defined as
```math
k(x, x') = \\cos(\\pi \\|x-x'\\|_2).
```
"""
struct CosineKernel <: SimpleKernel end

kappa(::CosineKernel, d::Real) = cospi(d)

metric(::CosineKernel) = Euclidean()

Base.show(io::IO, ::CosineKernel) = print(io, "Cosine Kernel")
