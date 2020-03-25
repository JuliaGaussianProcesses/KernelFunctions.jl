"""
    TensorProductKernel(kernel1, kernel2)

A tensor product kernel of the form
```math
k((x₁, x₂), (y₁, y₂)) = kernel1(x₁, y₁) * kernel2(x₂, y₂)
```
"""
struct TensorProductKernel{K1<:Kernel,K2<:Kernel} <: Kernel
    kernel1::K1
    kernel2::K2
end

kappa(kernel::TensorProductKernel, (x1, x2), (y1, y2)) =
    kappa(kernel.kernel1, x1, y1) * kappa(kernel.kernel2, x2, y2)

(kernel::TensorProductKernel)(x, y) = kappa(kernel, x, y)

Base.show(io::IO, κ::TensorProductKernel) = printshifted(io, κ, 0)

function printshifted(io::IO, κ::TensorProductKernel, shift::Int)
    print(io,"Tensor product kernel:")
    print(io,"\n"*("\t"^(shift+1))*"- ")
    printshifted(io, κ.kernel1, shift+2)
    print(io,"\n"*("\t"^(shift+1))*"- ")
    printshifted(io, κ.kernel2, shift+2)
end