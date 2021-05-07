@testset "Gabor" begin
    v1 = rand(3)
    v2 = rand(3)
    ell = rand()
    p = rand()
    k = gaborkernel(;
        sqexponential_transform=ScaleTransform(inv(ell)),
        cosine_transform=ScaleTransform(inv(p)),
    )
    @test k isa KernelProduct{
        <:Tuple{
            TransformedKernel{<:SqExponentialKernel,<:ScaleTransform},
            TransformedKernel{<:CosineKernel,<:ScaleTransform},
        },
    }
    @test k.kernels[1].transform.s[1] == inv(ell)
    @test k.kernels[2].transform.s[1] == inv(p)

    k_manual = exp(-sqeuclidean(v1, v2) / (2 * ell^2)) * cospi(euclidean(v1, v2) / p)
    @test k_manual ≈ k(v1, v2) atol = 1e-5

    lhs_manual = (SqExponentialKernel() ∘ ScaleTransform(1 / ell))(v1, v2)
    rhs_manual = (CosineKernel() ∘ ScaleTransform(1 / p))(v1, v2)
    @test lhs_manual * rhs_manual ≈ k(v1, v2) atol = 1e-5

    @test gaborkernel() isa KernelProduct{<:Tuple{<:SqExponentialKernel,<:CosineKernel}}

    test_ADs(
        x -> gaborkernel(;
            sqexponential_transform=ScaleTransform(x[1]),
            cosine_transform=ScaleTransform(x[2]),
        ),
        [ell, p],
    )

    # deprecated `GaborKernel`
    k2 = @test_deprecated GaborKernel(; ell=ell, p=p)
    @test k2.ell ≈ ell atol = 1e-5
    @test k2.p ≈ p atol = 1e-5
    @test k2(v1, v2) ≈ k(v1, v2)

    k3 = @test_deprecated GaborKernel()
    @test k3.ell ≈ 1.0 atol = 1e-5
    @test k3.p ≈ 1.0 atol = 1e-5
    @test repr(k3) == "Gabor Kernel (ell = 1, p = 1)"

    test_interface(k3, Vector{Float64})

    test_ADs(x -> GaborKernel(; ell=x[1], p=x[2]), [ell, p]; ADs=[:Zygote])

    # Tests are also failing randomly for ForwardDiff and ReverseDiff but randomly
end
