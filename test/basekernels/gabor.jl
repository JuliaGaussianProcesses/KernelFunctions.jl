@testset "Gabor" begin
    v1 = rand(3)
    v2 = rand(3)
    ell = abs(rand())
    p = abs(rand())
    k = GaborKernel(; ell=ell, p=p)
    @test k.ell ≈ ell atol = 1e-5
    @test k.p ≈ p atol = 1e-5

    k_manual = exp(-sqeuclidean(v1, v2) / (2 * k.ell^2)) * cospi(euclidean(v1, v2) / k.p)
    @test k(v1, v2) ≈ k_manual atol = 1e-5

    lhs_manual = (SqExponentialKernel() ∘ ScaleTransform(1 / k.ell))(v1, v2)
    rhs_manual = (CosineKernel() ∘ ScaleTransform(1 / k.p))(v1, v2)
    @test k(v1, v2) ≈ lhs_manual * rhs_manual atol = 1e-5

    k = GaborKernel()
    @test k.ell ≈ 1.0 atol = 1e-5
    @test k.p ≈ 1.0 atol = 1e-5
    @test repr(k) == "Gabor Kernel (ell = 1, p = 1)"

    test_interface(k, Vector{Float64})

    test_ADs(x -> GaborKernel(; ell=x[1], p=x[2]), [ell, p]; ADs=[:Zygote])

    # Tests are also failing randomly for ForwardDiff and ReverseDiff but randomly
end
