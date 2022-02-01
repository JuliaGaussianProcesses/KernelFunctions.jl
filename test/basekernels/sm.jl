@testset "sm" begin
    D_in = 5
    v1 = rand(D_in)
    v2 = rand(D_in)

    K = 3
    αs₁ = rand(K)
    αs₂ = rand(D_in, K)
    γs = rand(D_in, K)
    ωs = rand(D_in, K)

    k1 = SpectralMixtureKernel(αs₁, γs, ωs)
    k2 = spectral_mixture_product_kernel(αs₂, γs, ωs)

    t = v1 - v2

    @test k1(v1, v2) ≈ sum(
        αs₁[k] * exp(-norm(t .* γs[:, k])^2 / 2) * cospi(2 * dot(ωs[:, k], t)) for k in 1:K
    )

    @test isapprox(
        k2(v1, v2),
        prod(
            sum(
                αs₂[i, k] * exp(-(γs[i, k] * t[i])^2 / 2) * cospi(2 * ωs[i, k] * t[i]) for
                k in 1:K
            ) for i in 1:D_in
        ),
    )

    @test_throws DimensionMismatch SpectralMixtureKernel(rand(5), rand(4, 3), rand(4, 3))
    @test_throws DimensionMismatch SpectralMixtureKernel(rand(3), rand(4, 3), rand(5, 3))
    @test_throws DimensionMismatch spectral_mixture_product_kernel(
        rand(5, 3), rand(4, 3), rand(5, 3)
    )

    # Standardised tests. Choose input dims carefully.
    @testset "ColVecs" begin
        x0 = ColVecs(randn(D_in, 3))
        x1 = ColVecs(randn(D_in, 3))
        x2 = ColVecs(randn(D_in, 2))
        test_interface(k1, x0, x1, x2)
        test_interface(k2, x0, x1, x2)
    end
    @testset "RowVecs" begin
        x0 = RowVecs(randn(3, D_in))
        x1 = RowVecs(randn(3, D_in))
        x2 = RowVecs(randn(2, D_in))
        test_interface(k1, x0, x1, x2)
        test_interface(k2, x0, x1, x2)
    end
    # test_ADs(x->spectral_mixture_kernel(exp.(x[1:3]), reshape(x[4:18], 5, 3), reshape(x[19:end], 5, 3)), vcat(log.(αs₁), γs[:], ωs[:]), dims = [5,5])
    @test_broken "No tests passing (BaseKernel)"
end
