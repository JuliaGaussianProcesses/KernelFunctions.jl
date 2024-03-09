@testset "sm" begin
    D_in = 5
    v1 = rand(D_in)
    v2 = rand(D_in)

    αs₁ = rand(3)
    αs₂ = rand(D_in, 3)
    γs = rand(D_in, 3)
    ωs = rand(D_in, 3)

    k1 = spectral_mixture_kernel(αs₁, γs, ωs)
    k2 = spectral_mixture_product_kernel(αs₂, γs, ωs)

    t = v1 - v2

    @test k1(v1, v2) ≈ sum(αs₁ .* exp.(-(t' * γs)' .^ 2 ./ 2) .* cospi.((t' * ωs)')) atol =
        1e-5

    @test isapprox(
        k2(v1, v2),
        prod([
            sum(
                αs₂[i, :]' .* exp.(-(γs[i, :]' * t[i]) .^ 2 ./ 2) .*
                cospi.(ωs[i, :]' * t[i]),
            ) for i in 1:length(t)
        ],);
        atol=1e-5,
    )

    @test_throws DimensionMismatch spectral_mixture_kernel(rand(5), rand(4, 3), rand(4, 3))
    @test_throws DimensionMismatch spectral_mixture_kernel(rand(3), rand(4, 3), rand(5, 3))
    @test_throws DimensionMismatch spectral_mixture_product_kernel(
        rand(5, 3), rand(4, 3), rand(5, 3)
    )

    # Standardised tests. Choose input dims carefully.
    @testset "ColVecs" begin
        x0 = ColVecs(randn(D_in, 3))
        x1 = ColVecs(randn(D_in, 3))
        x2 = ColVecs(randn(D_in, 2))
        TestUtils.test_interface(k1, x0, x1, x2)
        TestUtils.test_interface(k2, x0, x1, x2)
    end
    @testset "RowVecs" begin
        x0 = RowVecs(randn(3, D_in))
        x1 = RowVecs(randn(3, D_in))
        x2 = RowVecs(randn(2, D_in))
        TestUtils.test_interface(k1, x0, x1, x2)
        TestUtils.test_interface(k2, x0, x1, x2)
    end

    @testset "Type stability given static arrays" begin
        αs = @SVector rand(3)
        γs = @SMatrix rand(D_in, 3)
        ωs = @SMatrix rand(D_in, 3)
        @inferred spectral_mixture_kernel(αs, γs, ωs)
    end

    # test_ADs(x->spectral_mixture_kernel(exp.(x[1:3]), reshape(x[4:18], 5, 3), reshape(x[19:end], 5, 3)), vcat(log.(αs₁), γs[:], ωs[:]), dims = [5,5])
    # No tests passing (BaseKernel)
    @test_broken false
end
