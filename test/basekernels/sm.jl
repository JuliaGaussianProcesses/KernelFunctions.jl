@testset "sm" begin
    v1 = rand(5)
    v2 = rand(5)

    αs₁ = rand(3)
    αs₂ = rand(5, 3)
    γs = rand(5, 3)
    ωs = rand(5, 3)

    k1 = spectral_mixture_kernel(αs₁, γs, ωs)
    k2 = spectral_mixture_product_kernel(αs₂, γs, ωs)

    t = v1 - v2

    @test k1(v1, v2) ≈ sum(αs₁ .* exp.(-(t' * γs)'.^2 ./ 2) .* cospi.((t' * ωs)')) atol=1e-5

    @test isapprox(
        k2(v1, v2),
        prod(
            [sum(αs₂[i,:]' .* exp.(-(γs[i,:]' * t[i]).^2 ./ 2) .*
            cospi.(ωs[i,:]' * t[i])) for i in 1:length(t)],
        );
        atol=1e-5,
    )

    @test_throws DimensionMismatch spectral_mixture_kernel(rand(5) ,rand(4,3), rand(4,3))
    @test_throws DimensionMismatch spectral_mixture_kernel(rand(3) ,rand(4,3), rand(5,3))
    @test_throws DimensionMismatch spectral_mixture_product_kernel(rand(5,3) ,rand(4,3), rand(5,3))
    # test_ADs(x->spectral_mixture_kernel(exp.(x[1:3]), reshape(x[4:18], 5, 3), reshape(x[19:end], 5, 3)), vcat(log.(αs₁), γs[:], ωs[:]), dims = [5,5])
    @test_broken "No tests passing (BaseKernel)"
end
