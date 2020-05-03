@testset "sm" begin
    v1 = rand(5)
    v2 = rand(5)
    h = SqExponentialKernel()

    αs₁ = rand(3)
    αs₂ = rand(5, 3)
    γs = rand(5, 3)
    ωs = rand(5, 3)

    k1 = spectral_mixture_kernel(h, αs₁, γs, ωs)
    k2 = spectral_mixture_product_kernel(h, αs₂, γs, ωs)

    t = v1 - v2

    @test k1(v1, v2) ≈ sum(αs₁ .* exp.(-(t' * γs)'.^2) .*
                           cospi.((t' * ωs)')) atol=1e-5

    @test k2(v1, v2) ≈ prod(sum(αs₂[i,:]' .* (exp.(-(γs[i,:]' * t[i]).^2) .* cospi.(ωs[i,:]' * t[i]))) for i in 1:length(t)) atol=1e-5
end
