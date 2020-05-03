@testset "sm" begin
    v1 = rand(5)
    v2 = rand(5)
    h = SqExponentialKernel()
    αs = rand(3) .+ 1e-3
    γs = [randn(5) for _ in 1:3]
    ωs = [randn(5) for _ in 1:3]

    k = SpectralMixtureKernel(h, αs, γs, ωs)
    t = v1 - v2

    @test k(v1, v2) ≈ sum(αs .* exp.(-(t' * hcat(γs...))'.^2) .*
                          cospi.((t' * hcat(ωs...))')) atol=1e-5
end
