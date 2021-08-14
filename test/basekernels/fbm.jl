@testset "FBM" begin
    rng = MersenneTwister(42)
    h = 0.3
    k = FBMKernel(; h=h)
    v1 = rand(rng, 3)
    v2 = rand(rng, 3)
    @test k(v1, v2) ≈
          (
        sqeuclidean(v1, zero(v1))^h + sqeuclidean(v2, zero(v2))^h -
        sqeuclidean(v1 - v2, zero(v1 - v2))^h
    ) / 2 atol = 1e-5
    @test repr(k) == "Fractional Brownian Motion Kernel (h = $(h))"

    test_interface(k)
    @test repr(k) == "Fractional Brownian Motion Kernel (h = $(h))"
    test_ADs(FBMKernel; ADs=[:ReverseDiff])

    # Tests failing for ForwardDiff and Zygote@0.6 (obtained with Julia > 1.3).
    # Related to: https://github.com/FluxML/Zygote.jl/issues/1036
    @test_broken !isinf(ForwardDiff.gradient(x -> x[1]^x[2], [0.0, 0.9])[1])
    # if VERSION >= v"1.4.0"
        f(x, y) = x^y
        @test_broken !isinf(
            Zygote.gradient((x, y) -> sum(f.(x, y)), zeros(1), fill(0.9, 1))[1][1]
        )
    # else
        # test_ADs(FBMKernel; ADs=[:Zygote])
    # end

    test_params(k, ([h],))
end
