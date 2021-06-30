@testset "scaletransform" begin
    rng = MersenneTwister(123456)
    s = rand(rng) + 1e-3
    t = ScaleTransform(s)

    x = randn(rng, 7)
    XV = [randn(rng, 6) for _ in 1:4]
    XC = ColVecs(randn(rng, 10, 5))
    XR = RowVecs(randn(rng, 6, 11))

    @testset "$(typeof(x))" for x in [x, XV, XC, XR]
        x′ = map(t, x)
        @test all([t(x[n]) ≈ s .* x[n] for n in eachindex(x)])
        @test all([t(x[n]) ≈ x′[n] for n in eachindex(x)])
    end

    s2 = 2.0
    KernelFunctions.set!(t, s2)
    @test t.s == [s2]
    @test isequal(ScaleTransform(s), ScaleTransform(s))
    @test repr(t) == "Scale Transform (s = $(s2))"
    test_ADs(x -> SEKernel() ∘ ScaleTransform(exp(x[1])), randn(rng, 1))
end
