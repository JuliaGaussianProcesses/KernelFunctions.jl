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
    test_interface_ad_perf(0.3, StableRNG(123456)) do c
        SEKernel() ∘ ScaleTransform(c)
    end

    @testset "median heuristic" begin
        for x in (x, XV, XC, XR), dist in (Euclidean(), Cityblock())
            n = length(x)
            t = median_heuristic_transform(dist, x)
            @test t isa ScaleTransform
            @test first(t.s) ≈
                inv(median(dist(x[i], x[j]) for i in 1:n, j in 1:n if i != j))

            y = map(t, x)
            @test median(dist(y[i], y[j]) for i in 1:n, j in 1:n if i != j) ≈ 1
        end
    end
end
