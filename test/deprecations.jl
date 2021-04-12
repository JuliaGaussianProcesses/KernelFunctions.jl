@testset "deprecations.jl" begin
    p = rand()
    v = rand(3)
    M = rand(3, 3)
    v1 = rand(3)
    v2 = rand(3)
    kernel = SqExponentialKernel()

    k1 = @test_deprecated transform(kernel, LinearTransform(M))
    @test k1(v1, v2) == (kernel ∘ LinearTransform(M))(v1, v2)

    k2 = @test_deprecated transform(kernel ∘ ScaleTransform(p), ARDTransform(v))
    @test k2(v1, v2) == (kernel ∘ ARDTransform(v) ∘ ScaleTransform(p))(v1, v2)

    k3 = @test_deprecated transform(kernel, p)
    @test k3(v1, v2) == (kernel ∘ ScaleTransform(p))(v1, v2)

    k4 = @test_deprecated transform(kernel, v)
    @test k4(v1, v2) == (kernel ∘ ARDTransform(v))(v1, v2)
end
