@testset "deprecations.jl" begin
    p = rand()
    v = rand(3)
    M = rand(3, 3)
    kernel = SqExponentialKernel()
    
    @test (@test_deprecated transform(kernel, LinearTransform(M))) ==
        kernel ∘ LinearTransform(M)
    @test (@test_deprecated transform(kernel ∘ ScaleTransform(p), ARDTransform(v))) ==
        kernel ∘ ARDTransform(v) ∘ ScaleTransform(p)
    @test (@test_deprecated transform(kernel, p)) == kernel ∘ ScaleTransform(p)
    @test (@test_deprecated transform(kernel, v)) == kernel ∘ ARDTransform(v)
end
