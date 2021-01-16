@testset "nn" begin
    using LinearAlgebra
    k = NeuralNetworkKernel()
    v1 = rand(3)
    v2 = rand(3)
    @test k(v1, v2) ≈ asin(v1' * v2 / sqrt((1 + v1' * v1) * (1 + v2' * v2))) atol = 1e-5

    # Standardised tests.
    TestUtils.test_interface(k, Float64)
    test_ADs(NeuralNetworkKernel)
    @test_broken "Zygote uncompatible with BaseKernel"
end
