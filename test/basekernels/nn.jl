@testset "nn" begin
    k = NeuralNetworkKernel()
    v1 = rand(3); v2 = rand(3)
    @test k(v1,v2) ≈ asin(v1' * v2 / sqrt((1 + v1' * v1) * (1 + v2' * v2))) atol=1e-5

    # kernelmatrix tests
    m1 = rand(3,3)
    m2 = rand(3,3)
    @test kernelmatrix(k, m1, m1) ≈ kernelmatrix(k, m1) atol=1e-5
    @test kernelmatrix(k, m1, m2) ≈ k(m1, m2) atol=1e-5


    x1 = rand()
    x2 = rand()
    @test kernelmatrix(k, x1*ones(1,1), x2*ones(1,1))[1] ≈ k(x1, x2) atol=1e-5
end
