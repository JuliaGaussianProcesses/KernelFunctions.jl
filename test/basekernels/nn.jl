@testset "nn" begin
    using LinearAlgebra
    k = NeuralNetworkKernel()
    v1 = rand(3); v2 = rand(3)
    @test k(v1,v2) ≈ asin(v1' * v2 / sqrt((1 + v1' * v1) * (1 + v2' * v2))) atol=1e-5

    # kernelmatrix tests
    m1 = rand(3,4)
    m2 = rand(3,4)
    @test kernelmatrix(k, m1, m1) ≈ kernelmatrix(k, m1) atol=1e-5
    @test_broken kernelmatrix(k, m1, m2) ≈ k(m1, m2) atol=1e-5


    x1 = rand()
    x2 = rand()
    @test kernelmatrix(k, x1*ones(1,1), x2*ones(1,1))[1] ≈ k(x1, x2) atol=1e-5

    @test k(v1, v2) ≈ k(v1, v2) atol=1e-5
    @test typeof(k(v1, v2)) <: Real

    @test_broken size(k(m1, m2)) == (4, 4)
    @test_broken size(k(m1)) == (4, 4)

    A1 = ones(4, 4)
    kernelmatrix!(A1, k, m1, m2)
    @test A1 ≈ kernelmatrix(k, m1, m2) atol=1e-5

    A2 = ones(4, 4)
    kernelmatrix!(A2, k, m1)
    @test A2 ≈ kernelmatrix(k, m1) atol=1e-5

    @test size(kerneldiagmatrix(k, m1)) == (4,)
    A3 = kernelmatrix(k, m1)
    @test kerneldiagmatrix(k, m1) ≈ [A3[i, i] for i in 1:LinearAlgebra.checksquare(A3)] atol=1e-5

    A4 = ones(4)
    kerneldiagmatrix!(A4, k, m1)
    @test kerneldiagmatrix(k, m1) ≈ A4 atol=1e-5

    A5 = ones(4,4)
    @test_throws AssertionError kernelmatrix!(A5, k, m1, m2, obsdim=3)
    @test_throws AssertionError kernelmatrix!(A5, k, m1, obsdim=3)
    @test_throws DimensionMismatch kernelmatrix!(A5, k, ones(4,3), ones(3,4))

    @test k([x1], [x2]) ≈ k(x1, x2) atol=1e-5
    test_ADs(NeuralNetworkKernel, ADs = [:ForwardDiff, :ReverseDiff])
    @test_broken "Zygote uncompatible with BaseKernel"
end
