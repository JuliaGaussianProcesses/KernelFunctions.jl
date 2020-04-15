@testset "tensorproduct" begin
    rng = MersenneTwister(123456)
    u1 = rand(rng, 10)
    u2 = rand(rng, 10)
    v1 = rand(rng, 5)
    v2 = rand(rng, 5)

    # kernels
    k1 = SqExponentialKernel()
    k2 = ExponentialKernel()
    kernel1 = TensorProduct(k1, k2)
    kernel2 = TensorProduct([k1, k2])

    @test kernel1.kernels === (k1, k2) === TensorProduct((k1, k2)).kernels

    for (x, y) in (((v1, u1), (v2, u2)), ([v1, u1], [v2, u2]))
        val = k1(x[1], y[1]) * k2(x[2], y[2])

        @test kernel1(x, y) == kernel2(x, y) == val
        @test KernelFunctions.kappa(kernel1, x, y) ==
            KernelFunctions.kappa(kernel2, x, y) == val
    end
end
