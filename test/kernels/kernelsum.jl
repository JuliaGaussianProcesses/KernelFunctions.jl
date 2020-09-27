@testset "kernelsum" begin
    rng = MersenneTwister(123456)
    x = rand(rng)*2
    v1 = rand(rng, 3)
    v2 = rand(rng, 3)

    k1 = LinearKernel()
    k2 = SqExponentialKernel()
    k3 = RationalQuadraticKernel()
    X = rand(rng, 2,2)

    k = KernelSum(k1,k2)
    ks1 = 2.0*k1
    ks2 = 0.5*k2
    @test length(k) == 2
    @test string(k) == (
        "Sum of 2 kernels:\n\tLinear Kernel (c = 0.0)\n\tSquared " *
        "Exponential Kernel"
    )
    @test k(v1, v2) == (k1 + k2)(v1, v2)
    @test (k + k3)(v1,v2) ≈ (k3 + k)(v1, v2)
    @test (k1 + k2)(v1, v2) == KernelSum(k1, k2)(v1, v2)
    @test (k + ks1)(v1, v2) ≈ (ks1 + k)(v1, v2)
    @test (k + k)(v1, v2) == KernelSum([k1, k2, k1, k2])(v1, v2)
    @test KernelSum([k1, k2]) == KernelSum((k1, k2)) == k1 + k2

    @test (KernelSum([k1, k2]) + KernelSum([k2, k1])).kernels == [k1, k2, k2, k1]
    @test (KernelSum([k1, k2]) + k3).kernels == [k1, k2, k3]
    @test (k3 + KernelSum([k1, k2])).kernels == [k3, k1, k2]

    @test (KernelSum((k1, k2)) + KernelSum((k2, k1))).kernels == (k1, k2, k2, k1)
    @test (KernelSum((k1, k2)) + k3).kernels == (k1, k2, k3)
    @test (k3 + KernelSum((k1, k2))).kernels == (k3, k1, k2)

    # Standardised tests.
    TestUtils.test_interface(k, Float64)
    test_ADs(
        x->KernelSum(SqExponentialKernel(), LinearKernel(c=x[1])), rand(1);
        ADs = [:ForwardDiff, :ReverseDiff, :Zygote],
    )

    test_params(k1 + k2, (k1, k2))
end
