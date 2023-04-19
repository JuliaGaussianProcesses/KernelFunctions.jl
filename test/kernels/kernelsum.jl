@testset "kernelsum" begin
    k1 = LinearKernel()
    k2 = SqExponentialKernel()
    k = KernelSum(k1, k2)
    @test k == KernelSum([k1, k2]) == KernelSum((k1, k2))
    for (_k1, _k2) in Iterators.product(
        (k1, KernelSum((k1,)), KernelSum([k1])), (k2, KernelSum((k2,)), KernelSum([k2]))
    )
        @test k == _k1 + _k2
    end
    @test length(k) == 2
    @test repr(k) == (
        "Sum of 2 kernels:\n" *
        "\tLinear Kernel (c = 0.0)\n" *
        "\tSquared Exponential Kernel (metric = Euclidean(0.0))"
    )

    # Standardised tests.
    test_interface(k, Float64)
    test_interface(ConstantKernel(; c=1.5) + WhiteKernel(), Vector{String})
    test_ADs(x -> KernelSum(SqExponentialKernel(), LinearKernel(; c=exp(x[1]))), rand(1))
    test_interface_ad_perf(2.4, StableRNG(123456)) do c
        KernelSum(SqExponentialKernel(), LinearKernel(; c=c))
    end

    test_params(k1 + k2, (k1, k2))

    # Regression tests for https://github.com//issues/458
    @testset for k in (
        RBFKernel() + RBFKernel() * LinearKernel(),
        RBFKernel() + RBFKernel() * ExponentialKernel(),
        RBFKernel() * (LinearKernel() + ExponentialKernel()),
    )
        test_type_stability(k)
    end
end
