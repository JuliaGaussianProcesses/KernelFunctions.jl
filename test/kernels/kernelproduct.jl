@testset "kernelproduct" begin
    k1 = LinearKernel()
    k2 = SqExponentialKernel()
    k = KernelProduct(k1, k2)
    @test k == KernelProduct([k1, k2]) == KernelProduct((k1, k2))
    for (_k1, _k2) in Iterators.product(
        (k1, KernelProduct((k1,)), KernelProduct([k1])),
        (k2, KernelProduct((k2,)), KernelProduct([k2])),
    )
        @test k == _k1 * _k2
    end
    @test length(k) == 2
    @test string(k) == (
        "Product of 2 kernels:\n\tLinear Kernel (c = 0.0)\n\tSquared " *
        "Exponential Kernel (metric = Euclidean(0.0))"
    )

    # Standardised tests.
    TestUtils.test_interface(k, Float64)
    TestUtils.test_interface(ConstantKernel(; c=1.0) * WhiteKernel(), Vector{String})
    test_ADs(
        x -> KernelProduct(SqExponentialKernel(), LinearKernel(; c=exp(x[1]))), rand(1)
    )
    test_interface_ad_perf(2.4, StableRNG(123456)) do c
        KernelProduct(SqExponentialKernel(), LinearKernel(; c=c))
    end
    test_params(k1 * k2, (k1, k2))
end
