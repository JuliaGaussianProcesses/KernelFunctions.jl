@testset "kernelproduct" begin
    k1 = LinearKernel()
    k2 = SqExponentialKernel()
    k = KernelProduct(k1, k2)
    @test k == KernelProduct([k1, k2]) == KernelProduct((k1, k2))
    @test length(k) == 2
    @test string(k) == (
        "Product of 2 kernels:\n\tLinear Kernel (c = 0.0)\n\tSquared " *
        "Exponential Kernel"
    )

    # Standardised tests.
    TestUtils.test_interface(k, Float64)
    test_ADs(
        x -> KernelProduct(SqExponentialKernel(), LinearKernel(; c=exp(x[1]))), rand(1)
    )

    test_params(k1 * k2, (k1, k2))
end
