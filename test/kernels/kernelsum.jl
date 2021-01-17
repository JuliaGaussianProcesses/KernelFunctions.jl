@testset "kernelsum" begin
    k1 = LinearKernel()
    k2 = SqExponentialKernel()
    k = KernelSum(k1, k2)
    @test k == KernelSum([k1, k2]) == KernelSum((k1, k2))
    @test length(k) == 2
    @test string(k) == (
        "Sum of 2 kernels:\n\tLinear Kernel (c = 0.0)\n\tSquared " * "Exponential Kernel"
    )

    # Standardised tests.
    TestUtils.test_interface(k, Float64)
    test_ADs(
        x -> KernelSum(SqExponentialKernel(), LinearKernel(; c=x[1])),
        rand(1);
        ADs=[:ForwardDiff, :ReverseDiff, :Zygote],
    )

    test_params(k1 + k2, (k1, k2))
end
