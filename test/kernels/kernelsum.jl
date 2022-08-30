@testset "kernelsum" begin
    k1 = LinearKernel()
    k2 = SqExponentialKernel()
    k = KernelSum(k1, k2)
    @test k == KernelSum([k1, k2]) == KernelSum((k1, k2))
    @test length(k) == 2
    @test string(k) == (
        "Sum of 2 kernels:\n" *
        "\tLinear Kernel (c = 0.0)\n" *
        "\tSquared Exponential Kernel (metric = Euclidean(0.0))"
    )

    # Standardised tests.
    TestUtils.test_interface(k, Float64)
    test_ADs(x -> KernelSum(SqExponentialKernel(), LinearKernel(; c=exp(x[1]))), rand(1))
    test_interface_ad_perf(2.4, StableRNG(123456)) do c
        KernelSum(SqExponentialKernel(), LinearKernel(; c=c))
    end

    test_params(k1 + k2, (k1, k2))

    @testset "Type stability" begin
        function check_type_stability(k)
            @inferred k(0.1, 0.2)
            x = rand(10)
            y = rand(10)
            @inferred kernelmatrix(k, x)
            @inferred kernelmatrix(k, x, y)
            @inferred kernelmatrix_diag(k, x)
            @inferred kernelmatrix_diag(k, x, y)
        end
        @testset for k in (
            RBFKernel() + RBFKernel() * LinearKernel(),
            RBFKernel() + RBFKernel() * ExponentialKernel(),
            RBFKernel() * (LinearKernel() + ExponentialKernel())
        )
            check_type_stability(k)
        end
    end
end
