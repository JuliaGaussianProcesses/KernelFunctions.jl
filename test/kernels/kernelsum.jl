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
    TestUtils.test_interface(ConstantKernel(c=1.5) * WhiteKernel(), Vector{String})
    test_ADs(x -> KernelSum(SqExponentialKernel(), LinearKernel(; c=exp(x[1]))), rand(1))
    test_interface_ad_perf(2.4, StableRNG(123456)) do c
        KernelSum(SqExponentialKernel(), LinearKernel(; c=c))
    end

    test_params(k1 + k2, (k1, k2))

    # Regression tests for https://github.com//issues/458
    @testset "Type stability" begin
        function check_type_stability(k)
            @test (@inferred k(0.1, 0.2)) isa Real
            x = rand(10)
            y = rand(10)
            @test (@inferred kernelmatrix(k, x)) isa Matrix{<:Real}
            @test (@inferred kernelmatrix(k, x, y)) isa Matrix{<:Real}
            @test (@inferred kernelmatrix_diag(k, x)) isa Vector{<:Real}
            @test (@inferred kernelmatrix_diag(k, x, y)) isa Vector{<:Real}
        end
        @testset for k in (
            RBFKernel() + RBFKernel() * LinearKernel(),
            RBFKernel() + RBFKernel() * ExponentialKernel(),
            RBFKernel() * (LinearKernel() + ExponentialKernel()),
        )
            check_type_stability(k)
        end
    end
end
