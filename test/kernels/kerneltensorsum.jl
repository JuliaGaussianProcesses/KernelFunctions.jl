@testset "kerneltensorsum" begin
    rng = MersenneTwister(123456)
    u1 = rand(rng, 10)
    u2 = rand(rng, 10)
    v1 = rand(rng, 5)
    v2 = rand(rng, 5)

    # kernels
    k1 = SqExponentialKernel()
    k2 = ExponentialKernel()
    kernel1 = KernelTensorSum(k1, k2)
    kernel2 = KernelTensorSum([k1, k2])

    @test kernel1 == kernel2
    @test kernel1.kernels == (k1, k2) === KernelTensorSum((k1, k2)).kernels
    for (_k1, _k2) in Iterators.product(
        (k1, KernelTensorSum((k1,)), KernelTensorSum([k1])),
        (k2, KernelTensorSum((k2,)), KernelTensorSum([k2])),
    )
        @test kernel1 == _k1 ⊕ _k2
    end
    @test length(kernel1) == length(kernel2) == 2
    @test string(kernel1) == (
        "Independent sum of 2 kernels:\n" *
        "\tSquared Exponential Kernel (metric = Euclidean(0.0))\n" *
        "\tExponential Kernel (metric = Euclidean(0.0))"
    )
    @test_throws DimensionMismatch kernel1(rand(3), rand(3))

    @testset "val" begin
        for (x, y) in (((v1, u1), (v2, u2)), ([v1, u1], [v2, u2]))
            val = k1(x[1], y[1]) + k2(x[2], y[2])

            @test kernel1(x, y) == kernel2(x, y) == val
        end
    end

    # Standardised tests.
    TestUtils.test_interface(kernel1, ColVecs{Float64})
    TestUtils.test_interface(kernel1, RowVecs{Float64})
    TestUtils.test_interface(
        KernelTensorSum(WhiteKernel(), ConstantKernel(; c=1.1)), ColVecs{String}
    )
    test_ADs(
        x -> KernelTensorSum(SqExponentialKernel(), LinearKernel(; c=exp(x[1]))),
        rand(1);
        dims=[2, 2],
    )
    types = [ColVecs{Float64,Matrix{Float64}}, RowVecs{Float64,Matrix{Float64}}]
    test_interface_ad_perf(2.1, StableRNG(123456), types) do c
        KernelTensorSum(SqExponentialKernel(), LinearKernel(; c=c))
    end
    test_params(KernelTensorSum(k1, k2), (k1, k2))

    @testset "single kernel" begin
        kernel = KernelTensorSum(k1)
        @test length(kernel) == 1

        @testset "eval" begin
            for (x, y) in (((v1,), (v2,)), ([v1], [v2]))
                val = k1(x[1], y[1])

                @test kernel(x, y) == val
            end
        end
    end
end
