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

    @test kernel1 == kernel2
    @test kernel1.kernels === (k1, k2) === TensorProduct((k1, k2)).kernels
    @test length(kernel1) == length(kernel2) == 2
    @test_throws DimensionMismatch kernel1(rand(3), rand(3))

    @testset "val" begin
        for (x, y) in (((v1, u1), (v2, u2)), ([v1, u1], [v2, u2]))
            val = k1(x[1], y[1]) * k2(x[2], y[2])

            @test kernel1(x, y) == kernel2(x, y) == val
        end
    end

    # Standardised tests.
    TestUtils.test_interface(kernel1, ColVecs{Float64})
    TestUtils.test_interface(kernel1, RowVecs{Float64})
    test_ADs(() -> TensorProduct(SqExponentialKernel(), LinearKernel()); dims=[2, 2]) # ADs = [:ForwardDiff, :ReverseDiff])
    test_params(TensorProduct(k1, k2), (k1, k2))

    @testset "single kernel" begin
        kernel = TensorProduct(k1)
        @test length(kernel) == 1

        @testset "eval" begin
            for (x, y) in (((v1,), (v2,)), ([v1], [v2]))
                val = k1(x[1], y[1])

                @test kernel(x, y) == val
            end
        end
    end
end
