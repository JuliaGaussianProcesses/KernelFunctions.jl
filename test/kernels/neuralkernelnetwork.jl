@testset "neural_kernel_network" begin
    rng, N, N′, D = MersenneTwister(123456), 5, 6, 2
    x0 = collect(range(-2.0, 2.0; length=N)) .+ 1e-3 .* randn(rng, N)
    x1 = collect(range(-1.7, 2.3; length=N)) .+ 1e-3 .* randn(rng, N)
    x2 = collect(range(-1.7, 3.3; length=N′)) .+ 1e-3 .* randn(rng, N′)

    X0 = ColVecs(randn(rng, D, N))
    X1 = ColVecs(randn(rng, D, N))
    X2 = ColVecs(randn(rng, D, N′))

    # Most of the NeuralKernelNetwork tests are currently broken.
    @testset "general test" begin

        # Specify primitives.
        k1 = 0.6 * (SEKernel() ∘ ScaleTransform(0.5))
        k2 = 0.4 * (Matern32Kernel() ∘ ScaleTransform(0.1))
        primitives = Primitive(k1, k2)

        # Build NKN Kernel.
        nkn = NeuralKernelNetwork(primitives, Chain(LinearLayer(2, 2), product))

        # Apply standard test suite.
        TestUtils.test_interface(nkn, Float64)
    end
    @testset "kernel composition test" begin
        rng = MersenneTwister(123456)

        # Specify primitives.
        k1 = rand(rng) * transform(SEKernel(), randn(rng))
        k2 = rand(rng) * transform(Matern32Kernel(), randn(rng))
        primitives = Primitive(k1, k2)

        @testset "LinearLayer" begin
            # Specify linear NKN and equivalent composite kernel.
            weights = rand(rng, 1, 2)
            nkn_add_kernel = NeuralKernelNetwork(primitives, LinearLayer(weights))
            sum_k = softplus(weights[1]) * k1 + softplus(weights[2]) * k2

            # Vector input.
            @test kernelmatrix_diag(nkn_add_kernel, x0) ≈ kernelmatrix_diag(sum_k, x0)
            @test kernelmatrix_diag(nkn_add_kernel, x0, x1) ≈
                kernelmatrix_diag(sum_k, x0, x1)

            # ColVecs input.
            @test kernelmatrix_diag(nkn_add_kernel, X0) ≈ kernelmatrix_diag(sum_k, X0)
            @test kernelmatrix_diag(nkn_add_kernel, X0, X1) ≈
                kernelmatrix_diag(sum_k, X0, X1)
        end
        @testset "product" begin
            nkn_prod_kernel = NeuralKernelNetwork(primitives, product)
            prod_k = k1 * k2

            # Vector input.
            @test kernelmatrix(nkn_prod_kernel, x0) ≈ kernelmatrix(prod_k, x0)
            @test kernelmatrix(nkn_prod_kernel, x0, x1) ≈ kernelmatrix(prod_k, x0, x1)

            # ColVecs input.
            @test kernelmatrix(nkn_prod_kernel, X0) ≈ kernelmatrix(prod_k, X0)
            @test kernelmatrix(nkn_prod_kernel, X0, X1) ≈ kernelmatrix(prod_k, X0, X1)
        end
    end
end
