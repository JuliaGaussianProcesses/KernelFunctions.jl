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

    @testset "kernelmatrix and kerneldiagmatrix" begin
        X = rand(rng, 2, 10)
        x_cols = ColVecs(X)
        x_rows = RowVecs(X')
        Y = rand(rng, 2, 10)
        y_cols = ColVecs(Y)
        y_rows = RowVecs(Y')

        trueX = kernelmatrix(k1, X[1, :]) .* kernelmatrix(k2, X[2, :])
        trueXY = kernelmatrix(k1, X[1, :], Y[1, :]) .* kernelmatrix(k2, X[2, :], Y[2, :])
        tmp = Matrix{Float64}(undef, 10, 10)
        tmp_diag = Vector{Float64}(undef, 10)

        for kernel in (kernel1, kernel2), (x, y) in ((x_cols, y_cols), (x_rows, y_rows))
            @test kernelmatrix(kernel, x) ≈ trueX

            @test kernelmatrix(kernel, x, y) ≈ trueXY

            fill!(tmp, 0)
            kernelmatrix!(tmp, kernel, x)
            @test tmp ≈ trueX

            fill!(tmp, 0)
            kernelmatrix!(tmp, kernel, x, y)
            @test tmp ≈ trueXY

            @test kerneldiagmatrix(kernel, x) ≈ diag(kernelmatrix(kernel, x))

            fill!(tmp_diag, 0)
            kerneldiagmatrix!(tmp_diag, kernel, x)
            @test tmp_diag ≈ diag(kernelmatrix(kernel, x))
        end
    end

    @testset "single kernel" begin
        kernel = TensorProduct(k1)
        @test length(kernel) == 1

        @testset "eval" begin
            for (x, y) in (((v1,), (v2,)), ([v1], [v2]))
                val = k1(x[1], y[1])

                @test kernel(x, y) == val
            end
        end

        @testset "kernelmatrix" begin
            N = 10

            x = randn(rng, N)
            y = randn(rng, N)
            vectors = (x, y)

            X = reshape(x, 1, :)
            x_cols = ColVecs(X)
            x_rows = RowVecs(X')
            Y = reshape(y, 1, :)
            y_cols = ColVecs(Y)
            y_rows = RowVecs(Y')

            trueX = kernelmatrix(k1, x)
            trueXY = kernelmatrix(k1, x, y)
            tmp = Matrix{Float64}(undef, N, N)
            tmp_diag = Vector{Float64}(undef, N)

            for (x, y) in ((x, y), (x_cols, y_cols), (x_rows, y_rows))

                @test kernelmatrix(kernel, x) ≈ trueX

                @test kernelmatrix(kernel, x, y) ≈ trueXY

                fill!(tmp, 0)
                kernelmatrix!(tmp, kernel, x)
                @test tmp ≈ trueX

                fill!(tmp, 0)
                kernelmatrix!(tmp, kernel, x, y)
                @test tmp ≈ trueXY

                @test kerneldiagmatrix(kernel, x) ≈ diag(kernelmatrix(kernel, x))

                fill!(tmp_diag, 0)
                kerneldiagmatrix!(tmp_diag, kernel, x)
                @test tmp_diag ≈ diag(kernelmatrix(kernel, x))
            end
        end
    end
    test_ADs(()->TensorProduct(SqExponentialKernel(), LinearKernel()), dims = [2, 2]) # ADs = [:ForwardDiff, :ReverseDiff])

    test_params(TensorProduct(k1, k2), (k1, k2))
end
