@testset "cuda" begin
    @testset "$T" for T in [Float32, Float64]
        @testset "($vec_type, $kernel)" for (vec_type, kernel) in Iterators.product(
            [ColVecs{T, CuMatrix{T}}, RowVecs{T, CuMatrix{T}}, CuVector{T}],
            [
                # Base Kernels
                ConstantKernel(c=T(2.3)),
                CosineKernel(),
                ExponentialKernel(),
                ExponentiatedKernel(),
                FBMKernel(T(0.5)),
                gaborkernel(),
                GammaExponentialKernel(; gamma=T(0.5)),
                GammaRationalKernel(; alpha=T(2.0), gamma=T(1.0)),
                LinearKernel(T(0.3)),
                # MaternKernel(T(0.7)), # doesn't work with AD, so not going to bother for now
                Matern12Kernel(),
                Matern32Kernel(),
                Matern52Kernel(),
                NeuralNetworkKernel(),
                # PeriodicKernel(r=rand(T, 2)), # has a `Vector` baked in
                PolynomialKernel(degree=2, c=T(0.3)),
                PolynomialKernel(degree=3, c=T(0.5)),
                RationalKernel(alpha=T(2.0)),
                RationalQuadraticKernel(alpha=T(2.0)),
                SEKernel(),
                # WhiteKernel(), # unclear how to compute delta metric efficiently
                # ZeroKernel(), # not able to get the target type out at the minute

                # Derived Kernels
                SEKernel() + Matern32Kernel(),
                SEKernel() + Matern32Kernel() + RationalKernel(alpha=T(2.0)),
                SEKernel() + Matern32Kernel() * RationalKernel(alpha=T(2.0)),

                SEKernel() * Matern32Kernel(),
                SEKernel() * Matern32Kernel() * RationalKernel(alpha=T(2.0)),
                SEKernel() * Matern32Kernel() + RationalKernel(alpha=T(2.0)),

                T(0.3) * SEKernel(),
                T(0.3) * (SEKernel() + Matern32Kernel()),

                SEKernel() ∘ ScaleTransform(T(0.3)),

                with_lengthscale(SEKernel(), T(0.5)),
            ],
        )
            TestUtils.test_gpu_against_cpu(StableRNG(123456), T, kernel, vec_type)
        end

        # Kernels which only really work for 1D inputs.
        @testset "(CuVector{$T}, $kernel)" for kernel in [
            WienerKernel{0}(), WienerKernel{1}(), WienerKernel{2}(), WienerKernel{3}()
        ]
            TestUtils.test_gpu_against_cpu(StableRNG(123456), T, kernel, CuVector{T})
        end

        # Kernels which only work for multi-dimensional inputs.
        @testset "$vec_type, $kernel" for (vec_type, kernel) in Iterators.product(
            [ColVecs{T, CuMatrix{T}}, RowVecs{T, CuMatrix{T}}],
            [
                KernelTensorProduct(SEKernel(), Matern32Kernel()),
                TransformedKernel(SEKernel(), LinearTransform(CuArray(randn(T, 3, 2)))),
                TransformedKernel(
                    SEKernel(),
                    ChainTransform((
                        LinearTransform(CuArray(randn(T, 2, 2))),
                        LinearTransform(CuArray(randn(T, 3, 2))),
                    )),
                ),
                TransformedKernel(SEKernel(), SelectTransform([1])),
            ],
        )
            TestUtils.test_gpu_against_cpu(StableRNG(123456), T, kernel, vec_type)
        end
    end
end
