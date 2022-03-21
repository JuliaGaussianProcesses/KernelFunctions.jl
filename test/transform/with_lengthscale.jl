@testset "with_lengthscale" begin
    @testset "ScaleTransform" begin
        l = exp(rand())
        kernel = @inferred(with_lengthscale(SqExponentialKernel(), l))

        @test kernel isa TransformedKernel{<:SqExponentialKernel,<:ScaleTransform}
        @test kernel.transform.s[1] ≈ inv(l)
    end

    @testset "ARDTransform" begin
        l = map(exp, rand(5))
        kernel = @inferred(with_lengthscale(SqExponentialKernel(), l))

        @test kernel isa TransformedKernel{<:SqExponentialKernel,<:ARDTransform}
        @test kernel.transform.v ≈ map(inv, l)
    end
end
