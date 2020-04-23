# minimal definition of a custom kernel
struct MyKernel <: SimpleKernel end

KernelFunctions.kappa(::MyKernel, d2::Real) = exp(-d2)
KernelFunctions.metric(::MyKernel) = SqEuclidean()

# some syntactic sugar
function (κ::MyKernel)(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    return KernelFunctions.kappa(Distances.evaluate(KernelFunctions.metric(κ), x, y))
end

@testset "custom" begin
    @test kappa(MyKernel(), 3) == kappa(SqExponentialKernel(), 3)
    @test kernelmatrix(MyKernel(), [1 2; 3 4], [5 6; 7 8]) == kernelmatrix(SqExponentialKernel(), [1 2; 3 4], [5 6; 7 8])
    @test kernelmatrix(MyKernel(), [1 2; 3 4]) == kernelmatrix(SqExponentialKernel(), [1 2; 3 4])

    @test MyKernel()([1, 2], [3, 4]) == SqExponentialKernel()([1, 2], [3, 4])
    @test MyKernel()([1 2; 3 4], [5 6; 7 8]) == SqExponentialKernel()([1 2; 3 4], [5 6; 7 8])
    @test MyKernel()([1 2; 3 4]) == SqExponentialKernel()([1 2; 3 4])
end
