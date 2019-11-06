using KernelFunctions

k = SqExponentialKernel()

@testset "Generic functions to test" begin
    @test length(k) == 1
    @test iterate(k) == (k,nothing)
    @test iterate(k,1) == nothing
end
