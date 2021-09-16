@testset "mokernel" begin
    struct TestMOKernel <: MOKernel end
    @test_throws ArgumentError matrixkernel(TestMOKernel(), rand(3), rand(3))

    out_dim = 3
    A = rand(out_dim, out_dim)
    A = A * A'
    k = IntrinsicCoregionMOKernel(GaussianKernel(), A)

    in_dim = 4
    x = rand(in_dim)
    y = rand(in_dim)
    @test matrixkernel(k, x, y, 3) ≈ matrixkernel(k, x, y)
end
