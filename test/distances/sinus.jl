@testset "sinus" begin
    A = rand(10)
    B = rand(10)
    p = rand(10)
    d = KernelFunctions.Sinus(p)
    @test Distances.parameters(d) == p
    @test evaluate(d, A, B) == sum(abs2.(sinpi.(A - B) ./ p))
    d1 = KernelFunctions.Sinus(first(p))
    @test d1(3.0, 2.0) == abs2(sinpi(3.0 - 2.0) / first(p))
end
