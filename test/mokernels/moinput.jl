@testset "moinput" begin
    
    x =  [rand(5) for _ in 1:4]
    mgpi = moinput(x, 3)
    
    @test length(mgpi) == 12
    @test size(mgpi) == (12,)
    @test size(mgpi, 1) == 12
    @test size(mgpi, 2) == 1
    @test lastindex(mgpi) == 12
    @test firstindex(mgpi) == 1
    @test iterate(mgpi) == (mgpi[1], (Base.OneTo(12), 1))
    @test iterate(mgpi, 2) == (mgpi[2], (2, nothing))
    @test_throws BoundsError mgpi[0]

    @test mgpi[2] == (x[2], 1)
    @test mgpi[5] == (x[1], 2)
    @test mgpi[7] == (x[3], 2)
    @test all([(x_, i) for i in 1:3 for x_ in x ] .== mgpi)

end
