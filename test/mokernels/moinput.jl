@testset "moinput" begin
    x = [rand(5) for _ in 1:4]
    type_1 =  AbstractVector{Tuple{Vector{Float64},Int}}
    type_2 = AbstractVector{Tuple{AbstractVector{Vector{Float64}},Int}}

    @testset "isotopicbyoutputs" begin
        mgpi_bo = IsotopicByOutputs(x, 3)

        @test isa(mgpi_bo, type_1) == true
        @test isa(mgpi_bo, type_2) == false

        @test length(mgpi_bo) == 12
        @test size(mgpi_bo) == (12,)
        @test size(mgpi_bo, 1) == 12
        @test size(mgpi_bo, 2) == 1
        @test lastindex(mgpi_bo) == 12
        @test firstindex(mgpi_bo) == 1
        @test_throws BoundsError mgpi_bo[0]

        @test mgpi_bo[2] == (x[2], 1)
        @test mgpi_bo[5] == (x[1], 2)
        @test mgpi_bo[7] == (x[3], 2)
        @test all([(x_, i) for i in 1:3 for x_ in x] .== mgpi_bo)
    end

    @testset "isotopicbyfeatures" begin
        mgpi_bf = IsotopicByFeatures(x, 3)

        @test isa(mgpi_bf, type_1) == true
        @test isa(mgpi_bf, type_2) == false

        @test length(mgpi_bf) == 12
        @test size(mgpi_bf) == (12,)
        @test size(mgpi_bf, 1) == 12
        @test size(mgpi_bf, 2) == 1
        @test lastindex(mgpi_bf) == 12
        @test firstindex(mgpi_bf) == 1
        @test_throws BoundsError mgpi_bf[0]

        @test mgpi_bf[2] == (x[1], 2)
        @test mgpi_bf[5] == (x[2], 2)
        @test mgpi_bf[7] == (x[3], 1)
        @test all([(x_, i) for x_ in x for i in 1:3] .== mgpi_bf)
    end
end
