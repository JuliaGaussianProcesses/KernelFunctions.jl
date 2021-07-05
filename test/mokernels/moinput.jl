@testset "moinput" begin
    x = [rand(5) for _ in 1:4]
    type_1 = AbstractVector{Tuple{Vector{Float64},Int}}
    type_2 = AbstractVector{Tuple{AbstractVector{Vector{Float64}},Int}}

    @testset "isotopicbyoutputs" begin
        ibo = isotopic_by_outputs(x, 3)

        @test ibo == KernelFunctions.IsotopicByOutputs(x, 3)

        @test isa(ibo, type_1) == true
        @test isa(ibo, type_2) == false

        @test length(ibo) == 12
        @test size(ibo) == (12,)
        @test size(ibo, 1) == 12
        @test size(ibo, 2) == 1
        @test lastindex(ibo) == 12
        @test firstindex(ibo) == 1
        @test_throws BoundsError ibo[0]

        @test ibo[2] == (x[2], 1)
        @test ibo[5] == (x[1], 2)
        @test ibo[7] == (x[3], 2)
        @test all([(x_, i) for i in 1:3 for x_ in x] .== ibo)
    end

    @testset "isotopicbyfeatures" begin
        ibf = isotopic_by_features(x, 3)

        @test ibf == KernelFunctions.IsotopicByFeatures(x, 3)

        @test isa(ibf, type_1) == true
        @test isa(ibf, type_2) == false

        @test length(ibf) == 12
        @test size(ibf) == (12,)
        @test size(ibf, 1) == 12
        @test size(ibf, 2) == 1
        @test lastindex(ibf) == 12
        @test firstindex(ibf) == 1
        @test_throws BoundsError ibf[0]

        @test ibf[2] == (x[1], 2)
        @test ibf[5] == (x[2], 2)
        @test ibf[7] == (x[3], 1)
        @test all([(x_, i) for x_ in x for i in 1:3] .== ibf)
    end
end
