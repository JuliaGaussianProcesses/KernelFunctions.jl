@testset "moinput" begin
    x = [rand(5) for _ in 1:4]
    type_1 = AbstractVector{Tuple{Vector{Float64},Int}}
    type_2 = AbstractVector{Tuple{AbstractVector{Vector{Float64}},Int}}

    @testset "isotopicbyoutputs_and_isotopicbyfeatures" begin

        ibo = KernelFunctions.MOInputIsotopicByOutputs(x, 3)
        ibo2 = KernelFunctions.MOInputIsotopicByOutputs(x, 2)
        ibf = KernelFunctions.MOInputIsotopicByFeatures(x, 3)

        @test_throws DimensionMismatch vcat(ibo, ibo2)
        @test_throws TypeError vcat(ibo, ibf)

        @testset "isotopicbyoutputs" begin

            @test ibo == MOInput(x, 3)

            @test isa(ibo, type_1) == true
            @test isa(ibo, type_2) == false

            @test length(ibo) == 12
            @test size(ibo) == (12,)
            @test size(ibo, 1) == 12
            @test size(ibo, 2) == 1
            @test lastindex(ibo) == 12
            @test firstindex(ibo) == 1
            @test_throws BoundsError ibo[0]
            @test vcat(ibo, ibo) == MOInput(vcat(x, x), 3)

            @test ibo[2] == (x[2], 1)
            @test ibo[5] == (x[1], 2)
            @test ibo[7] == (x[3], 2)
            @test all([(x_, i) for i in 1:3 for x_ in x] .== ibo)
        end

        @testset "isotopicbyfeatures" begin

            @test isa(ibf, type_1) == true
            @test isa(ibf, type_2) == false

            @test length(ibf) == 12
            @test size(ibf) == (12,)
            @test size(ibf, 1) == 12
            @test size(ibf, 2) == 1
            @test lastindex(ibf) == 12
            @test firstindex(ibf) == 1
            @test_throws BoundsError ibf[0]
            @test vcat(ibf, ibf) ==
                KernelFunctions.MOInputIsotopicByFeatures(vcat(x, x), 3)

            @test ibf[2] == (x[1], 2)
            @test ibf[5] == (x[2], 2)
            @test ibf[7] == (x[3], 1)
            @test all([(x_, i) for x_ in x for i in 1:3] .== ibf)
        end
    end

    @testset "prepare_isotopic_multi_output_data" begin
        @testset "ColVecs" begin
            N = 5
            P = 3

            x = randn(N)
            y = ColVecs(randn(P, N))

            x_canon, y_canon = prepare_isotopic_multi_output_data(x, y)

            @test x_canon isa KernelFunctions.MOInputIsotopicByFeatures
            @test length(x_canon) == N * P
            @test y_canon isa AbstractVector{<:Real}
            @test length(y_canon) == length(x_canon)
        end
        @testset "RowVecs" begin
            N = 5
            P = 3

            x = randn(N)
            y = RowVecs(randn(N, P))

            x_canon, y_canon = prepare_isotopic_multi_output_data(x, y)

            @test x_canon isa KernelFunctions.MOInputIsotopicByOutputs
            @test length(x_canon) == N * P
            @test y_canon isa AbstractVector{<:Real}
            @test length(y_canon) == length(x_canon)
        end
    end
end
