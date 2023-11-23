@testset "diffKernel" begin
    @testset "smoke test" begin
        k = MaternKernel()
        k(1, 1)
        k(1, DiffPt(1, (1, 1))) # Cov(Z(x), ∂₁∂₁Z(y)) where x=1, y=1
        k(DiffPt([1], 1), [2]) # Cov(∂₁Z(x), Z(y)) where x=[1], y=[2]
        k(DiffPt([1, 2], 1), DiffPt([1, 2], 2))# Cov(∂₁Z(x), ∂₂Z(y)) where x=[1,2], y=[1,2]
    end

    @testset "Sanity Checks with $k" for k in [SEKernel()]
        for x in [0, 1, -1, 42]
            # for stationary kernels Cov(∂Z(x) , Z(x)) = 0
            @test k(DiffPt(x, 1), x) ≈ 0

            # the slope should be positively correlated with a point further down
            @test k(
                DiffPt(x, 1), # slope
                x + 1e-1, # point further down
            ) > 0

            # correlation with self should be positive
            @test k(DiffPt(x, 1), DiffPt(x, 1)) > 0
        end
    end
end
