using TestEnv; TestEnv.activate()
using Distances: pairwise
using KernelFunctions: Sinus
using Zygote
using KernelFunctions.ChainRulesCore, Distances
using Test

function _pairwise!(metric::Sinus, r::AbstractMatrix, a::AbstractMatrix)
    require_one_based_indexing(r)
    n = size(a, 2)
    size(r) == (n, n) || throw(DimensionMismatch("Incorrect size of r."))
    @inbounds for j = 1:n
        for i = 1:(j - 1)
            r[i, j] = r[j, i]   # leveraging the symmetry of SemiMetric
        end
        r[j, j] = zero(eltype(r))
        aj = view(a, :, j)
        for i = (j + 1):n
            s = sinpi.(view(a, :, i) - aj) ./ p
            r[i, j] = dot(s, s)
        end
    end
    r
end


function ChainRulesCore.rrule(
    ::typeof(Distances.pairwise),
    d::Sinus,
    x::AbstractMatrix;
    dims = 2
)
    project_x = ProjectTo(x)
    function pairwise_pullback(Δ)
        n = size(x, 2)
        x̄ = zero(x)
        @inbounds for j in 1:n, i in 1:n
            xi = view(x, :, i)
            xj = view(x, :, j)
            ds = Δ[i, j] .* sinpi.(xi - xj) .* cospi.(xi - xj) ./ d.r
            x̄[:, i] = ds
            x̄[:, j] = -ds
        end
        NoTangent(), NoTangent(), x̄
    end
    return Distances.pairwise(d, x), pairwise_pullback
end
Zygote.refresh()

@testset "pairwise" begin

@testset "Single argument, vector" begin
    @testset "p = $p" for p in rand(5)
        testfun1(x) = sum(pairwise(Sinus(p), x))
        testfun_simple1(x) = sum(abs2.(sinpi.(x .- x') ./ p))
        x1 = rand(100)

        z1 = testfun1(x1)
        z_simple1 = testfun_simple1(x1)

        @test z1 ≈ z_simple1

        g1 = Zygote.gradient(testfun1, x1) |> only
        g_simple1 = Zygote.gradient(testfun_simple1, x1) |> only

        @test g1 ≈ g_simple1
    end
end

@testset "Double argument, vector" begin
    @testset "p = $p" for p in rand(5)
        testfun2(x, y) = sum(pairwise(Sinus(p), x, y))
        testfun_simple2(x, y) = sum(abs2.(sinpi.(x .- y') ./ p))
        x2 = rand(150)

        z2 = testfun2(x2[1:100], x2[101:150])
        z_simple2 = testfun_simple2(x2[1:100], x2[101:150])

        @test z2 ≈ z_simple2

        g2 = Zygote.gradient(_x -> testfun2(_x[1:100], _x[101:150]), x2) |> only
        g_simple2 = Zygote.gradient(_x -> testfun_simple2(_x[1:100], _x[101:150]), x2) |> only

        @test g2 ≈ g_simple2
    end
end

@testset "Single argument, matrix, dims = 2" begin
    @testset "p = $p" for p in [rand(2) for i in 1:5]
        testfun3(x) = sum(pairwise(Sinus(p), x; dims=2))
        function testfun_simple3(a)
            n = size(a, 2)
            r = zeros(n, n)
            @inbounds for j = 1:n
                for i = 1:(j - 1)
                    r[i, j] = r[j, i]   # leveraging the symmetry of SemiMetric
                end
                r[j, j] = zero(eltype(r))
                aj = view(a, :, j)
                for i = (j + 1):n
                    s = sinpi.(view(a, :, i) - aj) ./ p
                    r[i, j] = dot(s, s)
                end
            end
            sum(r)
        end
        x3 = rand(2, 100)

        z3 = testfun3(x3)
        z_simple3 = testfun_simple3(x3)

        @test z3 ≈ z_simple3

        g1 = Zygote.gradient(testfun3, x3) |> only
        # g_simple1 = Zygote.gradient(testfun_simple3, x3) |> only
        @show g1
        # @test g1 ≈ g_simple1
    end
end

end
