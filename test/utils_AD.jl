allapprox(x, y, tol = 1e-8) = all(isapprox.(x, y, atol = tol))
FDM = central_fdm(5, 1)

function gradient(::Val{:Zygote}, f::Function, args)
    first(Zygote.gradient(f, args))
end

function gradient(::Val{:Zygote}, f::Function, args::Zygote.Params)
    Zygote.gradient(f, args)
end

function gradient(::Val{:ForwardDiff}, f::Function, args)
    ForwardDiff.gradient(f, args)
end

function gradient(::Val{:ReverseDiff}, f::Function, args)
    ReverseDiff.gradient(f, args)
end

function gradient(::Val{:FiniteDiff}, f::Function, args)
    first(FiniteDifferences.grad(FDM, f, args))
end



function transform_AD(::Val{:Zygote}, t::Transform, A)
    ps = KernelFunctions.params(t)
    @test allapprox(
        first(Zygote.gradient(p -> transform_with_duplicate(p, t, A), ps)),
        first(FiniteDifferences.grad(
            FDM,
            p -> transform_with_duplicate(p, t, A),
            ps,
        )),
    )
    @test allapprox(
        first(Zygote.gradient(X -> sum(transform(t, X, 2)), A)),
        first(FiniteDifferences.grad(FDM, X -> sum(transform(t, X, 2)), A)),
    )
end

function transform_AD(::Val{:ForwardDiff}, t::Transform, A)
    ps = KernelFunctions.params(t)
    if t isa ScaleTransform
        @test allapprox(
            first(ForwardDiff.gradient(
                p -> transform_with_duplicate(first(p), t, A),
                [ps],
            )),
            first(FiniteDifferences.grad(
                FDM,
                p -> transform_with_duplicate(p, t, A),
                ps,
            )),
        )
    else
        @test allapprox(
            ForwardDiff.gradient(p -> transform_with_duplicate(p, t, A), ps),
            first(FiniteDifferences.grad(
                FDM,
                p -> transform_with_duplicate(p, t, A),
                ps,
            )),
        )
    end
    @test allapprox(
        ForwardDiff.gradient(X -> sum(transform(t, X, 2)), A),
        first(FiniteDifferences.grad(FDM, X -> sum(transform(t, X, 2)), A)),
    )
end
