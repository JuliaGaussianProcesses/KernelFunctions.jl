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
