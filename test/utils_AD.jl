allapprox(x,y,tol=1e-8) = all(isapprox.(x,y,atol=tol))
FDM = central_fdm(5,1)


function kappa_AD(::Val{:Zygote},k::Kernel,d::Real)
    first(Zygote.gradient(x->kappa(k,x),d))
end

function kappa_AD(::Val{:ForwardDiff},k::Kernel,d::Real)
    first(ForwardDiff.gradient(x->kappa(k,first(x)),[d]))
end

function kappa_fdm(k::Kernel,d::Real)
    first(FiniteDifferences.grad(FDM,x->kappa(k,x),d))
end


function transform_AD(::Val{:Zygote},t::Transform,A)
    ps = KernelFunctions.params(t)
    @test allapprox(first(Zygote.gradient(p->transform_with_duplicate(p,t,A),ps)),
        first(FiniteDifferences.grad(FDM,p->transform_with_duplicate(p,t,A),ps)))
    @test allapprox(first(Zygote.gradient(X->sum(transform(t,X,2)),A)),
            first(FiniteDifferences.grad(FDM,X->sum(transform(t,X,2)),A)))
end

function transform_AD(::Val{:ForwardDiff},t::Transform,A)
    ps = KernelFunctions.params(t)
    if t isa ScaleTransform
        @test allapprox(first(ForwardDiff.gradient(p->transform_with_duplicate(first(p),t,A),[ps])),
            first(FiniteDifferences.grad(FDM,p->transform_with_duplicate(p,t,A),ps)))
    else
        @test allapprox(ForwardDiff.gradient(p->transform_with_duplicate(p,t,A),ps),
            first(FiniteDifferences.grad(FDM,p->transform_with_duplicate(p,t,A),ps)))
    end
    @test allapprox(ForwardDiff.gradient(X->sum(transform(t,X,2)),A),
            first(FiniteDifferences.grad(FDM,X->sum(transform(t,X,2)),A)))
end

transform_with_duplicate(p,t,A) = sum(transform(KernelFunctions.duplicate(t,p),A,2))
