allapprox(x,y,tol=1e-8) = all(isapprox.(x,y,atol=tol))

function kappa_AD(::Val{:Zygote},k::Kernel,d::Real)
    first(Zygote.gradient(x->kappa(k,x),d))
end

function kappa_AD(::Val{:ForwardDiff},k::Kernel,d::Real)
    first(ForwardDiff.gradient(x->kappa(k,first(x)),[d]))
end

function kappa_fdm(k::Kernel,d::Real)
    central_fdm(5,1)(x->kappa(k,x),d)
end


function transform_AD(::Val{:Zygote},t::Transform,A)
    ps = KernelFunctions.params(t)
    @test allisapprox(first(Zygote.gradient(p->transform_with_duplicate(p,t,A),ps)),
            central_fdm(5,1)(p->transform_with_duplicate(p,t,A),ps))
    @test allisapprox(first(Zygote.gradient(X->sum(transform(t,X,2)),A))
            .≈ central_fdm(5,1)(X->sum(transform(t,X,2)),A))
end

function transform_AD(::Val{:ForwardDiff},t::Transform,A)
    ps = KernelFunctions.params(t)
    if t isa ScaleTransform
        @test allisapprox(first(ForwardDiff.gradient(p->transform_with_duplicate(first(p),t,A),[ps])),
            central_fdm(5,1)(p->transform_with_duplicate(p,t,A),ps))
    else
        @test allisapprox(ForwardDiff.gradient(p->transform_with_duplicate(p,t,A),ps),
            central_fdm(5,1)(p->transform_with_duplicate(p,t,A),ps))
    end
    @test allisapprox(ForwardDiff.gradient(X->sum(transform(t,X,2)),A),
            central_fdm(5,1)(X->sum(transform(t,X,2)),A))
end

transform_with_duplicate(p,t,A) = sum(transform(KernelFunctions.duplicate(t,p),A,2))
