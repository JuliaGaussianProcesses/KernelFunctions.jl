using Plots; pyplot();
using Distributions
using LinearAlgebra
using KernelFunctions
# Translational invariants kernels

default(lw=3.0,titlefontsize=28,tickfontsize=18)

x₀ = 0.0; l=0.1
n_grid = 101
fill(x₀,n_grid,1)
xrange = reshape(collect(range(-3,3,length=n_grid)),:,1)

k = SqExponentialKernel(1.0)
K1 = kernelmatrix(k,xrange,obsdim=1)
p = heatmap(K1,yflip=true,colorbar=false,framestyle=:none,background_color=RGBA(0.0,0.0,0.0,0.0))
savefig(joinpath(@__DIR__,"src","assets","heatmap_sqexp.png"))


k = Matern32Kernel(FunctionTransform(x->(sin.(x)).^2))
K2 = kernelmatrix(k,xrange,obsdim=1)
p = heatmap(K2,yflip=true,colorbar=false,framestyle=:none,background_color=RGBA(0.0,0.0,0.0,0.0))
savefig(joinpath(@__DIR__,"src","assets","heatmap_matern.png"))


k = PolynomialKernel(LowRankTransform(randn(3,1)),2.0,0.0)
K3 = kernelmatrix(k,xrange,obsdim=1)
p = heatmap(K3,yflip=true,colorbar=false,framestyle=:none,background_color=RGBA(0.0,0.0,0.0,0.0))
savefig(joinpath(@__DIR__,"src","assets","heatmap_poly.png"))

k = 0.5*SqExponentialKernel()*LinearKernel(0.5) + 0.4*Matern32Kernel(FunctionTransform(x->sin.(x)))
K4 = kernelmatrix(k,xrange,obsdim=1)
p = heatmap(K4,yflip=true,colorbar=false,framestyle=:none,background_color=RGBA(0.0,0.0,0.0,0.0))
savefig(joinpath(@__DIR__,"src","assets","heatmap_prodsum.png"))

plot(heatmap.([K1,K2,K3,K4],yflip=true,colorbar=false)...,layout=(2,2))
savefig(joinpath(@__DIR__,"src","assets","heatmap_combination.png"))


for k in [SqExponentialKernel,ExponentialKernel]
    K = kernelmatrix(k(),xrange,obsdim=1)
    v = rand(MvNormal(K+1e-7I))
    plot(xrange,v,lab="",title="f(x)",framestyle=:none) |> display
    savefig(joinpath(@__DIR__,"src","assets","GP_sample_$(k).png"))
    plot(xrange,kernel.(k(),x₀,xrange),lab="",ylims=(0,1.1),title="k(0,x)") |> display
    savefig(joinpath(@__DIR__,"src","assets","kappa_function_$(k).png"))
end

for k in [GammaExponentialKernel(1.0,1.5)]
    sparse =1
    while !isposdef(kernelmatrix(k,xrange*sparse,obsdim=1) + 1e-5I); sparse += 1; end
    v = rand(MvNormal(kernelmatrix(k,xrange*sparse,obsdim=1)+1e-7I))
    plot(xrange,v,lab="",title="f(x)",framestyle=:none) |> display
    savefig(joinpath(@__DIR__,"src","assets","GP_sample_GammaExponentialKernel.png"))
    plot(xrange,kernel.(k,x₀,xrange),lab="",ylims=(0,1.1),title="k(0,x)") |> display
    savefig(joinpath(@__DIR__,"src","assets","kappa_function_GammaExponentialKernel.png"))
end

for k in [MaternKernel,Matern32Kernel,Matern52Kernel]
    K = kernelmatrix(k(),xrange,obsdim=1)
    v = rand(MvNormal(K+1e-7I))
    plot(xrange,v,lab="",title="f(x)",framestyle=:none) |> display
    savefig(joinpath(@__DIR__,"src","assets","GP_sample_$(k).png"))
    plot(xrange,kernel.(k(),x₀,xrange),lab="",ylims=(0,1.1),title="k(0,x)") |> display
    savefig(joinpath(@__DIR__,"src","assets","kappa_function_$(k).png"))
end


for k in [RationalQuadraticKernel]
    K = kernelmatrix(k(),xrange,obsdim=1)
    v = rand(MvNormal(K+1e-7I))
    plot(xrange,v,lab="",title="f(x)",framestyle=:none) |> display
    savefig(joinpath(@__DIR__,"src","assets","GP_sample_$(k).png"))
    plot(xrange,kernel.(k(),x₀,xrange),lab="",ylims=(0,1.1),title="k(0,x)") |> display
    savefig(joinpath(@__DIR__,"src","assets","kappa_function_$(k).png"))
end
