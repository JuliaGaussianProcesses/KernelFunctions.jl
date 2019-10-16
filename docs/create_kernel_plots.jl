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
