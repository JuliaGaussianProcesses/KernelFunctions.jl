using KernelFunctions

SUITE["KernelFunctions"] = BenchmarkGroup()

kernelnames = ["SquaredExponentialKernel"]
kerneltypes = ["ARD","ISO"]
kernels=Dict{String,Dict{String,KernelFunctions.Kernel}}()
for k in kernelnames
    kernels[k] = Dict{String,KernelFunctions.Kernel}()
    SUITE["KernelFunctions"][k] = BenchmarkGroup()
    for kt in kerneltypes
        SUITE["KernelFunctions"][k][kt] = BenchmarkGroup()
        kernels[k][kt] = eval(Meta.parse("KernelFunctions."*k*"("*(kt == "ARD" ? "alpha*ones(Float64,dim)" : "alpha" )*")"))
    end
end

for k in kernelnames
    for kt in kerneltypes
        SUITE["KernelFunctions"][k][kt]["k(X,Y)"] = @benchmarkable kernelmatrix($(kernels[k][kt]),$X,$Y,obsdim=1)
        # SUITE["KernelFunctions"][k][kt]["k!(X,Y)"] = @benchmarkable kernelmatrix!(KXY,$(kernels[k][kt]),$X,$Y) setup=(KXY=copy($KXY))
        SUITE["KernelFunctions"][k][kt]["k(X)"] = @benchmarkable kernelmatrix($(kernels[k][kt]),$X,obsdim=1)
        # SUITE["KernelFunctions"][k][kt]["k!(X)"] = @benchmarkable kernelmatrix!(KX,$(kernels[k][kt]),$X) setup=(KX=copy($KX))
        # SUITE["KernelFunctions"][k][kt]["kdiag(X)"] = @benchmarkable kerneldiagmatrix($(kernels[k][kt]),$X)
        # SUITE["KernelFunctions"][k][kt]["kdiag!(X)"] = @benchmarkable kerneldiagmatrix!(kX,$(kernels[k][kt]),$X) setup=(kX=copy($kX))
    end
end
# results = run(SUITE)
