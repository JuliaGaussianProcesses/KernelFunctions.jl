using MLKernels

SUITE["MLKernels"] = BenchmarkGroup()

mlkernelnames = ["SquaredExponentialKernel"]
kernels=Dict{String,MLKernels.Kernel}()
for k in mlkernelnames
    SUITE["MLKernels"][k] = BenchmarkGroup()
    kernels[k] = eval(Meta.parse("MLKernels."*k*"(alpha)"))
end

for k in mlkernelnames
    SUITE["MLKernels"][k]["k(X,Y)"] = @benchmarkable MLKernels.kernelmatrix($(kernels[k]),$X,$Y)
    # SUITE["MLKernels"][k][kt]["k!(X,Y)"] = @benchmarkable MLKernels.kernelmatrix!(KXY,$(kernels[k][kt]),$X,$Y) setup=(KXY=copy($KXY))
    SUITE["MLKernels"][k]["k(X)"] = @benchmarkable MLKernels.kernelmatrix($(kernels[k]),$X)
    # SUITE["MLKernels"][k][kt]["k!(X)"] = @benchmarkable MLKernels.kernelmatrix!(KX,$(kernels[k][kt]),$X) setup=(KX=copy($KX))
    # SUITE["MLKernels"][k][kt]["kdiag(X)"] = @benchmarkable MLKernels.kerneldiagmatrix($(kernels[k][kt]),$X)
    # SUITE["MLKernels"][k][kt]["kdiag!(X)"] = @benchmarkable MLKernels.kerneldiagmatrix!(kX,$(kernels[k][kt]),$X) setup=(kX=copy($kX))
end
# results = run(SUITE)
