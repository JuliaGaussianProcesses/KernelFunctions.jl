using KernelFunctions
using Stheno
using Stheno: pw
using BenchmarkTools
using Zygote

# Ds = [1,2,5,10,20,50,100,200,500,1000]
Ds = [1,10,100,1000]
timestheno = zeros(Float64,length(Ds)); memstheno = similar(timestheno)
timekf = similar(timestheno); memkf = similar(timestheno)
@progress for (i,D) in enumerate(Ds)

    A = randn(D,1000)
    B = randn(D,1001)

    # Standardised eq kernel with length-scale 0.1.
    medkf = median(@benchmark KernelFunctions.kernelmatrix(SquaredExponentialKernel(0.01),$A,$B,obsdim=2))
    timekf[i] = medkf.time/1e6; memkf[i] = medkf.memory/2^20
    medstheno = median(@benchmark pw(eq(; l=0.1), ColsAreObs($A), ColsAreObs($B)))
    timestheno[i] = medstheno.time/1e6; memstheno[i] = medstheno.memory/2^20
end

using Plots
ptime = plot(Ds,timestheno,lab="Stheno",xaxis=:log,xlabel="D",ylabel="t [ms]",title="Time")
plot!(Ds,timekf,lab="KernelFunctions")
pmem = plot(Ds,memstheno,lab="Stheno",xaxis=:log,xlabel="D",ylabel="Mem [MB]",title="Memory Usage")
plot!(Ds,memkf,lab="KernelFunctions")
plot(ptime,pmem)
