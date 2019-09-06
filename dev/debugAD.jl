using KernelFunctions
using Zygote, ForwardDiff, Tracker
using Test, LinearAlgebra

dims = [10,5]
A = rand(dims...)
B = rand(dims...)
K = [zeros(dims[1],dims[1]),zeros(dims[2],dims[2])]
l = 0.1
vl = l*ones(dims[1])
testfunction(k,A,B) = det(kernelmatrix(k,A,B))
testfunction(k,A) = sum(kernelmatrix(k,A))
k = MaternKernel(vl)
KernelFunctions.kappa(k,3)
testfunction(SquaredExponentialKernel(vl),A)
testfunction(MaternKernel(vl),A)
@which kernelmatrix(MaternKernel(vl),A,B)
#For debugging
@info "Running Zygote gradients"
Zygote.refresh()
## Zygote
Zygote.gradient(x->testfunction(SquaredExponentialKernel(x),A),vl)
Zygote.gradient(x->testfunction(MaternKernel(x),A),vl)
Zygote.gradient(x->testfunction(SquaredExponentialKernel(x),A,B),vl)[1]
Zygote.gradient(x->testfunction(MaternKernel(x),A,B),vl)[1]
Zygote.gradient(x->testfunction(SquaredExponentialKernel(x),A,B),l)
Zygote.gradient(x->testfunction(MaternKernel(x),A,B),l)
Zygote.gradient(x->testfunction(SquaredExponentialKernel(x),A),l)
Zygote.gradient(x->testfunction(MaternKernel(x),A),l)
Zygote.gradient(x->testfunction(MaternKernel(x),A),l)
Zygote.gradient(x->kernelmatrix(MaternKernel(x,1.0),A)[1],l)
@info "Running Tracker gradients"
## Tracker
# Tracker.gradient(x->testfunction(SquaredExponentialKernel(vl),x,B),A)
# Tracker.gradient(x->testfunction(SquaredExponentialKernel(l),x[:,:]),A)
# # Tracker.gradient(x->testfunction(SquaredExponentialKernel(x),A,B),vl)
# Tracker.gradient(x->testfunction(SquaredExponentialKernel(x),A),vl)
# Tracker.gradient(x->testfunction(SquaredExponentialKernel(x),A,B),l)
# Tracker.gradient(x->testfunction(SquaredExponentialKernel(x),A),l)

@info "Running ForwardDiff gradients"
## ForwardDiff
ForwardDiff.gradient(x->testfunction(SquaredExponentialKernel(x),A,B),vl) #✓
ForwardDiff.gradient(x->testfunction(MaternKernel(x),A,B),vl) #✓
ForwardDiff.gradient(x->testfunction(SquaredExponentialKernel(x),A),vl) #✓
ForwardDiff.gradient(x->testfunction(MaternKernel(x),A),vl) #✓
ForwardDiff.gradient(x->testfunction(SquaredExponentialKernel(x[1]),A,B),[l])
ForwardDiff.gradient(x->testfunction(MaternKernel(x[1]),A,B),[l])
ForwardDiff.gradient(x->testfunction(SquaredExponentialKernel(x[1]),A),[l])
ForwardDiff.gradient(x->testfunction(MaternKernel(x[1]),A),[l])
