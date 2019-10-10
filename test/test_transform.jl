using Test
using KernelFunctions
using Random: seed!

seed!(42)

dims = (10,5)
X = rand(dims...)
##
s = 3.0
v1 = vcat(3.0,4.0*ones(dims[2]-1))
v2 = vcat(3.0,4.0*ones(dims[1]-1))
t = ScaleTransform(s)
vt1 = ScaleTransform(v1)
vt2 = ScaleTransform(v2)
@test all(KernelFunctions.transform(t,X).==s*X)
@test all(KernelFunctions.transform(vt1,X,1).==v1'.*X)
@test all(KernelFunctions.transform(vt2,X,2).==v2.*X)
##
P = rand(5,10)
tp = LowRankTransform(P)
@test all(KernelFunctions.transform(tp,X,2).==P*X)
##
f(x) = sin.(x)
tf = FunctionTransform(f)
KernelFunctions.transform(tf,X,1)
@test all(KernelFunctions.transform(tf,X,1).==f(X))
##
tchain = TransformChain([t,tp,tf])
t∘tp∘tf
TransformChain([t,tp])
@test all(KernelFunctions.transform(tchain,X,2).==f(P*(s*X)))
