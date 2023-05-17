using OneHotArrays: OneHotVector
import ForwardDiff as FD
import LinearAlgebra as LA

""" 
	DiffPt(x; partial=())

For a covariance kernel k of GP Z, i.e.
```julia
	k(x,y) # = Cov(Z(x), Z(y)),
```
a DiffPt allows the differentiation of Z, i.e.	
```julia
	k(DiffPt(x, partial=1), y) # = Cov(∂₁Z(x), Z(y))
```
for higher order derivatives partial can be any iterable, i.e.
```julia
	k(DiffPt(x, partial=(1,2)), y) # = Cov(∂₁∂₂Z(x), Z(y))
```
"""
struct DiffPt{Dim}
	pos # the actual position
	partial
end

DiffPt(x;partial=()) = DiffPt{length(x)}(x, partial) # convenience constructor

"""
Take the partial derivative of a function `fun`  with input dimesion `dim`.
If partials=(i,j), then (∂ᵢ∂ⱼ fun) is returned.
"""
function partial(fun, dim, partials=())
	if !isnothing(local next = iterate(partials))
		idx, state = next
		return partial(
			x -> FD.derivative(0) do dx
				fun(x .+ dx * OneHotVector(idx, dim))
			end,
			dim,
			Base.rest(partials, state),
		)
	end
	return fun 
end

"""
Take the partial derivative of a function with two dim-dimensional inputs,
i.e. 2*dim dimensional input
"""
function partial(k, dim; partials_x=(), partials_y=())
	local f(x,y) = partial(t -> k(t,y), dim, partials_x)(x)
	return (x,y) -> partial(t -> f(x,t), dim, partials_y)(y)
end




"""
	_evaluate(k::T, x::DiffPt{Dim}, y::DiffPt{Dim}) where {Dim, T<:Kernel}

implements `(k::T)(x::DiffPt{Dim}, y::DiffPt{Dim})` for all kernel types. But since
generics are not allowed in the syntax above by the dispatch system, this
redirection over `_evaluate` is necessary

unboxes the partial instructions from DiffPt and applies them to k,
evaluates them at the positions of DiffPt
"""
function _evaluate(k::T, x::DiffPt{Dim}, y::DiffPt{Dim}) where {Dim, T<:Kernel}
	return partial(
		k, Dim,
		partials_x=x.partial, partials_y=y.partial
	)(x.pos, y.pos)
end



#=
This is a hack to work around the fact that the `where {T<:Kernel}` clause is
not allowed for the `(::T)(x,y)` syntax. If we were to only implement
```julia
	(::Kernel)(::DiffPt,::DiffPt)
```
then julia would not know whether to use
`(::SpecialKernel)(x,y)` or `(::Kernel)(x::DiffPt, y::DiffPt)`
```
To avoid this hack, no kernel type T should implement
```julia
	(::T)(x,y)
```
and instead implement
```julia
	_evaluate(k::T, x, y)
```
Then there should be only a single
```julia
	(k::Kernel)(x,y) = evaluate(k, x, y)
```
which all the kernels would fall back to.

This ensures that evaluate(k::T, x::DiffPt{Dim}, y::DiffPt{Dim}) is always
more specialized and call beforehand.
=#
for T in [SimpleKernel, Kernel] #subtypes(Kernel)
	(k::T)(x::DiffPt{Dim}, y::DiffPt{Dim}) where {Dim} = evaluate(k, x, y)
	(k::T)(x::DiffPt{Dim}, y) where {Dim} = evaluate(k, x, DiffPt(y))
	(k::T)(x, y::DiffPt{Dim}) where {Dim} = evaluate(k, DiffPt(x), y)
end

