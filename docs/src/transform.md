# Input Transforms

`Transform` is the object that takes care of transforming the input data before distances are being computed. It can be as standard as `IdentityTransform` returning the same input, or multiplying the data by a scalar with `ScaleTransform` or by a vector with `ARDTransform`.
There is a more general `Transform`: `FunctionTransform` that uses a function and applies it on each vector via `mapslices`.
You can also create a pipeline of `Transform` via `TransformChain`. For example, `LowRankTransform(rand(10,5))∘ScaleTransform(2.0)`.

A transformation can be applied to a matrix or a vector via `KernelFunctions.apply(t::Transform, v::AbstractVecOrMat)`

Check the list on the [API page](@ref Transforms)
