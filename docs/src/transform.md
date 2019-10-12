# Transform

`Transform` is the object that takes care of transforming the input data before distances are being computed. It can be as standard as `IdentityTransform` returning the same input, can be a scalar with `ScaleTransform` multiplying the vectors by a scalar or a vector.
There is a more general `Transform`: `FunctionTransform` that uses a function and apply it on each vector via `mapslices`.
You can also create a pipeline of `Transform` via `TransformChain`. For example `LowRankTransform(rand(10,5))∘ScaleTransform(2.0)`.

One apply a transformation on a matrix or a vector via `transform(t::Transform,v::AbstractVecOrMat)`

## Transforms :

```@docs
  IdentityTransform
  ScaleTransform
  LowRankTransform
  FunctionTransform
  ChainTransform
```
